"""Wardrobe / outfit-suggestion tool. Follows tools/reminders.py's shape:
schema constants at module level, plain functions that open/close their
own core.db connection, and describe_* formatters for deterministic
(non-LLM) confirmation strings.

suggest_fit is the exception — it's registered with "speak_prose": "outfit"
instead of a describe_* formatter, so its result goes through
Orchestrator.format_outfit for spoken prose (see core/orchestrator.py).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

import config
from core import memory
from core.db import get_connection
from tools.weather import get_current_weather

ADD_WARDROBE_ITEM_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_wardrobe_items",
        "description": (
            "Save one or more new clothing items into the user's wardrobe. "
            "Call this when the user says they bought, got, or want to "
            "add garment(s) — e.g. 'add my navy peacoat to my wardrobe', "
            "'I got new white sneakers', 'add a black hat and a red "
            "scarf'. Do NOT call this to ask what to wear, or to change "
            "an item that's already saved."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "One entry per garment to add.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": config.WARDROBE_CATEGORIES,
                                "description": "Which slot the garment occupies.",
                            },
                            "name": {
                                "type": "string",
                                "description": (
                                    "Short name of the garment, e.g. 'navy wool peacoat'."
                                ),
                            },
                            "color": {
                                "type": "string",
                                "description": "Primary color. Empty string if unknown.",
                            },
                            "warmth_level": {
                                "type": "integer",
                                "description": (
                                    "1-5, 1 = lightest (tank top, shorts), 5 = heaviest "
                                    "(parka). 0 if unknown."
                                ),
                            },
                            "formality": {
                                "type": "integer",
                                "description": (
                                    "1-5, 1 = gym/loungewear, 5 = suit/formal. 0 if unknown."
                                ),
                            },
                            "tags": {
                                "type": "string",
                                "description": (
                                    "Comma-separated descriptors (fabric, fit, pattern). "
                                    "Empty string if none."
                                ),
                            },
                            "layer": {
                                "type": "string",
                                "enum": config.WARDROBE_LAYER_ROLES + [""],
                                "description": (
                                    "Only for 'top' category items meant to be layered on "
                                    "cold days: 'base' for a thin layer worn directly "
                                    "against skin (thermal, undershirt), 'mid' for an "
                                    "insulating layer worn over that (sweater, fleece). "
                                    "Empty string if not a layering piece."
                                ),
                            },
                        },
                        "required": ["category", "name"],
                    },
                },
            },
            "required": ["items"],
        },
    },
}

LIST_WARDROBE_ITEMS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_wardrobe_items",
        "description": (
            "List the clothes the user has already saved in their wardrobe, "
            "optionally filtered to one category. Call this when the user "
            "asks what clothes they own or what is in their closet — NOT "
            "when they ask what to wear."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": config.WARDROBE_CATEGORIES + ["all"],
                    "description": "Which category to list, or 'all'.",
                },
            },
            "required": ["category"],
        },
    },
}

UPDATE_WARDROBE_ITEM_SCHEMA = {
    "type": "function",
    "function": {
        "name": "update_wardrobe_items",
        "description": (
            "Update one or more items already saved in the user's "
            "wardrobe — change color, warmth, formality, tags, category, "
            "name, or layer role. Call this to edit, correct, or mark an "
            "existing item, e.g. 'set the peacoat's warmth to 5', 'mark "
            "my thermal shirt as a base layer', 'change my sweatshirt's "
            "color to blue'. Do NOT call this to add a brand new item."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "description": "One entry per item being changed.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {
                                "type": "integer",
                                "description": "The id of the item to change.",
                            },
                            "category": {
                                "type": "string",
                                "enum": config.WARDROBE_CATEGORIES,
                                "description": "New category. Omit if unchanged.",
                            },
                            "name": {
                                "type": "string",
                                "description": "New name. Omit if unchanged.",
                            },
                            "color": {
                                "type": "string",
                                "description": "New color. Omit if unchanged.",
                            },
                            "warmth_level": {
                                "type": "integer",
                                "description": "New warmth, 1-5. Omit if unchanged.",
                            },
                            "formality": {
                                "type": "integer",
                                "description": "New formality, 1-5. Omit if unchanged.",
                            },
                            "tags": {
                                "type": "string",
                                "description": (
                                    "New comma-separated tags (fabric, fit, pattern). "
                                    "Never put base/mid layer information here — that "
                                    "belongs in the 'layer' field instead. Omit if "
                                    "unchanged."
                                ),
                            },
                            "layer": {
                                "type": "string",
                                "enum": config.WARDROBE_LAYER_ROLES,
                                "description": (
                                    "This item's layering role for cold-weather "
                                    "outfits — 'base' for a base layer, 'mid' for a "
                                    "mid layer. Use THIS field, not 'tags', whenever "
                                    "the user says to mark/tag/set an item as a base "
                                    "or mid layer. Only meaningful for 'top' category "
                                    "items. Omit if unchanged."
                                ),
                            },
                        },
                        "required": ["item_id"],
                    },
                },
            },
            "required": ["updates"],
        },
    },
}

DELETE_WARDROBE_ITEM_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delete_wardrobe_items",
        "description": (
            "Delete or remove one or more items from the user's wardrobe "
            "by id, in a single call — e.g. 'remove wardrobe item 5', "
            "'delete items 5 and 6 from my wardrobe'. List every id "
            "mentioned in 'item_ids', even if there's only one. If an id "
            "isn't obvious from the conversation, call list_wardrobe_items "
            "first to look it up. Do NOT call this to add or list clothes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "The id(s) of the wardrobe item(s) to delete.",
                },
            },
            "required": ["item_ids"],
        },
    },
}

SUGGEST_FIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "suggest_fit",
        "description": (
            "Suggest an outfit to wear right now, chosen from the user's "
            "own saved wardrobe and the current local weather. Call this "
            "for any request about what to wear or what to put on — 'what "
            "should I wear today', 'pick me an outfit', 'do I need a "
            "jacket', 'what should I wear to dinner'. Do NOT call this to "
            "list or to add clothes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "occasion": {
                    "type": "string",
                    "description": (
                        "The occasion or activity the outfit is for, in the "
                        "user's own words (e.g. 'job interview', 'class', "
                        "'dinner date', 'the gym'). Use the empty string if "
                        "no occasion was mentioned."
                    ),
                },
            },
            "required": ["occasion"],
        },
    },
}

_WARDROBE_COLUMNS = (
    "id, category, name, color, warmth_level, formality, last_worn_at, tags, layer, embedding"
)


def _normalize_int(value: int | None, low: int = 1, high: int = 5) -> int | None:
    if not value or not (low <= value <= high):
        return None
    return value


def _normalize_layer(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().lower()
    return value if value in config.WARDROBE_LAYER_ROLES else None


def _infer_layer_from_tags(tags_value: str | None) -> str | None:
    """qwen2.5:7b sometimes writes 'base layer'/'mid layer' straight into
    the tags field instead of the dedicated layer field, despite the
    schema saying not to (observed empirically, even after tightening the
    schema description). Recover the intent here rather than relying on
    prompt wording alone to get it right every time."""
    if not tags_value:
        return None
    lowered = tags_value.lower()
    if "base layer" in lowered:
        return "base"
    if "mid layer" in lowered or "middle layer" in lowered:
        return "mid"
    return None


def _embedding_text(category: str, name: str, color: str | None, tags: str | None) -> str:
    parts = [name, category]
    if color:
        parts.append(color)
    if tags:
        parts.append(tags)
    return ", ".join(parts)


def add_wardrobe_item(
    category: str,
    name: str,
    color: str | None = None,
    warmth_level: int | None = None,
    formality: int | None = None,
    tags: str | None = None,
    layer: str | None = None,
) -> dict:
    """Insert a wardrobe item, embedding it before the insert so a failed
    embedding call leaves no partial row (same ordering as add_reminder
    parsing its datetime before opening a connection)."""
    category = (category or "").strip().lower()
    if category not in config.WARDROBE_CATEGORIES:
        raise ValueError(f"unknown category: {category!r}")

    name = (name or "").strip()
    if not name:
        raise ValueError("name is required")

    color = (color or "").strip() or None
    tags = (tags or "").strip() or None
    warmth_level = _normalize_int(warmth_level)
    formality = _normalize_int(formality)
    layer = _normalize_layer(layer)

    inferred_layer = _infer_layer_from_tags(tags)
    if inferred_layer:
        layer = layer or inferred_layer
        if tags and tags.lower() in (f"{inferred_layer} layer", "middle layer"):
            tags = None

    vector = memory.embed_text(_embedding_text(category, name, color, tags))
    blob = memory.pack_vector(vector)

    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO wardrobe_items "
            "(category, name, color, warmth_level, formality, tags, layer, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                category,
                name,
                color,
                warmth_level,
                formality,
                tags,
                layer,
                blob,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return {
            "id": cursor.lastrowid,
            "category": category,
            "name": name,
            "color": color,
            "warmth_level": warmth_level,
            "formality": formality,
            "tags": tags,
            "layer": layer,
        }
    finally:
        conn.close()


def describe_add_wardrobe_item(result: dict) -> str:
    return f"Added to your wardrobe: {result['name']} ({result['category']})."


def resolve_add_items(text: str, arguments: dict) -> dict:
    """Sanity-check add_wardrobe_items' extracted items against the
    user's own words. 'items' is a required schema field, so if the
    router ever misroutes a message into this tool (it should have gone
    to list/update/chat instead), extraction still has to produce
    *something* -- observed empirically to hallucinate a full copy of
    items visible in conversation history rather than an empty list,
    silently duplicating much of the wardrobe. Drop any item whose name
    doesn't actually appear in the user's current message, so a misroute
    produces a harmless empty add instead of a destructive one."""
    items = arguments.get("items") or []
    lowered_text = text.lower()
    kept = [item for item in items if (item.get("name") or "").strip().lower() in lowered_text]
    return {"items": kept}


def add_wardrobe_items(items: list[dict]) -> dict:
    """Registered tool entry point for one-or-more-items adds. Inserts
    each item via add_wardrobe_item individually — one bad item (a typo'd
    category, an empty name) is reported and skipped rather than losing
    the rest of the batch."""
    added = []
    failed = []
    for item in items or []:
        name = item.get("name") or "<unnamed>"
        try:
            added.append(
                add_wardrobe_item(
                    category=item.get("category", ""),
                    name=item.get("name", ""),
                    color=item.get("color"),
                    warmth_level=item.get("warmth_level"),
                    formality=item.get("formality"),
                    tags=item.get("tags"),
                    layer=item.get("layer"),
                )
            )
        except Exception as exc:
            failed.append({"name": name, "error": str(exc)})
    return {"added": added, "failed": failed}


def describe_add_wardrobe_items(result: dict) -> str:
    parts = []
    if result["added"]:
        names = ", ".join(f"{i['name']} ({i['category']})" for i in result["added"])
        parts.append(f"Added to your wardrobe: {names}.")
    if result["failed"]:
        names = ", ".join(f"{f['name']} ({f['error']})" for f in result["failed"])
        parts.append(f"Couldn't add: {names}.")
    return " ".join(parts) if parts else "No items specified."


def _apply_wardrobe_update(item_id: int, changes: dict) -> dict:
    """Apply a partial update to one wardrobe item. Only keys present in
    `changes` are touched — this relies on the grammar-constrained
    extraction actually omitting fields the user didn't mention (verified
    empirically: qwen2.5:7b reliably leaves non-required properties out
    of the JSON when there's nothing to fill them with), so an absent key
    means "leave alone", not "clear this field"."""
    conn = get_connection()
    try:
        row = conn.execute(
            f"SELECT {_WARDROBE_COLUMNS} FROM wardrobe_items WHERE id = ?", (item_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"no wardrobe item with id {item_id}")
        current = dict(row)

        if "category" in changes:
            category = (changes["category"] or "").strip().lower()
            if category not in config.WARDROBE_CATEGORIES:
                raise ValueError(f"unknown category: {category!r}")
            current["category"] = category
        if "name" in changes:
            name = (changes["name"] or "").strip()
            if not name:
                raise ValueError("name cannot be empty")
            current["name"] = name
        if "color" in changes:
            current["color"] = (changes["color"] or "").strip() or None
        if "warmth_level" in changes:
            current["warmth_level"] = _normalize_int(changes["warmth_level"])
        if "formality" in changes:
            current["formality"] = _normalize_int(changes["formality"])

        if "tags" in changes:
            new_tags = (changes["tags"] or "").strip() or None
            inferred_layer = _infer_layer_from_tags(new_tags)
            if inferred_layer:
                if "layer" not in changes:
                    current["layer"] = inferred_layer
                if new_tags and new_tags.lower() in (f"{inferred_layer} layer", "middle layer"):
                    new_tags = None
            current["tags"] = new_tags
        if "layer" in changes:
            current["layer"] = _normalize_layer(changes["layer"])

        # Re-embed only if something feeding the embedding text actually
        # changed — an update that's purely warmth/formality/layer
        # doesn't need a new /api/embed call.
        blob = current["embedding"]
        if any(k in changes for k in ("category", "name", "color", "tags")):
            vector = memory.embed_text(
                _embedding_text(current["category"], current["name"], current["color"], current["tags"])
            )
            blob = memory.pack_vector(vector)

        conn.execute(
            "UPDATE wardrobe_items SET category=?, name=?, color=?, warmth_level=?, "
            "formality=?, tags=?, layer=?, embedding=? WHERE id=?",
            (
                current["category"],
                current["name"],
                current["color"],
                current["warmth_level"],
                current["formality"],
                current["tags"],
                current["layer"],
                blob,
                item_id,
            ),
        )
        conn.commit()
        current.pop("embedding", None)
        return current
    finally:
        conn.close()


def update_wardrobe_items(updates: list[dict]) -> dict:
    """Registered tool entry point for one-or-more-items updates. A bad
    id or invalid value in one entry is reported and skipped rather than
    losing the rest of the batch."""
    updated = []
    failed = []
    for update in updates or []:
        item_id = update.get("item_id")
        try:
            changes = {k: v for k, v in update.items() if k != "item_id"}
            updated.append(_apply_wardrobe_update(item_id, changes))
        except Exception as exc:
            failed.append({"item_id": item_id, "error": str(exc)})
    return {"updated": updated, "failed": failed}


def describe_update_wardrobe_items(result: dict) -> str:
    parts = []
    if result["updated"]:
        names = ", ".join(f"#{i['id']} {i['name']}" for i in result["updated"])
        parts.append(f"Updated: {names}.")
    if result["failed"]:
        names = ", ".join(f"#{f['item_id']} ({f['error']})" for f in result["failed"])
        parts.append(f"Couldn't update: {names}.")
    return " ".join(parts) if parts else "No updates specified."


_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "top": ("top", "tops", "shirt", "shirts", "sweater", "sweaters", "sweatshirt",
            "sweatshirts", "hoodie", "hoodies", "jersey", "jerseys"),
    "bottom": ("bottom", "bottoms", "pant", "pants", "jean", "jeans", "trouser",
               "trousers", "short", "shorts"),
    "outerwear": ("outerwear", "jacket", "jackets", "coat", "coats", "parka",
                  "parkas", "peacoat", "peacoats"),
    "shoes": ("shoe", "shoes", "sneaker", "sneakers", "boot", "boots"),
    "accessory": ("accessory", "accessories", "hat", "hats", "scarf", "scarves",
                  "belt", "belts"),
}


def resolve_list_category(text: str, arguments: dict) -> dict:
    """Sanity-check list_wardrobe_items' extracted category against the
    user's own words. "category" is a required schema field, so the
    extraction model has to produce *something* even for a fully generic
    request like "what's in my wardrobe" that never names one -- observed
    empirically to default to a fixed wrong category ('outerwear') rather
    than 'all' in that case. Only trust a specific category if the text
    actually mentions it or one of its synonyms; otherwise fall back to
    listing everything, which is what a generic request actually wants."""
    category = arguments.get("category")
    if category and category != "all":
        lowered = text.lower()
        keywords = _CATEGORY_KEYWORDS.get(category, (category,))
        if not any(keyword in lowered for keyword in keywords):
            category = "all"
    return {"category": category}


def list_wardrobe_items(category: str | None = None) -> list[dict]:
    conn = get_connection()
    try:
        columns = (
            "id, category, name, color, warmth_level, formality, last_worn_at, tags, layer"
        )
        if category in (None, "", "all"):
            rows = conn.execute(
                f"SELECT {columns} FROM wardrobe_items ORDER BY category, name"
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {columns} FROM wardrobe_items WHERE category = ? ORDER BY name",
                (category,),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def describe_list_wardrobe_items(results: list[dict]) -> str:
    if not results:
        return "Your wardrobe is empty."

    def line(r: dict) -> str:
        suffix = f", {r['layer']} layer" if r.get("layer") else ""
        return f"#{r['id']}: {r['name']} ({r['category']}{suffix})"

    return "Your wardrobe:\n" + "\n".join(line(r) for r in results)


def delete_wardrobe_item(item_id: int) -> dict:
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM wardrobe_items WHERE id = ?", (item_id,))
        conn.commit()
        return {"deleted": cursor.rowcount > 0, "id": item_id}
    finally:
        conn.close()


def describe_delete_wardrobe_item(result: dict) -> str:
    if result["deleted"]:
        return f"Removed item #{result['id']} from your wardrobe."
    return f"Couldn't find wardrobe item #{result['id']}."


def delete_wardrobe_items(item_ids: list[int]) -> dict:
    """Registered tool entry point for one-or-more-ids deletes. Deletes
    each id individually (rather than one WHERE id IN (...) statement) so
    the confirmation can say exactly which ids existed and which didn't —
    a small personal-scale table, so N individual deletes cost nothing."""
    deleted = []
    not_found = []
    for item_id in item_ids or []:
        result = delete_wardrobe_item(item_id)
        (deleted if result["deleted"] else not_found).append(item_id)
    return {"deleted": deleted, "not_found": not_found}


def describe_delete_wardrobe_items(result: dict) -> str:
    parts = []
    if result["deleted"]:
        ids = ", ".join(f"#{i}" for i in result["deleted"])
        parts.append(f"Removed {ids} from your wardrobe.")
    if result["not_found"]:
        ids = ", ".join(f"#{i}" for i in result["not_found"])
        parts.append(f"Couldn't find {ids}.")
    return " ".join(parts) if parts else "No items specified."


def _target_warmth(temperature: float) -> int:
    for bound, warmth in config.WARDROBE_WARMTH_BANDS:
        if temperature <= bound:
            return warmth
    return config.WARDROBE_WARMTH_BANDS[-1][1]


def _target_formality(occasion: str | None) -> int | None:
    if not occasion:
        return None
    lowered = occasion.lower()
    matches = [
        value
        for keyword, value in config.WARDROBE_FORMALITY_KEYWORDS.items()
        if keyword in lowered
    ]
    return max(matches) if matches else None


def _load_candidates(category: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            f"SELECT {_WARDROBE_COLUMNS} FROM wardrobe_items WHERE category = ?",
            (category,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _filter_candidates(
    rows: list[dict], target_warmth: int | None, target_formality: int | None
) -> list[dict]:
    def keep(row: dict) -> bool:
        warmth_ok = (
            row["warmth_level"] is None
            or target_warmth is None
            or abs(row["warmth_level"] - target_warmth) <= config.WARDROBE_WARMTH_TOLERANCE
        )
        formality_ok = (
            row["formality"] is None
            or target_formality is None
            or abs(row["formality"] - target_formality) <= config.WARDROBE_FORMALITY_TOLERANCE
        )
        return warmth_ok and formality_ok

    filtered = [row for row in rows if keep(row)]
    return filtered if filtered else rows


def _rank_candidates(rows: list[dict], query: np.ndarray | None, limit: int) -> list[dict]:
    if not rows:
        return []

    by_id = {row["id"]: row for row in rows}

    if query is not None:
        pairs = [(row["id"], memory.unpack_vector(row["embedding"])) for row in rows]
        ranked = memory.cosine_top_k(query, pairs, k=limit)
        if ranked:
            picked = [dict(by_id[item_id], score=round(score, 3)) for item_id, score in ranked]
            for row in picked:
                row.pop("embedding", None)
            return picked

    # No occasion text, or nothing had a usable embedding: fall back to
    # recency — never-worn first, then longest-ago-worn.
    ordered = sorted(
        rows, key=lambda r: (r["last_worn_at"] is not None, r["last_worn_at"] or "")
    )
    picked = [dict(row) for row in ordered[:limit]]
    for row in picked:
        row.pop("embedding", None)
    return picked


def _pick_category(
    category: str,
    rows: list[dict],
    target_warmth: int | None,
    target_formality: int | None,
    query: np.ndarray | None,
    limit: int,
    notes: list[str],
    picks: dict,
    alternates: dict,
    pick_key: str | None = None,
) -> None:
    """Filter+rank `rows` and record the result under picks[pick_key or
    category] (and the matching alternates). Shared by suggest_fit's
    normal one-per-category slots and its base/mid layering picks."""
    key = pick_key or category
    if not rows:
        notes.append(f"no {category} saved in your wardrobe yet")
        return

    filtered = _filter_candidates(rows, target_warmth, target_formality)
    if filtered is rows:
        # _filter_candidates falls back to the unfiltered list (the same
        # object) only when filtering would have emptied it.
        notes.append("nothing matched the weather exactly, so this is the closest available")

    ranked = _rank_candidates(filtered, query, limit)
    if not ranked:
        return
    picks[key] = ranked[0]
    alternates[key] = ranked[1:]


def suggest_fit(occasion: str | None = None) -> dict:
    """Suggest an outfit from the user's own wardrobe, given the current
    weather and an optional free-text occasion. Returns a dict that is the
    LLM's only source of facts for Orchestrator.format_outfit — that
    prompt is explicitly told never to invent a garment/color/temperature
    beyond what's in here, so anything degraded or assumed goes in
    "notes" instead."""
    occasion = (occasion or "").strip() or None
    notes: list[str] = []

    weather = get_current_weather()
    target_warmth = None
    is_layering = False
    if "error" in weather:
        notes.append("couldn't reach the weather service, so warmth wasn't factored in")
        needs_outerwear = False
    else:
        target_warmth = _target_warmth(weather["temperature"])
        needs_outerwear = (
            weather["temperature"] <= config.WARDROBE_OUTERWEAR_MAX_TEMP
            or weather["precipitation_probability"] >= config.WARDROBE_RAIN_PROBABILITY_THRESHOLD
        )
        is_layering = weather["temperature"] <= config.WARDROBE_LAYERING_MAX_TEMP

    target_formality = _target_formality(occasion)

    query = None
    if occasion:
        try:
            query = memory.embed_text(occasion)
        except Exception:
            notes.append("couldn't use the occasion text for matching, using weather/fit only")

    picks: dict[str, dict] = {}
    alternates: dict[str, list[dict]] = {}
    limit = 1 + config.WARDROBE_ALTERNATES_PER_CATEGORY

    # "top" is handled separately from the other core slots: on a cold
    # enough day, try to pick a base + mid layer pair instead of a single
    # top — but only if the wardrobe actually has both roles tagged,
    # otherwise fall back to the normal single-top pick.
    top_rows = _load_candidates("top")
    layered = False
    if is_layering and top_rows:
        base_rows = [r for r in top_rows if r.get("layer") == "base"]
        mid_rows = [r for r in top_rows if r.get("layer") == "mid"]
        if base_rows and mid_rows:
            _pick_category(
                "top", base_rows, target_warmth, target_formality, query, limit,
                notes, picks, alternates, pick_key="top_base",
            )
            _pick_category(
                "top", mid_rows, target_warmth, target_formality, query, limit,
                notes, picks, alternates, pick_key="top_mid",
            )
            layered = "top_base" in picks and "top_mid" in picks
            if layered:
                notes.append("layered a base and a mid top for the cold")
        if not layered:
            notes.append(
                "no base+mid layer items tagged yet, so picking one warm top "
                "instead of layering"
            )

    if not layered:
        _pick_category("top", top_rows, target_warmth, target_formality, query, limit, notes, picks, alternates)

    remaining_categories = [c for c in config.WARDROBE_OUTFIT_CATEGORIES if c != "top"]
    if needs_outerwear:
        remaining_categories.append("outerwear")

    for category in remaining_categories:
        rows = _load_candidates(category)
        _pick_category(category, rows, target_warmth, target_formality, query, limit, notes, picks, alternates)

    if not picks:
        notes.append("no wardrobe items saved yet — add some with add_wardrobe_item")

    return {
        "occasion": occasion,
        "weather": weather,
        "target_warmth": target_warmth,
        "target_formality": target_formality,
        "picks": picks,
        "alternates": alternates,
        "notes": notes,
    }
