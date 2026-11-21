"""Wardrobe / outfit-suggestion tool. Follows tools/reminders.py's shape:
schema constants at module level, plain functions that open/close their
own core.db connection, and describe_* formatters for deterministic
(non-LLM) confirmation strings.

suggest_fit is the exception — it's registered with "speak_prose": "outfit"
instead of a describe_* formatter, so its result goes through
Orchestrator.format_outfit for spoken prose (see core/orchestrator.py).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

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
                            "item": {
                                "type": "string",
                                "description": (
                                    "Which saved item to change, in the user's own "
                                    "words (e.g. 'peacoat', 'grey hoodie'), or its "
                                    "id number only if the user said one. Never "
                                    "make up an id."
                                ),
                            },
                            "category": {
                                "type": "string",
                                "enum": config.WARDROBE_CATEGORIES,
                                "description": "New category. Omit if unchanged.",
                            },
                            "new_name": {
                                "type": "string",
                                "description": (
                                    "Only when the user asks to rename the item: "
                                    "the new name. Omit otherwise."
                                ),
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
                        "required": ["item"],
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
            "Delete or remove one or more items from the user's wardrobe, "
            "in a single call — e.g. 'remove my old peacoat', 'delete the "
            "grey hoodie and the black jeans', 'remove wardrobe item 5'. "
            "Do NOT call this to add or list clothes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "One entry per item to delete, in the user's own words "
                        "(e.g. 'peacoat', 'grey hoodie'), or its id number only "
                        "if the user said one. Never make up an id."
                    ),
                },
            },
            "required": ["items"],
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
                    # Tighter than the orchestrator's default: qwen3 pads
                    # this field (resolve_fit_occasion trims what's left).
                    "maxLength": 40,
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

MARK_WORN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mark_worn",
        "description": (
            "Record that the user is wearing, will wear, or wore certain "
            "clothes, so later outfit suggestions rotate — e.g. 'I'm "
            "wearing that', 'ok I'll wear that outfit', 'I wore my grey "
            "hoodie today'. Do NOT call this to ask what to wear, or to "
            "add, change, or remove clothes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The specific items worn, in the user's own words "
                        "(e.g. 'grey hoodie'). Use an empty list if they mean "
                        "the outfit that was just suggested ('that', 'it', "
                        "'the outfit')."
                    ),
                },
            },
            "required": ["items"],
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
    silently duplicating much of the wardrobe. Drop any item whose name's
    words don't all appear in the user's current message, so a misroute
    produces a harmless empty add instead of a destructive one.

    Compared word-by-word, not as one substring: an exact-substring check
    rejected "black henley top" for "add ...: black  henley top" (double
    space), which looked to the user like adding was just broken.

    Also corrects the category when the name itself says it (observed:
    "navy crewneck sweater" saved as outerwear) — see _category_from_name."""
    said = _ref_tokens(text)
    kept = []
    for item in arguments.get("items") or []:
        name_words = _ref_tokens(item.get("name"))
        if name_words and name_words <= said:
            category = _category_from_name(item.get("name"))
            kept.append({**item, "category": category} if category else item)
    return {"items": kept}


def _category_from_name(name: str | None) -> str | None:
    """The one category whose keywords appear in the garment's name, or
    None if zero or several do ('boot cut jeans' names both shoes and
    bottoms) — then the model's choice stands."""
    words = set(re.findall(r"[a-z0-9]+", (name or "").lower()))
    hits = {cat for cat, keywords in _CATEGORY_KEYWORDS.items() if words & set(keywords)}
    return hits.pop() if len(hits) == 1 else None


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
    if not parts:
        return (
            "I couldn't pick out an item to add from that — try naming it "
            "directly, like 'add a black henley top'."
        )
    return " ".join(parts)


_REF_STOPWORDS = {
    "my", "the", "a", "an", "item", "items", "wardrobe", "closet", "from",
    "that", "this", "one", "pair", "of", "old", "new", "number", "no", "to",
    # The model sometimes folds the field into the reference ("warmth of
    # my peacoat") — field words never identify a garment, so ignore them.
    "warmth", "level", "color", "colour", "formality", "layer", "tag",
    "tags", "name", "category", "set", "change", "rename", "mark",
}

# A change to one of these fields is only kept if the user's message
# gives some evidence for it — the extraction model has been observed
# adding fields nobody asked about ("mark my thermal shirt as a base
# layer" also came back with category='outerwear').
_FIELD_EVIDENCE: dict[str, tuple[str, ...]] = {
    "warmth_level": ("warm", "cold", "heavy", "light", "thick", "thin", "insulat"),
    "formality": ("formal", "casual", "dress", "fancy", "smart", "gym"),
    "layer": ("layer",),
}


def _ref_tokens(text: str | None) -> set[str]:
    """Lowercased words minus filler, with a crude plural strip so
    'sneakers' matches 'sneaker' and 'jeans' matches 'jean' — applied to
    both sides of a comparison, so it only has to be consistent, not
    linguistically right."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower().replace("'s", ""))
    out = set()
    for word in words:
        if word in _REF_STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.add(word)
    return out


def _match_item(ref: str, rows: list[dict], user_text: str) -> tuple[dict | None, str | None]:
    """Resolve one user reference ('peacoat', 'grey hoodie', '5') to
    exactly one saved row. Returns (row, None) on a unique match, or
    (None, reason) — the caller must not act on anything unresolved.

    This replaced asking the extraction model for an item_id directly:
    it never sees the wardrobe, so it invented ids (observed: item_id 1
    for "set my peacoat's warmth to 5", 12345 for "remove the peacoat"),
    which silently edited or deleted the wrong item. Matching names in
    Python against the real table can't hallucinate."""
    ref = (ref or "").strip()
    wanted = _ref_tokens(ref)
    if not wanted:
        return None, "I didn't catch which item you meant"

    if len(wanted) == 1 and next(iter(wanted)).isdigit():
        number = next(iter(wanted))
        # An id only counts if the user actually said that number — the
        # model is told never to invent one, but this doesn't rely on it.
        if not re.search(rf"(?<!\d){number}(?!\d)", user_text):
            return None, "I didn't catch which item you meant"
        row = next((r for r in rows if r["id"] == int(number)), None)
        return (row, None) if row else (None, f"there's no item #{number}")

    def item_tokens(row: dict) -> set[str]:
        return _ref_tokens(f"{row['name']} {row.get('color') or ''} {row['category']}")

    def covers(row: dict) -> bool:
        # Every word the user used must appear in the item, allowing a
        # word to sit inside a longer one ('coat' in 'peacoat').
        have = item_tokens(row)
        return all(any(w == h or (len(w) >= 3 and w in h) for h in have) for w in wanted)

    candidates = [row for row in rows if covers(row)]
    if len(candidates) > 1:
        exact = [row for row in candidates if _ref_tokens(row["name"]) == wanted]
        if len(exact) == 1:
            candidates = exact
    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return None, f"nothing in your wardrobe matches '{ref}'"
    options = ", ".join(f"#{row['id']} {row['name']}" for row in candidates[:5])
    return None, f"'{ref}' could be {options} — which one?"


def _value_leaked_from_ref(value, ref: str, user_text: str) -> bool:
    """True if a free-text change looks copied from the item reference
    rather than asked for — e.g. "change my grey hoodie to blue" coming
    back with color='grey', or name='peacoat' on a warmth-only request
    (observed with qwen2.5:7b). Such a value is part of the reference,
    not an instruction, so it's dropped."""
    if not isinstance(value, str) or not value.strip():
        return False
    lowered = value.strip().lower()
    return lowered in ref.lower() or lowered not in user_text.lower()


def _mentions(text: str, keywords) -> bool:
    """True if any keyword starts a word in text. Word-start, not plain
    substring: 'hat' matched inside "what's", so "what's in my closet"
    trusted a bogus 'accessory' category and listed nothing; and 'run'
    matched "brunch". Prefixes still work on purpose ('grocer' ->
    'grocery', 'warm' -> 'warmer')."""
    lowered = text.lower()
    return any(re.search(r"\b" + re.escape(keyword), lowered) for keyword in keywords)


def _has_evidence(field: str, value, user_text: str) -> bool:
    if field == "category":
        keywords = _CATEGORY_KEYWORDS.get(str(value), (str(value),))
    else:
        keywords = _FIELD_EVIDENCE.get(field)
        if keywords is None:
            return True
    return _mentions(user_text, keywords)


def _meaningful_refs(refs: list) -> list[str]:
    """Drop blank/filler-only references (the model sometimes adds an
    empty second entry), unless that would leave nothing — then keep them
    so the reply can say it didn't catch which item was meant."""
    refs = [str(ref or "") for ref in refs]
    meaningful = [ref for ref in refs if _ref_tokens(ref)]
    return meaningful or refs[:1]


def resolve_update_refs(text: str, arguments: dict) -> dict:
    """post_extract hook for update_wardrobe_items: turn each 'item'
    reference into a real item_id (see _match_item) and drop changes the
    user's words don't support. Unresolved entries are passed through as
    'unresolved' so the reply says why nothing happened."""
    rows = list_wardrobe_items()
    lowered_text = text.lower()
    entries = [e for e in arguments.get("updates") or [] if isinstance(e, dict)]
    keep = set(_meaningful_refs([e.get("item") for e in entries]))
    updates, unresolved = [], []
    for entry in entries:
        ref = str(entry.get("item") or "")
        if ref not in keep:
            continue
        row, reason = _match_item(ref, rows, lowered_text)
        if row is None:
            unresolved.append({"item": ref, "reason": reason})
            continue
        changes = {
            k: v for k, v in entry.items()
            if k not in ("item", "new_name") and _has_evidence(k, v, lowered_text)
        }
        if "new_name" in entry and not _value_leaked_from_ref(entry["new_name"], ref, lowered_text):
            changes["name"] = entry["new_name"]
        if "color" in changes and _value_leaked_from_ref(changes["color"], ref, lowered_text):
            del changes["color"]
        if changes:
            updates.append({"item_id": row["id"], **changes})
        else:
            unresolved.append({"item": ref, "reason": "I couldn't tell what to change"})
    return {"updates": updates, "unresolved": unresolved}


def resolve_delete_refs(text: str, arguments: dict) -> dict:
    """post_extract hook for delete_wardrobe_items — same name-to-id
    resolution as updates. Deleting is the one irreversible wardrobe
    action, so an ambiguous or unmatched reference deletes nothing."""
    rows = list_wardrobe_items()
    lowered_text = text.lower()
    item_ids, unresolved = [], []
    for ref in _meaningful_refs(arguments.get("items") or []):
        row, reason = _match_item(ref, rows, lowered_text)
        if row is None:
            unresolved.append({"item": ref, "reason": reason})
        elif row["id"] not in item_ids:
            item_ids.append(row["id"])
    return {"item_ids": item_ids, "unresolved": unresolved}


def _describe_unresolved(unresolved: list[dict] | None) -> str:
    if not unresolved:
        return ""
    reasons = "; ".join(entry["reason"] for entry in unresolved)
    return f"Didn't touch anything for that: {reasons}" + ("" if reasons.endswith("?") else ".")


# Words that, on their own, mean "the outfit you just suggested" rather
# than naming a garment.
_OUTFIT_REF_WORDS = {
    "it", "outfit", "fit", "those", "these", "them", "everything", "all",
    "clothe", "suggestion", "look",
}


def _record_suggestion(item_ids: list[int]) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO outfit_suggestions (suggested_at, item_ids) VALUES (?, ?)",
            (datetime.now(timezone.utc).isoformat(), json.dumps(item_ids)),
        )
        conn.commit()
    finally:
        conn.close()


def _last_suggestion_ids() -> list[int] | None:
    """Item ids from the most recent suggest_fit, or None if there isn't
    one recent enough (WARDROBE_SUGGESTION_TTL_HOURS) to be what "that"
    refers to."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT suggested_at, item_ids FROM outfit_suggestions ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    age = datetime.now(timezone.utc) - datetime.fromisoformat(row["suggested_at"])
    if age > timedelta(hours=config.WARDROBE_SUGGESTION_TTL_HOURS):
        return None
    return json.loads(row["item_ids"])


def resolve_worn_refs(text: str, arguments: dict) -> dict:
    """post_extract hook for mark_worn. An empty list, or references like
    'that'/'the outfit', mean the last suggestion; anything else resolves
    by name like updates/deletes do."""
    refs = [str(r or "") for r in arguments.get("items") or []]
    named = [r for r in refs if _ref_tokens(r) - _OUTFIT_REF_WORDS]
    if not named:
        ids = _last_suggestion_ids()
        if ids is None:
            reason = "I haven't suggested an outfit recently — which items are you wearing?"
            return {"item_ids": [], "unresolved": [{"item": "that", "reason": reason}]}
        return {"item_ids": ids, "unresolved": []}

    rows = list_wardrobe_items()
    lowered_text = text.lower()
    item_ids, unresolved = [], []
    for ref in named:
        row, reason = _match_item(ref, rows, lowered_text)
        if row is None:
            unresolved.append({"item": ref, "reason": reason})
        elif row["id"] not in item_ids:
            item_ids.append(row["id"])
    return {"item_ids": item_ids, "unresolved": unresolved}


def mark_worn(item_ids: list[int], unresolved: list[dict] | None = None) -> dict:
    """Registered tool entry point: stamp last_worn_at on each item so
    suggest_fit rotates away from it for WARDROBE_REWEAR_DAYS. Takes ids
    from resolve_worn_refs (the raw extraction has 'items', so a skipped
    hook fails closed)."""
    now = datetime.now(timezone.utc).isoformat()
    marked = []
    conn = get_connection()
    try:
        for item_id in item_ids or []:
            row = conn.execute("SELECT name FROM wardrobe_items WHERE id = ?", (item_id,)).fetchone()
            if row is None:  # deleted since it was suggested
                continue
            conn.execute("UPDATE wardrobe_items SET last_worn_at = ? WHERE id = ?", (now, item_id))
            marked.append({"id": item_id, "name": row["name"]})
        conn.commit()
    finally:
        conn.close()
    return {"marked": marked, "unresolved": unresolved or []}


def describe_mark_worn(result: dict) -> str:
    parts = []
    if result["marked"]:
        names = ", ".join(r["name"] for r in result["marked"])
        parts.append(f"Got it — marked as worn: {names}. I'll rotate those out for a few days.")
    if result["unresolved"]:
        parts.append(_describe_unresolved(result["unresolved"]))
    return " ".join(parts) if parts else "Nothing to mark as worn."


def _apply_wardrobe_update(item_id: int, changes: dict) -> dict:
    """Apply a partial update to one wardrobe item. Only keys present in
    `changes` are touched, so an absent key means "leave alone", not
    "clear this field". The extraction model doesn't reliably omit fields
    the user didn't mention, so resolve_update_refs strips values that
    just echo the item reference before they get here."""
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


def update_wardrobe_items(updates: list[dict], unresolved: list[dict] | None = None) -> dict:
    """Registered tool entry point for one-or-more-items updates. Expects
    item_ids already resolved by resolve_update_refs; if that hook didn't
    run, entries have no item_id and fail closed. A bad value in one
    entry is reported and skipped rather than losing the rest."""
    updated = []
    failed = []
    for update in updates or []:
        item_id = update.get("item_id")
        try:
            if item_id is None:
                raise ValueError("no item resolved")
            changes = {k: v for k, v in update.items() if k != "item_id"}
            updated.append(_apply_wardrobe_update(item_id, changes))
        except Exception as exc:
            failed.append({"item_id": item_id, "error": str(exc)})
    return {"updated": updated, "failed": failed, "unresolved": unresolved or []}


def describe_update_wardrobe_items(result: dict) -> str:
    parts = []
    if result["updated"]:
        names = ", ".join(f"#{i['id']} {i['name']}" for i in result["updated"])
        parts.append(f"Updated: {names}.")
    if result["failed"]:
        names = ", ".join(f"#{f['item_id']} ({f['error']})" for f in result["failed"])
        parts.append(f"Couldn't update: {names}.")
    if result.get("unresolved"):
        parts.append(_describe_unresolved(result["unresolved"]))
    return " ".join(parts) if parts else "No updates specified."


_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "top": ("top", "tops", "shirt", "shirts", "sweater", "sweaters", "sweatshirt",
            "sweatshirts", "hoodie", "hoodies", "jersey", "jerseys", "henley",
            "henleys", "tee", "tees", "polo", "polos", "blouse", "blouses",
            "tank", "tanks", "cardigan", "cardigans", "crewneck", "crewnecks",
            "turtleneck", "turtlenecks", "flannel", "flannels"),
    "bottom": ("bottom", "bottoms", "pant", "pants", "jean", "jeans", "trouser",
               "trousers", "short", "shorts", "chinos", "joggers", "sweatpants",
               "skirt", "skirts", "leggings"),
    "outerwear": ("outerwear", "jacket", "jackets", "coat", "coats", "parka",
                  "parkas", "peacoat", "peacoats", "windbreaker", "windbreakers",
                  "puffer", "puffers", "raincoat", "raincoats"),
    "shoes": ("shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "loafers",
              "sandals", "slides", "heels", "cleats"),
    "accessory": ("accessory", "accessories", "hat", "hats", "scarf", "scarves",
                  "belt", "belts", "beanie", "beanies", "cap", "caps", "gloves",
                  "watch", "watches", "sunglasses"),
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
        keywords = _CATEGORY_KEYWORDS.get(category, (category,))
        if not _mentions(text, keywords):
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


def speak_list_wardrobe_items(results: list[dict]) -> str:
    """Spoken counterpart to describe_list_wardrobe_items: reading every
    item and id aloud is slow and useless, so say counts and point at the
    screen. Short lists are still read out by name."""
    if not results:
        return "Your wardrobe is empty."
    if len(results) <= 3:
        return "You have " + ", ".join(r["name"] for r in results) + "."
    counts: dict[str, int] = {}
    for r in results:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    parts = [f"{n} {_category_noun(cat, n)}" for cat, n in counts.items()]
    return f"You have {len(results)} items: {', '.join(parts)}. The full list is on screen."


def _category_noun(category: str, count: int) -> str:
    singular, plural = {
        "accessory": ("accessory", "accessories"),
        "shoes": ("pair of shoes", "pairs of shoes"),
        "outerwear": ("outerwear piece", "outerwear pieces"),
    }.get(category, (category, category + "s"))
    return singular if count == 1 else plural


def delete_wardrobe_item(item_id: int) -> dict:
    conn = get_connection()
    try:
        row = conn.execute("SELECT name FROM wardrobe_items WHERE id = ?", (item_id,)).fetchone()
        cursor = conn.execute("DELETE FROM wardrobe_items WHERE id = ?", (item_id,))
        conn.commit()
        return {"deleted": cursor.rowcount > 0, "id": item_id, "name": row["name"] if row else None}
    finally:
        conn.close()


def describe_delete_wardrobe_item(result: dict) -> str:
    if result["deleted"]:
        return f"Removed item #{result['id']} from your wardrobe."
    return f"Couldn't find wardrobe item #{result['id']}."


def delete_wardrobe_items(item_ids: list[int], unresolved: list[dict] | None = None) -> dict:
    """Registered tool entry point for one-or-more-items deletes. Takes
    item_ids resolved by resolve_delete_refs (the raw extraction has
    'items', not 'item_ids', so a skipped hook fails closed with a
    TypeError instead of deleting anything). Deletes each id
    individually so the confirmation can name exactly what was removed."""
    deleted = []
    not_found = []
    for item_id in item_ids or []:
        result = delete_wardrobe_item(item_id)
        (deleted if result["deleted"] else not_found).append(result)
    return {"deleted": deleted, "not_found": not_found, "unresolved": unresolved or []}


def describe_delete_wardrobe_items(result: dict) -> str:
    parts = []
    if result["deleted"]:
        names = ", ".join(f"#{r['id']} {r['name']}" for r in result["deleted"])
        parts.append(f"Removed from your wardrobe: {names}.")
    if result["not_found"]:
        ids = ", ".join(f"#{r['id']}" for r in result["not_found"])
        parts.append(f"Couldn't find {ids}.")
    if result.get("unresolved"):
        parts.append(_describe_unresolved(result["unresolved"]))
    return " ".join(parts) if parts else "No items specified."


def _target_warmth(temperature: float) -> int:
    for bound, warmth in config.WARDROBE_WARMTH_BANDS:
        if temperature <= bound:
            return warmth
    return config.WARDROBE_WARMTH_BANDS[-1][1]


def _target_formality(occasion: str | None) -> int | None:
    if not occasion:
        return None
    hits = {
        keyword: value
        for keyword, value in config.WARDROBE_FORMALITY_KEYWORDS.items()
        if _mentions(occasion, (keyword,))
    }
    # 'work' also starts 'workout' — when one hit is a prefix of another,
    # the longer, more specific keyword wins (workout = gym, not office).
    matches = [
        value
        for keyword, value in hits.items()
        if not any(other != keyword and other.startswith(keyword) for other in hits)
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


def _drop_recently_worn(rows: list[dict]) -> list[dict]:
    """Skip anything worn within WARDROBE_REWEAR_DAYS so suggestions
    rotate — ranking (embedding or recency) is otherwise deterministic and
    would return the same outfit every time. Falls back to the full list
    if everything was worn recently (e.g. only one pair of shoes)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.WARDROBE_REWEAR_DAYS)

    def fresh(row: dict) -> bool:
        worn = row.get("last_worn_at")
        return not worn or datetime.fromisoformat(worn) < cutoff

    kept = [row for row in rows if fresh(row)]
    return kept if kept else rows


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

    ranked = _rank_candidates(_drop_recently_worn(filtered), query, limit)
    if not ranked:
        return
    picks[key] = ranked[0]
    alternates[key] = ranked[1:]


# Words that show up in almost every what-to-wear request and say nothing
# about the occasion itself.
_OCCASION_FILLER = {
    "what", "should", "i", "me", "my", "a", "an", "the", "to", "for", "and",
    "or", "wear", "wearing", "put", "on", "outfit", "fit", "something",
    "clothes", "today", "tonight", "now", "pick", "help", "suggest", "give",
    "idea", "do", "need", "is", "it", "go", "going",
}


def resolve_fit_occasion(text: str, arguments: dict) -> dict:
    """post_extract hook for suggest_fit: keep only occasion words the
    user actually said. qwen3 pads this field with invented context —
    "what should I wear today" came back as "daily wear/office work/...",
    which then matched the 'office'/'work' formality keywords and dressed
    a plain request up. Grounding it in the user's words makes the
    occasion (and the formality/embedding it drives) model-independent."""
    said = re.findall(r"[a-z0-9']+", text.lower())
    said_set = set(said)
    extracted = re.findall(r"[a-z0-9']+", str(arguments.get("occasion") or "").lower())
    kept: list[str] = []
    for word in extracted:
        if word in said_set and word not in _OCCASION_FILLER and word not in kept:
            kept.append(word)
    return {"occasion": " ".join(kept)}


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
    else:
        _record_suggestion([pick["id"] for pick in picks.values()])

    return {
        "occasion": occasion,
        "weather": weather,
        "target_warmth": target_warmth,
        "target_formality": target_formality,
        "picks": picks,
        "alternates": alternates,
        "notes": notes,
    }
