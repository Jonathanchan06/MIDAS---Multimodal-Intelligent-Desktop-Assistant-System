"""Synthetic training data for the LoRA argument extractor.

Every example is built from a template plus vocabulary, and its target
JSON is computed from the same pieces -- so labels are correct by
construction (no LLM labelling its own output, nothing leaves the
machine). Prompts are rendered by core.orchestrator.build_extraction_messages,
the function the live extraction call uses, so training and inference
see byte-identical prompts.

Targets follow the *current* tool schemas and the rules in CLAUDE.md
("Extraction design rule"): user's own words in free text, no invented
ids, no fields nobody asked to change, color in `color`, reminder times as
slots (never calendar math). If a schema's parameters change, regenerate.

Leakage guard: any example too close to an eval case (core or hard set)
is dropped, so the eval stays a held-out test. Never add templates or
vocabulary *because* a hard-set case needs them — general variety
(typos, spelled-out numbers, casual phrasing) is fine; copying the test
is not.

Usage:
    python training/gen_extraction_data.py              # ~400/tool
    python training/gen_extraction_data.py --per-tool 600 --seed 7
Writes training/data/extraction_{train,val}.jsonl in chat "messages"
format (system, [history], user, assistant).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.orchestrator import build_extraction_messages  # noqa: E402
from tools.reminders import ADD_REMINDER_SCHEMA, DELETE_REMINDER_SCHEMA  # noqa: E402
from tools.wardrobe import (  # noqa: E402
    ADD_WARDROBE_ITEM_SCHEMA,
    DELETE_WARDROBE_ITEM_SCHEMA,
    LIST_WARDROBE_ITEMS_SCHEMA,
    MARK_WORN_SCHEMA,
    SUGGEST_FIT_SCHEMA,
    UPDATE_WARDROBE_ITEM_SCHEMA,
)

# Keyed like main.TOOL_REGISTRY / the eval cases.
SCHEMAS = {
    "add_reminder": ADD_REMINDER_SCHEMA,
    "delete_reminder": DELETE_REMINDER_SCHEMA,
    "suggest_fit": SUGGEST_FIT_SCHEMA,
    "add_wardrobe_item": ADD_WARDROBE_ITEM_SCHEMA,
    "list_wardrobe_items": LIST_WARDROBE_ITEMS_SCHEMA,
    "update_wardrobe_item": UPDATE_WARDROBE_ITEM_SCHEMA,
    "delete_wardrobe_item": DELETE_WARDROBE_ITEM_SCHEMA,
    "mark_worn": MARK_WORN_SCHEMA,
}
EVAL_CASES = [ROOT / "eval" / "extraction_cases.json", ROOT / "eval" / "extraction_cases_hard.json"]
OUT_DIR = Path(__file__).parent / "data"

# --- vocabulary -------------------------------------------------------------

TASKS = [
    "call mom", "call dad", "call grandma", "text Jake back", "email my landlord",
    "pay the electric bill", "pay the credit card", "book a haircut", "feed the cat",
    "walk the dog", "submit my timesheet", "renew the car registration", "buy milk",
    "pick up the dry cleaning", "start the laundry", "take the chicken out of the freezer",
    "water the garden", "check the mail", "stretch", "drink some water",
    "lock the back door", "charge my laptop", "pack my gym bag", "cancel the free trial",
    "return the library books", "schedule a vet appointment", "take my vitamins",
    "pick up Emma from practice", "move the car", "take out the recycling",
    "order more coffee filters", "back up my phone", "study for the chem exam",
    "send the invoice", "call the pharmacy", "refill my prescription", "clean the fridge",
    "reply to Priya", "join the standup call", "grab the package from the front desk",
    "change the air filter", "practice guitar", "prep lunch for tomorrow",
    "turn off the sprinklers", "file my expense report", "call the insurance company",
    "pick up groceries", "wish Sam happy birthday", "bring the umbrella", "rotate the tires",
]
# Noun-phrase reminders ("remind me about ...").
EVENTS = [
    "the team meeting", "my flight", "the vet appointment", "the doctor's appointment",
    "my haircut", "the parent teacher conference", "the soccer game", "trash day",
    "the conference call", "my interview", "the rent", "the book club", "the oil change",
    "Mia's recital", "the farmers market", "my therapy session", "the project deadline",
]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]

COLORS = ["black", "white", "grey", "gray", "navy", "blue", "red", "green", "olive",
          "brown", "tan", "khaki", "beige", "cream", "burgundy", "maroon", "pink",
          "purple", "yellow", "orange", "charcoal", "light blue", "dark green"]
MATERIALS = ["wool", "cotton", "linen", "denim", "leather", "suede", "cashmere",
             "flannel", "corduroy", "knit", "fleece"]
FITS = ["slim fit", "oversized", "striped", "plaid", "relaxed fit", "cropped", "vintage"]

# garment -> (category, warmth 1-5, formality 1-5, layer role or "")
GARMENTS = {
    "t-shirt": ("top", 1, 1, ""), "tee": ("top", 1, 1, ""), "henley": ("top", 2, 2, ""),
    "oxford shirt": ("top", 2, 4, ""), "dress shirt": ("top", 2, 5, ""),
    "polo": ("top", 2, 3, ""), "flannel shirt": ("top", 3, 2, ""),
    "sweater": ("top", 3, 3, "mid"), "crewneck": ("top", 3, 2, "mid"),
    "cardigan": ("top", 3, 3, "mid"), "hoodie": ("top", 3, 1, "mid"),
    "sweatshirt": ("top", 3, 1, "mid"), "turtleneck": ("top", 3, 4, ""),
    "thermal": ("top", 2, 1, "base"), "base layer top": ("top", 2, 1, "base"),
    "long sleeve tee": ("top", 2, 1, "base"), "tank top": ("top", 1, 1, ""),
    "quarter zip": ("top", 3, 3, "mid"), "fleece pullover": ("top", 4, 2, "mid"),
    "button down": ("top", 2, 4, ""),
    "jeans": ("bottom", 2, 2, ""), "chinos": ("bottom", 2, 3, ""),
    "dress pants": ("bottom", 2, 5, ""), "joggers": ("bottom", 2, 1, ""),
    "sweatpants": ("bottom", 3, 1, ""), "shorts": ("bottom", 1, 1, ""),
    "cargo pants": ("bottom", 2, 2, ""), "corduroys": ("bottom", 3, 3, ""),
    "trousers": ("bottom", 2, 4, ""), "leggings": ("bottom", 2, 1, ""),
    "parka": ("outerwear", 5, 2, ""), "puffer jacket": ("outerwear", 5, 2, ""),
    "denim jacket": ("outerwear", 3, 2, ""), "bomber jacket": ("outerwear", 3, 2, ""),
    "trench coat": ("outerwear", 4, 4, ""), "windbreaker": ("outerwear", 2, 1, ""),
    "overcoat": ("outerwear", 5, 5, ""), "leather jacket": ("outerwear", 3, 3, ""),
    "blazer": ("outerwear", 3, 5, ""), "raincoat": ("outerwear", 2, 2, ""),
    "fleece jacket": ("outerwear", 4, 1, ""), "shacket": ("outerwear", 3, 2, ""),
    "sneakers": ("shoes", 2, 2, ""), "running shoes": ("shoes", 2, 1, ""),
    "chelsea boots": ("shoes", 3, 4, ""), "loafers": ("shoes", 2, 4, ""),
    "oxfords": ("shoes", 2, 5, ""), "sandals": ("shoes", 1, 1, ""),
    "hiking boots": ("shoes", 4, 1, ""), "high tops": ("shoes", 2, 2, ""),
    "slip ons": ("shoes", 2, 2, ""), "work boots": ("shoes", 4, 2, ""),
    "baseball cap": ("accessory", 1, 1, ""), "cap": ("accessory", 1, 1, ""),
    "scarf": ("accessory", 4, 3, ""), "gloves": ("accessory", 5, 2, ""),
    "belt": ("accessory", 1, 3, ""), "watch": ("accessory", 1, 3, ""),
    "sunglasses": ("accessory", 1, 2, ""), "bucket hat": ("accessory", 1, 1, ""),
    "tie": ("accessory", 1, 5, ""), "beanie": ("accessory", 4, 1, ""),
}
# Leading words that refer to "the outfit you just suggested".
DEICTIC = ["that", "it", "those", "that outfit", "the outfit you picked",
           "what you suggested", "the fit you picked", "that look", "your pick"]

OCCASIONS = [
    "a job interview", "class", "dinner with friends", "the gym", "a wedding",
    "a first date", "church", "a funeral", "brunch", "a hike", "work", "a presentation",
    "a concert", "a baseball game", "the office", "a birthday party", "running errands",
    "yoga", "a business meeting", "a night out", "a barbecue", "a lecture",
    "a networking event", "a picnic", "a dinner party", "my sister's graduation",
    "a holiday party", "the movies", "a client lunch", "a lab session", "a road trip",
]
CATEGORY_WORDS = {
    "top": ["shirts", "tops", "sweaters", "t-shirts", "hoodies", "tees"],
    "bottom": ["pants", "jeans", "shorts", "bottoms", "trousers"],
    "outerwear": ["jackets", "coats", "outerwear"],
    "shoes": ["shoes", "sneakers", "boots", "footwear", "loafers"],
    "accessory": ["accessories", "hats", "scarves", "belts", "watches"],
}

# --- helpers ----------------------------------------------------------------

def strip_article(phrase: str) -> str:
    return re.sub(r"^(a|an|the|my)\s+", "", phrase)


def maybe(rng: random.Random, p: float) -> bool:
    return rng.random() < p


def garment_phrase(rng: random.Random, with_color: float = 0.7, with_material: float = 0.25):
    """Returns (spoken phrase without article, garment, color, tag)."""
    garment = rng.choice(list(GARMENTS))
    color = rng.choice(COLORS) if maybe(rng, with_color) else ""
    tag = ""
    if maybe(rng, with_material):
        tag = rng.choice(MATERIALS + FITS)
    words = [w for w in (color, tag, garment) if w]
    return " ".join(words), garment, color, tag


def a_or_an(phrase: str) -> str:
    return ("an " if phrase[0] in "aeiou" else "a ") + phrase


def item_ref(rng: random.Random) -> str:
    """A reference to a saved item the way people say it: 'peacoat',
    'grey hoodie', 'wool scarf'."""
    phrase, garment, color, tag = garment_phrase(rng, with_color=0.5, with_material=0.15)
    return phrase


def join_list(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def fmt_hour(hour: int, minute: int, rng: random.Random) -> str:
    """Spoken clock time: '5pm', '5:30 pm', '17:00', 'noon'."""
    if hour == 12 and minute == 0 and maybe(rng, 0.4):
        return "noon"
    h12 = hour % 12 or 12
    suffix = "am" if hour < 12 else "pm"
    style = rng.random()
    if style < 0.55:
        return f"{h12}{':%02d' % minute if minute else ''}{suffix}"
    if style < 0.85:
        return f"{h12}{':%02d' % minute if minute else ''} {suffix}"
    return f"{h12}:{minute:02d}{suffix}"


# --- per-tool generators: each returns (user_text, target, history) -------

def gen_add_reminder(rng: random.Random):
    """Targets mirror resolve_when's contract: day enum, date MM-DD only
    with day='date', time HH:MM, in_minutes only for relative times."""
    kind = rng.random()
    slots: dict = {"day": "", "time": ""}
    if kind < 0.2:                                   # relative
        minutes, spoken = rng.choice([
            (10, "in 10 minutes"), (15, "in 15 minutes"), (20, "in 20 minutes"),
            (30, "in 30 minutes"), (30, "in half an hour"), (45, "in 45 minutes"),
            (60, "in an hour"), (60, "in one hour"), (90, "in 90 minutes"),
            (90, "in an hour and a half"), (120, "in 2 hours"), (120, "in two hours"),
            (180, "in 3 hours"), (240, "in four hours"), (5, "in 5 minutes"),
        ])
        when, slots = spoken, {"day": "", "time": "", "in_minutes": minutes}
    elif kind < 0.4:                                 # clock time only
        hour, minute = rng.randint(6, 22), rng.choice([0, 0, 0, 15, 30, 45])
        when = f"at {fmt_hour(hour, minute, rng)}"
        slots["time"] = f"{hour:02d}:{minute:02d}"
    elif kind < 0.55:                                # tomorrow / today / tonight
        choice = rng.choice(["tomorrow", "today", "tonight", "tomorrow morning",
                             "tomorrow afternoon", "tomorrow evening", "this evening",
                             "this afternoon", "tomorrow night"])
        day = "tomorrow" if choice.startswith("tomorrow") else "today"
        part = {"morning": "09:00", "afternoon": "14:00", "evening": "18:00",
                "night": "21:00", "tonight": "21:00"}
        time = next((v for k, v in part.items() if k in choice), "")
        if maybe(rng, 0.5) and choice in ("tomorrow", "today", "tonight"):
            hour = rng.randint(7, 22) if choice != "tonight" else rng.randint(19, 23)
            minute = rng.choice([0, 0, 30])
            when = f"{choice} at {fmt_hour(hour, minute, rng)}"
            time = f"{hour:02d}:{minute:02d}"
        else:
            when = choice
        slots = {"day": day, "time": time}
    elif kind < 0.85:                                # weekday
        weekday = rng.choice(WEEKDAYS)
        lead = rng.choice(["on ", "next ", "this ", ""])
        spoken_day = lead + weekday.capitalize()
        time = ""
        if maybe(rng, 0.7):
            hour, minute = rng.randint(7, 21), rng.choice([0, 0, 30])
            when = f"{spoken_day} at {fmt_hour(hour, minute, rng)}"
            time = f"{hour:02d}:{minute:02d}"
        elif maybe(rng, 0.5):
            part, time = rng.choice([("morning", "09:00"), ("afternoon", "14:00"),
                                     ("evening", "18:00"), ("night", "21:00")])
            when = f"{lead}{weekday.capitalize()} {part}"
        else:
            when = spoken_day
        slots = {"day": weekday, "time": time}
    else:                                            # calendar date
        month = rng.randint(1, 12)
        dom = rng.randint(1, 28)
        suffix = "th" if 10 <= dom % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(dom % 10, "th")
        spoken = f"{MONTHS[month - 1]} {dom}{suffix if maybe(rng, 0.4) else ''}"
        time = ""
        if maybe(rng, 0.6):
            hour = rng.randint(8, 20)
            when = f"on {spoken} at {fmt_hour(hour, 0, rng)}"
            time = f"{hour:02d}:00"
        else:
            when = f"on {spoken}"
        slots = {"day": "date", "date": f"{month:02d}-{dom:02d}", "time": time}

    if maybe(rng, 0.2):
        what = rng.choice(EVENTS)
        templates = [
            "remind me about {what} {when}", "{when} remind me about {what}",
            "set a reminder for {what} {when}", "can you remind me of {what} {when}",
            "don't let me forget {what} {when}", "heads up about {what} {when} please",
        ]
    else:
        what = rng.choice(TASKS)
        templates = [
            "remind me to {what} {when}", "{when} remind me to {what}",
            "can you remind me to {what} {when}", "set a reminder to {what} {when}",
            "don't let me forget to {what} {when}", "ping me {when} to {what}",
            "I need to {what} {when}, remind me", "please remind me {when} to {what}",
            "hey remind me to {what} {when}", "remind me {when} that I need to {what}",
            "reminder to {what} {when}", "could you set a reminder to {what} {when}",
            "nudge me {when} to {what}", "I have to {what} {when}, don't let me forget",
        ]
    text = rng.choice(templates).format(what=what, when=when)
    if maybe(rng, 0.3):
        text = text[0].upper() + text[1:]
    # Key order follows the schema's property order, like Ollama's grammar.
    target = {"text": strip_article(what) if what in EVENTS else what, "day": slots["day"]}
    if slots.get("date"):
        target["date"] = slots["date"]
    target["time"] = slots["time"]
    if slots.get("in_minutes"):
        target["in_minutes"] = slots["in_minutes"]
    return text, target, []


def _reminder_ref(rng: random.Random):
    """(spoken reference, target reference) for an existing reminder."""
    if maybe(rng, 0.5):
        task = rng.choice(TASKS)
        return rng.choice([f"the reminder to {task}", f"my reminder to {task}",
                           f"the {task} reminder"]), task
    event = strip_article(rng.choice(EVENTS))
    return rng.choice([f"the {event} reminder", f"my {event} reminder",
                       f"the reminder about {event}", f"the reminder for {event}"]), event


def gen_delete_reminder(rng: random.Random):
    kind = rng.random()
    if kind < 0.3:                                   # by number
        n = rng.randint(1, 60)
        text = rng.choice([
            "delete reminder {n}", "cancel reminder #{n}", "remove reminder number {n}",
            "get rid of reminder {n}", "reminder {n} can go", "scratch reminder #{n}",
            "kill reminder {n}, I already did it", "please delete #{n}",
            "drop reminder number {n} from my list",
        ]).format(n=n)
        return text, {"reminders": [str(n)]}, []
    if kind < 0.45:                                  # two numbers
        a, b = rng.sample(range(1, 40), 2)
        text = rng.choice(["delete reminders {a} and {b}", "cancel #{a} and #{b}",
                           "remove reminders {a} and {b} please"]).format(a=a, b=b)
        return text, {"reminders": [str(a), str(b)]}, []
    if kind < 0.6:                                   # from a listing in history
        tasks = rng.sample(TASKS, 3)
        ids = rng.sample(range(1, 30), 3)
        listing = "Your reminders:\n" + "\n".join(
            f"#{i}: {t} — due 2026-12-0{d}T{h}:00:00" for i, t, d, h in
            zip(ids, tasks, (2, 3, 4), (9, 14, 18))
        )
        history = [{"role": "user", "content": rng.choice(["what are my reminders",
                                                           "show my reminders"])},
                   {"role": "assistant", "content": listing}]
        pick = rng.choice(tasks)
        key = " ".join(pick.split()[-2:]) if len(pick.split()) > 2 else pick
        text = rng.choice(["cancel the {k} one", "delete the one about {k}",
                           "get rid of the {k} reminder", "remove {k}"]).format(k=key)
        # The user named it; the id is only in history, which the resolver
        # can't check — so the target is the words, not the number.
        return text, {"reminders": [key]}, history
    if kind < 0.75:                                  # two by description
        (sa, ta), (sb, tb) = _reminder_ref(rng), _reminder_ref(rng)
        text = rng.choice(["cancel {a} and {b}", "delete {a} and {b}",
                           "I don't need {a} or {b} anymore"]).format(a=sa, b=sb)
        return text, {"reminders": [ta, tb]}, []
    spoken, target = _reminder_ref(rng)
    text = rng.choice([
        "cancel {r}", "delete {r}", "never mind {r}", "remove {r}",
        "get rid of {r}", "I don't need {r} anymore", "scratch {r}",
        "forget {r}", "you can drop {r}", "clear {r}", "please cancel {r}, it's done",
    ]).format(r=spoken)
    return text, {"reminders": [target]}, []


def gen_suggest_fit(rng: random.Random):
    if maybe(rng, 0.35):
        text = rng.choice([
            "what should I wear", "what should I wear today", "pick an outfit for me",
            "what should I put on", "do I need a coat today", "is it jacket weather",
            "dress me", "what's a good outfit for today", "help me pick clothes",
            "it's freezing, what do I wear", "what do I wear right now",
            "choose my clothes for today", "outfit check, what should I go with",
            "what should I throw on", "is it shorts weather", "do I need layers today",
            "give me an outfit", "what's the move for clothes today",
        ])
        return text, {"occasion": ""}, []
    occasion = rng.choice(OCCASIONS)
    text = rng.choice([
        "what should I wear to {o}", "outfit for {o}", "I have {o} later, what should I wear",
        "dress me for {o}", "what do I wear for {o}", "pick something for {o}",
        "what's a good outfit for {o}", "help me dress for {o}",
        "what should I put on for {o} tonight", "I'm going to {o}, what should I wear",
        "need an outfit for {o}", "what to wear for {o} tomorrow",
    ]).format(o=occasion)
    return text, {"occasion": strip_article(occasion)}, []


def _new_item(rng: random.Random) -> tuple[str, dict]:
    phrase, garment, color, tag = garment_phrase(rng)
    category, warmth, formality, layer = GARMENTS[garment]
    target = {
        "category": category, "name": phrase, "color": color,
        "warmth_level": warmth, "formality": formality, "tags": tag,
        "layer": layer if category == "top" else "",
    }
    # Plural garments ("jeans", "sneakers") take no article.
    spoken = phrase if garment.endswith("s") else a_or_an(phrase)
    if maybe(rng, 0.25):
        spoken = "my new " + phrase
    return spoken, target


def gen_add_wardrobe_item(rng: random.Random):
    count = rng.choices([1, 2, 3], weights=[6, 3, 1])[0]
    pieces = [_new_item(rng) for _ in range(count)]
    listed = join_list([s for s, _ in pieces])
    text = rng.choice([
        "add {x} to my wardrobe", "I got {x}", "I bought {x}", "save {x}",
        "add {x}", "new in my closet: {x}", "put {x} in my wardrobe",
        "I just picked up {x}", "add {x} to my closet", "can you save {x}",
        "log {x} in my wardrobe", "got {x} today, add it",
    ]).format(x=listed)
    return text, {"items": [t for _, t in pieces]}, []


def gen_list_wardrobe_items(rng: random.Random):
    if maybe(rng, 0.35):
        text = rng.choice([
            "what's in my wardrobe", "show me everything in my closet", "list my clothes",
            "what clothes do I have", "what do I own", "show my wardrobe",
            "list everything in my wardrobe", "what have I got saved",
            "read me my closet", "what's saved in my wardrobe", "list all my clothes",
        ])
        return text, {"category": "all"}, []
    category = rng.choice(list(CATEGORY_WORDS))
    word = rng.choice(CATEGORY_WORDS[category])
    text = rng.choice([
        "list my {w}", "what {w} do I have", "show me my {w}", "do I have any {w}",
        "what {w} are saved", "which {w} do I own", "show all my {w}",
        "what {w} are in my closet", "list the {w} in my wardrobe",
    ]).format(w=word)
    return text, {"category": category}, []


def _change(rng: random.Random, garment_is_top: bool):
    """One requested change: (spoken template with {i}, field, value)."""
    options = ["warmth", "formality", "color", "tags", "rename", "category"]
    if garment_is_top:
        options += ["layer", "layer"]
    kind = rng.choice(options)
    if kind == "warmth":
        n = rng.randint(1, 5)
        return rng.choice(["set the warmth of my {i} to %d", "make my {i} warmth %d",
                           "my {i} should be warmth level %d", "change {i} warmth to %d"]) % n, \
            "warmth_level", n
    if kind == "formality":
        n = rng.randint(1, 5)
        return rng.choice(["set my {i}'s formality to %d", "make the {i} formality %d",
                           "the {i} is a formality %d"]) % n, "formality", n
    if kind == "color":
        c = rng.choice(COLORS)
        return rng.choice(["change my {i}'s color to %s", "my {i} is actually %s",
                           "set the color of the {i} to %s", "recolor my {i} to %s"]) % c, \
            "color", c
    if kind == "tags":
        t = rng.choice(MATERIALS + FITS)
        return rng.choice(["tag my {i} as %s", "add the tag %s to my {i}",
                           "mark the {i} as %s"]) % t, "tags", t
    if kind == "rename":
        new = item_ref(rng)
        return rng.choice(["rename my {i} to %s", "call the {i} '%s' instead",
                           "change the name of my {i} to %s"]) % new, "new_name", new
    if kind == "category":
        cat = rng.choice(list(CATEGORY_WORDS))
        word = {"top": "a top", "bottom": "bottoms", "outerwear": "outerwear",
                "shoes": "shoes", "accessory": "an accessory"}[cat]
        return rng.choice(["the {i} should be %s", "move my {i} to %s",
                           "file the {i} under %s"]) % word, "category", cat
    role = rng.choice(["base", "mid"])
    return rng.choice(["mark my {i} as a %s layer", "make the {i} a %s layer",
                       "my {i} is a %s layer", "set the {i} as a %s layer"]) % role, "layer", role


def gen_update_wardrobe_item(rng: random.Random):
    count = rng.choices([1, 2], weights=[4, 1])[0]
    clauses, updates = [], []
    for _ in range(count):
        if maybe(rng, 0.12):
            ref = str(rng.randint(1, 60))
            spoken_ref, garment_is_top = f"item {ref}", False
        else:
            phrase, garment, _, _ = garment_phrase(rng, with_color=0.4, with_material=0.1)
            ref, spoken_ref, garment_is_top = phrase, phrase, GARMENTS[garment][0] == "top"
        template, field, value = _change(rng, garment_is_top)
        clause = template.format(i=spoken_ref)
        if spoken_ref.startswith("item ") and "my item" in clause:
            clause = clause.replace("my item", "item")
        clauses.append(clause)
        updates.append({"item": ref, field: value})
    text = " and ".join(clauses)
    return text, {"updates": updates}, []


def gen_delete_wardrobe_item(rng: random.Random):
    if maybe(rng, 0.15):
        ids = [str(n) for n in rng.sample(range(1, 60), rng.choice([1, 1, 2]))]
        text = (f"remove wardrobe item {ids[0]}" if len(ids) == 1
                else f"delete items {ids[0]} and {ids[1]}")
        return text, {"items": ids}, []
    refs = [item_ref(rng) for _ in range(rng.choices([1, 2, 3], weights=[6, 3, 1])[0])]
    listed = join_list([f"{rng.choice(['my', 'the'])} {r}" for r in refs])
    text = rng.choice([
        "remove {x}", "delete {x}", "get rid of {x}", "throw out {x}", "toss {x}",
        "take {x} out of my wardrobe", "I don't have {x} anymore, remove it",
        "delete {x} from my closet", "I donated {x}, take it off the list",
    ]).format(x=listed)
    return text, {"items": refs}, []


def gen_mark_worn(rng: random.Random):
    history = []
    if maybe(rng, 0.4):
        if maybe(rng, 0.5):
            history = [{"role": "user", "content": "what should I wear today"},
                       {"role": "assistant", "content": "It's 51°F and cloudy — go with "
                        f"your {item_ref(rng)}, {item_ref(rng)} and {item_ref(rng)}."}]
        text = rng.choice([
            "I'm wearing {d}", "ok I'll wear {d}", "going with {d}", "wearing {d} today",
            "sounds good, I'll wear {d}", "yep, I'll go with {d}", "I'll take {d}",
            "perfect, wearing {d}", "putting on {d} now",
        ]).format(d=rng.choice(DEICTIC))
        return text, {"items": []}, history
    refs = [item_ref(rng) for _ in range(rng.choices([1, 2, 3], weights=[5, 3, 1])[0])]
    listed = join_list([f"{rng.choice(['my', 'the'])} {r}" for r in refs])
    text = rng.choice([
        "I'm wearing {x}", "I wore {x} today", "today I'm in {x}", "I put on {x}",
        "I wore {x} yesterday", "wearing {x} right now", "I've got {x} on",
        "had {x} on today", "rocking {x} today",
    ]).format(x=listed)
    return text, {"items": refs}, history


GENERATORS = {
    "add_reminder": gen_add_reminder,
    "delete_reminder": gen_delete_reminder,
    "suggest_fit": gen_suggest_fit,
    "add_wardrobe_item": gen_add_wardrobe_item,
    "list_wardrobe_items": gen_list_wardrobe_items,
    "update_wardrobe_item": gen_update_wardrobe_item,
    "delete_wardrobe_item": gen_delete_wardrobe_item,
    "mark_worn": gen_mark_worn,
}

# --- leakage guard ----------------------------------------------------------

_FILLER = {"a", "an", "the", "my", "me", "i", "to", "for", "of", "and", "on", "in", "at", "it"}


def _content(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _FILLER)


def too_close(text: str, eval_sets: list[frozenset[str]], threshold: float) -> bool:
    words = _content(text)
    for other in eval_sets:
        union = words | other
        if union and len(words & other) / len(union) >= threshold:
            return True
    return False


# --- main -------------------------------------------------------------------

def random_now(rng: random.Random) -> datetime:
    """A random moment in 2026-2027, daytime-ish, so the model never ties
    its output to one fixed 'now' in the prompt."""
    start = datetime(2026, 1, 1, 7, 0)
    return (start + timedelta(days=rng.randint(0, 600), minutes=rng.randint(0, 15 * 60))).replace(second=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-tool", type=int, default=400)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leak-threshold", type=float, default=0.75,
                        help="drop examples whose content-word Jaccard with any eval case is >= this")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    eval_cases = [c for path in EVAL_CASES for c in json.loads(path.read_text(encoding="utf-8"))]
    eval_sets = [_content(c["text"]) for c in eval_cases]

    examples, dropped = [], Counter()
    for tool, gen in GENERATORS.items():
        seen: set[str] = set()
        attempts = 0
        while len(seen) < args.per_tool and attempts < args.per_tool * 30:
            attempts += 1
            text, target, history = gen(rng)
            text = re.sub(r"\s+", " ", text).strip()
            if text.lower() in seen:
                dropped["duplicate"] += 1
                continue
            if too_close(text, eval_sets, args.leak_threshold):
                dropped["near_eval_case"] += 1
                continue
            seen.add(text.lower())
            function_schema = SCHEMAS[tool]["function"]
            messages = build_extraction_messages(function_schema, text, history, random_now(rng))
            messages.append({"role": "assistant",
                             "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))})
            examples.append({"tool": tool, "messages": messages})
        if len(seen) < args.per_tool:
            print(f"  note: {tool} produced only {len(seen)} unique examples")

    rng.shuffle(examples)
    n_val = int(len(examples) * args.val_fraction)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("val", examples[:n_val]), ("train", examples[n_val:])):
        path = OUT_DIR / f"extraction_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows):>5} examples -> {path.relative_to(ROOT)}")
    print("per tool:", dict(Counter(e["tool"] for e in examples)))
    print("dropped:", dict(dropped))


if __name__ == "__main__":
    main()
