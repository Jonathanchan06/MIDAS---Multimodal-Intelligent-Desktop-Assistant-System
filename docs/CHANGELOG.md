# Changelog

## Planned
1. ~~Personal agent personality~~ — done: TTS now speaks every chat reply, not just the briefing.
2. ~~Connect to iPhone~~ — done via Tailscale (see "Phone-to-PC API"). "Give access to computer" was explicitly discussed and shelved — needs careful scoping (allowlisted actions vs. open command execution) before any code gets written, not something to build casually. "Connect to Google account API" not started.
3. ~~Is not aware of current news~~ — done: chat now has real date/time injected into its system prompt, and news/market questions correctly route to live data instead of the model guessing or hallucinating.
4. Ship


## History

7/20/2025

Added morning debrief feature.
Planning to build overnight messages summarization feature. To overcome the obstacle of instagram not having an inherent API to return user messages and apple not having a notification reader function, the plan is to take screenshots everytime a notification pops up and sends it to qwen OCR to analyze it,then puts it in a DB to summarize in the morning

7/26/2026

Built v1 of the overnight notification summarization feature described above — screenshot capture via a Shortcuts automation, `qwen2.5vl:3b` vision extraction, SQLite log, dedup in the morning briefing (see "Overnight notification summary"). 

9/23/2026

Built the wardrobe/outfit-suggestion feature (see "Wardrobe / outfit suggestions"): `core/memory.py` as a generic numpy-based embedding/retrieval primitive (no vector DB — personal-scale data, a linear cosine scan is effectively free), a `wardrobe_items` table, and `suggest_fit`/`add_wardrobe_item`/`update_wardrobe_item`/`delete_wardrobe_item`/`list_wardrobe_items` tools, all batch-capable (one message can add/update/remove several items at once). Added base/mid layering support for cold-weather outfits. Chat-history RAG (the other half of the original embeddings idea) stayed deliberately unbuilt — real payoff given the tiny `CHAT_NUM_CTX` budget, but not worth it until it's actually felt.

Also landed three fixes worth remembering: (1) a `post_extract` registry hook, since a *required* schema field (needed just to make extraction run at all) can force the model to guess when the user's text doesn't actually specify one — `resolve_list_category` sanity-checks that guess against the user's own words; (2) `ROUTER_NUM_CTX` was too tight for the grown route list — only `schema.function.description` feeds `arch-router`'s one shared prompt, and letting one tool's description balloon silently corrupts *every* tool's routing, not just its own; (3) a real incident where "list wardrobe now" got misrouted to `add_wardrobe_item`, and since `items` is required, extraction hallucinated ~19 items straight out of conversation history rather than admitting there was nothing to add — silently duplicated 18 items before being caught. `resolve_add_items` now drops any extracted item whose name doesn't actually appear in the user's own message, so a future misroute into a *mutating* tool produces a harmless no-op instead of corrupting data — this pattern (verify extraction against the raw text, not just the schema) is worth applying to any future tool whose required fields could get invented under a misroute, not just this one. See "Adding a new tool" for all three.
