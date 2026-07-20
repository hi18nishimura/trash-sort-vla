"""Offline rule/grammar parser: English command -> goal-spec wire form.

The no-API fallback for :class:`tamp.nl.grounder.GoalGrounder`, and — because it is
the layer that is *always* available — the one that must **fail closed**.

Safety rules (an audit found the earlier version violated all three):
1. **Never invert an instruction.** A placement goal requires an imperative
   placement verb. "take cup_1 *off* the tray" previously parsed as "put cup_1 *on*
   the tray"; it now refuses.
2. **Never act on a prohibition or a question.** Any negation ("not", "don't",
   "never") or interrogative ("is …?", "where …") refuses outright. Previously
   "do not put cup_1 on the tray" put cup_1 on the tray.
3. **Never guess when it's ambiguous.** The destination is taken from *sentence*
   position (the last surface named), not from world-list order; a class-scoped
   command ("all the *cups*") never becomes an all-objects `gather`; and
   "clear the table" only resolves when exactly one destination is possible.

Returning ``None`` (refuse) is always preferable to a plausible-but-wrong goal:
the TAMP layer will happily execute a wrong-but-feasible plan.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from tamp.types import WorldState

_ALL_WORDS = ("all", "every", "everything", "each", "both")
_PUT = ("put", "place", "move", "set", "drop", "stack", "load", "transfer")
# A destination only counts when introduced by one of these. "away FROM the tray"
# names the tray but is not a placement — this is what prevents the inversion bug.
_PLACE_PREPS = ("on", "onto", "in", "into", "inside")
# Any of these => refuse. Cheaper to ask the user to rephrase than to invert them.
_NEGATIONS = ("not", "never", "dont", "cant", "cannot", "without", "avoid", "except")
_QUESTION_WORDS = ("is", "are", "was", "were", "do", "does", "did", "can", "could",
                   "will", "would", "should", "what", "where", "which", "who", "why", "how")
# Removal/relocation-away language. The DSL has no "remove from X" goal, and reading
# these as a placement is exactly the inversion bug — so they only survive if the
# sentence ALSO contains an explicit placement verb naming a destination.
_REMOVAL = ("off", "remove", "away", "unstack", "detach", "unload")


def _boundary(word: str) -> str:
    """Whole-word match that also works for ids whose first/last char isn't a word
    char (e.g. 'cup(1)'), where \\b would silently fail to match."""
    return rf"(?<!\w){re.escape(word.lower())}(?!\w)"


def _word_in(text: str, word: str) -> bool:
    return re.search(_boundary(word), text) is not None


def _ids_in_order(text: str, world: WorldState) -> List[str]:
    """Object ids mentioned, in order of first appearance in the *sentence*."""
    hits = []
    for o in world.objects:
        m = re.search(_boundary(o.object_id), text)
        if m:
            hits.append((m.start(), o.object_id))
    return [oid for _, oid in sorted(hits)]


def _surfaces_in_order(text: str, world: WorldState) -> List[Tuple[int, str]]:
    """(position, surface_id) for every surface named, ordered by sentence position.
    A surface may be named by its id or by its class word ('tray', 'table')."""
    hits = {}
    for s in world.surfaces:
        for word in (s.surface_id, s.class_name):
            if not word:
                continue
            m = re.search(_boundary(word), text)
            if m and (s.surface_id not in hits or m.start() < hits[s.surface_id]):
                hits[s.surface_id] = m.start()
    return sorted((pos, sid) for sid, pos in hits.items())


def _named_surface(text: str, world: WorldState) -> Optional[str]:
    """The first surface merely *named* (no placement sense). Used for the surface
    being cleared in "clear the table"."""
    surfs = _surfaces_in_order(text, world)
    return surfs[0][1] if surfs else None


def _placement_dest(text: str, world: WorldState) -> Optional[str]:
    """The destination = the LAST surface preceded by a *placement preposition*
    ("on/onto/in/into the tray").

    Requiring the preposition — not merely that a surface is named somewhere — is
    what stops the inversion bug: "move cup_1 *away from* the tray" names the tray
    but never places anything on it, so it yields no destination and the caller
    refuses. "take the cup off the table and put it *on the tray*" correctly yields
    `tray` (the last placement), not whichever surface comes first in the world."""
    preps = "|".join(_PLACE_PREPS)
    best = None
    for s in world.surfaces:
        for word in (s.surface_id, s.class_name):
            if not word:
                continue
            pat = (rf"(?<!\w)(?:{preps})\s+(?:the\s+|a\s+|an\s+)?"
                   rf"{re.escape(word.lower())}(?!\w)")
            for m in re.finditer(pat, text):
                if best is None or m.start() > best[0]:
                    best = (m.start(), s.surface_id)
    return best[1] if best else None


def _classes_in(text: str, world: WorldState) -> List[str]:
    found = []
    for cls in {o.class_name for o in world.objects if o.class_name}:
        if _word_in(text, cls) or _word_in(text, cls + "s"):
            found.append(cls)
    return found


def _must_refuse(c: str) -> bool:
    """Prohibitions and questions are never turned into motion."""
    if "?" in c:
        return True
    if any(_word_in(c, w) for w in _NEGATIONS) or "n't" in c:
        return True
    # a leading interrogative ("is cup_1 on the tray", "where is the cup")
    first = c.strip().split()
    return bool(first) and first[0] in _QUESTION_WORDS


def refusal_reason(command: str) -> Optional[str]:
    """Why a command is refused outright, or None if it isn't. Public so the
    grounder can report *why* nothing was grounded (prohibitions and questions
    are decisions, not parse failures)."""
    c = " " + command.lower().strip() + " "
    if "?" in c:
        return "the command is a question, not a goal"
    if any(_word_in(c, w) for w in _NEGATIONS) or "n't" in c:
        return "the command is a prohibition, not a goal"
    first = c.strip().split()
    if first and first[0] in _QUESTION_WORDS:
        return "the command is a question, not a goal"
    return None


def parse(command: str, world: WorldState) -> Optional[Dict]:
    c = " " + command.lower().strip() + " "
    if _must_refuse(c):
        return None                                    # fail closed

    ids = _ids_in_order(c, world)
    classes = _classes_in(c, world)
    has_put = any(_word_in(c, v) for v in _PUT)

    # STACK — "stack A on B [on C]" / "stack the blocks"
    if _word_in(c, "stack"):
        if len(ids) >= 2:
            return {"type": "stack", "order": ids}
        if classes:
            order = [o.object_id for o in world.objects if o.class_name in classes]
            if len(order) >= 2:
                return {"type": "stack", "order": order}
        return None

    # CLEAR THE <surface> — move everything off it. Resolvable only when exactly one
    # destination is possible; with 2+ candidates the destination would be a guess.
    if _word_in(c, "clear"):
        src = _named_surface(c, world)
        if src is None:
            return None
        others = [s.surface_id for s in world.surfaces if s.surface_id != src]
        return {"type": "gather", "surface": others[0]} if len(others) == 1 else None

    # Every remaining goal is a PLACEMENT and therefore needs a real destination
    # (introduced by a placement preposition). No destination -> refuse.
    dest = _placement_dest(c, world)
    if dest is None:
        return None

    wants_all = any(_word_in(c, w) for w in _ALL_WORDS) \
        or _word_in(c, "gather") or _word_in(c, "collect") or _word_in(c, "tidy")

    # "all the CUPS on the tray" is class-scoped — it must NOT become an all-objects
    # `gather`, which would sweep up every other object in the scene too.
    if wants_all and classes:
        return {"type": "sort_by_class",
                "assignments": [{"class": cls, "surface": dest} for cls in classes]}
    if wants_all:
        return {"type": "gather", "surface": dest}

    # Placement goals REQUIRE an imperative placement verb (rule 1).
    if not has_put:
        return None
    if ids:
        return {"type": "on", "object": ids[0], "surface": dest}
    if classes:
        return {"type": "place_class", "class": classes[0], "surface": dest}
    return None
