"""The goal-spec DSL shared by the LLM grounder and the offline grammar parser.

A goal spec is a small dict consumed by ``tamp.pddl.problem_generator`` (types:
``stack`` / ``on`` / ``on_any`` / ``clear`` / ``gather`` / ``place_class`` /
``sort_by_class``). This module owns three things so both grounding paths agree:

* ``GOAL_JSON_SCHEMA`` — a structured-output schema (flat, all-optional fields,
  ``additionalProperties: false``) that constrains the Claude response.
* ``validate_goal(spec, world)`` — semantic validation against the *actual*
  perceived objects/surfaces. The LLM can hallucinate an id; this rejects it.
* ``normalize_goal(spec, world)`` — fold the schema-friendly wire form (e.g. an
  ``assignments`` list) into the exact dict the problem generator expects.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from tamp.types import WorldState

GOAL_TYPES = ["stack", "on", "on_any", "clear", "gather", "place_class",
              "sort_by_class", "none"]

# Structured-output schema for the Claude response. Kept flat with every field
# optional (only `type` required) and additionalProperties:false, which is the
# subset the structured-outputs feature supports (no oneOf-by-type, no maps with
# arbitrary keys — hence `assignments` is a list, not an object).
GOAL_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": GOAL_TYPES},
        "order": {"type": "array", "items": {"type": "string"}},      # stack: top->bottom
        "base_surface": {"type": "string"},                            # stack (optional)
        "object": {"type": "string"},                                  # on / on_any / clear
        "surface": {"type": "string"},                                 # on / gather / place_class
        "surfaces": {"type": "array", "items": {"type": "string"}},    # on_any
        "class": {"type": "string"},                                   # place_class
        "assignments": {                                               # sort_by_class
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"class": {"type": "string"}, "surface": {"type": "string"}},
                "required": ["class", "surface"],
                "additionalProperties": False,
            },
        },
        "reason": {"type": "string"},   # why `none`, or a one-line gloss of the goal
    },
    "required": ["type"],
    "additionalProperties": False,
}


def _classes(world: WorldState) -> set:
    return {o.class_name for o in world.objects if o.class_name}


def validate_goal(spec: Dict, world: WorldState) -> Tuple[bool, str]:
    """Return (ok, error). Checks the spec references only real objects/surfaces.

    This function is TOTAL: LLM output is untrusted input, so it must **reject, never
    raise**. An audit found it previously threw `AttributeError` on a non-dict
    assignment and `TypeError` on an unhashable id, which escaped the grounder's
    fallback chain and killed the whole grounding. Every field is type-guarded, and
    every under-specified spec that would later crash the solver (empty `on_any`
    surfaces, a *string* `order`, duplicate stack ids) is rejected here.
    """
    if not isinstance(spec, dict) or "type" not in spec:
        return False, "missing 'type'"
    t = spec["type"]
    oids = set(world.object_ids())
    sids = {s.surface_id for s in world.surfaces}

    def need(ids, pool, what):
        """Membership check that tolerates junk (non-str / unhashable) values."""
        missing = [i for i in ids if not isinstance(i, str) or i not in pool]
        return f"unknown {what}: {missing}" if missing else ""

    def id_list(val, field):
        """Require an actual list of strings — a bare str would otherwise iterate
        character-by-character and 'validate' (e.g. order="AB" -> ['A','B'])."""
        if not isinstance(val, list):
            return None, f"'{field}' must be a list, got {type(val).__name__}"
        return val, ""

    if t == "none":
        return False, spec.get("reason", "no goal")

    if t == "stack":
        order, err = id_list(spec.get("order"), "order")
        if err:
            return False, err
        if len(order) < 2:
            return False, "stack needs >=2 objects in 'order'"
        if len(set(order)) != len(order):
            return False, f"stack 'order' has duplicate objects: {order}"
        if (e := need(order, oids, "object")):
            return False, e
        if "base_surface" in spec and (e := need([spec["base_surface"]], sids, "surface")):
            return False, e
        return True, ""

    if t == "on":
        return _check_all([need([spec.get("object")], oids, "object"),
                           need([spec.get("surface")], sids, "surface")])

    if t == "on_any":
        surfaces = spec.get("surfaces")
        if not surfaces:                     # missing OR empty -> solver would IndexError
            return False, "on_any needs at least one surface"
        surfaces, err = id_list(surfaces, "surfaces")
        if err:
            return False, err
        return _check_all([need([spec.get("object")], oids, "object"),
                           need(surfaces, sids, "surface")])

    if t == "clear":
        return _check_all([need([spec.get("object")], oids, "object")])

    if t == "gather":
        if not world.objects:                # solver would IndexError on an empty conj
            return False, "gather needs at least one object in the world"
        return _check_all([need([spec.get("surface")], sids, "surface")])

    if t == "place_class":
        cls = spec.get("class")
        if cls not in _classes(world):
            return False, f"no object of class '{cls}' present"
        return _check_all([need([spec.get("surface")], sids, "surface")])

    if t == "sort_by_class":
        asg = spec.get("assignments")
        if not isinstance(asg, list) or not asg:
            return False, "sort_by_class needs a non-empty 'assignments' list"
        if not all(isinstance(a, dict) for a in asg):   # guard: LLM may emit ["cup"]
            return False, "each 'assignments' entry must be an object {class, surface}"
        errs = [need([a.get("surface")], sids, "surface") for a in asg]
        bad_cls = [a.get("class") for a in asg if a.get("class") not in _classes(world)]
        if bad_cls:
            errs.append(f"no objects of class {bad_cls}")
        return _check_all(errs)

    return False, f"unknown goal type '{t}'"


def _check_all(errs: List[str]) -> Tuple[bool, str]:
    real = [e for e in errs if e]
    return (False, "; ".join(real)) if real else (True, "")


def normalize_goal(spec: Dict, world: WorldState) -> Optional[Dict]:
    """Fold the wire form into the problem-generator dict. Returns None if invalid."""
    ok, _ = validate_goal(spec, world)
    if not ok:
        return None
    t = spec["type"]
    if t == "sort_by_class":
        # class->surface, expanded to object_id->surface for every matching object.
        cls_to_surf = {a["class"]: a["surface"] for a in spec["assignments"]}
        assignment = {o.object_id: cls_to_surf[o.class_name]
                      for o in world.objects if o.class_name in cls_to_surf}
        return {"type": "sort_by_class", "assignment": assignment}
    if t == "place_class":
        # resolve to the first matching detected object (problem_generator expects this)
        match = next((o.object_id for o in world.objects if o.class_name == spec["class"]), None)
        return {"type": "place_class", "class": spec["class"],
                "surface": spec["surface"], "_resolved": match}
    return {k: v for k, v in spec.items() if k != "reason"}
