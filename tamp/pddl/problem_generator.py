"""Generate a unified-planning ``Problem`` from a ``WorldState`` + goal spec.

The init state is derived from geometry by :mod:`tamp.reasoning.object_reasoner`
(for ``on`` / ``clear`` / ``stacked_on``). The geometric bridge fluents
(``reachable`` / ``graspable`` / ``stackable`` / ``placeable``) are seeded
**optimistically** (True) and later falsified by the refiner in the TAMP loop.

Goal specs are small dicts (loaded from ``goals/*.yaml``):

    {type: stack, order: [A, B, C]}        # A on B on C (C on table)
    {type: on, object: red_box, surface: tray}
    {type: clear, object: blue_box}
    {type: sort_by_class, mapping: {cup: tray, bolt: bin}}
"""
from __future__ import annotations

from typing import Dict

from unified_planning.model import Object as UPObject
from unified_planning.model import Problem
from unified_planning.shortcuts import And

from tamp.pddl.domain_model import build_domain
from tamp.reasoning.object_reasoner import infer_relations
from tamp.types import WorldState


DEFAULT_GRIPPER = "gripper0"


def generate_problem(
    world: WorldState,
    goal_spec: Dict,
    gripper: str = DEFAULT_GRIPPER,
    optimistic_geometry: bool = True,
    default_surface: str | None = None,
) -> Problem:
    """Build a fully-specified UP ``Problem`` for *world* and *goal_spec*."""
    problem = Problem("tabletop_problem")
    d = build_domain(problem)
    Object, SurfaceT, Gripper = (d["types"]["object"], d["types"]["surface"], d["types"]["gripper"])
    f = d["fluents"]

    # --- objects ---
    obj_objs = {o.object_id: problem.add_object(o.object_id, Object) for o in world.objects}
    surf_objs = {s.surface_id: problem.add_object(s.surface_id, SurfaceT) for s in world.surfaces}
    grip = problem.add_object(gripper, Gripper)

    # --- initial symbolic state from geometry ---
    rel = infer_relations(world)
    problem.set_initial_value(f["handempty"](grip), True)

    for o in world.objects:
        oid = o.object_id
        support = rel.support_of.get(oid)  # surface id or object id this rests on
        # Fallback: objects with no inferred support rest on the default surface.
        # Robust for YOLO-derived scenes where the z/support heuristic is imprecise.
        if support is None and default_surface in surf_objs:
            support = default_surface
        if support in surf_objs:
            problem.set_initial_value(f["on"](obj_objs[oid], surf_objs[support]), True)
        elif support in obj_objs:
            problem.set_initial_value(f["stacked_on"](obj_objs[oid], obj_objs[support]), True)
        # `clear` means "nothing rests ON me" — which ``infer_relations`` derives from
        # the on-top map, independently of whether *my own* support was inferred. It
        # must NOT be widened by the default-surface fallback: a previous
        # `or support == default_surface` fired for every *correctly*-inferred support
        # equal to the default surface, so an object with another object stacked on it
        # was asserted clear() and the planner would emit PICK on a buried object.
        if oid in rel.clear:
            problem.set_initial_value(f["clear"](obj_objs[oid]), True)

    # --- geometric bridge fluents (seeded optimistically) ---
    if optimistic_geometry:
        for o in world.objects:
            problem.set_initial_value(f["reachable"](obj_objs[o.object_id]), True)
            problem.set_initial_value(f["graspable"](obj_objs[o.object_id]), True)
        for top in world.objects:
            for bottom in world.objects:
                if top.object_id != bottom.object_id:
                    problem.set_initial_value(
                        f["stackable"](obj_objs[top.object_id], obj_objs[bottom.object_id]), True)
        for o in world.objects:
            for s in world.surfaces:
                problem.set_initial_value(
                    f["placeable"](obj_objs[o.object_id], surf_objs[s.surface_id]), True)

    # --- goal ---
    problem.add_goal(_build_goal(goal_spec, f, obj_objs, surf_objs))
    return problem


def _build_goal(spec: Dict, f, obj_objs, surf_objs):
    kind = spec["type"]
    if kind == "stack":
        order = spec["order"]  # top -> ... -> bottom
        conj = [f["stacked_on"](obj_objs[order[i]], obj_objs[order[i + 1]])
                for i in range(len(order) - 1)]
        if "base_surface" in spec:
            conj.append(f["on"](obj_objs[order[-1]], surf_objs[spec["base_surface"]]))
        return And(*conj) if len(conj) > 1 else conj[0]
    if kind == "on":
        return f["on"](obj_objs[spec["object"]], surf_objs[spec["surface"]])
    if kind == "on_any":
        # object on ANY of the listed surfaces (disjunctive) — lets the planner
        # pick an alternative when geometry falsifies one option.
        from unified_planning.shortcuts import Or
        opts = [f["on"](obj_objs[spec["object"]], surf_objs[s]) for s in spec["surfaces"]]
        return Or(*opts) if len(opts) > 1 else opts[0]
    if kind == "clear":
        return f["clear"](obj_objs[spec["object"]])
    if kind == "gather":
        # Put every movable object onto the given surface (multi-object PnP).
        surf = surf_objs[spec["surface"]]
        conj = [f["on"](o, surf) for o in obj_objs.values()]
        return And(*conj) if len(conj) > 1 else conj[0]
    if kind == "place_class":
        # Place the first detected object whose class matches onto the surface.
        # Lets a YOLO-driven goal say "put a cup on the tray" without fixed IDs.
        cls, sid = spec["class"], spec["surface"]
        oid = spec.get("_resolved")  # caller resolves class -> concrete object id
        if oid is None:
            raise ValueError("place_class goal must be resolved to an object id first")
        return f["on"](obj_objs[oid], surf_objs[sid])
    if kind == "sort_by_class":
        # Each object goes onto the surface mapped from its class.
        # mapping resolved by caller-provided object->surface assignment.
        conj = [f["on"](obj_objs[oid], surf_objs[sid]) for oid, sid in spec["assignment"].items()]
        return And(*conj) if len(conj) > 1 else conj[0]
    raise ValueError(f"unknown goal type: {kind!r}")


def write_pddl(problem: Problem, domain_path: str, problem_path: str) -> None:
    """Serialize *problem* to ``.pddl`` files for inspection / repro."""
    from unified_planning.io import PDDLWriter

    w = PDDLWriter(problem)
    w.write_domain(domain_path)
    w.write_problem(problem_path)
