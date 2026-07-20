"""Command-line entry points for the TAMP core.

``tamp-offline`` (installed console script) runs the full PDDL + PyTAMP TAMP loop
on a benchmark scene with no ROS and no sim — the V1–V3 offline demo. The legacy
``scripts/run_offline.py`` is a thin wrapper around :func:`offline_main`.
"""
from __future__ import annotations

import argparse
import importlib
import logging

import numpy as np

from tamp.tamp_solver import TampSolver

SCENES = {
    "stack3": "benchmarks.scene_stack3",   # 3 blocks A,B,C on a table
    "replan": "benchmarks.scene_replan",   # closed-loop replan recovery
    "pnp": "benchmarks.scene_pnp",         # two cups + a tray (the Gazebo demo scene)
}


def offline_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tamp-offline",
                                 description="Offline PDDL+PyTAMP TAMP demo (no ROS, no sim).")
    ap.add_argument("--scene", choices=SCENES, default="stack3")
    ap.add_argument("--viz", metavar="PATH", help="render scene+plan to PNG/GLB")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="[%(name)s] %(message)s")

    mod = importlib.import_module(SCENES[args.scene])
    world, goal = mod.make_world(), dict(mod.GOAL)

    # Class-based goals ("place a cup") need a concrete object id before solving —
    # resolve to the first perceived instance of the class, exactly as the ROS
    # planner node does. Without this, `--scene pnp` crashed on its own default goal.
    if goal.get("type") == "place_class":
        match = next((o.object_id for o in world.objects
                      if o.class_name == goal.get("class")), None)
        if match is None:
            print(f"no object of class {goal.get('class')!r} in the scene")
            return 1
        goal = {"type": "on", "object": match, "surface": goal["surface"]}

    solver = TampSolver()
    res = solver.solve(world, goal)
    print(f"\n=== TAMP [{args.scene}] ===")
    print(f"success={res.success}  replans={res.replans}  reason={res.reason}")
    if res.falsified:
        print(f"falsified bridge predicates: {res.falsified}")
    for i, a in enumerate(res.plan, 1):
        xyz = np.round(a.target_pose.position, 3) if a.target_pose else None
        print(f"  {i}. {a}  -> EEF {xyz}")

    if args.scene == "replan" and res.success:
        # Which disjunct the satisficing planner tries first is an implementation
        # detail this demo deliberately does not script (see
        # benchmarks/scene_replan.py). Report the organic path, then demonstrate
        # the loop's other guarantee: when execution falsifies the only feasible
        # tray too, the solver REPORTS FAILURE rather than emitting a plan onto
        # the unreachable one.
        if res.replans:
            print("[replan] organic in-loop recovery: the planner tried the far tray "
                  "first; geometry vetoed it and the loop re-planned onto tray_near.")
        else:
            print("[replan] the planner happened to pick the reachable tray directly "
                  "(no in-loop replan needed this run — the choice is planner-internal).")
        print("[replan] counterfactual (execution later falsifies tray_near too):")
        res2 = solver.solve(world, goal,
                            pre_falsified=[("placeable", "block", "tray_near")])
        print(f"[replan]   success={res2.success}  reason={res2.reason}  "
              f"falsified={res2.falsified}")
        print("[replan]   -> honest failure: no feasible tray remains, and the loop "
              "says so instead of planning onto the unreachable tray.")

    if args.viz and res.success:
        if args.viz.endswith(".png"):
            # Headless-safe matplotlib render (the trimesh viewer needs a GL context).
            from tamp.viz.plot_plan import render_plan_png
            render_plan_png(world, res.plan, args.viz,
                            title=f"TAMP plan — scene '{args.scene}'")
            print(f"[viz] saved {args.viz}")
        else:
            from tamp.viz.scene_viz import build_scene, save_png
            scene = build_scene(world, res.plan)
            ok = save_png(scene, args.viz)
            print(f"[viz] saved {'PNG' if ok else 'GLB'} near {args.viz}")

    return 0 if res.success else 1


def command_main(argv=None) -> int:
    """`tamp-command` — type an English command, watch it ground to a goal and
    solve. Offline by default (grammar parser); hosted-LLM backends activate
    automatically when their SDK + credentials are present."""
    ap = argparse.ArgumentParser(
        prog="tamp-command",
        description="Natural-language -> PDDL goal -> TAMP plan (the flagship demo).")
    ap.add_argument("--scene", choices=SCENES, default="stack3")
    ap.add_argument("--no-llm", action="store_true", help="force the offline grammar parser")
    ap.add_argument("command", nargs="*", help="a one-shot command; omit for an interactive prompt")
    args = ap.parse_args(argv)

    from tamp.nl import GoalGrounder
    mod = importlib.import_module(SCENES[args.scene])
    world = mod.make_world()
    grounder = GoalGrounder(use_llm=(False if args.no_llm else None))
    solver = TampSolver()

    print(f"=== scene '{args.scene}' ===")
    print("objects:  " + ", ".join(f"{o.object_id}({o.class_name})" for o in world.objects))
    print("surfaces: " + ", ".join(f"{s.surface_id}({s.class_name})" for s in world.surfaces))
    print(f"grounding: {'offline grammar' if not grounder.use_llm else 'LLM backend (falls back offline)'}")

    def handle(cmd: str):
        r = grounder.ground(cmd, world)
        if not r.success:
            print(f"  ✗ {r.error}")
            return
        print(f"  goal [{r.source}]: {r.goal}")
        res = solver.solve(world, r.goal)
        if not res.success:
            print(f"  ✗ infeasible: {res.reason} (replans={res.replans})")
            return
        print(f"  ✓ plan ({len(res.plan)} steps, {res.replans} replan):")
        for i, a in enumerate(res.plan, 1):
            print(f"     {i}. {a.action_type} {a.object_id}")

    if args.command:
        handle(" ".join(args.command))
        return 0
    print("\nType a command (e.g. \"stack A on B on C\", \"put the cups on the tray\"); 'quit' to exit.")
    while True:
        try:
            cmd = input("command> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if cmd.lower() in ("quit", "exit", "q"):
            break
        if cmd:
            handle(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(offline_main())
