"""M1 unit tests: object reasoning, problem generation, symbolic solve."""
from benchmarks.scene_stack3 import GOAL, make_world
from tamp.pddl.problem_generator import generate_problem
from tamp.planning.symbolic_planner import solve
from tamp.reasoning.object_reasoner import infer_relations


def test_reasoner_all_on_table_and_clear():
    rel = infer_relations(make_world())
    assert rel.support_of == {"A": "table", "B": "table", "C": "table"}
    assert rel.clear == {"A", "B", "C"}


def test_stack3_solves():
    res = solve(generate_problem(make_world(), GOAL))
    assert res.success
    assert [a.action_type for a in res.actions] == ["PICK", "STACK", "PICK", "STACK"]


def test_stack3_plan_achieves_goal_order():
    res = solve(generate_problem(make_world(), GOAL))
    stacks = [(a.params["top"], a.params["bottom"]) for a in res.actions if a.action_type == "STACK"]
    assert ("B", "C") in stacks and ("A", "B") in stacks


def test_unsolvable_returns_failure():
    # Goal references stacking onto itself is impossible; use an empty-hand goal
    # that can't hold given no such fluent in goal — instead test a contradictory
    # stack of a single missing relation by removing objects.
    w = make_world()
    w.objects = w.objects[:1]  # only A present
    bad_goal = {"type": "stack", "order": ["A", "A"]}
    # A on A is blocked by (not (= top bottom)); planner must fail.
    res = solve(generate_problem(w, bad_goal))
    assert not res.success
