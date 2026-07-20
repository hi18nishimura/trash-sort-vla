"""M3 tests: the full TAMP plan↔refine↔replan loop."""
from benchmarks.scene_replan import GOAL as REPLAN_GOAL
from benchmarks.scene_replan import make_world as replan_world
from benchmarks.scene_stack3 import GOAL as STACK_GOAL
from benchmarks.scene_stack3 import make_world as stack_world
from tamp.geometry.robot_model import RobotModel
from tamp.tamp_solver import TampSolver

# Build the robot once (URDF parse is the slow part).
_ROBOT = RobotModel()


def _solver():
    return TampSolver(robot=_ROBOT)


def test_stack3_solves_without_replan():
    res = _solver().solve(stack_world(), STACK_GOAL)
    assert res.success
    assert res.replans == 0
    assert [a.action_type for a in res.plan] == ["PICK", "STACK", "PICK", "STACK"]
    assert all(a.target_pose is not None for a in res.plan)


def test_stack_heights_increase_with_shadow_scene():
    # The second STACK (A on B) must target a higher EEF z than the first
    # (B on C), proving the shadow scene tracked the growing tower.
    res = _solver().solve(stack_world(), STACK_GOAL)
    stacks = [a for a in res.plan if a.action_type == "STACK"]
    assert stacks[1].target_pose.position[2] > stacks[0].target_pose.position[2]


def test_replan_never_lands_on_the_unreachable_tray():
    """RIG-FREE recovery property (this test previously asserted `replans == 1`,
    which only held because the unreachable tray was NAMED to exploit Fast
    Downward's tie-break — renaming it silently made the 'recovery' vanish).
    What the loop actually guarantees, independent of which disjunct the
    satisficing planner explores first: the final plan lands on the reachable
    tray, and IF the planner tried the far tray first, it was falsified."""
    res = _solver().solve(replan_world(), REPLAN_GOAL)
    assert res.success
    place = [a for a in res.plan if a.action_type == "PLACE"][0]
    assert place.params["s"] == "tray_near"
    if res.replans:                       # organic in-loop recovery happened
        assert ("placeable", "block", "tray_far") in res.falsified
    else:                                 # planner picked the feasible tray directly
        assert res.falsified == []


def test_replan_loop_recovers_from_execution_falsification():
    """Deterministic recovery test through the loop's real entry point: execution
    reports the far tray failed (pre_falsified) -> the solve must land on
    tray_near without ever proposing tray_far."""
    res = _solver().solve(replan_world(), REPLAN_GOAL,
                          pre_falsified=[("placeable", "block", "tray_far")])
    assert res.success
    place = [a for a in res.plan if a.action_type == "PLACE"][0]
    assert place.params["s"] == "tray_near"


def test_replan_reports_failure_when_no_feasible_tray_remains():
    """The no-fake-success guarantee: with the REACHABLE tray falsified as well,
    the only symbolic option is the unreachable one — the solver must FAIL
    (geometry vetoes it) rather than return a plan onto a tray 2.5 m away."""
    res = _solver().solve(replan_world(), REPLAN_GOAL,
                          pre_falsified=[("placeable", "block", "tray_near")])
    assert not res.success
    # ...and the in-loop veto machinery demonstrably ran: forced onto tray_far,
    # the refiner falsified it organically before the solver gave up.
    assert ("placeable", "block", "tray_far") in res.falsified
