"""Execution-time feedback that closes the plan→execute→replan loop (M-A / A2).

Geometric refinement (``tamp.geometry``) falsifies a bridge predicate when an
action is *kinematically* infeasible. This package handles the other half: when
an action that geometry judged feasible *fails at execution* (the arm didn't
reach, the object slipped, a runtime collision), it maps that failure to the same
kind of bridge predicate so the planner can re-solve and recover.
"""
