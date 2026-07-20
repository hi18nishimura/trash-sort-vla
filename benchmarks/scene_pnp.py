"""Scene fixture matching worlds/tabletop_pnp.sdf — used to drive the pick-and-place
manipulation stack from a known /world_state (deterministic, camera-free) so the
TAMP -> MoveIt -> grasp_manager -> Gazebo chain can be verified without rendering.
"""
from tamp.types import ObjectState, Pose, Surface, WorldState


def make_world() -> WorldState:
    workspace = Surface("workspace", Pose(position=(0.5, 0.0, 0.40)), dimensions=(0.8, 0.8, 0.02),
                        class_name="table")
    # class_name must be "tray", not the default "table" — otherwise the word "table"
    # in a command matches the tray as well, and any surface-by-name lookup is ambiguous.
    tray = Surface("tray", Pose(position=(0.55, 0.0, 0.415)), dimensions=(0.34, 0.34, 0.01),
                   class_name="tray")
    # Positions MUST match worlds/tabletop_pnp_nocam.sdf. z=0.445 rests the cups flat
    # on the table (top 0.410 + half-height 0.035); y=±0.20 clears the tray edge
    # (y=±0.17) so bullet-featherstone doesn't topple them, while staying within the
    # arm's precise reach (y=±0.24 missed by 42 mm).
    cup = ObjectState("cup_1", "cup", Pose(position=(0.45, 0.20, 0.445)), dimensions=(0.04, 0.04, 0.07))
    cup2 = ObjectState("cup_2", "cup", Pose(position=(0.45, -0.20, 0.445)), dimensions=(0.04, 0.04, 0.07))
    return WorldState(objects=[cup, cup2], surfaces=[workspace, tray])


# pick a cup and place it on the tray (class-driven, like the YOLO goal)
GOAL = {"type": "place_class", "class": "cup", "surface": "tray"}
