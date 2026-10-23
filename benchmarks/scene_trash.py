"""Scene fixture matching worlds/trash_sort.sdf (trash sorting task).
Positions MUST match the world file and config/scene_trash.yaml.
z=0.445 rests each item flat on the table (table top 0.410 + half-height 0.035)."""
from tamp.types import ObjectState, Pose, Surface, WorldState


def make_world() -> WorldState:
    workspace = Surface("workspace", Pose(position=(0.5, 0.0, 0.40)),
                        dimensions=(0.8, 0.8, 0.02), class_name="table")
    bin_blue = Surface("bin_blue", Pose(position=(0.56, 0.13, 0.415)),
                       dimensions=(0.10, 0.10, 0.01), class_name="tray")
    bin_green = Surface("bin_green", Pose(position=(0.56, 0.0, 0.415)),
                        dimensions=(0.10, 0.10, 0.01), class_name="tray")
    bin_orange = Surface("bin_orange", Pose(position=(0.56, -0.13, 0.415)),
                         dimensions=(0.10, 0.10, 0.01), class_name="tray")

    can = ObjectState("can_1", "can", Pose(position=(0.45, 0.20, 0.445)),
                      dimensions=(0.04, 0.04, 0.07))
    bottle = ObjectState("bottle_1", "bottle", Pose(position=(0.45, 0.0, 0.445)),
                         dimensions=(0.04, 0.04, 0.07))
    paper = ObjectState("paper_1", "paper", Pose(position=(0.45, -0.20, 0.445)),
                        dimensions=(0.04, 0.04, 0.07))
    return WorldState(objects=[can, bottle, paper],
                      surfaces=[workspace, bin_blue, bin_green, bin_orange])


# put the can into the blue bin
GOAL = {"type": "place_class", "class": "can", "surface": "bin_blue"}
