"""Instruction phrasings for the trash-sorting task.

canonical : the single phrasing used in the baseline (v1) dataset
train     : paraphrases mixed into the v2 training data
unseen    : paraphrases NEVER used in training (evaluation only)
"""
COLORS = {"bin_blue": "blue", "bin_green": "green", "bin_orange": "orange"}
CANONICAL_ITEM = {"can": "can", "bottle": "plastic bottle", "paper": "paper cup"}

TRAIN_ITEM = {"can": ["can", "soda can", "aluminum can", "empty can"],
              "bottle": ["plastic bottle", "PET bottle", "water bottle", "bottle"],
              "paper": ["paper cup", "used paper cup", "cup", "paper trash"]}
TRAIN_BIN = ["bin", "box", "container", "tray"]
TRAIN_TEMPLATES = [
    "put the {item} in the {color} {bin}",
    "place the {item} into the {color} {bin}",
    "move the {item} to the {color} {bin}",
    "drop the {item} in the {color} {bin}",
    "pick up the {item} and put it in the {color} {bin}",
    "the {item} goes in the {color} {bin}",
    "throw the {item} into the {color} {bin}",
    "{item} to the {color} {bin}, please",
]

UNSEEN_ITEM = {"can": ["beverage can", "tin can"],
               "bottle": ["drink bottle", "plastic drink bottle"],
               "paper": ["disposable cup", "paper drinking cup"]}
UNSEEN_BIN = ["bucket", "basket"]
UNSEEN_TEMPLATES = [
    "could you sort the {item} into the {color} {bin}?",
    "I want the {item} in the {color} {bin}",
    "please dispose of the {item} in the {color} {bin}",
    "grab the {item} and leave it in the {color} {bin}",
]


def canonical(item, bin_id):
    return f"put the {CANONICAL_ITEM[item]} in the {COLORS[bin_id]} bin"


def sample(item, bin_id, rng, kind):
    if kind == "canonical":
        return canonical(item, bin_id)
    if kind == "train":
        items, bins, templates = TRAIN_ITEM, TRAIN_BIN, TRAIN_TEMPLATES
    else:
        items, bins, templates = UNSEEN_ITEM, UNSEEN_BIN, UNSEEN_TEMPLATES
    return rng.choice(templates).format(item=rng.choice(items[item]),
                                        color=COLORS[bin_id], bin=rng.choice(bins))
