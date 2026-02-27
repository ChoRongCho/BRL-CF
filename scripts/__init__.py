# Global variable

GT_MODEL_CONFIDENCE = {
    "at": 0.99,
    "pose": 0.99,
    
    "predidcate1": 0.2,
    "predidcate2": 0.3,
    "predidcate3": 0.4,
    
    "ripe": 0.8,
    "unripe": 0.85,
}

ACTIONS = {
    "pick_tomato1": {
        "preconditions":["ripe_tomato1", "pose_tomato1_0.1_8.2_-0.1", "at_stem1_tomato1"],
        "gt_vals": 0
    },
    "pick_tomato2": {
        "preconditions":["ripe_tomato2", "pose_tomato2_0.3_8.1_-0.2", "at_stem1_tomato2"],
        "gt_vals": 0
    },
    "pick_tomato3": {
        "preconditions":["unripe_tomato3", "pose_tomato3_0.4_6.2_0.3", "at_stem1_tomato3"],
        "gt_vals": 0
    },
    "place_tomato1": {
        "preconditions":["at_robot1_stem1", "at_stem1_tomato1"],
        "gt_vals": 0
    },
    "place_tomato2": {
        "preconditions":["at_robot1_stem1", "at_stem1_tomato2"],
        "gt_vals": 0
    },
    "place_tomato3": {
        "preconditions":["at_robot1_stem1", "at_stem1_tomato3"],
        "gt_vals": 0
    },
    "action1": {
        "preconditions":["predidcate1_dummy1"],
        "gt_vals": 0
    },
    "action2": {
        "preconditions":["predidcate1_dummy2"],
        "gt_vals": 0
    },
    "action3": {
        "preconditions":["predidcate2_dummy1"],
        "gt_vals": 0
    },
    "action4": {
        "preconditions":["predidcate2_dummy3"],
        "gt_vals": 0
    },
    "action5": {
        "preconditions":["predidcate3_dummy3"],
        "gt_vals": 0
    },
    "action6": {
        "preconditions":["predidcate1_dummy1", "predidcate1_dummy2"],
        "gt_vals": 0
    },
    "action7": {
        "preconditions":["predidcate2_dummy1", "predidcate3_dummy3"],
        "gt_vals": 0
    }
}

