"""Saved poses from keyboard control"""

import torch

# Saved at: 2025-10-15 20:57:58

poses = {
    "objects": {
        "cube": {
            "pos": torch.tensor([0.590000, -0.200001, 0.049892]),
            "rot": torch.tensor([1.000000, 0.000000, -0.000000, 0.000001]),
        },
        "sphere": {
            "pos": torch.tensor([0.400013, -0.599987, 0.099633]),
            "rot": torch.tensor([1.000000, -0.000054, 0.000051, 0.000000]),
        },
        "bbq_sauce": {
            "pos": torch.tensor([0.700018, -0.299995, 0.109172]),
            "rot": torch.tensor([1.000000, 0.000000, 0.000000, -0.000085]),
        },
        "box_base": {
            "pos": torch.tensor([0.500016, 0.200004, 0.075224]),
            "rot": torch.tensor([-0.000002, 0.707104, -0.000002, 0.707110]),
            "dof_pos": {"box_joint": -0.2539360821247101},
        },
    },
    "robots": {
        "franka": {
            "pos": torch.tensor([0.000000, 0.000000, 0.000000]),
            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
            "dof_pos": {
                "panda_finger_joint1": 0.040922,
                "panda_finger_joint2": 0.040922,
                "panda_joint1": -0.020000,
                "panda_joint2": -0.785398,
                "panda_joint3": -0.000000,
                "panda_joint4": -2.356194,
                "panda_joint5": 0.000000,
                "panda_joint6": 1.570796,
                "panda_joint7": 0.785398,
            },
        },
    },
}
