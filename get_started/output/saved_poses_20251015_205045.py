"""Saved poses from keyboard control"""

import torch

# Saved at: 2025-10-15 20:50:45

poses = {
    "objects": {
        "cube": {
            "pos": torch.tensor([0.300000, -0.200001, 0.049892]),
            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000001]),
        },
        "sphere": {
            "pos": torch.tensor([0.400007, -0.599992, 0.099633]),
            "rot": torch.tensor([1.000000, -0.000026, 0.000025, 0.000000]),
        },
        "bbq_sauce": {
            "pos": torch.tensor([0.700018, -0.299995, 0.109172]),
            "rot": torch.tensor([1.000000, 0.000000, 0.000000, -0.000086]),
        },
        "box_base": {
            "pos": torch.tensor([0.500010, 0.200000, 0.075224]),
            "rot": torch.tensor([-0.000005, 0.707104, 0.000001, 0.707109]),
            "dof_pos": {"box_joint": -0.2539360523223877},
        },
    },
    "robots": {
        "franka": {
            "pos": torch.tensor([0.090000, 0.000000, 0.000000]),
            "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
            "dof_pos": {
                "panda_finger_joint1": 0.040922,
                "panda_finger_joint2": 0.040922,
                "panda_joint1": 0.000000,
                "panda_joint2": -0.785398,
                "panda_joint3": -0.000000,
                "panda_joint4": -2.356194,
                "panda_joint5": -0.000000,
                "panda_joint6": 1.570796,
                "panda_joint7": 0.785398,
            },
        },
    },
}
