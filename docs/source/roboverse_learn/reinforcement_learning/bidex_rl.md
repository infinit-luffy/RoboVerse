# Bidex RL

We are migrating the bimanual dexterous hand (bidex) reinforcement learning tasks from the [DexterousHands GitHub repository](https://github.com/PKU-MARL/DexterousHands) into our framework.

RL algorithm: `PPO`

Simulator: `IsaacGym` and `IsaacLab` will be supported soon

## Installation

```bash
pip install wandb
```

Wandb login, enter your wandb account token.

```bash
wandb login
```

## Training

- IsaacGym:

    ```bash
    python roboverse_learn/bidex/train.py
    --task ShadowHandOver # task to be trained
    --num_envs 128
    --episode_length 75
    --objects ["egg", "cube"] # default egg, please refer to specific task configuration for objects supported
    --seed # default 42
    --use_wandb # to use wandb
    --headless # to train in headless server
    --test # to test with trained checkpoints
    --model_dir logs/ShadowHandOver/ppo_seed42_cube/model_0.pt # to load and continue training or test with trained checkpoints
    ```
    For the **Shadow Hand Over** task, after training for around 1–2k steps, the success rate reaches approximately **20%**. A success demo and the reference curve are provided below.
<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 68%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/bidex_rl/shadow_hand_over_demo.mp4" type="video/mp4" alt="IsaacGym Success Demo" style="width: 88%;">
            </video>
            <!-- <p style="margin-top: 5px;">Isaac Gym</p> -->
        </div>
        <div style="width: 68%; text-align: center;">
            <img src="https://roboverse.wiki/_static/standard_output/bidex_rl/shadow_hand_over_isaacgym_curve.png" alt="IsaacGym Training Curve" style="width: 72%;">
            <!-- <p style="margin-top: 5px;">Isaac Lab</p> -->
        </div>
    </div>
</div>

## Observation Space
Following the original repo, the observation of all tasks is composed of three parts: the state values of the left and right hands, and the information of objects and target. The state values of the left and right hands were the same for each task, including hand joint and finger positions, velocity, and force information. The state values of the object and goal are different for each task, which we will describe in the following. Here gives the specific information of the left-hand and right-hand state values. Note that the observation is slightly different in the HandOver and ReOreintation task due to the fixed base.
#### <span id="obs_normal">Observation space of dual shadow hands</span>
| Index | Description |
|  :----:  | :----:  |
| 0 - 23 | right shadow hand dof position |
| 24 - 47 |	right shadow hand dof velocity |
| 48 - 71 | right shadow hand dof force |
| 72 - 136 |	right shadow hand fingertip pose, linear velocity, angle velocity (5 x 13) |
| 137 - 166 |	right shadow hand fingertip force, torque (5 x 6) |
| 167 - 169 |	right shadow hand base position |
| 170 - 172 |	right shadow hand base rotation |
| 173 - 198 |	right shadow hand actions |
| 199 - 222 |	left shadow hand dof position |
| 223 - 246 |	left shadow hand dof velocity |
| 247 - 270 |   left shadow hand dof force |
| 271 - 335 |	left shadow hand fingertip pose, linear velocity, angle velocity (5 x 13) |
| 336 - 365 |	left shadow hand fingertip force, torque (5 x 6) |
| 366 - 368 |	left shadow hand base position |
| 369 - 371 |	left shadow hand base rotation |
| 372 - 397 |	left shadow hand actions |

## Tasks

### ShadowHandOver
This environment consists of two shadow hands with palms facing up, opposite each other, and an object that needs to be passed. In the beginning, the object will fall randomly in the area of the shadow hand on the right side. Then the hand holds the object and passes the object to the other hand. Note that the base of the hand is fixed. More importantly, the hand which holds the object initially can not directly touch the target, nor can it directly roll the object to the other hand, so the object must be thrown up and stays in the air in the process.
There are 398-dimensional observations and 40-dimensional actions in the task. Additionally, the reward function is related to the pos error between the object and the target. When the pos error gets smaller, the reward increases dramatically. In addition to the pose-based reward, we introduce a postive throw bonus component granted when the object is detected to be airborne (i.e., not in contact with any hand or surface), indicating a successful throw attempt.To use the HandOver environment, pass `--task=ShadowHandOver`
#### <span id="obs1">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 373 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 374 - 380 |	object pose |
| 381 - 383 |	object linear velocity |
| 384 - 386 |	object angle velocity |
| 387 - 393 |	goal pose |
| 394 - 397 |	goal rot - object rot |

#### <span id="action1">Action Space</span>

| Index | Description |
|  ----  | ----  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 39 |	left shadow hand actuated joint |

#### <span id="r1">Rewards</span>
Rewards is the pose distance between object and goal, and the specific formula is as follows:
```python
diff_xy = target_pos[:, :2] - object_pos[:, :2]
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
reward_dist = goal_dist

reward = torch.exp(-0.2*(reward_dist * dist_reward_scale))
thrown = (diff_xy[:, 1] >= -0.1) & (diff_xy[:, 1] <= -0.06) & (object_pos[:, 2] >= 0.4)& env_throw_bonus
reward = torch.where(thrown, reward + throw_bonus, reward)
```

### ShadowHandCatchUnderarm
In this problem, two shadow hands with palms facing upwards are controlled to pass an object from one palm to the other. What makes it more difficult than the Handover problem is that the hands' translation and rotation degrees of freedom are no longer frozen but are added into the action space. To use the HandCatchUnderarm environment, pass `--task=ShadowHandCatchUnderarm`
#### <span id="obs2">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	object pose |
| 405 - 407 |	object linear velocity |
| 408 - 410 |	object angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |

#### <span id="action2">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 2 |	right shadow hand base translation |
| 3 - 5 |	right shadow hand base rotation |
| 6 - 25 |	right shadow hand actuated joint |
| 26 - 28 |	left shadow hand base translation |
| 29 - 31 |	left shadow hand base rotation |
| 32 - 51 |	left shadow hand actuated joint |

#### <span id="r2">Rewards</span>

Rewards is similar to that of task ShadowHandOver, and the specific formula is as follows:
```python
diff_xy = target_pos[:, :2] - object_pos[:, :2]
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
reward_dist = goal_dist

reward = torch.exp(-0.2*(reward_dist * dist_reward_scale))
thrown = (diff_xy[:, 1] >= -0.25) & (diff_xy[:, 1] <= -0.1) & (object_pos[:, 2] >= 0.4)
reward = torch.where(thrown, reward + throw_bonus, reward)
```

###  ShadowHandOver2Underarm
This environment is like made up of half Hand Over and Catch Underarm, the object needs to be thrown from the vertical hand to the palm-up hand. To use the HandCatchUnderarm environment, pass `--task=ShadowHandCatchOver2Underarm`
#### <span id="obs3">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	object pose |
| 405 - 407 |	object linear velocity |
| 408 - 410 |	object angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |

#### <span id="action3">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 2 |	right shadow hand base translation |
| 3 - 5 |	right shadow hand base rotation |
| 6 - 25 |	right shadow hand actuated joint |
| 26 - 28 |	left shadow hand base translation |
| 29 - 31 |	left shadow hand base rotation |
| 32 - 51 |	left shadow hand actuated joint |

#### <span id="r3">Rewards</span>

Rewards is the pose distance between object and goal, and the specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
# Orientation alignment for the cube in hand and goal cube
quat_diff = quat_mul(object_rot, quat_conjugate(target_rot)

rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

dist_rew = goal_dist

reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))
```

### ShadowHandTwoCatchUnderarm Environments

This environment is similar to Catch Underarm, but with an object in each hand and the corresponding goal on the other hand. Therefore, the environment requires two objects to be thrown into the other hand at the same time, which requires a higher manipulation technique than the environment of a single object. To use the HandCatchUnderarm environment, pass `--task=ShadowHandTwoCatchUnderarm`

#### <span id="obs4">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	object1 pose |
| 405 - 407 |	object1 linear velocity |
| 408 - 410 |	object1 angle velocity |
| 411 - 417 |	goal1 pose |
| 418 - 421 |	goal1 rot - object1 rot |
| 422 - 428 |	object2 pose |
| 429 - 431 |	object2 linear velocity |
| 432 - 434 |	object2 angle velocity |
| 435 - 441 |	goal2 pose |
| 442 - 445 |	goal2 rot - object2 rot |

#### <span id="action4">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 2 |	right shadow hand base translation |
| 3 - 5 |	right shadow hand base rotation |
| 6 - 25 |	right shadow hand actuated joint |
| 26 - 28 |	left shadow hand base translation |
| 29 - 31 |	left shadow hand base rotation |
| 32 - 51 |	left shadow hand actuated joint |

#### <span id="r4">Rewards</span>

Rewards is the pose distance between two object and  two goal, this means that both objects have to be thrown in order to be swapped over. We also introduce a postive throw bonus component granted when each objects are detected to be airborne. The specific formula is as follows:
```python
diff_xy = right_goal_pos[:, :2] - right_object_pos[:, :2]
goal_dist = torch.norm(right_object_pos - right_goal_pos, p=2, dim=-1)
reward_dist = goal_dist
diff_another_xy = left_goal_pos[:, :2] - left_object_pos[:, :2]
goal_another_dist = torch.norm(left_object_pos - left_goal_pos, p=2, dim=-1)
reward_another_dist = goal_another_dist
reward = torch.exp(-0.2*(reward_dist * dist_reward_scale)) + torch.exp(-0.2*(reward_another_dist * dist_reward_scale))

right_thrown = (diff_xy[:, 1] >= -0.25) & (diff_xy[:, 1] <= -0.1) & (right_object_pos[:, 2] >= 0.4)
reward = torch.where(right_thrown, reward + throw_bonus, reward)
left_thrown = (diff_another_xy[:, 1] <= 0.25) & (diff_another_xy[:, 1] >= 0.1) & (left_object_pos[:, 2] >= 0.4)
reward = torch.where(left_thrown, reward + throw_bonus, reward)
```

### ShadowHandCatchAbreast Environments

This environment consists of two shadow hands placed side by side in the same direction and an object that needs to be passed. Compared with the previous environment which is more like passing objects between the hands of two people, this environment is designed to simulate the two hands of the same person passing objects, so different catch techniques are also required and require more hand translation and rotation techniques. To use the HandCatchAbreast environment, pass `--task=ShadowHandCatchAbreast`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	object pose |
| 405 - 407 |	object linear velocity |
| 408 - 410 |	object angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 2 |	right shadow hand base translation |
| 3 - 5 |	right shadow hand base rotation |
| 6 - 25 |	right shadow hand actuated joint |
| 26 - 28 |	left shadow hand base translation |
| 29 - 31 |	left shadow hand base rotation |
| 32 - 51 |	left shadow hand actuated joint |

#### <span id="r5">Rewards</span>

Rewards is similar to that of task ShadowHandCatchUnderarm, and the specific formula is as follows:
```python
diff_xy = target_pos[:, :2] - object_pos[:, :2]
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
reward_dist = goal_dist
reward = torch.exp(-0.2 * (dist_rew * dist_reward_scale)) - action_penalty * action_penalty_scale

thrown = (diff_xy[:, 1] >= -0.40) & (diff_xy[:, 1] <= -0.1) & (object_pos[:, 2] >= 0.4)
reward = torch.where(thrown, reward + throw_bonus, reward)
```

### ShadowHandDoorOpenOutward/DoorCloseInward Environments

These two environments require a closed/opened door to be opened/closed and the door can only be pushed outward or initially open inward. Both these two environments only need to do the push behavior, so it is relatively simple. To use the Door Open Outward/Door Close Inward environment, pass `--task=ShadowHandDoorOpenOutward/DoorCloseInward`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 400 |   right handle position |
| 401 - 403 |   left handle position |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 2 |	right shadow hand base translation |
| 3 - 5 |	right shadow hand base rotation |
| 6 - 25 |	right shadow hand actuated joint |
| 26 - 28 |	left shadow hand base translation |
| 29 - 31 |	left shadow hand base rotation |
| 32 - 51 |	left shadow hand actuated joint |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to the left handle, the distance from the right hand to the right handle, and the distance from the object to the target point.

```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

right_hand_dist = torch.norm(door_right_handle_pos - right_hand_pos, p=2, dim=-1)
left_hand_dist = torch.norm(door_left_handle_pos - left_hand_pos, p=2, dim=-1)

right_hand_finger_dist = (torch.norm(door_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(door_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(door_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(door_right_handle_pos - right_hand_lf_pos, p=2, dim=-1)
                        + torch.norm(door_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
left_hand_finger_dist = (torch.norm(door_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(door_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(door_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(door_left_handle_pos - left_hand_lf_pos, p=2, dim=-1)
                        + torch.norm(door_left_handle_pos - left_hand_th_pos, p=2, dim=-1))

right_hand_dist_rew = right_hand_finger_dist
left_hand_dist_rew = left_hand_finger_dist

up_rew = torch.zeros_like(right_hand_dist_rew)
# if door open outward:
up_rew = torch.where(right_hand_finger_dist < 0.5,
                torch.where(left_hand_finger_dist < 0.5,
                                torch.abs(door_right_handle_pos[:, 1] - door_left_handle_pos[:, 1]) * 2, up_rew), up_rew)
# if door close inward:
up_rew = torch.where(right_hand_finger_dist < 0.5,
                torch.where(left_hand_finger_dist < 0.5,
                                1 - torch.abs(door_right_handle_pos[:, 1] - door_left_handle_pos[:, 1]) * 2, up_rew), up_rew)

reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew
```

### ShadowHandDoorOpenInward/DoorCloseOutward Environments

These two environments also require a closed/opened door to be opened/closed and the door can only be pushed inward or initially open outward, but because they can't complete the task by simply pushing, which need to catch the handle by hand and then open or close it, so it is relatively difficult. To use the Door Open Outward/Door Close Inward environment, pass `--task=ShadowHandDoorOpenInward/DoorCloseOutward`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	object pose |
| 405 - 407 |	object linear velocity |
| 408 - 410 |	object angle velocity |
| 411 - 417 |	goal pose |
| 418 - 421 |	goal rot - object rot |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 2 |	right shadow hand base translation |
| 3 - 5 |	right shadow hand base rotation |
| 6 - 25 |	right shadow hand actuated joint |
| 26 - 28 |	left shadow hand base translation |
| 29 - 31 |	left shadow hand base rotation |
| 32 - 51 |	left shadow hand actuated joint |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to the left handle, the distance from the right hand to the right handle, and the distance from the object to the target point.

```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

right_hand_dist = torch.norm(door_right_handle_pos - right_hand_pos, p=2, dim=-1)
left_hand_dist = torch.norm(door_left_handle_pos - left_hand_pos, p=2, dim=-1)

right_hand_finger_dist = (torch.norm(door_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(door_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(door_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(door_right_handle_pos - right_hand_lf_pos, p=2, dim=-1)
                        + torch.norm(door_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
left_hand_finger_dist = (torch.norm(door_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(door_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(door_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(door_left_handle_pos - left_hand_lf_pos, p=2, dim=-1)
                        + torch.norm(door_left_handle_pos - left_hand_th_pos, p=2, dim=-1))

right_hand_dist_rew = right_hand_finger_dist
left_hand_dist_rew = left_hand_finger_dist

up_rew = torch.zeros_like(right_hand_dist_rew)
# if door close outward:
up_rew = torch.where(right_hand_finger_dist < 0.5,
                torch.where(left_hand_finger_dist < 0.5,
                                1 - torch.abs(door_right_handle_pos[:, 1] - door_left_handle_pos[:, 1]) * 2, up_rew), up_rew)

reward = 6 - right_hand_dist_rew - left_hand_dist_rew + up_rew

# if door open inward:
up_rew = torch.where(right_hand_finger_dist < 0.5,
                torch.where(left_hand_finger_dist < 0.5,
                                torch.abs(door_right_handle_pos[:, 1] - door_left_handle_pos[:, 1]) * 2, up_rew), up_rew)

reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew
```

### ShadowHandTurnButton Environments

This environment involves dual hands and two buttons, we need to use dual hand fingers to press the desired button. To use the ShadowHandTurnButton environment, pass `--task=ShadowHandTurnButton`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 400 |  right button handle position |
| 401 - 403 |   left button handle position |

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 2 |	right shadow hand base translation |
| 3 - 5 |	right shadow hand base rotation |
| 6 - 25 |	right shadow hand actuated joint |
| 26 - 28 |	left shadow hand base translation |
| 29 - 31 |	left shadow hand base rotation |
| 32 - 51 |	left shadow hand actuated joint |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to the left switch, the distance from the right hand to the right switch, and the distance between the button and button's desired goal.

```python
right_hand_dist = torch.norm(right_object_pos - right_hand_pos, p=2, dim=-1)
left_hand_dist = torch.norm(left_object_pos - left_hand_pos, p=2, dim=-1)

right_hand_finger_dist = (torch.norm(right_object_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(right_object_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(right_object_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(right_object_pos - right_hand_lf_pos, p=2, dim=-1)
                        + torch.norm(right_object_pos - right_hand_th_pos, p=2, dim=-1))
left_hand_finger_dist = (torch.norm(left_object_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(left_object_pos - left_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(left_object_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(left_object_pos - left_hand_lf_pos, p=2, dim=-1)
                        + torch.norm(left_object_pos - left_hand_th_pos, p=2, dim=-1))


right_hand_dist_rew = torch.exp(-10 * right_hand_finger_dist)
left_hand_dist_rew = torch.exp(-10 * left_hand_finger_dist)

up_rew = torch.zeros_like(right_hand_dist_rew)
up_rew = (1.4 - (switch_right_handle_pos[:, 2] + switch_left_handle_pos[:, 2])) * 50

reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew
```

### ShadowHandBlockStack Environments

This environment involves dual hands and two blocks, and we need to stack the block as a tower. To use the Stack Block environment, pass `--task=ShadowHandBlockStack`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	block1 pose |
| 405 - 407 |	block1 linear velocity |
| 408 - 410 |	block1 angle velocity |
| 411 - 417 |	block2 pose
| 418 - 420 |	block2 linear velocity
| 421 - 423 |	block2 angle velocity

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 2 |	right shadow hand base translation |
| 3 - 5 |	right shadow hand base rotation |
| 6 - 25 |	right shadow hand actuated joint |
| 26 - 28 |	left shadow hand base translation |
| 29 - 31 |	left shadow hand base rotation |
| 32 - 51 |	left shadow hand actuated joint |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to block1, the distance from the right hand to block2, and the distance between the block and desired goal.

```python
stack_pos1 = target_pos.clone()
stack_pos2 = target_pos.clone()

stack_pos1[:, 1] -= 0.1
stack_pos2[:, 1] -= 0.1
stack_pos1[:, 2] += 0.05

goal_dist1 = torch.norm(stack_pos1 - block_left_handle_pos, p=2, dim=-1)
goal_dist2 = torch.norm(stack_pos2 - block_right_handle_pos, p=2, dim=-1)

right_hand_dist = torch.norm(right_object_pos - right_hand_pos, p=2, dim=-1)
left_hand_dist = torch.norm(left_object_pos - left_hand_pos, p=2, dim=-1)

right_hand_finger_dist = (torch.norm(right_object_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(right_object_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(right_object_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(right_object_pos - right_hand_lf_pos, p=2, dim=-1)
                        + torch.norm(right_object_pos - right_hand_th_pos, p=2, dim=-1))
left_hand_finger_dist = (torch.norm(left_object_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(left_object_pos - left_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(left_object_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(left_object_pos - left_hand_lf_pos, p=2, dim=-1)
                        + torch.norm(left_object_pos - left_hand_th_pos, p=2, dim=-1))

right_hand_dist_rew = right_hand_finger_dist
left_hand_dist_rew = left_hand_finger_dist

up_rew = torch.zeros_like(right_hand_dist_rew)
up_rew = torch.where(right_hand_finger_dist < 0.5,
                torch.where(left_hand_finger_dist < 0.5,
                    (0.24 - goal_dist1 - goal_dist2) * 2, up_rew), up_rew)

stack_rew = torch.zeros_like(right_hand_dist_rew)
stack_rew = torch.where(goal_dist2 < 0.07,
                torch.where(goal_dist1 < 0.07,
                    (0.05-torch.abs(stack_pos1[:, 2] - block_left_handle_pos[:, 2])) * 50 ,stack_rew),stack_rew)

reward = 1.5 - right_hand_dist_rew - left_hand_dist_rew + up_rew + stack_rew
```

### ShadowHandPushBlock Environments

This environment involves two hands and two blocks, we need to use both hands to reach and push the block to the desired goal separately. This is a relatively simple task. To use the Push Block environment, pass `--task=ShadowHandPushBlock`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	block1 pose |
| 405 - 407 |	block1 linear velocity |
| 408 - 410 |	block1 angle velocity |
| 411 - 417	|   block2 pose
| 418 - 420 |	block2 linear velocity
| 421 - 423 |	block2 angle velocity
| 424 - 426 |   left goal position
| 427 - 429 |   right goal position

#### <span id="action5">Action Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 2 |	right shadow hand base translation |
| 3 - 5 |	right shadow hand base rotation |
| 6 - 25 |	right shadow hand actuated joint |
| 26 - 28 |	left shadow hand base translation |
| 29 - 31 |	left shadow hand base rotation |
| 32 - 51 |	left shadow hand actuated joint |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left hand to block1, the distance from the right hand to block2, and the distance between the block and desired goal.

```python
left_goal_dist = torch.norm(target_pos - block_left_handle_pos, p=2, dim=-1)
right_goal_dist = torch.norm(target_pos - block_right_handle_pos, p=2, dim=-1)

right_hand_finger_dist = (torch.norm(block_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(block_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(block_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(block_right_handle_pos - right_hand_lf_pos, p=2, dim=-1)
                        + torch.norm(block_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
left_hand_finger_dist = (torch.norm(block_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(block_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                        + torch.norm(block_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(block_left_handle_pos - left_hand_lf_pos, p=2, dim=-1)
                        + torch.norm(block_left_handle_pos - left_hand_th_pos, p=2, dim=-1))

right_hand_dist_rew = torch.exp(-10*right_hand_finger_dist)
left_hand_dist_rew = torch.exp(-10*left_hand_finger_dist)

up_rew = torch.zeros_like(right_hand_dist_rew)
up_rew = (torch.exp(-10*left_goal_dist) + torch.exp(-10*right_goal_dist)) * 2

reward = right_hand_dist_rew + left_hand_dist_rew + up_rew
```
### ShadowHandReOrientation Environment

This environment involves two hands and two objects. Each hand holds an object and we need to reorient the object to the target orientation. To use the Re Orientation environment, pass `--task=ShadowHandReOrientation`

#### <span id="obs5">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 397 |	dual hands observation shown in [Observation space of dual shadow hands](#obs_normal)|
| 398 - 404 |	object1 pose |
| 405 - 407 |	object1 linear velocity |
| 408 - 410 |	object1 angle velocity |
| 411 - 417 |	goal1 pose |
| 418 - 421 |	goal1 rot - object1 rot |
| 422 - 428 |	object2 pose |
| 429 - 431 |	object2 linear velocity |
| 432 - 434 |	object2 angle velocity |
| 435 - 441 |	goal2 pose |
| 442 - 445 |	goal2 rot - object2 rot |

#### <span id="action5">Action Space</span>

| Index | Description |
|  ----  | ----  |
| 0 - 19 |	right shadow hand actuated joint |
| 20 - 39 |	left shadow hand actuated joint |

#### <span id="r5">Rewards</span>

The reward consists of three parts: the distance from the left object to the left object goal, the distance from the right object to the right object goal, and the distance between the object and desired goal.

```python
goal_dist = torch.norm(right_object_pos - right_goal_pos, p=2, dim=-1)
goal_another_dist = torch.norm(left_object_pos - left_goal_pos, p=2, dim=-1)

quat_diff = math.quat_mul(
    right_object_rot, math.quat_inv(right_goal_rot)
)  # (num_envs, 4)
rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))

quat_another_diff = math.quat_mul(
    left_object_rot, math.quat_inv(left_goal_rot)
)  # (num_envs, 4)
rot_another_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_another_diff[:, 1:4], p=2, dim=-1), max=1.0))

dist_rew = goal_dist * dist_reward_scale + goal_another_dist * dist_reward_scale
rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale + 1.0/(torch.abs(rot_another_dist) + rot_eps) * rot_reward_scale

reward = dist_rew + rot_rew + action_penalty * action_penalty_scale
```

- [ ]  ShadowHandBottleCap
    - To train
- [ ]  ShadowHandLiftUnderarm
    - To train
- [ ]  ShadowHandOpenScissors
    - To train
- [ ]  ShadowHandOpenPenCap
    - To train
- [ ]  ShadowHandSwingCup
    - To train
- [ ]  ShadowHandPourWater
    - Isn't fully solved by original repo
- [ ]  ShadowHandGraspAndPlace
    - Isn't fully solved by original repo
