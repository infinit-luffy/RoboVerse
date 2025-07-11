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
    --test #to test with trained checkpoints
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

## Task list

- [x]  ShadowHandOver
- [x]  ShadowHandCatchUnderarm
- [x]  ShadowHandOver2Underarm
- [ ]  ShadowHandPushBlock
    - Still training bugs, can't be successfully solved
- [x]  ShadowHandTurnBotton
- [ ]  ShadowHandCatchAbreast
    - Isn't be fully solved by original repo
- [ ]  ShadowHandLiftUnderarm
- [ ]  ShadowHandSwingCup
- [ ]  ShadowHandCatchTwoCatchUnderarm
    - Isn't be fully solved by original repo
- [ ]  ShadowHandDoorOpenInward
- [ ]  ShadowHandDoorOpenOutward
- [ ]  ShadowHandDoorCloseInward
- [ ]  ShadowHandBottleCap
- [ ]  ShadowHandOpenScissors
- [ ]  ShadowHandOpenPenCap
- [ ]  ShadowHandSwingCup
- [ ]  ShadowHandGrasp AndPlace