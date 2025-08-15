# python roboverse_learn/bidex/train.py \
# --num_envs 2048 \
# --task ShadowHandPushBlock \
# --headless \
# --device cuda:0 \
# --use_wandb \
# # --experiment Tensorized_reset

# python roboverse_learn/bidex/train.py \
# --num_envs 2048 \
# --task ShadowHandOver \
# --headless \
# --device cuda:1 \
# # --use_wandb \
# --objects egg \
# # --experiment Tensorized_reset

# python roboverse_learn/bidex/train.py \
# --num_envs 2048 \
# --task ShadowHandOver2Underarm \
# --headless \
# --device cuda:2 \
# --use_wandb \
# --objects egg \
# # --experiment Tensorized_reset

# python roboverse_learn/bidex/train.py \
# --num_envs 128 \
# --task ShadowHandTurnButton \
# --headless \
# --device cuda:3 \
# # --use_wandb \
# # --experiment Tensorized_reset

# python roboverse_learn/bidex/train.py \
# --num_envs 2048 \
# --task ShadowHandCatchAbreast \
# --headless \
# --device cuda:6 \
# --use_wandb \
# --objects egg \
# --experiment orig_reward \
# --model_dir /home/user/RoboVerse/logs/ShadowHandCatchAbreast/ppo_seed42_egg/model_1000.pt \

# python roboverse_learn/bidex/train.py \
# --num_envs 128 \
# --task ShadowHandLiftUnderarm \
# --headless \
# --device cuda:7 \
# --use_wandb \
# --experiment orig_reward \
# --model_dir /home/user/RoboVerse/logs/ShadowHandCatchAbreast/ppo_seed42_egg/model_1000.pt \

# python roboverse_learn/bidex/train.py \
# --num_envs 128 \
# --task ShadowHandSwingCup \
# --headless \
# --device cuda:7 \
# --use_wandb \

# python roboverse_learn/bidex/train.py \
# --num_envs 2048 \
# --task ShadowHandPushBlock \
# --headless \
# --device cuda:0 \
# --obs_type state \
# --use_wandb \

# python roboverse_learn/bidex/train.py \
# --num_envs 2048 \
# --task ShadowHandOpenOutward \
# --headless \
# --device cuda:0 \
# --obs_type state \
# --use_wandb \

# python roboverse_learn/bidex/train.py \
# --num_envs 128 \
# --task ShadowHandStackBlock \
# --headless \
# --device cuda:2 \
# --obs_type rgb \
# --use_wandb \

# python roboverse_learn/bidex/train.py \
# --num_envs 128 \
# --task ShadowHandOpenOutward \
# --headless \
# --device cuda:6 \
# --use_wandb \
# --model_dir /home/user/RoboVerse/logs/ShadowHandOpenOutward/ppo_seed42_door/model_7000.pt \
# --use_wandb \

# s
# --use_wandb \
# --experiment orig_reward \

# python roboverse_learn/bidex/train.py \
# --num_envs 128 \
# --task ShadowHandOpenInward \
# --headless \
# --device cuda:7 \
# --use_wandb \

# python roboverse_learn/bidex/train.py \
# --num_envs 1 \
# --task ShadowHandScissor \
# --headless \
# --device cuda:3 \
# --use_wandb \

python roboverse_learn/bidex/train.py \
--num_envs 128 \
--task ShadowHandOver \
--headless \
--device cuda:1 \
--obs_type rgb \
--use_wandb \
--no_prio \

# python roboverse_learn/bidex/train.py \
# --num_envs 2048 \
# --task ShadowHandStackBlock \
# --headless \
# --device cuda:4 \
# --use_wandb \
# --experiment collision_filter_reset \

# python roboverse_learn/bidex/train.py \
# --num_envs 128 \
# --task ShadowHandReOrientation \
# --headless \
# --device cuda:0 \
# --use_wandb \

# python roboverse_learn/bidex/train.py \
# --num_envs 128 \
# --task ShadowHandTwoCatchUnderarm \
# --headless \
# --device cuda:0 \

# python roboverse_learn/bidex/train.py \
# --num_envs 256 \
# --task ShadowHandBottle \
# --headless \
# --device cuda:0 \

# python roboverse_learn/bidex/train.py \
# --num_envs 128 \
# --task ShadowHandOver \
# --headless \
# --device cuda:0 \
# --obs_type rgb \
# --no_prio \
# --use_wandb \
