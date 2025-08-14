# python roboverse_learn/bidex/train.py \
# --num_envs 4 \
# --task ShadowHandPushBlock \
# --headless \
# --device cuda:1 \
# --model_dir /home/user/RoboVerse/logs/ShadowHandPushBlock/ppo_seed42_cube/model_33000.pt \
# --test

# python roboverse_learn/bidex/train.py \
# --num_envs 1024 \
# --task ShadowHandCatchAbreast \
# --headless \
# --device cuda:0 \
# --model_dir /home/user/RoboVerse/logs/ShadowHandCatchAbreast/ppo_seed42_egg/model_8000.pt \
# --test

# python roboverse_learn/bidex/train.py \
# --num_envs 128 \
# --task ShadowHandOver2Underarm \
# --headless \
# --device cuda:1 \
# --model_dir logs/ShadowHandOver2Underarm/ppo_seed42_egg/model_4000.pt \
# --test


python roboverse_learn/bidex/train.py \
--num_envs 2048 \
--task ShadowHandStackBlock \
--headless \
--device cuda:1 \
--model_dir /home/user/RoboVerse/logs/ShadowHandStackBlock/ppo_seed42_cube_state/model_9000.pt \
--test \
