export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python roboverse_learn/dexbench_rvrl/train.py \
--num_envs 128 \
--env_id dexbench/PushBlock \
--device cuda:0 \
# --robot allegro \

# python roboverse_learn/dexbench_rvrl/train.py \
# --num_envs 128 \
# --env_id dexbench/DoorCloseInward \
# --device cuda:0 \
# --algo sac \
# --obs_type rgb \
# --headless \

# python roboverse_learn/dexbench_rvrl/train.py \
# --num_envs 16 \
# --env_id dexbench/HandOver \
# --device cuda:0 \
# --headless \
# --algo dm3 \
# --obs_type rgb \

# python roboverse_learn/dexbench_rvrl/train.py \
# --num_envs 8 \
# --env_id dexbench/HandOver \
# --device cuda:0 \
# --algo tdmpc2 \
# --obs_type rgb \
# --no_prio \
