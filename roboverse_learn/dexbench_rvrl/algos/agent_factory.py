from ppo import PPO


def create_agent(
    algo: str,
    env,
    train_cfg,
    device="cpu",
    log_dir="run",
    model_dir=None,
    is_testing=False,
    print_log=True,
    wandb_run=None,
):
    """Create an RL agent.
    Args:
        algo: Algorithm to use. Currently only supports "PPO".
        env: Vectorized environment.
        train_cfg: Training configuration dictionary.
        device: Device to run on.
        log_dir: Directory to save logs and models.
        model_dir: Directory to load model from.
        is_testing: Whether the agent is in testing mode.
        print_log: Whether to print logs to console.
        wandb_run: Wandb run object for logging."

    Returns:
        Agent object.
    """
    ALGO_MAP = {
        "PPO": PPO,
    }

    assert algo.upper() in ALGO_MAP, f"Algorithm {algo} not supported. Supported algorithms: {list(ALGO_MAP.keys())}"

    agent = ALGO_MAP[algo.upper()](
        vec_env=env,
        cfg_train=train_cfg,
        device=device,
        log_dir=log_dir,
        model_dir=model_dir,
        is_testing=is_testing,
        print_log=print_log,
        wandb_run=wandb_run,
    )

    return agent
