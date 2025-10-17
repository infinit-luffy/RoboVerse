"""Sub-module containing utilities for setting up the environment."""

from __future__ import annotations

import importlib

from metasim.utils import is_camel_case, is_snake_case, to_camel_case, to_snake_case

from .base_cfg import BaseTaskCfg


def get_task(task_id: str) -> BaseTaskCfg:
    """Get the task cfg instance from the task id.

    Args:
        task_id: The id of the task.

    Returns:
        The task cfg instance.
    """
    if ":" in task_id:
        prefix, task_name = task_id.split(":")
    else:
        prefix, task_name = None, task_id

    if is_camel_case(task_name):
        task_name_camel = task_name
        task_name_snake = to_snake_case(task_name)
    elif is_snake_case(task_name):
        task_name_camel = to_camel_case(task_name)
        task_name_snake = task_name
    else:
        raise ValueError(f"Invalid task name: {task_id}, should be in either camel case or snake case")

    import_path = "roboverse_pack.tasks.dexbench"
    module = importlib.import_module(import_path)
    task_cls = getattr(module, f"{task_name_camel}Cfg")
    return task_cls
