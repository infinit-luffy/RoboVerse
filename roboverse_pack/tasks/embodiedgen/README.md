# EmbodiedGen Tasks

This directory contains tasks using assets from EmbodiedGen demo assets.

## Available Tasks

- **Put Banana Task** (`embodiedgen.put_banana`) - Pick up banana and place it inside the mug (multi-object scene)
- **Put Mug Task** (`embodiedgen.put_mug`) - Pick and place mug to target location

---

## Put Banana Task

### Description

The `PutBananaTask` requires a Franka Panda robot to pick up a banana and place it inside a mug. The scene contains multiple objects on the table (book, lamp, remote control, rubik's cube, vase) to create a realistic and challenging environment.

### Task Details

- **Task Name**: `embodiedgen.put_banana` (registry), `put_banana` (alias)
- **Max Episode Steps**: 250
- **Robot**: Franka Panda
- **Objects**: 
  - Table (fixed geometry)
  - Banana (movable rigid body - to be picked)
  - Mug (movable rigid body - target container)
  - Book, Lamp, Remote Control, Rubik's Cube, Vase (scene objects)

### Initial State

- **Table**: Position `[0.4, -0.2, 0.4]`
- **Banana**: Position `[0.28, -0.58, 0.825]` (left side of table)
- **Mug** (target): Position `[0.68, -0.34, 0.863]` (right side of table)
- **Other objects**: Distributed across the table (book, lamp, remote, rubik's cube, vase)
- **Robot**: Position `[0.8, -0.8, 0.78]` with default joint configuration

### Success Criteria

The task is considered successful when the banana is placed inside the mug:
- The banana must be within ±0.04m (X, Y) and -0.03m to +0.06m (Z) relative to the mug center (at 0.05m height)
- This ensures the banana is actually inside the mug container
- Orientation is ignored (banana can be placed in any orientation)

### Usage

#### As a registered task:

```python
from metasim.task.registry import make_task

# Create the task
task = make_task("embodiedgen.put_banana", num_envs=4, device="cuda")

# Reset and run
obs = task.reset()
action = ...  # Your action here
obs, reward, terminated, truncated, info = task.step(action)
```

#### Direct instantiation:

```python
from roboverse_pack.tasks.embodiedgen import PutBananaTask

# Create task
task = PutBananaTask(scenario=PutBananaTask.scenario, device="cuda")

# Use with simulator
from metasim.sim import make_sim_context

with make_sim_context(backend="mujoco", num_envs=4) as sim:
    task.handler = sim
    obs = task.reset()
    # ... run your task
```

### Testing

Run the test script to verify the task works:

```bash
python roboverse_pack/tasks/embodiedgen/test_put_banana.py
```

### Assets Used

All assets are from the EmbodiedGen demo assets located in:
```
roboverse_data/assets/EmbodiedGenData/demo_assets/
```

Objects used in the scene:
- `table/` - Table model (URDF, USD, MJCF)
- `banana/` - Banana model (object to manipulate)
- `mug/` - Mug model (target container)
- `book/`, `lamp/`, `remote_control/`, `rubik's_cube/`, `vase/` - Scene objects for realism

### Notes

- This task does NOT use trajectory files - all initial states are defined manually
- The task can be easily extended to use other objects from the demo assets
- The checker uses `RelativeBboxDetector` to determine task success

---

## Put Mug Task

### Description

The `PutMugTask` requires a Franka Panda robot to pick up a mug from one location on a table and place it at a target location marked on the table. This task is slightly more challenging than put_banana due to the mug's handle.

### Task Details

- **Task Name**: `embodiedgen.put_mug` (registry), `put_mug` (alias)
- **Max Episode Steps**: 250
- **Robot**: Franka Panda
- **Objects**: 
  - Table (fixed geometry)
  - Mug (movable rigid body)
  - Target marker (vase scaled down, visual only)

### Initial State

- **Table**: Position `[0.4, -0.2, 0.4]`
- **Mug**: Position `[0.68, -0.34, 0.863]` (right side of table)
- **Target**: Position `[0.30, 0.05, 0.82]` (left side of table)
- **Robot**: Position `[0.8, -0.8, 0.78]` with default joint configuration

### Success Criteria

The task is considered successful when the mug is placed within a bounding box around the target location:
- Tolerance: ±0.10m in X and Y directions, ±0.06m in Z direction (larger than banana due to handle)
- Orientation is ignored (mug can be placed in any orientation)

### Usage

Same as Put Banana Task, but use `"embodiedgen.put_mug"` or import `PutMugTask`.

---

## General Notes

- All tasks in this package do NOT use trajectory files - initial states are defined manually
- Tasks can be easily extended to use other objects from the demo assets
- All checkers use `RelativeBboxDetector` to determine task success
- The base class `EmbodiedGenBaseTask` supports both trajectory-based and manual initial states

