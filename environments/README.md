# Custom Environments for Virtual Asimov 1

VASIMOV allows you to drop custom environment files (`.yaml`, `.yml`, or `.xml`) directly into this `environments/` directory or load them from any path on your machine.

---

## 1. Quick Start

### Run simulator with a custom environment:
```bash
# By name (auto-discovered from environments/)
./run_sim.sh --env corridor
./run_sim.sh --env arena
./run_sim.sh --env obstacle_course

# By relative/absolute file path
./run_sim.sh --env environments/corridor.yaml
./run_sim.sh --env environments/arena.xml
```

### Switch environment inside interactive console:
```text
asimov> env list                    # Shows all built-in presets and discovered custom files
asimov> env load corridor           # Loads corridor.yaml
asimov> env load arena              # Loads arena.xml
asimov> env load path/to/my_env.xml # Loads from arbitrary file path
```

---

## 2. Environment Formats

### Format A: Simple YAML (`.yaml` / `.yml`)
A clean, readable format for specifying ground friction, gravity, robot mass scaling, and dynamic/static obstacles:

```yaml
name: my_test_env
description: "My custom training course"

# Ground friction (default 1.0; low=ice, high=rubber)
ground_friction: 1.0

# Robot mass scaling (default 1.0; e.g. 1.15 for +15% payload)
mass_scale: 1.0

# Gravity vector [x, y, z] (default [0, 0, -9.81])
gravity: [0.0, 0.0, -9.81]

# List of obstacles
obstacles:
  # Dynamic box with 6-DoF freejoint (can be pushed, kicked, knocked over)
  - name: "crate_1"
    type: "box"            # box, sphere, cylinder
    pos: [1.2, 0.0, 0.15]  # [x, y, z] in meters
    size: [0.15, 0.15, 0.15]
    mass: 3.5              # kg
    friction: 1.0
    dynamic: true          # true = movable with physics, false = anchored
    rgba: [0.85, 0.35, 0.15, 1.0]

  # Dynamic rolling ball
  - name: "ball_1"
    type: "sphere"
    pos: [0.8, -0.2, 0.12]
    size: [0.12]           # radius
    mass: 1.8
    dynamic: true
    rgba: [0.2, 0.7, 0.9, 1.0]

  # Static platform / barrier
  - name: "static_wall"
    type: "box"
    pos: [2.5, 0.0, 0.25]
    size: [0.1, 1.0, 0.25]
    dynamic: false
    rgba: [0.5, 0.5, 0.5, 1.0]
```

### Format B: Pure MuJoCo XML (`.xml`)
Drop raw MuJoCo MJCF elements directly into an XML file:

```xml
<mujoco>
  <worldbody>
    <!-- Add perimeter walls, ramps, stairs, or obstacles -->
    <geom name="wall_north" type="box" size="3 0.1 0.5" pos="0 3 0.5" rgba="0.6 0.6 0.7 0.8" />
    <geom name="wall_south" type="box" size="3 0.1 0.5" pos="0 -3 0.5" rgba="0.6 0.6 0.7 0.8" />

    <!-- 6-DoF Dynamic body -->
    <body name="push_box" pos="1.5 0 0.15">
      <freejoint name="push_box_joint" />
      <geom name="push_box_geom" type="box" size="0.15 0.15 0.15" mass="4.0" rgba="0.9 0.4 0.1 1" />
    </body>
  </worldbody>
</mujoco>
```

---

## 3. First-Person & Chase Cameras
The robot model includes follow cameras rigidly locked to the robot floating base:
- **`chase`** (`chase_camera`): Elevated third-person camera 1.5m behind the robot. Turns and moves with robot heading.
- **`fpv_behind`** (`first_person_behind`): Close over-the-shoulder follow camera 0.8m behind robot.
- **`fpv`** (`first_person_camera`): True eye-level first-person view from the robot's head.

Launch with:
```bash
./run_sim.sh --camera chase
./run_sim.sh --camera fpv
```
Toggle live during teleoperation (`drive` mode) by pressing **`V`**!
