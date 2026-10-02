# Official Menlo Asimov 1 Locomotion Policy

- **Policy Identifier**: `official_locomotion`
- **Origin**: Hugging Face `Menlo/asimov1-locomotion-0818`
- **Category**: `locomotion`
- **Control Frequency**: 50.0 Hz (period 0.020 s, 4 physics steps @ 200 Hz)

## Observation Specification
78-dimensional float32 vector:
- `base_ang_vel` (3 dims, scale 0.25): IMU angular velocity in pelvis local frame
- `projected_gravity` (3 dims, scale 1.0): Unit gravity vector in pelvis local frame
- `velocity_commands` (3 dims, scale 1.0): Commanded twist `[vx, vy, vyaw]`
- `joint_pos_slot01` (9 dims): Relative position `q - q_default`
- `joint_pos_slot23` (8 dims): Relative position `q - q_default`
- `joint_pos_slot45` (6 dims): Relative position `q - q_default`
- `joint_vel_slot01` (9 dims, scale 0.10): Scaled velocity
- `joint_vel_slot23` (8 dims, scale 0.10): Scaled velocity
- `joint_vel_slot45` (6 dims, scale 0.10): Scaled velocity
- `actions` (23 dims): Previous step's raw action vector

## Action Specification
23-dimensional float32 vector:
- Action type: `joint_position`
- Target transform: `q_target = default_joint_pos + 0.25 * raw_action`
- Actuates all 23 primary humanoid joints (neck pitch and yaw unactuated).
