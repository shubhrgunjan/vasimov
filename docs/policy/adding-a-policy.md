# Guide: Adding a New Policy to VASIMOV

This guide walks through adding a new policy to the VASIMOV simulation platform. The policy-pluggable architecture allows you to drop in a new policy into the `policies/` directory without modifying the simulator core.

---

## Step 1: Create the Policy Directory

Create a directory under `policies/official/` or `policies/custom/`:

```bash
mkdir -p policies/custom/my_new_policy
```

Your directory structure should look like this:

```text
policies/custom/my_new_policy/
├── policy.onnx          # Model weights artifact
├── manifest.yaml        # Self-describing policy contract
└── README.md            # Documentation, author, provenance
```

---

## Step 2: Add the Policy Model Artifact

Place your exported model (e.g. `policy.onnx`) into the directory.

Verify model input and output shapes using Python or `onnx`:
```python
import onnxruntime as ort
session = ort.InferenceSession("policies/custom/my_new_policy/policy.onnx")
print("Inputs :", [(i.name, i.shape, i.type) for i in session.get_inputs()])
print("Outputs:", [(o.name, o.shape, o.type) for o in session.get_outputs()])
```

---

## Step 3: Create the Policy Manifest (`manifest.yaml`)

Every policy requires a machine-readable `manifest.yaml`.

```yaml
name: my_new_policy
version: "1.0"
category: locomotion  # locomotion | recovery | balance | manipulation
description: "My custom Asimov 1 controller"

runtime:
  type: onnx
  execution_provider: cpu

model:
  path: policy.onnx

control:
  frequency_hz: 50.0  # Decimation = 4 for 200 Hz physics loop

observation:
  dim: 78
  dtype: float32
  history_steps: 1
  requires:
    - base_ang_vel_body
    - projected_gravity
    - command_vel
    - joint_pos
    - joint_vel
    - previous_actions

action:
  dim: 23
  dtype: float32
  type: joint_position
  joint_order:
    - left_hip_pitch_joint
    - left_hip_roll_joint
    - left_hip_yaw_joint
    - left_knee_joint
    - left_ankle_pitch_joint
    - left_ankle_roll_joint
    - right_hip_pitch_joint
    - right_hip_roll_joint
    - right_hip_yaw_joint
    - right_knee_joint
    - right_ankle_pitch_joint
    - right_ankle_roll_joint
    - waist_yaw_joint
    - left_shoulder_pitch_joint
    - left_shoulder_roll_joint
    - left_shoulder_yaw_joint
    - left_elbow_joint
    - left_wrist_roll_joint
    - right_shoulder_pitch_joint
    - right_shoulder_roll_joint
    - right_shoulder_yaw_joint
    - right_elbow_joint
    - right_wrist_roll_joint

default_joint_pos:
  left_hip_pitch_joint: -0.2
  left_hip_roll_joint: 0.0
  left_hip_yaw_joint: 0.0
  left_knee_joint: 0.4
  left_ankle_pitch_joint: -0.2
  left_ankle_roll_joint: 0.0
  right_hip_pitch_joint: -0.2
  right_hip_roll_joint: 0.0
  right_hip_yaw_joint: 0.0
  right_knee_joint: 0.4
  right_ankle_pitch_joint: -0.2
  right_ankle_roll_joint: 0.0
  waist_yaw_joint: 0.0
  left_shoulder_pitch_joint: 0.2
  left_shoulder_roll_joint: 0.15
  left_shoulder_yaw_joint: 0.0
  left_elbow_joint: 0.5
  left_wrist_roll_joint: 0.0
  right_shoulder_pitch_joint: 0.2
  right_shoulder_roll_joint: -0.15
  right_shoulder_yaw_joint: 0.0
  right_elbow_joint: 0.5
  right_wrist_roll_joint: 0.0
```

---

## Step 4: Implement an Adapter (If Non-Standard)

If your policy uses standard 78-D locomotion or 375-D recovery observations, built-in adapters (`OfficialLocomotionAdapter` and `GetupSafefallAdapter`) handle conversion automatically.

If your policy introduces novel inputs or action transforms, implement a subclass of `BasePolicyAdapter` in `edge/policy_adapter.py`:

```python
from edge.policy_adapter import BasePolicyAdapter
from edge.robot_state import RobotState
from edge.robot_command import UnifiedRobotCommand

class MyCustomAdapter(BasePolicyAdapter):
    def build_observation(self, state: RobotState) -> np.ndarray:
        # Assemble custom observation tensor from physical RobotState
        terms = [
            state.base_ang_vel_body,
            state.projected_gravity,
            ...
        ]
        return np.concatenate(terms, axis=0).astype(np.float32).reshape(1, -1)

    def process_action(self, raw_action: np.ndarray, state: RobotState) -> UnifiedRobotCommand:
        # Convert raw network actions into physical joint targets
        targets = {}
        for i, jn in enumerate(self.manifest.action.joint_order):
            targets[jn] = float(self.manifest.default_joint_pos[jn] + 0.25 * raw_action[0, i])
        return UnifiedRobotCommand(joint_targets=targets, policy_name=self.manifest.name)
```

---

## Step 5: Validate Policy Contract

Run the built-in validation tool to verify weights, tensor shapes, and kinematic consistency against the MuJoCo Asimov 1 model:

```bash
./run_sim.sh --policy my_new_policy --validate
```

Example validation output:
```text
── POLICY VALIDATION REPORT: my_new_policy ──
  ✓ Manifest loaded: 'my_new_policy' (v1.0)
  ✓ Manifest schema & configuration valid
  ✓ Model artifact present: policy.onnx (812.3 KB)
  ✓ ONNX runtime initialized: inputs=['obs'], outputs=['actions']
  ✓ Input tensor shapes match observation dim (78)
  ✓ Output tensor shapes match action dim (23)

✓ POLICY VALIDATION PASSED — Ready for simulation.
```

If validation fails, the report pinpoints missing model inputs, shape mismatches, or invalid joint names.

---

## Step 6: Launch Simulation with Your Policy

Run the simulator directly targeting your policy:

```bash
./run_sim.sh --policy my_new_policy
```

You can also list all discovered policies at any time:

```bash
./run_sim.sh --list-policies
```

---

## Step 7: Inspect and Hot-Swap at Runtime

In the interactive console, manage policies dynamically without restarting the simulator:

```text
asimov❯ policy list
asimov❯ policy info my_new_policy
asimov❯ policy select my_new_policy
asimov❯ policy active
```
