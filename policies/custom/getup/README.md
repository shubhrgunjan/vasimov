# Asimov 1 Get-Up Policy

- **Policy Identifier**: `getup`
- **Origin**: OpenHorizon Labs (`asimov1-getup-safefall`)
- **Category**: `recovery`
- **Control Frequency**: 50.0 Hz (period 0.020 s, 4 physics steps @ 200 Hz)

## Purpose & Behavior
Recovers the Asimov 1 humanoid from any fallen or prone ground posture (back, stomach, side, seated, kneeling) back to the standard standing posture.

## Observation Specification
375-dimensional float32 vector:
- History: 5 timesteps stacked term-major.
- Per-step 75-D term composition:
  1. `base_ang_vel` (3 dims, $\times 0.25$)
  2. `projected_gravity` (3 dims, $\times 1.0$)
  3. `joint_pos` (23 dims, relative to default)
  4. `joint_vel` (23 dims, $\times 0.10$)
  5. `actions` (23 dims, previous step raw action)

## Action Specification
23-dimensional float32 vector:
- Target transform: `q_target = default_joint_pos + 0.25 * raw_action`
- Reaches upright stance ready for handover to the official walking policy.
