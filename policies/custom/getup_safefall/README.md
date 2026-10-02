# Asimov 1 Get-Up & Safe-Fall Recovery Suite

- **Policy Identifier**: `getup_safefall`
- **Origin**: OpenHorizon Labs (`asimov1-getup-safefall`)
- **Category**: `recovery`
- **Control Frequency**: 50.0 Hz (period 0.020 s, 4 physics steps @ 200 Hz)

## Overview
Composite recovery package containing both:
1. `safefall.onnx`: Impact-mitigating fall response active upon sudden tilt/disturbance.
2. `getup.onnx`: Dynamic standing recovery active once the robot settles on the ground.

## Arbitration Cycle
$$\text{Locomotion} \xrightarrow{\text{Disturbance / Tilt} > 30^\circ} \text{Safe-Fall} \xrightarrow{\text{Settled}} \text{Get-Up} \xrightarrow{\text{Upright}} \text{Locomotion}$$

See [docs/policy/getup-safefall-compatibility.md](../../../docs/policy/getup-safefall-compatibility.md) for full contract analysis.
