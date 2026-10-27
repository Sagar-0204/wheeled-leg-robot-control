# Evaluation: ppo_main @ 21500000 steps

## 1. Max recoverable push (N, 0.1 s, forward and backward)

| payload [kg] | RL | LQR-nominal | LQR-oracle |
|---|---|---|---|
| 1 * | 200 | 200 | 200 |
| 2 | 200 | 200 | 200 |
| 4 | 280 | 240 | 240 |
| 6 | 320 | 280 | 280 |
| 8 | 360 | 320 | 320 |
| 10 * | 440 | 320 | 360 |
| 12 * | 480 | 360 | 360 |

\* outside the RL training range (2-8 kg)

## 2. Height changes under pushes (6 kg payload)

| controller | fell | max pitch deviation [rad] | max displacement [m] |
|---|---|---|---|
| RL | no | 0.137 | 0.16 |
| LQR-nominal | no | 0.199 | 0.14 |
| LQR-oracle | no | 0.198 | 0.15 |

## 3. Fall rate: full domain randomization + random pushes from any direction every 2-3 s
(standing at a random height, 30 episodes x 20 s per band)

| push strength | RL | LQR-nominal | LQR-oracle |
|---|---|---|---|
| 40-80 N | 0% | 10% | 10% |
| 80-120 N | 0% | 57% | 57% |
| 120-160 N | 3% | 90% | 93% |
| 160-200 N | 33% | 100% | 100% |

## 4. RL command tracking (50 episodes x 20 s, full randomization + pushes)

- fall rate: 0%
- mean |velocity error|: 0.065 m/s
- mean |yaw-rate error|: 0.074 rad/s
- mean |height error|: 0.94 cm
