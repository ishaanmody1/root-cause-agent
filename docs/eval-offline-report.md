## Eval results (48 scenarios, 40 alarms)

### Detector

| Metric | Value |
|---|---|
| False alarms on normal runs | 0.0% of 6 |
| Alarms on benign setpoint changes | 66.7% of 6 |
| Fault detection rate | 100.0% of 36 |
| Median detection delay | 17 ticks |
| First alarm at the wrong stage (misleading) | 30.6% |

### Diagnosis (on scenarios where the detector fired)

| Metric | Naive (blame first alarm) | Rule-based tracer |
|---|---|---|
| Root-cause stage accuracy | 69.4% | 94.4% |
| Fault-type accuracy | 94.4% | 91.7% |
| Both correct | 63.9% | 91.7% |
| Both correct, misleading cases only | 0.0% | 100.0% |
| Both correct when answered (excl. abstentions) | 63.9% | 91.7% |
| Abstained (insufficient evidence) | 0.0% | 0.0% |
| Confident errors (wrong at conf >= 0.75) | 0.0% | 5.0% |
| False positives on no-fault alarms | 100.0% | 0.0% |
| Brier score (lower is better) | 0.26 | 0.089 |
| Expected calibration error | 0.125 | 0.123 |
| Mean tool calls | 0.0 | 5.2 |
| Mean model calls | 0.0 | 2.6 |
| Mean tokens (in+out) | 0 | 278 |
| Mean cost per diagnosis | 0.0 USD | 0.0 USD |
| p50 / p90 latency | 0.0 / 0.0 s | 0.0 / 0.0 s |

### Both-correct rate by scenario kind

| Kind | n | Naive (blame first alarm) | Rule-based tracer |
|---|---|---|---|
| clog | 12 | 8.3% | 100.0% |
| normal_setpoint_change | 4 | 0.0% | 100.0% |
| overheating | 6 | 100.0% | 50.0% |
| sensor_drift | 6 | 66.7% | 100.0% |
| sensor_failure | 6 | 100.0% | 100.0% |
| supply_drop | 6 | 100.0% | 100.0% |

### Calibration: Naive (blame first alarm)

| Confidence | n | Mean conf | Accuracy |
|---|---|---|---|
| 0.6-0.8 | 40 | 0.7 | 0.57 |

### Calibration: Rule-based tracer

| Confidence | n | Mean conf | Accuracy |
|---|---|---|---|
| 0.4-0.6 | 7 | 0.56 | 0.86 |
| 0.6-0.8 | 1 | 0.7 | 1.0 |
| 0.8-1.0 | 32 | 0.86 | 0.94 |


### Per-scenario

| Scenario | Truth | First alarm | Naive (blame first alarm) | Rule-based tracer |
|---|---|---|---|---|
| clog-8505 | clog@feed | heater.flow ⚡ | ❌ clog@heater (0.70) | ✅ clog@feed (0.92) |
| overheating-6368 | overheating@press | press.temperature | ✅ overheating@press (0.70) | ✅ overheating@press (0.70) |
| sensor_drift-5110 | sensor_drift@heater | heater.power | ✅ sensor_drift@heater (0.70) | ✅ sensor_drift@heater (0.50) |
| sensor_failure-2697 | sensor_failure@heater | heater.power | ✅ sensor_failure@heater (0.70) | ✅ sensor_failure@heater (0.90) |
| supply_drop-3077 | supply_drop@feed | feed.flow | ✅ supply_drop@feed (0.70) | ✅ supply_drop@feed (0.85) |
| clog-0165 | clog@heater | press.flow ⚡ | ❌ clog@press (0.70) | ✅ clog@heater (0.85) |
| clog-1752 | clog@heater | press.flow ⚡ | ❌ clog@press (0.70) | ✅ clog@heater (0.85) |
| overheating-8131 | overheating@heater | heater.power | ✅ overheating@heater (0.70) | ✅ overheating@heater (0.85) |
| sensor_drift-6493 | sensor_drift@press | press.temperature | ✅ sensor_drift@press (0.70) | ✅ sensor_drift@press (0.50) |
| sensor_failure-9126 | sensor_failure@feed | feed.flow | ✅ sensor_failure@feed (0.70) | ✅ sensor_failure@feed (0.90) |
| supply_drop-5035 | supply_drop@feed | feed.flow | ✅ supply_drop@feed (0.70) | ✅ supply_drop@feed (0.85) |
| normal_setpoint_change-9706 | none@- | heater.power | ❌ overheating@heater (0.70) | ✅ no_fault (0.80) |
| clog-7294 | clog@feed | heater.flow ⚡ | ❌ clog@heater (0.70) | ✅ clog@feed (0.92) |
| clog-6322 | clog@heater | press.flow ⚡ | ❌ clog@press (0.70) | ✅ clog@heater (0.85) |
| overheating-5435 | overheating@feed | feed.temperature | ✅ overheating@feed (0.70) | ❌ no_fault (0.80) |
| sensor_drift-5598 | sensor_drift@feed | feed.temperature | ✅ sensor_drift@feed (0.70) | ✅ sensor_drift@feed (0.50) |
| sensor_failure-9349 | sensor_failure@heater | heater.power | ✅ sensor_failure@heater (0.70) | ✅ sensor_failure@heater (0.90) |
| supply_drop-2773 | supply_drop@feed | feed.flow | ✅ supply_drop@feed (0.70) | ✅ supply_drop@feed (0.85) |
| normal_setpoint_change-6708 | none@- | heater.power | ❌ overheating@heater (0.70) | ✅ no_fault (0.80) |
| clog-0027 | clog@heater | press.flow ⚡ | ❌ clog@press (0.70) | ✅ clog@heater (0.85) |
| clog-3941 | clog@heater | press.flow ⚡ | ❌ clog@press (0.70) | ✅ clog@heater (0.85) |
| overheating-8573 | overheating@heater | heater.power | ✅ overheating@heater (0.70) | ✅ overheating@heater (0.85) |
| sensor_drift-5542 | sensor_drift@finish | finish.flow | ❌ clog@finish (0.70) | ✅ sensor_drift@finish (0.60) |
| sensor_failure-0335 | sensor_failure@feed | feed.flow | ✅ sensor_failure@feed (0.70) | ✅ sensor_failure@feed (0.90) |
| supply_drop-7648 | supply_drop@feed | feed.flow | ✅ supply_drop@feed (0.70) | ✅ supply_drop@feed (0.85) |
| normal_setpoint_change-8464 | none@- | heater.power | ❌ sensor_drift@heater (0.70) | ✅ no_fault (0.80) |
| clog-1756 | clog@press | press.pressure | ✅ clog@press (0.70) | ✅ clog@press (0.85) |
| clog-0892 | clog@feed | heater.flow ⚡ | ❌ clog@heater (0.70) | ✅ clog@feed (0.92) |
| overheating-8630 | overheating@feed | feed.temperature | ✅ overheating@feed (0.70) | ❌ sensor_drift@feed (0.60) |
| sensor_drift-0220 | sensor_drift@finish | finish.quality | ✅ sensor_drift@finish (0.70) | ✅ sensor_drift@finish (0.60) |
| sensor_failure-5414 | sensor_failure@press | press.pressure | ✅ sensor_failure@press (0.70) | ✅ sensor_failure@press (0.90) |
| supply_drop-0803 | supply_drop@feed | feed.flow | ✅ supply_drop@feed (0.70) | ✅ supply_drop@feed (0.85) |
| clog-4226 | clog@feed | heater.flow ⚡ | ❌ clog@heater (0.70) | ✅ clog@feed (0.92) |
| clog-4031 | clog@heater | press.flow ⚡ | ❌ clog@press (0.70) | ✅ clog@heater (0.85) |
| overheating-0283 | overheating@press | press.temperature | ✅ overheating@press (0.70) | ❌ no_fault (0.80) |
| sensor_drift-0053 | sensor_drift@finish | finish.flow | ❌ clog@finish (0.70) | ✅ sensor_drift@finish (0.60) |
| sensor_failure-1242 | sensor_failure@heater | heater.flow | ✅ sensor_failure@heater (0.70) | ✅ sensor_failure@heater (0.90) |
| supply_drop-0082 | supply_drop@feed | feed.flow | ✅ supply_drop@feed (0.70) | ✅ supply_drop@feed (0.85) |
| normal_setpoint_change-5255 | none@- | heater.power | ❌ overheating@heater (0.70) | ✅ no_fault (0.80) |
| clog-6471 | clog@heater | press.flow ⚡ | ❌ clog@press (0.70) | ✅ clog@heater (0.85) |