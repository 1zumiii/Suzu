# Evaluation suite

Cells: mixed accuracy / pure-model accuracy (percent). Sets: wiki eval (16000), tech (old) (221), anime (old) (211), chat (old) (205), tech (new hold-out) (220), anime (new hold-out) (193), chat (new hold-out) (221)

| model | wiki eval | tech (old) | anime (old) | chat (old) | tech (new hold-out) | anime (new hold-out) | chat (new hold-out) |
|---|---|---|---|---|---|---|---|
| Suzu-2.1 | 84.1 / 84.9 | 89.6 / 91.4 | 90.0 / 90.0 | 98.5 / 98.5 | 94.5 / 95.0 | 90.7 / 90.7 | 95.5 / 95.0 |
| Suzu-2.2 (original run) | 84.2 / 85.2 | 89.1 / 91.0 | 91.9 / 91.9 | 97.6 / 99.0 | 94.5 / 94.5 | 92.2 / 91.7 | 95.9 / 96.8 |
| t6_seed1 | 84.1 / 85.0 | 90.5 / 92.3 | 91.9 / 91.5 | 97.1 / 98.0 | 95.5 / 95.0 | 92.7 / 93.3 | 97.3 / 96.8 |
| t6_seed2 | 84.1 / 85.0 | 89.6 / 91.9 | 91.5 / 91.0 | 98.0 / 98.5 | 95.0 / 96.4 | 93.3 / 92.7 | 95.9 / 95.0 |
| t6_seed3 | 84.2 / 85.0 | 91.0 / 90.5 | 91.9 / 91.0 | 97.6 / 98.0 | 94.5 / 95.5 | 92.2 / 92.7 | 95.9 / 97.7 |
| **max difference between models** | 0.2 / 0.2 | 1.8 / 1.8 | 1.9 / 1.9 | 1.5 / 1.0 | 0.9 / 1.8 | 2.6 / 2.6 | 1.8 / 2.7 |

Paired difference to the first model (Suzu-2.1), mixed accuracy, points (95% interval):

| model | wiki eval | tech (old) | anime (old) | chat (old) | tech (new hold-out) | anime (new hold-out) | chat (new hold-out) |
|---|---|---|---|---|---|---|---|
| Suzu-2.2 (original run) | +0.1 (±0.3) | -0.5 (±2.7) | +1.9 (±1.8) | -1.0 (±1.3) | +0.0 (±1.8) | +1.6 (±3.0) | +0.5 (±2.0) |
| t6_seed1 | -0.0 (±0.3) | +0.9 (±2.2) | +1.9 (±2.3) | -1.5 (±1.6) | +0.9 (±1.8) | +2.1 (±3.2) | +1.8 (±1.8) |
| t6_seed2 | -0.0 (±0.3) | +0.0 (±2.2) | +1.4 (±2.1) | -0.5 (±1.0) | +0.5 (±2.0) | +2.6 (±3.0) | +0.5 (±2.0) |
| t6_seed3 | +0.1 (±0.3) | +1.4 (±2.7) | +1.9 (±1.8) | -1.0 (±1.3) | +0.0 (±1.8) | +1.6 (±3.4) | +0.5 (±2.0) |
