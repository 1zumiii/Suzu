# Task G: labeled slang evaluation (pure model)

Sentences: 6760 (pos 3405, ctrl 3355). Intervals: bootstrap 95% over sentences.

## Lambda = 0 (pure model)

| half | model | n pos / ctrl | slang acc | control acc | balanced (95% CI) |
|---|---|---|---|---|---|
| eval | Suzu-2 | 1647 / 1619 | 51.1 | 53.4 | 52.3 (50.5 to 54.1) |
| eval | Suzu-2.1 | 1647 / 1619 | 65.9 | 46.3 | 56.1 (54.5 to 57.8) |
| eval | Suzu-2.2 | 1647 / 1619 | 69.0 | 45.3 | 57.2 (55.6 to 58.9) |
| seed | Suzu-2 | 1758 / 1736 | 51.4 | 53.7 | 52.6 (50.9 to 54.2) |
| seed | Suzu-2.1 | 1758 / 1736 | 65.9 | 44.6 | 55.3 (53.7 to 56.8) |
| seed | Suzu-2.2 | 1758 / 1736 | 87.3 | 32.1 | 59.7 (58.4 to 61.1) |
| all | Suzu-2 | 3405 / 3355 | 51.3 | 53.6 | 52.4 (51.3 to 53.6) |
| all | Suzu-2.1 | 3405 / 3355 | 65.9 | 45.5 | 55.7 (54.5 to 56.8) |
| all | Suzu-2.2 | 3405 / 3355 | 78.4 | 38.5 | 58.5 (57.4 to 59.5) |

## Paired balanced-accuracy differences (lambda = 0)

| half | comparison | difference, points (95% CI) |
|---|---|---|
| eval | Suzu-2.1 minus Suzu-2 | +3.8 (+2.1 to +5.5) |
| eval | Suzu-2.2 minus Suzu-2.1 | +1.1 (+0.0 to +2.1) |
| eval | Suzu-2.2 minus Suzu-2 | +4.9 (+3.1 to +6.6) |
| seed | Suzu-2.1 minus Suzu-2 | +2.7 (+1.2 to +4.2) |
| seed | Suzu-2.2 minus Suzu-2.1 | +4.4 (+3.0 to +6.0) |
| seed | Suzu-2.2 minus Suzu-2 | +7.1 (+5.5 to +8.9) |

## Dictionary-prior sweep: score + lambda * ln(weight)

| model | lambda | eval-half slang | eval-half control | eval-half balanced | seed-half balanced | wiki acc | wiki change vs 0 (95% CI) |
|---|---|---|---|---|---|---|---|
| Suzu-2 | 0 | 51.1 | 53.4 | 52.3 | 52.6 | 84.5 | +0.00 (±0.00) |
| Suzu-2 | 0.25 | 54.5 | 51.9 | 53.2 | 52.9 | 82.9 | -1.63 (±0.28) |
| Suzu-2 | 0.5 | 56.4 | 50.4 | 53.4 | 53.2 | 80.6 | -3.85 (±0.38) |
| Suzu-2 | 1 | 58.2 | 48.4 | 53.3 | 53.2 | 76.8 | -7.71 (±0.50) |
| Suzu-2 | 2 | 60.3 | 43.9 | 52.1 | 53.2 | 71.1 | -13.38 (±0.62) |
| Suzu-2.1 | 0 | 65.9 | 46.3 | 56.1 | 55.3 | 84.9 | +0.00 (±0.00) |
| Suzu-2.1 | 0.25 | 68.2 | 45.8 | 57.0 | 55.6 | 83.7 | -1.26 (±0.28) |
| Suzu-2.1 | 0.5 | 68.4 | 45.5 | 56.9 | 56.1 | 81.3 | -3.59 (±0.39) |
| Suzu-2.1 | 1 | 68.5 | 43.1 | 55.8 | 54.8 | 77.6 | -7.37 (±0.50) |
| Suzu-2.1 | 2 | 66.7 | 40.3 | 53.5 | 53.8 | 71.9 | -13.01 (±0.62) |
| Suzu-2.2 | 0 | 69.0 | 45.3 | 57.2 | 59.7 | 85.2 | +0.00 (±0.00) |
| Suzu-2.2 | 0.25 | 71.0 | 44.5 | 57.8 | 59.0 | 83.6 | -1.56 (±0.28) |
| Suzu-2.2 | 0.5 | 71.3 | 43.7 | 57.5 | 58.0 | 81.8 | -3.37 (±0.38) |
| Suzu-2.2 | 1 | 70.4 | 41.9 | 56.2 | 57.0 | 77.6 | -7.52 (±0.51) |
| Suzu-2.2 | 2 | 68.9 | 39.6 | 54.3 | 55.2 | 72.0 | -13.12 (±0.62) |

## Recommended lambda range

Rule: eval-half balanced accuracy within 1 point of the best lambda for that model, and wiki change not below -0.3 points.

- Suzu-2: best eval-half balanced 53.4; lambda values that pass: none
- Suzu-2.1: best eval-half balanced 57.0; lambda values that pass: [0]
- Suzu-2.2: best eval-half balanced 57.8; lambda values that pass: [0]
