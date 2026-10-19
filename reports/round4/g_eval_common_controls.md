# Task G: labeled slang evaluation (pure model)

Sentences: 1684 (pos 847, ctrl 837). Intervals: bootstrap 95% over sentences.

## Lambda = 0 (pure model)

| half | model | n pos / ctrl | slang acc | control acc | balanced (95% CI) |
|---|---|---|---|---|---|
| eval | Suzu-2 | 392 / 384 | 34.2 | 75.3 | 54.7 (51.5 to 57.8) |
| eval | Suzu-2.1 | 392 / 384 | 56.1 | 62.8 | 59.4 (55.8 to 62.8) |
| eval | Suzu-2.2 | 392 / 384 | 54.6 | 67.2 | 60.9 (57.4 to 64.2) |
| seed | Suzu-2 | 455 / 453 | 27.7 | 77.7 | 52.7 (49.8 to 55.4) |
| seed | Suzu-2.1 | 455 / 453 | 50.1 | 67.8 | 58.9 (55.8 to 62.0) |
| seed | Suzu-2.2 | 455 / 453 | 71.6 | 64.2 | 67.9 (64.9 to 71.0) |
| all | Suzu-2 | 847 / 837 | 30.7 | 76.6 | 53.6 (51.6 to 55.7) |
| all | Suzu-2.1 | 847 / 837 | 52.9 | 65.5 | 59.2 (56.8 to 61.4) |
| all | Suzu-2.2 | 847 / 837 | 63.8 | 65.6 | 64.7 (62.4 to 66.9) |

## Paired balanced-accuracy differences (lambda = 0)

| half | comparison | difference, points (95% CI) |
|---|---|---|
| eval | Suzu-2.1 minus Suzu-2 | +4.7 (+1.1 to +8.0) |
| eval | Suzu-2.2 minus Suzu-2.1 | +1.4 (-0.3 to +3.3) |
| eval | Suzu-2.2 minus Suzu-2 | +6.2 (+2.5 to +9.6) |
| seed | Suzu-2.1 minus Suzu-2 | +6.2 (+3.2 to +9.4) |
| seed | Suzu-2.2 minus Suzu-2.1 | +9.0 (+6.0 to +11.9) |
| seed | Suzu-2.2 minus Suzu-2 | +15.2 (+11.9 to +18.5) |

## Dictionary-prior sweep: score + lambda * ln(weight)

| model | lambda | eval-half slang | eval-half control | eval-half balanced | seed-half balanced | wiki acc | wiki change vs 0 (95% CI) |
|---|---|---|---|---|---|---|---|
| Suzu-2 | 0 | 34.2 | 75.3 | 54.7 | 52.7 | 84.5 | +0.00 (±0.00) |
| Suzu-2 | 0.25 | 33.2 | 76.8 | 55.0 | 54.0 | 82.9 | -1.63 (±0.28) |
| Suzu-2 | 0.5 | 32.4 | 77.6 | 55.0 | 54.6 | 80.6 | -3.85 (±0.38) |
| Suzu-2 | 1 | 30.4 | 77.3 | 53.9 | 54.8 | 76.8 | -7.71 (±0.50) |
| Suzu-2 | 2 | 29.1 | 76.6 | 52.8 | 54.1 | 71.1 | -13.38 (±0.62) |
| Suzu-2.1 | 0 | 56.1 | 62.8 | 59.4 | 58.9 | 84.9 | +0.00 (±0.00) |
| Suzu-2.1 | 0.25 | 56.6 | 64.6 | 60.6 | 57.6 | 83.7 | -1.26 (±0.28) |
| Suzu-2.1 | 0.5 | 54.6 | 66.1 | 60.4 | 58.9 | 81.3 | -3.59 (±0.39) |
| Suzu-2.1 | 1 | 51.0 | 68.8 | 59.9 | 57.6 | 77.6 | -7.37 (±0.50) |
| Suzu-2.1 | 2 | 43.4 | 69.3 | 56.3 | 57.2 | 71.9 | -13.01 (±0.62) |
| Suzu-2.2 | 0 | 54.6 | 67.2 | 60.9 | 67.9 | 85.2 | +0.00 (±0.00) |
| Suzu-2.2 | 0.25 | 55.4 | 67.7 | 61.5 | 65.6 | 83.6 | -1.56 (±0.28) |
| Suzu-2.2 | 0.5 | 53.1 | 68.8 | 60.9 | 64.3 | 81.8 | -3.37 (±0.38) |
| Suzu-2.2 | 1 | 48.7 | 70.8 | 59.8 | 61.9 | 77.6 | -7.52 (±0.51) |
| Suzu-2.2 | 2 | 43.4 | 69.8 | 56.6 | 60.0 | 72.0 | -13.12 (±0.62) |

## Recommended lambda range

Rule: eval-half balanced accuracy within 1 point of the best lambda for that model, and wiki change not below -0.3 points.

- Suzu-2: best eval-half balanced 55.0; lambda values that pass: [0]
- Suzu-2.1: best eval-half balanced 60.6; lambda values that pass: none
- Suzu-2.2: best eval-half balanced 61.5; lambda values that pass: [0]
