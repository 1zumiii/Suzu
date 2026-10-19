# T4 single-character held-out evaluation (pure model, top-10 candidates)

Held-out items: 572. Paired bootstrap 95% over items (2000 resamples).

| model | accuracy |
|---|---|
| Suzu-2.1 | 69.1 |
| 2.2-recipe seed0 | 71.3 |
| 2.2-recipe seed1 | 72.0 |
| 2.2-recipe seed2 | 71.9 |
| 2.2-recipe seed3 | 71.9 |
| Suzu-2.3 (single-char training) | 73.1 |

| comparison | difference, points (95% CI) |
|---|---|
| 2.2-recipe seed0 minus Suzu-2.1 | +2.3 (+0.5 to +4.0) |
| 2.2-recipe seed1 minus Suzu-2.1 | +3.0 (+1.0 to +4.9) |
| 2.2-recipe seed2 minus Suzu-2.1 | +2.8 (+1.2 to +4.5) |
| 2.2-recipe seed3 minus Suzu-2.1 | +2.8 (+1.0 to +4.7) |
| Suzu-2.3 (single-char training) minus Suzu-2.1 | +4.0 (+2.3 to +5.9) |
