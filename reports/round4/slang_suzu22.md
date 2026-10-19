# Slang suppression baseline (stage A)

Terms with competitors: 428 (eval half 207, seed half 221). Word split over the whole draft list: seed 631, eval 627; category-stratified, md5 alternate.
Pure model, no frequency fusion. Suppressed = lead-in-dictionary terms the model ranks first in fewer than half of the contexts (not first in a strict majority). Lifted = non-lead terms the model ranks first in at least half of the contexts. Gap = term score minus best competitor, median over term x context.

## 8 contexts / all

| model | lead terms | suppressed | rate | non-lead | lifted | gap median |
|---|---|---|---|---|---|---|
| Suzu-2 | 289 | 115 | 39.8% | 139 | 23 | -0.56 |
| Suzu-2.1 | 289 | 70 | 24.2% | 139 | 47 | +0.75 |
| Suzu-2.2 | 289 | 37 | 12.8% | 139 | 64 | +2.00 |

## 8 contexts / eval half

| model | lead terms | suppressed | rate | non-lead | lifted | gap median |
|---|---|---|---|---|---|---|
| Suzu-2 | 140 | 55 | 39.3% | 67 | 14 | -0.45 |
| Suzu-2.1 | 140 | 36 | 25.7% | 67 | 22 | +0.88 |
| Suzu-2.2 | 140 | 30 | 21.4% | 67 | 20 | +1.16 |

## 8 contexts / seed half

| model | lead terms | suppressed | rate | non-lead | lifted | gap median |
|---|---|---|---|---|---|---|
| Suzu-2 | 149 | 60 | 40.3% | 72 | 9 | -0.62 |
| Suzu-2.1 | 149 | 34 | 22.8% | 72 | 25 | +0.66 |
| Suzu-2.2 | 149 | 7 | 4.7% | 72 | 44 | +2.58 |

## 18 contexts / all

| model | lead terms | suppressed | rate | non-lead | lifted | gap median |
|---|---|---|---|---|---|---|
| Suzu-2 | 289 | 122 | 42.2% | 139 | 25 | -0.50 |
| Suzu-2.1 | 289 | 71 | 24.6% | 139 | 44 | +0.86 |
| Suzu-2.2 | 289 | 39 | 13.5% | 139 | 66 | +2.11 |

## 18 contexts / eval half

| model | lead terms | suppressed | rate | non-lead | lifted | gap median |
|---|---|---|---|---|---|---|
| Suzu-2 | 140 | 59 | 42.1% | 67 | 14 | -0.44 |
| Suzu-2.1 | 140 | 36 | 25.7% | 67 | 23 | +0.92 |
| Suzu-2.2 | 140 | 31 | 22.1% | 67 | 21 | +1.23 |

## 18 contexts / seed half

| model | lead terms | suppressed | rate | non-lead | lifted | gap median |
|---|---|---|---|---|---|---|
| Suzu-2 | 149 | 63 | 42.3% | 72 | 11 | -0.58 |
| Suzu-2.1 | 149 | 35 | 23.5% | 72 | 21 | +0.80 |
| Suzu-2.2 | 149 | 8 | 5.4% | 72 | 45 | +2.80 |
