# T2 candidate-count sweep and T3 error profile (Suzu-2.1, pure model)

Pool: 7191 items ({'wiki': 4000, 'single': 1500, 'domain': 723, 'holdout': 968}); 6967 have dictionary candidates. Candidates are rime-ice same-pinyin same-length words by weight. Hit = the model's first choice among the first K candidates equals the answer; an answer outside the first K is a miss. Latency is estimated by scaling the measured 8-candidate batch cost (9.7 ms, 7 threads) linearly with K; it is not a separate measurement.

## T2

| group | N | K | answer in top K | pure-model hit | dictionary-first | est. batch ms |
|---|---|---|---|---|---|---|
| 1 syllable | 1574 | 3 | 71.9 | 64.2 | 50.9 | 3.6 |
| 1 syllable | 1574 | 5 | 84.3 | 71.5 | 50.9 | 6.1 |
| 1 syllable | 1574 | 8 | 91.2 | 74.1 | 50.9 | 9.7 |
| 1 syllable | 1574 | 10 | 92.9 | 74.2 | 50.9 | 12.1 |
| 1 syllable | 1574 | 12 | 94.3 | 73.7 | 50.9 | 14.5 |
| 2 syllables | 5488 | 3 | 86.5 | 78.7 | 58.6 | 3.6 |
| 2 syllables | 5488 | 5 | 92.4 | 82.1 | 58.6 | 6.1 |
| 2 syllables | 5488 | 8 | 95.0 | 83.7 | 58.6 | 9.7 |
| 2 syllables | 5488 | 10 | 95.7 | 84.0 | 58.6 | 12.1 |
| 2 syllables | 5488 | 12 | 95.9 | 84.2 | 58.6 | 14.5 |
| 3+ syllables | 129 | 3 | 89.9 | 84.5 | 80.6 | 3.6 |
| 3+ syllables | 129 | 5 | 89.9 | 84.5 | 80.6 | 6.1 |
| 3+ syllables | 129 | 8 | 89.9 | 84.5 | 80.6 | 9.7 |
| 3+ syllables | 129 | 10 | 89.9 | 84.5 | 80.6 | 12.1 |
| 3+ syllables | 129 | 12 | 89.9 | 84.5 | 80.6 | 14.5 |

## T3 (errors at K = 10)

Items: 7191; errors incl. answer outside candidates: 1302 (18.1%).

| category | count | share of errors |
|---|---|---|
| wrong pick is common, context mismatch | 934 | 71.7% |
| answer not in the first 10 candidates (word not in dictionary) | 224 | 17.2% |
| answer not in the first 10 candidates | 137 | 10.5% |
| wrong pick is rare in corpus (count < 3) | 7 | 0.5% |

Among 941 wrong picks inside candidates: corpus count 0 for 7; the wrong pick is the dictionary-first word in 400; mean weight rank of the wrong pick 1.57; length distribution {2: 639, 3: 7, 1: 295}.

### Corpus-frequency floor simulation (K = 10)

penalty P is subtracted from the score of candidates whose corpus count is below the floor.

| floor | P | hit rate | change vs no penalty (points) |
|---|---|---|---|
| 0 | 0 | 81.89% of all items | +0.00 |
| 1 | 1 | 81.92% of all items | +0.03 |
| 1 | 2 | 81.92% of all items | +0.03 |
| 1 | 4 | 81.92% of all items | +0.03 |
| 3 | 1 | 81.92% of all items | +0.03 |
| 3 | 2 | 81.92% of all items | +0.03 |
| 3 | 4 | 81.92% of all items | +0.03 |
| 10 | 1 | 81.96% of all items | +0.07 |
| 10 | 2 | 81.95% of all items | +0.06 |
