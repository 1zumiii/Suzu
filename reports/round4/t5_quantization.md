# T5 quantization comparison (Suzu-2.1 structure)

Accuracy: pure model in PyTorch with weights dequantized from each file; difference is against the unquantized PyTorch checkpoint. Latency: student_check.exe, 7 threads, 30-token prefix, 8 candidates, batched path (ms per item), prefix / candidate-batch split in the last column.

| file | size MiB | single holdout | run3 holdout | wiki eval (4000) | batched ms | prefix / candidates ms |
|---|---|---|---|---|---|---|
| PyTorch checkpoint (fp32) | - | 69.1 | 93.7 | 85.8 | - | - |
| suzu21.Q8_0 | 41.9 | 68.9 (-0.2) | 93.7 (+0.0) | 85.7 (-0.0) | 25.32 | 13.24 / 11.18 |
| suzu21.Q6_K | 32.5 | 68.7 (-0.3) | 93.7 (+0.0) | 85.8 (+0.1) | 15.85 | 6.90 / 8.01 |
| suzu21.Q5_K_M | 29.1 | 68.9 (-0.2) | 93.5 (-0.2) | 85.9 (+0.1) | 15.14 | 6.50 / 7.67 |
| suzu21.Q4_K_M | 25.9 | 69.6 (+0.5) | 93.7 (+0.0) | 85.8 (+0.1) | 12.23 | 4.99 / 6.34 |
