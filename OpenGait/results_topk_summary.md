# Tuning Top-K Summary

- top_k: 10
- best_trial: 32
- best_metric: 58.087

## Top-K Trials

| rank | trial_id | metric | mask_ratio | lambda | lambda_edge | edge_thr_ratio | edge_thr_abs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 58.087 | 0.3 | 7.0 | 6.0 | 0.15 | 0.01 |
| 2 | 36 | 58.027 | 0.3 | 2.5 | 4.5 | 0.2 | 0.01 |
| 3 | 40 | 57.977 | 0.5 | 11.0 | 1.0 | 0.35 | 0.03 |
| 4 | 34 | 57.967 | 0.2 | 5.0 | 7.0 | 0.15 | 0.0 |
| 5 | 21 | 57.917 | 0.2 | 14.5 | 7.0 | 0.25 | 0.01 |
| 6 | 12 | 57.863 | 0.4 | 4.0 | 7.5 | 0.15 | 0.01 |
| 7 | 20 | 57.783 | 0.4 | 20.0 | 5.5 | 0.35 | 0.03 |
| 8 | 6 | 57.727 | 0.3 | 2.5 | 1.5 | 0.3 | 0.02 |
| 9 | 37 | 57.720 | 0.5 | 13.5 | 10.0 | 0.1 | 0.01 |
| 10 | 11 | 57.693 | 0.3 | 4.5 | 3.0 | 0.35 | 0.02 |

## Parameter Frequency In Top-K

- model_cfg.recon.mask_ratio: [(0.3, 4), (0.5, 2), (0.2, 2), (0.4, 2)]
- model_cfg.recon.lambda: [(2.5, 2), (7.0, 1), (11.0, 1), (5.0, 1), (14.5, 1), (4.0, 1), (20.0, 1), (13.5, 1), (4.5, 1)]
- model_cfg.recon.lambda_edge: [(7.0, 2), (6.0, 1), (4.5, 1), (1.0, 1), (7.5, 1), (5.5, 1), (1.5, 1), (10.0, 1), (3.0, 1)]
- model_cfg.recon.edge_sobel_thr_ratio: [(0.15, 3), (0.35, 3), (0.2, 1), (0.25, 1), (0.3, 1), (0.1, 1)]
- model_cfg.recon.edge_sobel_thr_abs: [(0.01, 5), (0.03, 2), (0.02, 2), (0.0, 1)]

## Repeated Edge Threshold Pairs

- edge_sobel_thr_ratio=0.15, edge_sobel_thr_abs=0.01: n=5, mean_metric=57.259, best_metric=58.087
- edge_sobel_thr_ratio=0.3, edge_sobel_thr_abs=0.02: n=4, mean_metric=57.000, best_metric=57.727
- edge_sobel_thr_ratio=0.35, edge_sobel_thr_abs=0.03: n=3, mean_metric=57.591, best_metric=57.977
- edge_sobel_thr_ratio=0.25, edge_sobel_thr_abs=0.0: n=3, mean_metric=57.030, best_metric=57.263
- edge_sobel_thr_ratio=0.2, edge_sobel_thr_abs=0.01: n=2, mean_metric=57.655, best_metric=58.027
- edge_sobel_thr_ratio=0.15, edge_sobel_thr_abs=0.0: n=2, mean_metric=57.495, best_metric=57.967
- edge_sobel_thr_ratio=0.25, edge_sobel_thr_abs=0.01: n=2, mean_metric=57.537, best_metric=57.917
- edge_sobel_thr_ratio=0.15, edge_sobel_thr_abs=0.02: n=2, mean_metric=57.198, best_metric=57.647
- edge_sobel_thr_ratio=0.3, edge_sobel_thr_abs=0.01: n=2, mean_metric=57.165, best_metric=57.617
- edge_sobel_thr_ratio=0.15, edge_sobel_thr_abs=0.03: n=2, mean_metric=57.218, best_metric=57.467
- edge_sobel_thr_ratio=0.35, edge_sobel_thr_abs=0.01: n=2, mean_metric=57.338, best_metric=57.340
- edge_sobel_thr_ratio=0.3, edge_sobel_thr_abs=0.0: n=2, mean_metric=57.032, best_metric=57.247
- edge_sobel_thr_ratio=0.2, edge_sobel_thr_abs=0.0: n=2, mean_metric=56.975, best_metric=56.997

## One-Parameter Trend Slices

- vary mask_ratio with fixed [lambda=2.5, lambda_edge=4.5, edge_sobel_thr_ratio=0.2] -> n=2, values=0.3: 58.027, 0.5: 56.950
- vary lambda with fixed [mask_ratio=0.3, edge_sobel_thr_ratio=0.15, edge_sobel_thr_abs=0.01] -> n=3, values=7.0: 58.087, 11.0: 56.530, 17.5: 56.173
- vary lambda_edge with fixed [mask_ratio=0.3, edge_sobel_thr_ratio=0.15, edge_sobel_thr_abs=0.01] -> n=3, values=3.0: 56.173, 6.0: 58.087, 7.5: 56.530
- vary edge_sobel_thr_ratio with fixed [mask_ratio=0.3, lambda=2.5, edge_sobel_thr_abs=0.01] -> n=2, values=0.2: 58.027, 0.25: 57.157
- vary edge_sobel_thr_abs with fixed [lambda=2.5, lambda_edge=4.5, edge_sobel_thr_ratio=0.2] -> n=2, values=0.01: 58.027, 0.03: 56.950
