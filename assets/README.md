# Assets

Static images used by the project README. Two sources:

- `paper/` — figures taken from the paper, [arXiv:2311.16519v2](https://arxiv.org/abs/2311.16519).
  Figures 1, 2, 3, 5 and 7 are raster images embedded in the PDF and were extracted at their
  native resolution; Figures 4, 6, 8, 9 and 10 are vector graphics and were rendered at 200 dpi
  from a clip of the page above their caption.
- `results/` — plots produced by the original research code, decoded from the base64 `image/png`
  cell outputs of its workflow notebooks (`src/workflow_*.ipynb`, last present in commit
  [`ebd03bb`](https://github.com/moodykong/bayesian-lstm-mionet/tree/ebd03bb/src)). The notebooks
  in [`notebooks/`](../notebooks) regenerate the same kinds of plot with the current package.

All files are PNG and under 1 MB.

## Inventory

| File | What it shows | Provenance | Dimensions |
| --- | --- | --- | --- |
| `paper/fig1_deeponet.png` | DeepONet architecture: branch net encodes the input function, trunk net encodes the evaluation coordinate. | Paper Figure 1, page 7 (embedded raster) | 4276x2086 |
| `paper/fig2_mionet.png` | MIONet architecture: multiple branch nets encode distinct input functions, combined with a trunk net. | Paper Figure 2, page 8 (embedded raster) | 1710x735 |
| `paper/fig3_lstm_mionet_architecture.png` | LSTM-enhanced MIONet architecture: Branch 1 encodes the current state, Branch 2 (FNN + RNN/LSTM + FNN) encodes the length-variant input function history, trunk encodes the step size `h`. | Paper Figure 3, page 9 (embedded raster, downscaled from 5206x3234) | 2400x1491 |
| `paper/fig4_lorentz_trajectories.png` | LSTM-MIONet prediction vs. true trajectory of the autonomous Lorentz system, states `x(t)` and `y(t)`. | Paper Figure 4, page 14 (vector, 200 dpi clip) | 1215x989 |
| `paper/fig5_lorentz_uq.png` | B-LSTM-MIONet 0.95 confidence interval for the Lorentz problem; the true trajectory stays inside the interval. | Paper Figure 5, page 15 (embedded raster) | 954x720 |
| `paper/fig6_pendulum_trajectories.png` | LSTM-MIONet prediction vs. true trajectory of the non-autonomous pendulum state `θ(t)` for four input functions. | Paper Figure 6, page 17 (vector, 200 dpi clip) | 1222x990 |
| `paper/fig7_pendulum_uq.png` | B-LSTM-MIONet 0.95 confidence interval for the pendulum problem. | Paper Figure 7, page 18 (embedded raster) | 547x418 |
| `paper/fig8_ausgrid_trajectories.png` | LSTM-MIONet prediction vs. true power generation `x(t)` for Ausgrid customers, in kWh over hours. | Paper Figure 8, page 19 (vector, 200 dpi clip) | 1204x991 |
| `paper/fig9_extrapolation.png` | Temporal extrapolation: L2 relative error of `θ` as the prediction horizon `T` grows beyond the training range. | Paper Figure 9, page 21 (vector, 200 dpi clip) | 789x676 |
| `paper/fig10_step_size.png` | Step-size study: L2 relative error of `θ` as a function of the time step `h`. | Paper Figure 10, page 22 (vector, 200 dpi clip) | 782x676 |
| `results/lorentz_infer_x0.png` | One-step inference on the Lorentz system, state component 0 (`x`), true vs. predicted. | `src/workflow_lorentz.ipynb`, cell 14 | 564x426 |
| `results/lorentz_recursive_teacher_forcing_1.0.png` | Recursive Lorentz rollout with teacher forcing probability 1.0. | `src/workflow_lorentz.ipynb`, cell 16 | 564x426 |
| `results/lorentz_recursive_teacher_forcing_0.5.png` | Recursive Lorentz rollout with teacher forcing probability 0.5. | `src/workflow_lorentz.ipynb`, cell 18 | 564x426 |
| `results/lorentz_recursive_teacher_forcing_0.0.png` | Fully recursive Lorentz rollout, teacher forcing probability 0.0 (no ground truth fed back). | `src/workflow_lorentz.ipynb`, cell 20 | 564x426 |
| `results/lorentz_deeponet_baseline.png` | Plain DeepONet baseline on the Lorentz system (state and step size only, no input function). | `src/workflow_lorentz.ipynb`, cell 22 | 564x426 |
| `results/lorentz_lstm_deeponet_baseline.png` | LSTM-DeepONet baseline on the Lorentz system, one-step inference. | `src/workflow_lorentz.ipynb`, cell 23 | 564x426 |
| `results/lorentz_lstm_deeponet_recursive_teacher_forcing_0.0.png` | LSTM-DeepONet baseline, fully recursive Lorentz rollout (teacher forcing 0.0). | `src/workflow_lorentz.ipynb`, cell 24 | 564x426 |
| `results/pendulum_infer.png` | One-step inference on the non-autonomous pendulum with GRF input functions. | `src/workflow_pendulum.ipynb`, cell 14 | 558x426 |
| `results/pendulum_recursive_teacher_forcing_1.0.png` | Recursive pendulum rollout with teacher forcing probability 1.0. | `src/workflow_pendulum.ipynb`, cell 16 | 558x426 |
| `results/pendulum_recursive_teacher_forcing_0.5.png` | Recursive pendulum rollout with teacher forcing probability 0.5. | `src/workflow_pendulum.ipynb`, cell 18 | 558x426 |
| `results/pendulum_recursive_teacher_forcing_0.0.png` | Fully recursive pendulum rollout, teacher forcing probability 0.0. | `src/workflow_pendulum.ipynb`, cell 20 | 556x426 |
| `results/pendulum_ood_sin_t_over_2.png` | Out-of-distribution generalization: pendulum driven by `u = sin(t/2)`, unseen during training. | `src/workflow_pendulum.ipynb`, cell 24 | 567x426 |
| `results/pendulum_deeponet_local_baseline.png` | DeepONet_Local baseline on the pendulum (uses the next input value `u` directly). | `src/workflow_pendulum.ipynb`, cell 26 | 558x426 |
| `results/ausgrid_customers_51_60.png` | Inference on the Ausgrid power generation data set, customers 51-60. | `src/workflow_Ausgrid.ipynb`, cell 15 | 559x426 |
| `results/ausgrid_customers_61_70.png` | Inference on the Ausgrid power generation data set, customers 61-70. | `src/workflow_Ausgrid.ipynb`, cell 17 | 559x426 |
