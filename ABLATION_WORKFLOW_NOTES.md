# Ablation workflow notes (Burgers) — working notes, not part of the paper

Written 2026-08-02 after building the `oneshot_shock_ablation_{add,remove}` and
`oneshot_onlybdry_ablation_{add,remove}` experiments, to save re-discovery time
before starting on `examples/Darcy`. Delete this file once it's no longer useful.

## 1. Conda environments — what to use for what

| env | what it's for | gotcha |
|---|---|---|
| `keql_legacy` | JAX/KEQL/RKHS code: `onestep`, `twostep`, and the data-prep half of `PINNSR/compute.py` | **Must be properly activated** (`conda activate keql_legacy`, or `conda run -n keql_legacy`) before running anything on GPU. Invoking the interpreter by absolute path with an unmodified shell `PATH` makes JAX fall back to the system's `/usr/bin/ptxas` (CUDA 12.2), which can't assemble the `.version 8.3` PTX this JAX build emits → cryptic `ptxas ... Unsupported .version 8.3` crash. Once activated properly, GPU results reproduced the shock ablation's committed NRMSE to 9 significant figures — it's a real, verified fix, not a workaround. |
| `pinnsr_legacy` | TF1.15 (`tensorflow-gpu==1.15.0`) PINN-SR training (`coefficients[_test].py` in `examples/Burgers_PINNSR/...`) | Its bundled cuBLAS 10.0 **cannot run GEMM ops on Ada Lovelace (RTX 4090, sm_89) at all** — `CUBLAS_STATUS_EXECUTION_FAILED`. This is a real hardware/library ceiling, not fixable by env vars. **Always run this on CPU**: `CUDA_VISIBLE_DEVICES="" python coefficients.py`. It's fast enough (~1–2 min for the whole ADO loop, tiny 9-layer MLP) that GPU isn't worth it anyway. |
| `pinnsr_gpu` | New env I created today (`conda_envs/pinnsr_gpu_env.yml`, Python 3.8 + `nvidia-tensorflow==1.15.5+nv23.03`). Actually runs PINN-SR training on the RTX 4090s. | **Do NOT use this to reproduce a specific published/expected result.** Full-scale test: same script, same seed, `pinnsr_legacy`-CPU reproduced the committed `u_error` bit-for-bit; `pinnsr_gpu`-GPU found a *different active PDE term* in the STRidge sparse regression (not just rounding noise) — GPU-side non-determinism (cuDNN/cuBLAS kernel/reduction order) that a fixed seed doesn't protect against. Only reach for `pinnsr_gpu` for brand-new PINN-SR work where matching an existing run isn't the goal. |

Rule of thumb: **onestep/twostep → keql_legacy on GPU. PINN-SR → pinnsr_legacy on CPU.**
Only fall back to `pinnsr_gpu` if CPU is genuinely too slow for a *new* experiment (it wasn't for anything Burgers-shaped so far).

## 2. The PINN-SR cross-directory handoff (the non-obvious mechanic)

For each experiment, PINN-SR is split across two directories that don't talk to each
other automatically — you have to shuttle two `.npy` files by hand:

1. `examples/Burgers/<exp>/PINNSR/compute.py` (or `compute_part1.py`, see below), run
   under `keql_legacy` — builds the problem, formats it for PINN-SR, saves
   `data_in_PINNSR[_test].npy`. **This will crash if run in full**, because later in
   the same script it immediately tries to load `data_out_PINNSR[_test].npy`, which
   doesn't exist yet. Truncate the script right after the `data_in` save (I made
   `compute_part1.py` files for this) to run just the prep half.
2. Copy `data_in_PINNSR[_test].npy` → `examples/Burgers_PINNSR/<exp>/`.
3. There, run `init_data_out_PINNSR[_test].py` (writes a placeholder) then
   `coefficients[_test].py` (the actual TF1 training) under `pinnsr_legacy`, CPU only.
   → produces `data_out_PINNSR[_test].npy`. (`run_exp[_test].sh` does these two
   steps, but it calls bare `python` — make sure that resolves to `pinnsr_legacy`,
   e.g. invoke the two commands directly with the env's full python path instead of
   trusting the script's shebang-less `python`.)
4. Copy `data_out_PINNSR[_test].npy` back to `examples/Burgers/<exp>/PINNSR/`.
5. Re-run the **full**, untruncated `compute.py` under `keql_legacy` — it redoes the
   (deterministic, cheap) data prep, then successfully loads `data_out` this time and
   writes the final `PINNSR/data.npy` that `figs/plot.ipynb` consumes.
6. Run `figs/plot.ipynb` (as a plain script works fine, see §4) under `keql_legacy` —
   loads `onestep/data.npy`, `twostep/data.npy`, `PINNSR/data.npy`, prints NRMSEs,
   writes the PDFs.

## 3. Ablation folder conventions (add / remove)

Two derivative-feature sets, applied consistently everywhere they appear:

- **add**: `feature_operators = (eval_k, dx_k, dxx_k, dtx_k, dtt_k)` → 21-term PINN-SR
  library (`1, u, u_x, u_xx, u_tx, u_tt` + all pairwise products). Needs
  `dtx_k, dtt_k` added to the `KernelTools` import wherever `feature_operators` is
  defined.
- **remove**: `feature_operators = (eval_k, dx_k)` → 6-term library
  (`1, u, u_x, u², u·u_x, u_x²`).
- **twostep is the fixed baseline** at `feature_operators = (eval_k, dx_k, dxx_k)`
  (10-term library) in *both* the add and remove folders — confirmed byte-identical
  between shock's `_add`/`_remove` twostep scripts. Since nothing changes for it,
  **just copy the unablated experiment's `twostep/data.npy` instead of recomputing**.
- `u_operators` (used to build the `CholInducedRKHS` basis) does **not** need to
  change — `evaluate_operators` can evaluate any derivative combination regardless of
  which operators were used to construct the basis (verified: shock's onestep uses
  `u_operators=(eval_k,dx_k,dt_k)` while `feature_operators` goes up to `dtx_k,dtt_k`
  with no issue).
- **The one piece of real engineering, not a mechanical copy**: any experiment that
  does an "operator learning" / new-IC generalization step (onlybdry has this, shock
  doesn't) hand-builds the PINN-SR feature matrix in Python
  (`get_grid_features` inside `get_u_pde_adj`, in `examples/Burgers/<exp>/PINNSR/compute.py`)
  to match the *discovered* `coeffs` vector's column order. This has to be rewritten
  by hand to match each ablation's library (21 or 6 columns), in **exactly** the same
  order as the `Phi` construction in the corresponding `Burgers_PINNSR/<exp>/coefficients.py`'s
  `net_f`. Getting the column order wrong silently gives nonsense predictions (no
  error, just wrong numbers) — double check by comparing to a case with a known
  working library, e.g. the 10-term baseline or the 21/6-term shock versions I
  already built.
- On the `Burgers_PINNSR/<exp>/coefficients.py` side, the library-size change touches
  5 spots, all easy to miss: `lambda_history_Adam`, `lambda_normalized_history_STRidge`,
  `lambda_history_STRidge`, `lambda_history_Pretrain` (`np.zeros((N,1))`), and
  `self.lambda1` (`tf.zeros([N, 1], ...)`), plus the `net_f`/`library_description`
  block itself.

## 4. How I built + verified new ablation folders (recipe for Darcy, if it turns out to have an analogous structure)

1. Copy the *entire* unablated experiment folder (both `examples/Burgers/<exp>` and
   its `examples/Burgers_PINNSR/<exp>` twin) into a scratch dir outside the repo,
   strip stale `data*.npy`/`*.png`/`*.pdf` outputs.
2. Patch **only** the specific cells/lines that need to change — I loaded the
   original `.ipynb` as JSON and did targeted `cell['source']` string replacement so
   the cell structure and everything else (grid, obs sampling, kernel choices,
   optimizer settings) stayed byte-identical to the original. Mirrored the same
   edits into the paired `.py` script.
3. Run a **quick smoke test first** (reduced LM `max_iter`, reduced PINN-SR
   `model.train()`/Adam-loop counts) to confirm the pipeline mechanics work
   (imports resolve, file handoffs line up, shapes match) before spending real
   compute on a full run.
4. Run the **full-scale** version and sanity-check NRMSEs are in a plausible range
   (compare against the unablated baseline's NRMSE if one exists).
5. Only then copy the verified folder tree into the real `examples/` — never
   build/test directly in the repo.
6. Clean up scratch-only helper files (`compute_part1.py`, `plot_extracted.py`, etc.)
   before copying into the repo — they're testing aids, not part of the convention
   used elsewhere in the repo.

## 5. Reproducibility results actually measured today (for reference)

- Shock ablation, full-scale, GPU (`keql_legacy` properly activated):
  `onestep` NRMSE 0.37045481618833914 vs committed 0.3704548160949337 (9 sig figs).
- Shock ablation PINN-SR, `pinnsr_legacy` CPU: `u_error` matched the committed value
  bit-for-bit for both `_add` (0.323702738664326) and `_remove` (0.3626896271522685),
  in under 2 minutes each.
- Shock ablation PINN-SR, `pinnsr_gpu` GPU: same seed, found a *different* active
  term (`u_xx`, coeff ≈0.070) than the committed run (`u_tx**2`, coeff ≈0.00011,
  i.e. noise) — this is why PINN-SR training stays on CPU.

## 6. Darcy — NOT investigated yet, don't assume the above transfers directly

Quick look at `examples/Darcy/` (2026-08-02, structure only, no code read):
- No `onestep`/`twostep`/`PINNSR` subfolder split like Burgers has.
- No `Darcy_PINNSR` (or similar) companion directory — PINN-SR baseline may not
  even apply to Darcy.
- Folders are instead `in_distribution/`, `in_sample/`, `operator_learning/`,
  `out_distribution/`, each with `compute_errors.py`, `errors.npy`, `one_run.ipynb`.
- Darcy is an elliptic (time-independent) PDE, so the whole "add/remove `dtx_k`,
  `dtt_k`" ablation framing (which is about *time*-derivative features) likely
  doesn't apply as-is — would need to figure out what the analogous feature-ablation
  axis even is for Darcy before doing anything.
- `keql_tools/darcy_data.py` and `keql_tools/pde_solvers` likely hold the relevant
  data-generation code — haven't opened them yet.

**Next session: read `examples/Darcy/*/one_run.ipynb` and `compute_errors.py` from
scratch before assuming any Burgers convention applies.**

## 7. Loose ends from today, not yet resolved

- `conda_envs/pinnsr_gpu_env.yml` is created but **not committed**.
- `examples/Burgers/oneshot_onlybdry_ablation_{add,remove}` and their
  `Burgers_PINNSR` twins are created but **not committed**.
- No changes were made to `examples/Burgers_PINNSR/oneshot_shock_ablation_*` (the
  jax→numpy swap and `pinnsr_gpu` switch were only prototyped in a scratch copy,
  then explicitly NOT applied to the real repo — see §1, PINN-SR stays on
  `pinnsr_legacy`/CPU there).
