[![DOI](https://zenodo.org/badge/197470755.svg)](https://zenodo.org/badge/latestdoi/197470755)

# Wind turbine main bearing fatigue — physics-informed neural network

A reproduction, on Linux/WSL with TensorFlow 2.15, of

> Y. A. Yucesan, F. A. C. Viana, *A physics-informed neural network for wind turbine
> main bearing fatigue*, International Journal of Prognostics and Health Management,
> 11(1), 2020.

plus one experiment the paper does not run: **replacing the physics with a neural
network**, to measure what the physics is actually worth.

---

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# MANDATORY: pml-pinn does not run on TF 2.15 as published
patch -d "$(python -c 'import pinn,os;print(os.path.dirname(os.path.dirname(pinn.__file__)))')" \
      -p0 < pinn-tf215.patch

# dataset (~190 MB) from Harvard Dataverse, then link it
cd deterministic_grease_inspection/ijphm_2020
ln -s /path/to/wind_bearing_dataset/data   data
ln -s /path/to/wind_bearing_dataset/tables tables
```

Dataset: <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ENNXLZ>

### The two commands

```bash
cd deterministic_grease_inspection/ijphm_2020/advanced

# WITH physics — reproduces the paper
MPLBACKEND=Agg python train.py --physics

# WITHOUT physics — the bearing is predicted by an MLP instead of the SKF formula
MPLBACKEND=Agg python train.py --no-physics

# pick the initialisation (1-10, default 3)
MPLBACKEND=Agg python train.py --physics --case 3
```

Run them from that directory (the scripts resolve data as `../data`), and keep
`MPLBACKEND=Agg` on WSL, where there is no display.

`--bearing-epochs` changes step 4's budget (default 300). Everything else follows the
paper's values and is deliberately not exposed as an option.

### The ten cases

The paper draws the initial plane ten times and reports very different outcomes: 4 good,
3 fair, 3 poor. `--case N` selects one. `α₀` sets the plane's **floor** — the slowest
degradation it allows — which training can never push below.

| case | α₀ | floor | | case | α₀ | floor |
|---|---|---|---|---|---|---|
| 1 | 0.0051 | 3.5e-06 | | 6 | 0.0021 | 6.1e-06 |
| 2 | 0.0025 | 8.4e-06 | | 7 | 0.0148 | 6.8e-06 |
| 3 | 0.0068 | 6.8e-06 | | 8 | 0.0182 | 8.3e-06 |
| 4 | 0.0099 | 6.4e-06 | | 9 | 0.0118 | 4.1e-06 |
| 5 | 0.0015 | 7.6e-06 | | 10 | 0.0167 | 6.9e-06 |

```bash
# sweep all ten
for c in 1 2 3 4 5 6 7 8 9 10; do
  MPLBACKEND=Agg python train.py --physics --case $c
done
```

**Verification.** `--physics --case 3` must print
`grasso_rmse_test = 0.05311429793267597` and
`cuscinetto_rmse = 0.0006088197193056023`. If those match, the environment is correct.

---

## Pipeline order

Both commands run the same four steps. **Only step 4 differs.**

```
   ┌─ 1 ── initial plane ──────────────────────────────────────────────┐
   │      Δd = α₀ + α₁·temp + α₂·wind + α₃·d,  coefficients from --case │
   └───────────────────────────────┬───────────────────────────────────┘
                                   ▼
   ┌─ 2 ── pre-train the MLP on that plane ────────────────────────────┐
   │      MLP#1: 40 sigmoid → 20 elu → 10 elu → 5 elu → 1 sigmoid       │
   │      RMSprop lr 0.01, 500 epochs, plain MSE                        │
   └───────────────────────────────┬───────────────────────────────────┘
                                   ▼
   ┌─ 3 ── fine-tune it inside the cumulative-damage RNN ──────────────┐
   │      d_t = d_{t-1} + Δd_t over 25,920 ten-minute steps             │
   │      RMSprop lr 5e-4, 50 epochs, MASKED MSE on 6 inspections       │
   │      → grease damage for every turbine                             │
   └───────────────────────────────┬───────────────────────────────────┘
                                   ▼
   ┌─ 4 ── bearing fatigue, from the PREDICTED grease ─────────────────┐
   │  --physics     SKF chain: κ → η_c → a_SKF → L10 → Palmgren-Miner   │
   │                0 parameters, 0 data, no training                   │
   │  --no-physics  an MLP with the same 4 inputs computes the          │
   │                increment, trained on 6 true damage values          │
   └───────────────────────────────────────────────────────────────────┘
```

Timings on an 8-core CPU: steps 1–3 ≈ 285 s; step 4 is 3 s with physics,
≈ 1,600 s without.

### What the network predicts, and why it is hard

The network does **not** predict damage. It predicts the **increment** — how much
degradation happens in these 10 minutes — and the accumulation turns that stream into
the total. This makes damage irreversible by construction: a decreasing damage is not
representable, so it never has to be penalised.

The catch: **the increment is never observed.** Only the accumulated value is measured,
and only **six times in six months**, when a grease sample goes to a lab. With 10
turbines that is **60 numbers in total**. Training compares only the *running sum*
against those six values — rather like inferring a car's instantaneous speed from a
monthly odometer reading.

### Where the physics lives

Not in the loss. The loss is a plain **masked MSE** (Eq. 10 of the paper) with no
physical term — no regulariser, no penalty. The physics is wired into the
**architecture** as frozen layers: the SKF catalogue charts are loaded with
`set_weights()` into `TableInterpolation` layers, **17,155 non-trainable parameters**.
They are layers only so gradients can pass through them.

That makes the constraint *hard* — unviolable, and with no weighting to tune — but it
only works for physics you can write as a forward computation.

### How the swap is implemented

`CumulativeDamageCell` accepts any Keras `Model` in its `model=` argument and calls it
once per timestep. Its whole contract is `.weights` plus being callable, so swapping
physics for a network changes one line and nothing else:

```python
# --physics   (basic/models_and_functions.py:175 — the authors' own code)
CDMCellHybrid = CumulativeDamageCell(model=functionalModel, ...)          # SKF chain

# --no-physics (advanced/train.py:202)
cb = CumulativeDamageCell(model=Model(inputs=[phb], outputs=[ob]), ...)   # MLP
```

Two details keep the comparison fair: the MLP is blocked from reading the accumulated
damage (`Lambda(z[:, 1:])`), because the SKF chain never reads it either; and its output
is rescaled onto `[0, 10·max(n/L10)]`, an interval computable from load and cycles alone,
so it does not leak the answer.

---

## Repository layout

| path | what it is |
|---|---|
| `README.md` | this file |
| `requirements.txt` | exact pinned environment (Python 3.11) |
| `pinn-tf215.patch` | **mandatory** fixes to the `pinn` package for TF 2.15 |
| `LICENSE` | MIT, from the original authors |
| `2594-Full-Length Manuscripts-*.pdf` | the paper (gitignored) |
| `wind_bearing_dataset/` | the dataset, downloaded separately (gitignored) |
| `deterministic_grease_inspection/` | IJPHM 2020 — **the work reproduced here** |
| `probabilistic_grease_inspection/` | the authors' other papers, untouched |

### `deterministic_grease_inspection/ijphm_2020/`

| path | what it does |
|---|---|
| `data` → | symlink to the dataset time series (load, temperature, grease, cycles) |
| `tables` → | symlink to the SKF catalogue charts (`kappa`, `etac`, `aSKF`) |
| **`advanced/train.py`** | **the entry point** — the full pipeline, `--physics` / `--no-physics` / `--case` |
| `advanced/runs/` | one timestamped folder per run (gitignored) |
| `advanced/pinn_model.py` | the authors' SKF chain for the 30-year forecast |
| `advanced/case_config.py` | the authors' `CASE=<n>` switch for `run01`–`run04` |
| `advanced/run01_random_plane_generator.py` | authors' step 1 — generates the initial plane |
| `advanced/run02_train_mlp_with_plane.py` | authors' step 2 — pre-trains the MLP |
| `advanced/run03_train_rnn.py` | authors' step 3 — fine-tunes inside the RNN |
| `advanced/run04_predict_fatigue_life.py` | authors' step 4 — 30-year bearing forecast |
| `basic/models_and_functions.py` | **the SKF chain and the RNN builders** — used by `train.py` |
| `basic/run01_train_rnn.py` | authors' 6-month demo, grease training |
| `basic/run02_predict_pinn.py` | authors' 6-month demo, bearing prediction |
| `basic/models/MLP_PLANE.h5py` | pre-trained MLP shipped with the dataset — **not generated** |

`run01`–`run04` and the `basic/` scripts are the authors' original code, modified only
to run on Linux with TF 2.15. They still work, but `train.py` supersedes them.

### What a run produces

`advanced/runs/<timestamp>_case<N>_<method>/`

| file | content |
|---|---|
| `config.json` | arguments, seed, hyperparameters, Python and TF versions |
| `metrics.json` | metrics split by branch — grease and bearing |
| `loss_mlp_piano.csv` | loss per epoch, step 2 (500 rows) |
| `loss_grasso.csv` | loss per epoch, step 3 (50 rows) |
| `loss_cuscinetto.csv` | loss per epoch, step 4 — `--no-physics` only (300 rows) |
| `grasso_predictions_test.csv` | 4 held-out turbines × 6 inspections: predicted, true, error |
| `grasso_predictions_train.csv` | the same for the 10 training turbines |
| `grasso_predictions_daily.csv` | day-by-day curve on the test turbines |
| `cuscinetto_predictions.csv` | bearing damage: predicted, true, error |
| `models/` | trained MLP, RNN weights, the initial plane |
| `plots/` | loss, predicted-vs-actual, test curves, bearing damage |

Every CSV carries a `metodo` and `caso` column, so runs never get confused.

---

## Results

### The bearing branch: physics wins decisively

Same inputs, same target; only the increment calculation differs. Fed the *true* grease
damage, to isolate the bearing stage:

| method | physics | RMSE | observations | training | final error |
|---|---|---|---|---|---|
| SKF chain | yes | **0.00052** | **0** | **0 s** | **+0.4%** |
| MLP, 6 obs. | no | 0.00964 | 6 | 1,568 s | +50.4% |
| MLP, 180 obs. | no | 0.01059 | 180 | 1,455 s | +56.5% |

**Thirty times more data makes it slightly worse.** The gap is not data scarcity, it is
the substitution itself: the network settles on a constant wear rate and ignores load and
temperature, which is the easiest solution when the only signal is a running total.

### End to end, the two commands compared

Steps 1–3 are identical, so the grease result is **bit-identical**; only step 4 differs.

| | case 3 `--physics` | case 3 `--no-physics` | case 4 `--physics` | case 4 `--no-physics` |
|---|---|---|---|---|
| grease RMSE | 0.05311 | 0.05311 | 0.10899 | 0.10899 |
| bearing RMSE | **0.000609** | 0.001396 | **0.001400** | 0.005389 |
| bearing final error | **+4.66%** | −6.11% | **+8.85%** | +21.87% |
| bearing runtime | **3 s** | 1,440 s | **3 s** | 1,584 s |

The advantage **widens as the upstream input degrades** — 2.3× on case 3, 3.9× on
case 4. With a worse grease estimate the SKF chain degrades gracefully; the MLP does not.

Note also that the sign flips: physics overestimates, the MLP underestimates. For a
maintenance decision these are not equivalent — one replaces early, the other arrives late.

### Initialisation dominates the grease branch

Across the ten unconstrained cases (the repository's own draw):

| | RMSE on held-out turbines |
|---|---|
| best | 0.127 |
| median | 0.898 |
| worst | 1.461 |
| within the paper's stated 0.010–0.018 | **0 of 10** |
| overestimating the damage | **10 of 10** |

The correlation between the plane's floor and the final error is **0.998** — it is the
only variable that matters. A null model (always predicting the mean) scores 0.305, so
**8 of the 10 cases are worse than doing nothing**. With the constraint restored, the
best case reaches 0.054.

`α₀` sets the slowest degradation the plane allows, and training can never push below it.
Real early-life increments are around 4e-06; above that, the model overestimates forever.
The paper says the coefficients are chosen "using engineering judgment"; the published
code draws `np.random.random(4)` with no constraint, which puts the floor too high about
97% of the time. Footnote 5 of the paper is candid about it: training from purely random
weights *"proved to be extremely hard and we had no success"*.

### Physics as a data multiplier

Same loss, same held-out turbines, varying the training set:

| training data | observations | PINN | ANN (no physics) | gap |
|---|---|---|---|---|
| 20% | 12 | **0.00832** | 0.02600 | 3.1× |
| 40% | 24 | **0.00624** | 0.02321 | 3.7× |
| 60% | 36 | **0.00359** | 0.01001 | 2.8× |
| 80% | 48 | **0.00276** | 0.00606 | 2.2× |
| 100% | 60 | **0.00291** | 0.00532 | 1.8× |

The physics does not merely lower the error — it **shifts the curve left**. With 12
observations it reaches what the plain network needs about 48 to match: two turbines
sampled for six months instead of eight.

### Honest limitations

- **The paper's headline numbers were not reached.** No run lands in the stated
  0.010–0.018 band; the best is 0.054. The authors' own pre-trained demo reaches 0.109.
- **Only the grease has a loss.** Bearing damage appears in no cost function, here or in
  the paper: it is not measurable in the field. Its predictions are never validated
  during training.
- **Bearing ground truth exists for one turbine only**, and that turbine is inside the
  grease training set. That comparison rests on a single time series.
- **A bug in the repository**: `run02_predict_pinn.py` compares turbine 1's prediction
  against turbine 8's ground truth. Corrected, the physics error drops from +43% to +0.4%.
- **A reproducibility bug**: `pyDOE.lhs` uses `np.random.default_rng()` and ignores
  `np.random.seed()`, so the same case produced a different plane on every run until
  seeded explicitly. This affected the original code too.

---

## Portability fixes

The original code targets Windows and TensorFlow 2.0 from 2019. None of these change the
numerics:

| # | problem | fix |
|---|---|---|
| 1 | `pinn` imports from `tensorflow.python.keras`, the old internal copy, no longer interoperable with `tensorflow.keras` in TF 2.15 | imports repointed |
| 2 | `TableInterpolation.call` overwrites its own Variables with graph tensors, leaking a dead tensor across FuncGraphs on the second trace | read into locals |
| 3 | `arrange_table` builds a ragged NumPy array — a hard error since NumPy 1.24 | kept as a list |
| 4 | Windows paths, `plt.show()` with no display, an `xticks` call with 175 locations and 7 labels | POSIX paths, `Agg`, matched ticks |
| 5 | `pyDOE.lhs` ignores `np.random.seed()` | seeded explicitly |

Fixes 1 and 2 live in `pinn-tf215.patch` and apply to the installed package, not to this
repository.

---

## Citing the original work

    @misc{2019_yucesan_viana_python_main_bearing,
        author    = {Y. A. Yucesan and F. A. C. Viana},
        title     = {Python Scripts for Wind Turbine Main Bearing Fatigue Life
                     Estimation with Physics-informed Neural Networks},
        month     = Aug,
        year      = 2019,
        doi       = {10.5281/zenodo.3355725},
        version   = {0.0.1},
        publisher = {Zenodo},
        url       = {https://github.com/PML-UCF/pinn_wind_bearing}
    }

Other papers from the same group that use this repository:

- Yucesan & Viana, "[A hybrid physics-informed neural network for main bearing fatigue
  prognosis under grease quality variation](https://www.sciencedirect.com/science/article/pii/S088832702200070X),"
  *Mechanical Systems and Signal Processing*, 171, 108875, 2022.
- Yucesan, Dourado & Viana, "[A survey of modeling for prognosis and health management of
  industrial equipment](https://www.sciencedirect.com/science/article/pii/S1474034621001567),"
  *Advanced Engineering Informatics*, 50, 101404, 2021.
- Yucesan & Viana, "[Hybrid physics-informed neural networks for main bearing fatigue
  prognosis with visual grease inspection](https://www.sciencedirect.com/science/article/pii/S0166361520306205),"
  *Computers in Industry*, 125, 103386, 2021.
- Viana, Nascimento, Dourado & Yucesan, "[Estimating model inadequacy in ordinary
  differential equations with physics-informed neural networks](https://www.sciencedirect.com/science/article/pii/S0045794920302613),"
  *Computers and Structures*, 245, 106458, 2021.
- Yucesan & Viana, "[A physics-informed neural network for wind turbine main bearing
  fatigue](http://www.phmsociety.org/node/2736)," *IJPHM*, 11(1), 2020 — **the paper
  reproduced here**.
