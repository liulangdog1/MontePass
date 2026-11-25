# How the QMC settings are chosen

A DMC run needs a time step, a walker count, a block length and a number of blocks. The stage
`autotune` chooses them from short pilot runs of the molecule itself, so that the user does not
need experience with similar systems. Every chosen value is written to
`work/autotune/qmc_settings.json` together with the measurement behind it, and the report shows a
table of them.

Three modes (`[qmc] mode`):

| Mode | What happens |
|---|---|
| `auto` | One round of pilot runs, then the rules below. |
| `iterative` | As `auto`. If a check fails, the affected pilot runs are repeated with a changed setting, at most `max_repeats` times. |
| `fixed` | No pilot run. The values of the input file are checked for consistency and used as they are. |

## 1. The pilot runs

At the starting geometry of the first target the package runs one short DMC run for each
candidate time step: 0.02, 0.01 and 0.005 Ha^-1, with 512 walkers. Each run has its own Jastrow
optimization. The runs are stopped together as soon as the data allow the decision: either the
slope of the energy against the time step is known to 25 %, or more data could not change the
choice.

Measured from each run: the acceptance, the spread of the walker population, the error bar of the
energy (with the autocorrelation time of the blocks), the seconds per block, and the variance of
the VMC run that made the walkers.

## 2. The time step

A candidate time step is allowed if all of these hold:

| Rule | Condition | Default limit |
|---|---|---|
| R1 | mean DMC acceptance | at least 0.99 |
| R2 | DMC variance / VMC variance | at most 1.10 |
| R3 | relative spread of the walker population | at most 5 % at 2048 walkers |
| R4 | predicted time-step bias of an energy difference, `0.15 |k_E| tau` | at most `sigma_target` |

`k_E` is the measured slope of the total DMC energy against the time step. The factor 0.15 is the
measured ratio between the time-step slope of an energy difference of two nearby geometries and
`k_E`. If the slope is not resolved to 25 %, the rule uses `|k_E| + 2 sigma`.

The slope is a straight-line fit through the energies of all pilot runs with usable data. A run
that fails R1 to R3 is not a candidate, but its energy is still in the fit: the run at the largest
time step gives the fit most of its lever arm. If a straight line does not describe the energies
(`chi^2` above its 95 % point), the largest time step is left out of the fit.

The largest allowed candidate is taken, because the cost of a run is proportional to `1 / tau`.
A rule whose input was not measured is recorded as "not checked" and does not exclude a
candidate.

## 3. Block length, blocks and walkers

With the variance `V`, the effective correlation time `T_c` (defined by
`SEM^2 = 2 T_c V / (N_w T)` for `T` Ha^-1 of data and `N_w` walkers) and the wanted error bar
`sigma` in hartree:

    imaginary time needed      T = 2 T_c V / (N_w sigma^2)
    blocks                     B = n_discard + 1.2 T / T_b

- A block is 2.5 Ha^-1 long (`steps = 2.5 / tau`), which is 20 to 30 correlation times.
- The first 10 Ha^-1 of a run are discarded (`warmup_steps = 10 / tau`), and the analysis discards
  further start-up blocks if the data ask for it.
- The product `2 T_c V` is not guessed: it is taken from the error bar of the pilot run.
- A run keeps at least 64 blocks, so that its error bar can be estimated.
- The walker target is 2048, rounded up to a multiple of the MPI ranks.
- `sigma_target` (default 0.2 kcal/mol) is the wanted error bar of the final energy of one
  geometry. The number of blocks of a final run follows from it. Scan points may use fewer blocks:
  the search stops them when its own precision target is reached.

## 4. The Jastrow factor

One-body and two-body B-spline Jastrow factors. Their cutoffs follow the size of the molecule
(the largest distance between two atoms). The optimization uses the linear method with 8 cycles
of 100,000 samples. Among the cycles whose variance is within 1.5 times the smallest one, the
cycle with the lowest energy is used for DMC.

## 5. Checks on every run

`monitor.py` holds checks that work on the block trace of a run.

| Check | What it looks for |
|---|---|
| M1 | no new block for 5 expected block times (a hang) |
| M2 | a block takes more than 2.5 times the expected time |
| M3 | the start-up part is not over after the planned discard |
| M4 | the walker population is off target, spreads too much or drifts |
| M5 | a block energy more than 6 robust standard deviations from the median |
| M6 | the acceptance falls during the run: one block far below the first blocks, or 10 blocks in a row below them. "Far" is measured with the scatter of the block acceptance of the run itself |
| M7 | the mean acceptance is below 0.99 |
| M8 | the DMC variance is more than 1.10 times the VMC variance, or changes during the run |
| M9 | two runs of one geometry share random seeds |
| M10 | no checkpoint is written |

The pilot applies M3 to M8 to its runs. In mode `iterative` a finding leads to a repeat: a
transient that is not over doubles the discarded time; bad population or outlier data lead to a
new run with another seed; a DMC variance above the limit leads to a Jastrow optimization with
twice the samples.

## 6. Where the default numbers come from

The limits and first guesses were measured on 188 production DMC traces of 9 closed-shell
molecules with 28 to 62 valence electrons (elements H, C, N, O, S; ccECP pseudopotentials; one
B3LYP determinant with a one-body and two-body Jastrow factor; 2048 walkers; time step 0.005).
Measured ranges: variance / |E| = 0.0204 to 0.0228 hartree; `T_c` = 0.114 to 0.158 Ha^-1;
acceptance 0.9976 to 0.9984; population spread 1.6 to 1.9 %; `k_E` = -270 to -380 kcal/mol per
Ha^-1.

## 7. Examples from the test runs

| | water (8 valence electrons) | ammonia (8 valence electrons) |
|---|---|---|
| acceptance at 0.02 / 0.01 / 0.005 | 0.984 / 0.993 / 0.997 | 0.990 / 0.996 / 0.998 |
| slope `k_E` (kcal/mol per Ha^-1) | -183 +/- 38 | -7 +/- 29 (not resolved; the rule uses 65) |
| chosen time step | 0.005 (0.01 fails R4: predicted bias 0.275 kcal/mol) | 0.02 |
| steps per block, walkers | 500, 2048 | 125, 2048 |
| blocks of a final run | 106 | 113 |
| cost of the pilot runs | 2.1 core-hours | 2.6 core-hours |

For water the rules arrive at the time step that is commonly chosen by hand. For ammonia they
allow a time step four times larger, which makes every DMC run four times cheaper. The ammonia
barrier was also measured in a run at 0.005: 5.08 +/- 0.19 kcal/mol against 5.27 +/- 0.22 at 0.02
(both at the B3LYP geometries), a difference of 0.19 +/- 0.29.

**Is the choice the same when the pilot is repeated?** Independent pilot runs of the same molecule:

| Molecule | Slope, kcal/mol per Ha^-1 (3 time steps) | Chosen time step |
|---|---|---|
| water, B3LYP geometry | -183 +/- 38, -184 +/- 35 | 0.005, 0.005 |
| water, O-H bonds longer by 0.095 angstrom | -200 +/- 35 | 0.005 |
| ammonia | -7 +/- 29, -22 +/- 21 | 0.02, 0.02 |

Five other pilots of these molecules and of HCN chose 0.0016 to 0.005, because an earlier form of
check M6 excluded their run at 0.02. With the slope from 2 time steps the error of the slope was
70 to 175, and the rule `|k_E| + 2 sigma` then asked for a smaller time step. Their stored traces, analysed again with the corrected check,
give 0.005 for water (3 pilots) and HCN, and 0.01 for ammonia.

## 8. Limits

- The choice depends on how well the pilot runs resolve the slope `k_E`. When the slope is not
  resolved to 25 %, the rule uses `|k_E| + 2 sigma`, so a pilot with a large error chooses a
  smaller time step than needed. The result is safe and costs more. The log names the slope, its
  error and the value that the rule used.
- For ammonia the predicted bias at 0.02 is 0.195 kcal/mol, just below the limit of 0.20. A pilot
  with slightly different data can choose 0.01.
- The factor 0.15 of rule R4 and the default limits were measured for molecules of first-row and
  second-row elements with ccECP pseudopotentials. They were not tested for transition metals,
  open shells or other pseudopotentials.
- The walker count is not optimized. For a small molecule fewer walkers would reach the same
  error bar at a lower cost; the log says so when that is the case.
- A user who wants the conventional settings regardless of the pilot uses `mode = fixed`.
