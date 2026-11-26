# The search methods and how they compare

All four methods optimize one geometry (a minimum or a saddle) with DMC energies, in `d` search
directions. They differ in which points they compute, in how they turn energies into a step, and
in when they stop.

Symbols as in [methods.md](methods.md): `t` displacement along a direction (angstrom), `g` slope
(kcal/mol/angstrom), `k` force constant (kcal/mol/angstrom^2, `E = k t^2 / 2`). `E+`, `E-` and `E0`
are the DMC energies at `+t`, `-t` and at the centre; `S+`, `S-`, `S0` are the surrogate energies
of the same three geometries.

## 1. Classic line search (`method = classic`)

1. For every direction the surrogate energy is scanned at 17 points. The scan is wide: its
   half-width is 0.69 to 1.14 angstrom for water.
2. A "sizer" chooses for every direction a window and a noise level for the DMC points, so that a
   cubic fit through 7 noisy points on the surrogate curve gives the stationary point within a
   tolerance.
3. Every iteration computes 6 DMC points per direction plus the shared centre, fits a cubic per
   direction and moves all directions together.
4. After a fixed number of iterations (4) it stops and runs full-length DMC at the start and at
   the end.

The method has no convergence test. The package adds two checks that do not change a step:

- **Bracket check.** Is the stationary point of the cubic inside the grid?
- **Stability audit.** For each direction a quadratic fit, a cubic fit and one cubic fit per
  left-out point are compared. If their stationary points spread by more than
  `max(2 tolerance, 0.01 angstrom)`, the direction is `UNRESOLVED`: the step is then a choice of the
  fit model and not a measurement.

## 2. Dense scans, protocol v1 (`method = dense-v1`)

Protocol v1 is the set of rules that was fixed in advance for an earlier application of the method
to a larger molecule. The module `search/protocol_v1.py` is a line-by-line port of these rules and
must not be changed. The test that replays the stored block traces of that application through the
port needs data that are not part of this repository, so this repository has no such test.

- Points: two shells per direction at `+-R/2` and `+-R`, with `R = sqrt(2 e_cap / k)` and
  `e_cap = 5 kcal/mol`. Two full-length runs at every centre.
- Estimator: the odd part `E(+t) - E(-t)` is fitted with a line through zero; its slope gives
  the energy slope. The even part gives the curvature.
- Rule A: the slope is at least 4 standard errors from zero and the line fits
  (`chi^2` per degree of freedom at most 3). The direction is `DISPLACED`.
- Rule B: the energy that could still be gained along the direction,
  `k (|m| + 2 sigma_m)^2 / 2`, is below `eps_total / d`. The direction is `ZERO`.
- Blocks are requested in chunks of 100 up to 400 per point. 50 blocks are discarded.
- A direction whose line does not fit gets a third shell at `1.5 R` and a cubic model.
- The step of all directions is composed with the DMC curvatures on the diagonal and the
  surrogate couplings off the diagonal.
- Convergence: the sum of the remaining-energy bounds of all directions is below `eps_total`.

## 3. Dense scans, protocol v2 (`method = dense-v2`, the default)

Protocol v2 keeps the symmetric scans and changes ten things.

**1. One shell and the centre.** The slope needs only the outer pair. The curvature comes from
the outer pair and the centre energy. That halves the number of scan points.

**2. The surrogate is subtracted (control variate).** Every DMC point has a surrogate energy at
the same geometry at no extra cost, because the trial orbitals come from that calculation. The
estimator works on "DMC minus surrogate" and adds the exact surrogate slope and curvature:

    g = g_sur + [ (E+ - E-) - (S+ - S-) ] / (2 t)
    k = k_sur + [ (E+ + E- - 2 E0) - (S+ + S- - 2 S0) ] / t^2

A slope from a symmetric pair is exact for a quadratic function. The cubic part of the scanned
function adds an error. Where the surrogate has the shape of the DMC surface, the cubic parts
cancel in the difference. Example from the water test: along the O-H stretch the pair of DMC points
alone gives a slope of 5.4 kcal/mol/angstrom. The cubic part of the surrogate curve over the same
pair is 9.3 kcal/mol/angstrom, and the estimator gives 14.6 +/- 3.6 kcal/mol/angstrom.

The surrogate is used at a centre only while it is close to its own stationary point:
`sum_j g_sur_j^2 / (2 |k_sur_j|) <= e_cap`. Far from that point its cubic part is a different one,
and subtracting it adds more error than it removes. At such a centre the plain DMC differences
are used, and neither the surrogate curvatures nor its couplings. Two things are still taken from
the surrogate there, both from its own stationary point (the first centre of the search):

- its force constant along the direction, with an error of 100 %, as the value that is assumed
  before the scan. Without it a curvature measurement that is mostly noise would be used as it is;
- the cubic part of its slope along the direction (see item 4). It is added to the slope of the
  plain DMC pair, because near its own minimum the DMC surface has nearly the cubic part that the
  surrogate has near its minimum.

**3. Curvature.** The measured and the surrogate force constant are combined by their
uncertainties where they agree within 3 standard errors. Where they disagree, the measurement is
used, and it is remembered: at the next centres it takes the place of the surrogate value. Without
this memory a scan with a short span, whose own curvature measurement is weak, went back to the
surrogate value. Where that value is 2.5 times too small, the steps were too long and the search
oscillated (found on a model with a rotated surrogate Hessian, section 5). A remembered value is
kept only near the stationary point (predicted gain at most `e_cap`) and only with the right sign:
far away, or beyond an inflection point, a force constant is not a property of the next centre.
A later measurement that disagrees with the remembered value replaces it only if it is at least as
precise, or smaller in size. A force constant that is too large makes the predicted gain and the
bound too small, so the search could end too early; one that is too small only makes the search
more careful.

**4. One stopping rule for all directions together.** With the slope vector `g`, its covariance
`S` and the model Hessian `H`:

    G = g^T |H|^-1 g / 2                 energy that the step is predicted to gain
    N = tr(|H|^-1 S) / 2                 expected energy error of the step from the slope errors
    V = g^T |H|^-1 S |H|^-1 g + tr((|H|^-1 S)^2) / 2        variance of G

- `G - N - 2 sqrt(V) > g_move` (0.5 kcal/mol): the gain is clear. Stop the points, move, scan again.
- `c N < eps_total`: the precision is reached. The factor `c = chi^2(0.95, nu) / nu` turns the
  expected error into a joint 95 % bound. `nu` is the effective number of directions that carry
  the noise energy, `nu = tr(|H|^-1 S)^2 / tr((|H|^-1 S)^2)`. It equals the number of directions `n`
  only if all of them carry the same noise energy. At a symmetric centre, or with a reaction
  coordinate that is scanned at a smaller energy, fewer directions carry it, and the factor is
  larger (water at a symmetric centre: 2 of 3 directions, factor 3.00 in place of 2.61). Then the
  predicted gain decides:
  - `G - N <= eps_total`: the search takes the last step and ends as `CONVERGED`. The last step is
    not checked by a scan, and its gain is only as good as the model Hessian, so it must be small.
  - `G - N - sqrt(V) > eps_total`: the gain of a further step is resolved. The search moves.
  - in between: `G` may be noise. The search takes more blocks at this centre and decides again. A
    search that moved on every such `G` wandered around the minimum at the level of the noise.
- No blocks are left (limit `cap` per point) and the precision is not reached: the search takes the
  step and repeats the scan with more blocks per point. The number follows from the noise energy
  that was measured, with a margin of 1.5, and is at most 4 times `cap`. If more blocks are not
  allowed (`[qmc] mode = fixed`, where `cap_blocks` is the limit that the user set) or would not
  help, the search ends as `NOT_CONVERGED`.

The slope error in `S` has a statistical part and a shape part. The shape part is
`shape_mismatch` (0.3) times the cubic part of the surrogate along the scan, which is known
exactly. It covers a surrogate whose cubic part differs from that of the DMC surface. More blocks
do not reduce it. If it keeps the bound above the tolerance (alone, or as more than half of the
tolerance when more blocks cannot bring the rest down), the next centre is scanned with shorter
spans: the shape energy falls with the fourth power of the span. Where the surrogate is not
subtracted (item 2), the slope of a plain pair carries the whole cubic part of the DMC surface.
The cubic part that the surrogate had at its own stationary point is then added to the slope, and
the shape part is 0.5 times that cubic part. If no such value exists, nothing is added and the
shape part is 1.0 times the cubic part of the surrogate at the centre. On the ammonia model the
uncorrected systematic error costs 0.07 kcal/mol at the normal spans and 0.005 at half spans.
Spans are shortened for this reason only near the stationary point: far away the task is to get
there, and full spans allow longer steps.

**5. Surrogate couplings only when they predict well.** The model Hessian has the measured
force constants on the diagonal. The surrogate couplings between directions are added only if the
model with couplings predicts the slope change measured along the last step at least as well as
the model without them.

**6. Safeguards for a start far from the minimum.** A step is never longer than the scanned span
times a trust factor. The factor starts at 1 and follows the ratio of measured to predicted energy
change of the last step (limits 0.25 and 4). A step that raises the centre energy by more than 3
standard errors is rejected and halved. Along a direction whose measured curvature has the wrong
sign, the step goes downhill by the scanned span. After the first scan a span is at most 1.5 times
the span that the direction had in the first scan: a force constant that was measured far too low
would otherwise give a scan that reaches far up the walls, and a step of the same length. If the
iteration limit is reached with a rejected step, the search ends at the centre that the step
started from, with full-length runs there, and reports `NOT_CONVERGED`.

**7. Fewer discarded blocks.** The number is found for every trace (rule MSER-5) and limited to
10 to 50 blocks.

**8. Symmetry.** At a symmetric centre the slope along a direction with no totally symmetric part
is zero. Such a direction gets no pair of points. If a symmetry operation reverses the direction,
one point measures its curvature, so an unstable symmetric geometry is still found.

**9. A point does not keep its cores longer than the decision needs.** The decision of item 4
needs data from every direction. On a busy cluster only some of the points start at once, and
without a limit they would run on to `cap` blocks while the other points wait. With the noise
`sigma` of one block, measured on the points that have data, the precision target is expected to
need

    B = c sum_j sigma^2 / (4 t_j^2 |k_j|) / eps_total

usable blocks per point (`t_j` is the span and `k_j` the force constant of direction `j`; usable
blocks are the blocks after the discarded ones). A running point is stopped when it holds
`plan_factor` (3) times `B` blocks plus the 50 blocks that the analysis may discard. The factor
covers the error of the noise estimate, which comes from short traces. The 50 blocks make sure
that two stopped points of a pair, which are read at their common number of blocks, always have
usable data. Its cores go to the points that wait. The full-length
runs at the starting geometry are never stopped, and no point is stopped if the limit is above
`cap`.

**10. A search that is stopped and started again continues with the same decisions.** A decision
(the step and the new centre, or the final geometry) is written to the state file before any job
of the next stage is submitted. A point that exists already with another geometry is an error,
not a silent reuse. A failed run at a centre and a failed surrogate calculation end the search as
`NEEDS_ATTENTION` with the reason; the command `retry` submits such jobs again.

Five options exist and are off by default, because they made the result less reliable on the
model surfaces of section 5 or saved no blocks:

| Option | What it does | Why it is off |
|---|---|---|
| `pool` | carries "DMC slope minus surrogate slope" to the next centre | in a traced run far from the minimum the carried value was wrong by 3.7 of its claimed standard errors, and the search then moved away from a centre that was already at the minimum |
| `secant` | corrects the model Hessian with a rank-one update from measured slope changes | in a traced run with noisy slopes it gave force constants of 655 and 32,145 where the true values were 2,684 and 2,153 |
| `shrink_final` | multiplies the last step by `max(0, 1 - N / G)` | the bound `c N` is a statement about the full step; with the factor a run could stop without moving |
| `coupling_fit` | fits the couplings between the directions to the slope changes measured along the steps between centres, when the model Hessian does not predict them (99.9 % level), with the surrogate couplings as the prior | on the model with a turned surrogate surface (section 5) 24 of 32 runs (water) and 28 of 32 (ammonia) ended within the tolerance, against 32 and 31 without the option; with only the harmonic part turned, 29 and 30 against 28 and 30. The fitted couplings were close to the true ones, but a correct Hessian passes a systematic slope error on to the geometry in full |
| `shape_distrust` | uses a shape part of 1.0 (in place of 0.3) times the cubic part of the surrogate along a direction where DMC contradicted the surrogate force constant | on the four models with a turned surrogate (32 runs each) it raised the number of runs within the tolerance from 28 to 29 of 32 in one model (water with the harmonic part turned) and left it unchanged in the other three; it cost up to 43 % more search blocks (1,352 against 945, in that model) |

## 4. `method = auto`

One iteration of the classic search. If the audit resolves every direction and every fitted step
is inside its grid, the classic search continues. Otherwise protocol v2 starts from the same
starting geometry and uses the full-length runs at that geometry again.

## 5. Comparison on model surfaces

`examples/model_study/compare_protocols.py` runs all methods on analytic model molecules: water
(3 atoms, 3 search directions) and ammonia (4 atoms, 6 search directions). A "true" surface stands
for DMC, and Gaussian noise of 5 kcal/mol per block is added. A second surface stands for the surrogate. The true stationary point is known,
so the error of a final geometry can be checked: it is the true energy of that geometry above the
true stationary point. These are model surfaces and not QMC runs.

The columns of the tables:

- **classic**: the classic line search with its default options. Every point discards 50 blocks and
  keeps at least 30. With the noise of this study these two lower limits, and not the noise rule of
  the search, set the number of blocks of most points.
- **classic-lean**: the same search with 10 discarded blocks and at least 8 kept blocks, so that
  its own noise rule sets the number of blocks. This is the fair cost of the classic search on
  these models.
- **dense v1**: the rules of the first application.
- **dense v2** at its default tolerance of 0.2 kcal/mol, and at a tolerance of 0.05 kcal/mol, which
  gives about the error that the classic search reaches.

A cell holds three numbers: the mean number of search blocks; the runs whose error is below 0.2
kcal/mol; the mean error in kcal/mol. "Search blocks" are all DMC blocks without the full-length
runs at the starting and at the final geometry. Every case has 16 runs with different noise. A run
"left the molecule" if it ended more than 1 angstrom (root mean square) away from the minimum, or
more than 0.5 kcal/mol below it in energy (the model surface is not bounded where two atoms meet). The iteration limits are the
defaults of the search classes: 4 for the classic search, 5 for v1 and 10 for v2. A real run of the
package allows v2 20 iterations.

**Table A. Start at the stationary point of the surrogate** (as a real calculation starts at the
DFT geometry).

| Case; error of the start (kcal/mol) | classic | classic-lean | dense v1 | dense v2 (tolerance 0.2) | dense v2 (tolerance 0.05) |
|---|---|---|---|---|---|
| water, start at the surrogate minimum; 0.2 | 6,000; 16 of 16; 0.015 | 1,779; 16 of 16; 0.020 | 3,868; 16 of 16; 0.016 | 555; 16 of 16; 0.035 | 1,400; 16 of 16; 0.015 |
| ammonia, start at the surrogate minimum; 0.2 | 14,082; 16 of 16; 0.048 | 6,474; 16 of 16; 0.066 | 15,911; 16 of 16; 0.079 | 493; 16 of 16; 0.030 | 2,441; 16 of 16; 0.008 |
| ammonia inversion saddle; 0.3 | 20,424; 16 of 16; 0.057 | 14,016; 16 of 16; 0.088 | 16,190; 16 of 16; 0.027 | 522; 16 of 16; 0.012 | 984; 16 of 16; 0.003 |
| water, surrogate with a wrong cubic term; 0.2 | 6,000; 16 of 16; 0.015 | 1,827; 16 of 16; 0.012 | 4,365; 16 of 16; 0.017 | 474; 16 of 16; 0.037 | 1,281; 16 of 16; 0.018 |
| ammonia, surrogate with a wrong cubic term; 0.2 | 14,082; 12 of 16; 0.161 | 6,066; 14 of 16; 0.146 | 15,816; 16 of 16; 0.079 | 456; 16 of 16; 0.024 | 2,501; 16 of 16; 0.011 |

**Table B. The surrogate is the true surface, moved along one direction.** The start looks like a
stationary point to the surrogate and is 2 to 147 kcal/mol above the true minimum.

| Case; error of the start (kcal/mol) | classic | classic-lean | dense v1 | dense v2 (tolerance 0.2) | dense v2 (tolerance 0.05) |
|---|---|---|---|---|---|
| water, start 0.05 A away along the antisymmetric stretch; 3.4 | 6,000; 16 of 16; 0.012 | 1,776; 16 of 16; 0.019 | 9,139; 15 of 16; 0.086 | 861; 16 of 16; 0.051 | 2,212; 16 of 16; 0.010 |
| water, start 0.10 A away along the antisymmetric stretch; 13.9 | 6,000; 16 of 16; 0.015 | 1,800; 16 of 16; 0.010 | 13,522; 7 of 16; 1.100 | 1,178; 16 of 16; 0.047 | 3,046; 16 of 16; 0.013 |
| water, start 0.20 A away along the antisymmetric stretch; 59.5 | 6,000; 16 of 16; 0.008 | 1,800; 16 of 16; 0.025 | 23,846; 0 of 16; 39 | 1,616; 16 of 16; 0.039 | 3,706; 16 of 16; 0.012 |
| water, start 0.30 A away along the antisymmetric stretch; 147.3 | 6,000; 16 of 16; 0.010 | 1,851; 16 of 16; 0.023 | 34,400; 0 of 16; 147 | 2,046; 16 of 16; 0.028 | 4,066; 16 of 16; 0.015 |
| water, start 0.10 A away along the symmetric stretch (bonds longer); 9.4 | 6,000; 16 of 16; 0.012 | 1,800; 16 of 16; 0.016 | 12,968; 6 of 16; 1.513 | 923; 16 of 16; 0.035 | 2,097; 16 of 16; 0.008 |
| water, start 0.10 A away along the symmetric stretch (bonds shorter); 13.0 | 6,000; 16 of 16; 0.012 | 1,800; 16 of 16; 0.017 | 10,718; 16 of 16; 0.034 | 949; 16 of 16; 0.035 | 2,779; 16 of 16; 0.010 |
| water, start 0.20 A away along the symmetric stretch (bonds longer); 31.5 | 6,000; 16 of 16; 0.013 | 1,776; 16 of 16; 0.019 | 18,945; 0 of 16; 26 | 1,212; 16 of 16; 0.029 | 2,753; 16 of 16; 0.010 |
| water, start 0.20 A away along the symmetric stretch (bonds shorter); 59.9 | 6,000; 16 of 16; 0.012 | 1,800; 16 of 16; 0.017 | 24,166; 0 of 16; 20 | 1,396; 16 of 16; 0.022 | 2,727; 16 of 16; 0.009 |
| water, start 0.10 A away along the bend; 2.0 | 6,000; 16 of 16; 0.012 | 1,800; 16 of 16; 0.016 | 6,009; 16 of 16; 0.012 | 559; 16 of 16; 0.026 | 1,092; 16 of 16; 0.006 |
| water, start 0.30 A away along the bend; 19.1 | 6,000; 16 of 16; 0.013 | 1,776; 16 of 16; 0.019 | 19,074; 1 of 16; 18 | 912; 16 of 16; 0.035 | 2,104; 16 of 16; 0.008 |

**Table C. The surrogate is turned by 34 degrees.** Two normal directions of the true minimum are
turned into each other, and the surrogate minimum is 1 kcal/mol above the true minimum. In the
rows "surface" the whole surrogate surface is turned, so its anharmonic part is wrong as well. In
the rows "Hessian only" the anharmonic part is that of the true surface.

| Case; error of the start (kcal/mol) | classic | classic-lean | dense v1 | dense v2 (tolerance 0.2) | dense v2 (tolerance 0.05) |
|---|---|---|---|---|---|
| water, surrogate surface turned by 34 degrees; 1.0 | 6,000; 0 of 16; 0.455 | 1,800; 0 of 16; 0.456 | 9,158; 16 of 16; 0.033 | 1,054; 16 of 16; 0.058 | 3,416; 16 of 16; 0.036 |
| ammonia, surrogate surface turned by 34 degrees; 1.0 | 14,298; 0 of 16; 1.312 | 6,786; 0 of 16; 1.416 | 24,750; 16 of 16; 0.052 | 1,437; 16 of 16; 0.058 | 3,972; 16 of 16; 0.020 |
| water, surrogate Hessian only turned by 34 degrees; 1.0 | 6,000; 14 of 16; 0.120 | 2,352; 14 of 16; 0.089 | 9,345; 16 of 16; 0.026 | 908; 14 of 16; 0.092 | 3,936; 16 of 16; 0.033 |
| ammonia, surrogate Hessian only turned by 34 degrees; 1.0 | 15,189; 0 of 16; 19 | 7,821; 0 of 16; 20 | 24,848; 15 of 16; 0.064 | 956; 15 of 16; 0.127 | 3,441; 16 of 16; 0.029 |

**Table D. Water: every atom is moved at random** by the given root mean square displacement per
atom, and the surrogate is moved with it. Every run has its own displacement.

| Case; error of the start (kcal/mol) | classic | classic-lean | dense v1 | dense v2 (tolerance 0.2) | dense v2 (tolerance 0.05) |
|---|---|---|---|---|---|
| water, every atom moved, rms 0.02 A; 1.0 | 6,000; 16 of 16; 0.014 | 1,779; 16 of 16; 0.020 | 6,512; 16 of 16; 0.032 | 636; 16 of 16; 0.032 | 2,052; 16 of 16; 0.013 |
| water, every atom moved, rms 0.05 A; 5.8 | 6,000; 16 of 16; 0.013 | 1,779; 16 of 16; 0.017 | 9,913; 15 of 16; 0.078 | 973; 16 of 16; 0.049 | 2,745; 16 of 16; 0.011 |
| water, every atom moved, rms 0.10 A; 23.9 | 6,000; 16 of 16; 0.014 | 1,784; 16 of 16; 0.024 | 16,979; 5 of 16; 13 | 1,279; 16 of 16; 0.042 | 3,186; 16 of 16; 0.011 |
| water, every atom moved, rms 0.15 A; 57.4 | 6,000; 15 of 16; 0.425 | 1,798; 15 of 16; 1 left the molecule | 21,984; 2 of 16; 39 | 1,506; 16 of 16; 0.037 | 3,681; 16 of 16; 0.010 |
| water, every atom moved, rms 0.20 A; 111.0 | 6,000; 11 of 16; 1 left the molecule | 1,760; 12 of 16; 1 left the molecule | 25,432; 1 of 16; 85 | 1,834; 16 of 16; 0.049 | 3,567; 16 of 16; 0.012 |
| water, every atom moved, rms 0.30 A; 311.6 | 6,003; 8 of 16; 2 left the molecule | 1,729; 7 of 16; 2 left the molecule | 29,297; 1 of 16; 296 | 2,609; 14 of 16; 0.117 | 4,845; 14 of 16; 0.075 |

**Table E. Ammonia: every atom is moved at random.**

| Case; error of the start (kcal/mol) | classic | classic-lean | dense v1 | dense v2 (tolerance 0.2) | dense v2 (tolerance 0.05) |
|---|---|---|---|---|---|
| ammonia, every atom moved, rms 0.02 A; 1.2 | 14,102; 16 of 16; 0.048 | 6,460; 16 of 16; 0.070 | 21,945; 16 of 16; 0.076 | 1,658; 16 of 16; 0.065 | 9,067; 16 of 16; 0.014 |
| ammonia, every atom moved, rms 0.05 A; 6.5 | 14,109; 16 of 16; 0.070 | 6,435; 16 of 16; 0.067 | 27,129; 11 of 16; 0.204 | 2,394; 16 of 16; 0.061 | 11,168; 16 of 16; 0.017 |
| ammonia, every atom moved, rms 0.10 A; 27.8 | 14,221; 14 of 16; 1 left the molecule | 6,697; 13 of 16; 1 left the molecule | 32,019; 4 of 16; 6.986 | 3,409; 16 of 16; 0.071 | 13,891; 16 of 16; 0.014 |
| ammonia, every atom moved, rms 0.15 A; 68.8 | 14,315; 10 of 16; 4 left the molecule | 6,848; 10 of 16; 5 left the molecule | 41,034; 0 of 16; 45 | 4,239; 14 of 16; 0.385 | 14,259; 15 of 16; 0.352 |
| ammonia, every atom moved, rms 0.20 A; 135.6 | 14,273; 8 of 16; 7 left the molecule | 6,887; 7 of 16; 7 left the molecule | 46,026; 1 of 16; 112 | 5,328; 15 of 16; 0.061 | 15,018; 15 of 16; 0.038 |
| ammonia, every atom moved, rms 0.30 A; 359.4 | 13,544; 1 of 16; 9 left the molecule | 6,194; 1 of 16; 9 left the molecule | 50,716; 0 of 16; 337 | 5,871; 14 of 16; 1.488 | 15,162; 14 of 16; 1.387 |

What the tables say:

1. **A geometry that is wrong in a random direction (tables D and E).** The classic search ends
   within 0.2 kcal/mol in every run up to a displacement of 0.10 angstrom per atom for water and
   0.05 angstrom for ammonia. Beyond that it fails in a growing part of the runs: for water in 1
   of 16 runs at 0.15 angstrom, 4 or 5 at 0.20 and 8 or 9 at 0.30; for ammonia in 2 or 3 of 16
   runs at 0.10 angstrom, 6 at 0.15, 8 or 9 at 0.20 and 15 at 0.30. Up to 9 of 16 of these runs
   left the molecule. Protocol v2 ends within 0.2 kcal/mol in all 16 runs up to 0.20 angstrom for
   water and up to 0.10 angstrom for ammonia, and in 14 or 15 of 16 runs for the larger
   displacements. In these four cases 7 runs of v2 ended outside the tolerance, and 6 runs reached
   the limit of 10 iterations and were reported as `NOT_CONVERGED`. With 20 iterations, the default
   of a real run, all 48 runs of the three largest cases (water 0.30, ammonia 0.20 and 0.30
   angstrom) converged within the tolerance; that check was run on a laptop and is not part of the
   batch jobs. Protocol v1 fails in a part of the runs from 0.05 angstrom on.
2. **A surrogate with the wrong shape (table C; the last two rows of table A).** With a turned
   surrogate surface the classic search ended outside the tolerance in all 16 runs, for both
   molecules; for ammonia its mean error (1.3 to 1.4 kcal/mol) is larger than the error of the start
   (1.0). With only the Hessian turned it ended within the tolerance in 14 of 16 runs for water and
   in none for ammonia. Protocol v2 ended within the tolerance in 16, 16, 14 and 15 of 16 runs. With
   a wrong cubic term of the surrogate the classic search failed in 2 to 4 of 16 runs for ammonia;
   protocol v2 in none.
3. **Cost at the stationary point of the surrogate (table A).** For ammonia protocol v2 needs 493
   search blocks, the lean classic search 6,474 (13 times more) and the default classic search
   14,082 (29 times more), and the error of v2 is smaller (0.030 against 0.066 and 0.048 kcal/mol).
   For the ammonia saddle the numbers are 522 against 14,016 and 20,424. For water they are 555
   against 1,779 and 6,000, and here v2 has the larger error (0.035 against 0.020 and 0.015). At
   the tolerance 0.05 protocol v2 reaches the error of the classic search (0.015) with 1,400 blocks.
4. **Where the classic search is as good or better (table B).** For water with a start that is
   wrong along one direction, the lean classic search needs 1,776 to 1,851 blocks and ends with a
   mean error of 0.010 to 0.025 kcal/mol in every row. Protocol v2 needs 559 to 2,046 blocks at the
   tolerance 0.2 (error 0.022 to 0.051) and 1,092 to 4,066 blocks at the tolerance 0.05 (error
   0.006 to 0.015). So for an equal error the lean classic search needs fewer blocks in most of
   these rows. Its windows are several times wider than the spans of v2 (in the real water runs of
   this package 0.37 to 0.70 angstrom against 0.06 to 0.16), so a slope needs fewer blocks, and on
   these models a cubic fit over the wide window is correct, because the surrogate has the shape of
   the true surface. Statements 1 and 2 show what happens when that is not so.
5. **Protocol v1** fails in most runs from a start that is 0.10 angstrom away along a stretch
   direction with longer bonds, or at random (5 to 7 of 16 runs within the tolerance for water),
   and in all runs at 0.20 angstrom along a stretch direction. Its points stay within one span of
   the centre, and it has at most 5 iterations.
6. **`auto`.** The stability audit marked the first classic iteration `UNRESOLVED` in every run
   of every case, also where the classic search ended within the tolerance. The audit is strict. So
   `auto` always switched to protocol v2 and paid for one classic iteration (1,902 to 9,493 search
   blocks). For a new molecule `dense-v2` is the cheaper choice.

**Is the `CONVERGED` statement true?** `examples/model_study/calibrate_v2.py` runs 96 random seeds
for each of 22 cases, at the default tolerance and at 0.05 kcal/mol.

| Case | tolerance 0.2: converged; within the tolerance; 95th percentile of the error; search blocks | tolerance 0.05: the same |
|---|---|---|
| water, start at the surrogate minimum | 96 of 96; 96 (100 %); 0.070; 403 | 96 of 96; 94 (98 %); 0.040; 1,361 |
| ammonia, start at the surrogate minimum | 96 of 96; 96 (100 %); 0.073; 516 | 96 of 96; 93 (97 %); 0.041; 2,468 |
| ammonia inversion saddle | 96 of 96; 96 (100 %); 0.045; 535 | 96 of 96; 93 (97 %); 0.037; 992 |
| water, 0.05 A along the antisymmetric stretch | 96 of 96; 96 (100 %); 0.106; 835 | 96 of 96; 96 (100 %); 0.030; 2,328 |
| water, 0.10 A along the antisymmetric stretch | 96 of 96; 96 (100 %); 0.105; 1,108 | 96 of 96; 95 (99 %); 0.033; 3,043 |
| water, 0.20 A along the antisymmetric stretch | 96 of 96; 96 (100 %); 0.111; 1,629 | 96 of 96; 95 (99 %); 0.030; 3,648 |
| water, 0.30 A along the antisymmetric stretch | 95 of 96; 94 (99 %); 0.105; 2,127 | 94 of 96; 94 (100 %); 0.034; 4,228 |
| water, +0.10 A along the symmetric stretch | 96 of 96; 96 (100 %); 0.078; 890 | 96 of 96; 96 (100 %); 0.025; 2,073 |
| water, -0.10 A along the symmetric stretch | 96 of 96; 96 (100 %); 0.102; 942 | 96 of 96; 96 (100 %); 0.028; 2,336 |
| water, +0.20 A along the symmetric stretch | 96 of 96; 96 (100 %); 0.087; 1,235 | 96 of 96; 95 (99 %); 0.031; 2,808 |
| water, -0.20 A along the symmetric stretch | 96 of 96; 96 (100 %); 0.092; 1,452 | 96 of 96; 96 (100 %); 0.028; 2,833 |
| water, 0.30 A along the bend | 96 of 96; 96 (100 %); 0.098; 961 | 96 of 96; 95 (99 %); 0.029; 2,158 |
| water, surrogate surface turned by 34 deg | 96 of 96; 95 (99 %); 0.116; 1,047 | 94 of 96; 64 (68 %); 0.102; 3,534 |
| water, only the surrogate Hessian turned by 34 deg | 96 of 96; 88 (92 %); 0.230; 1,082 | 93 of 96; 76 (82 %); 0.086; 3,866 |
| ammonia, surrogate surface turned by 34 deg | 96 of 96; 94 (98 %); 0.139; 1,386 | 96 of 96; 91 (95 %); 0.049; 4,005 |
| ammonia, only the surrogate Hessian turned by 34 deg | 96 of 96; 90 (94 %); 0.202; 1,143 | 95 of 96; 73 (77 %); 0.094; 3,646 |
| water, every atom moved at random, rms 0.05 A | 96 of 96; 96 (100 %); 0.118; 951 | 96 of 96; 94 (98 %); 0.037; 2,692 |
| water, every atom moved at random, rms 0.10 A | 96 of 96; 96 (100 %); 0.098; 1,274 | 96 of 96; 96 (100 %); 0.029; 3,261 |
| water, every atom moved at random, rms 0.20 A | 95 of 96; 95 (100 %); 0.136; 1,872 | 95 of 96; 95 (100 %); 0.032; 3,795 |
| ammonia, every atom moved at random, rms 0.05 A | 96 of 96; 96 (100 %); 0.132; 2,233 | 96 of 96; 96 (100 %); 0.035; 10,571 |
| ammonia, every atom moved at random, rms 0.10 A | 96 of 96; 96 (100 %); 0.120; 3,394 | 96 of 96; 95 (99 %); 0.035; 14,404 |
| ammonia, every atom moved at random, rms 0.20 A | 88 of 96; 86 (98 %); 0.494; 5,295 | 84 of 96; 82 (98 %); 0.191; 17,008 |

At the tolerance 0.2 kcal/mol 2,102 of 2,112 runs ended as `CONVERGED`, and 2,082 of these were
within the tolerance (99.0 %). At the tolerance 0.05 kcal/mol 2,091 of 2,112 runs ended as
`CONVERGED`, and 2,000 of these were within the tolerance (95.6 %). The statement holds at the stated
level of 95 % in every case in which the surrogate has the right couplings (18 cases, lowest value
97 %). It does not hold in every case with a turned surrogate: 92 % and 94 % at the tolerance 0.2
(the two cases "Hessian only"), and 68 % to 82 % at the tolerance 0.05 (three of the four turned
cases). Section 6 has more on this limit.

The numbers of this section are from batch jobs with the frozen code of 2026-11-22 (Python 3.9.21,
numpy 1.26.4). Another computer can give other numbers for the same seeds. One case showed it:
"water, Hessian only" with 32 runs gave 28 of 32 within the tolerance on the cluster and 32 of 32 on
a laptop. The cause was not traced. One known source is the sign of a search direction, which
depends on the linear-algebra library; it decides which scan point gets which random numbers. The
96-run value for this case is 88 of 96.

## 6. Limits

- **The bound assumes that the model Hessian is right.** The scans measure the slope and the force
  constant along every direction, but not the couplings between two directions; those come from
  the surrogate. Where the surrogate Hessian is turned by 34 degrees against the true one, the
  predicted gain and the noise energy are too small. At the default tolerance 92 % and 94 % of the
  converged runs were within the tolerance (the two cases "Hessian only" of the calibration table),
  and at the tolerance 0.05 only 68 % to 82 % in three of the four turned cases. The search still
  removed most of the error: the mean error at the end was 0.05 to 0.09 kcal/mol from a start of
  1.0. Five repairs were tried on 2026-11-22 and none helped: couplings fitted to the measured slope
  changes, used in the step (option `coupling_fit`) or only in the stopping rule; a larger shape
  term where DMC contradicted the surrogate force constant (option `shape_distrust`); a stopping
  rule scaled by the ratio of model curvature to measured curvature along the last step; and longer
  spans. Three of them (the fit used only in the stopping rule, the scaled stopping rule, the longer
  spans) were tried in a copy of the code and are not in the package. A repair probably needs a
  direct measurement of the couplings, for example scan points off the axes; that was not tried.
- **The shape part of the slope error** assumes that the cubic part of the DMC surface is within
  30 % of that of the surrogate. The assumption was checked on one real data set only. Other
  molecules and surrogates were not measured.
- **Cost.** Protocol v2 scans over short spans (an energy rise of 5 kcal/mol), so that the result
  does not depend on the shape of the surrogate far from the centre. A slope then needs more blocks
  than in a wide window. For a small molecule whose surrogate has the right shape, the classic
  search with short points can be cheaper for an equal error (statement 4 of section 5).
- **The numbers of section 5 are for model surfaces with Gaussian noise.** They are not results of
  real DMC runs. The results of real DMC runs are in [validation.md](validation.md).
