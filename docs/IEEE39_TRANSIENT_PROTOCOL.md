# IEEE-39 Transient Benchmark Protocol

## Source

- Dataset: Sokolovic, Zivko (2024), *Dataset for Transient Stability
  Assessment of IEEE 39-Bus System*, Mendeley Data, Version 1.
- DOI: `10.17632/p992nhb8ss.1`
- License: CC BY 4.0.
- Inspected file: `data/ieee39_transient/raw/tsa_data.pkl`
- SHA256: `f190efdd64ea7652a1f8c84c3823b3f3477c80a83f32861dd45ffdf1e968a711`
- File size: 328,613,651 bytes.

## Verified Pickle Schema

The top-level object is a two-element tuple:

1. A list of 12,852 `pandas.DataFrame` trajectories.
2. A `pandas.Series` of 12,852 `uint8` stability labels.

Every trajectory has shape `[60, 50]`, uses `float64`, and has an index named
`t in s`. All trajectories have equal length and exactly 0.01 s spacing. Six
real timestamp grids occur, beginning at 0.11, 0.16, 0.19, 0.21, 0.26, or
0.31 s and ending 0.59 s later. The processed cache preserves the timestamp
grid for every scenario.

Columns are measurement-major. Each block contains G01 through G10:

1. `P in MW`: active power.
2. `ut in p.u.`: terminal voltage.
3. `ie in p.u.`: excitation current.
4. `xspeed in p.u.`: rotor speed.
5. `firel in deg`: rotor angle relative to G02.

This ordering can be mapped unambiguously to
`[scenario, time, generator, feature] = [12852, 60, 10, 5]` by grouping the
five documented measurement blocks and transposing the generator/feature
axes. The numeric payload is 308,448,000 bytes before processed-cache
overhead.

The data contain no NaN or infinite values. Stability labels are:

- unstable (`0`): 5,392 scenarios;
- stable (`1`): 7,460 scenarios.

SHA256 hashes over each timestamp index and trajectory matrix found 12,852
unique hashes and no exact duplicate trajectories.

## Missing Metadata

The pickle contains no explicit scenario identifiers beyond row position. It
also contains no per-scenario fields for:

- generation/load operating condition;
- faulted line or fault location;
- fault clearing time;
- original CSV filename or simulation identifier.

Consequently, deterministic stability-stratified splitting is possible, but
the requested operating-condition/fault/clearing-time stratification and
group-aware leakage prevention cannot be verified from this file.

## Mapping-Free Generator Graph

The dataset identifies generators only as G01 through G10 and states that G02
is the reference machine. It does not provide generator bus numbers.

`pandapower.networks.case39()` contains dynamic sources at one-based buses
30 through 39. Its external grid is at bus 31 and its other generator rows are
at buses 30 and 32 through 39, but all source names are `None`. The reference
at bus 31 is consistent with G02 being the reference machine, but this does
not independently verify every G01-G10 assignment.

No bus assignment is assumed. G01 through G10 are graph nodes in dataset
column order. The benchmark uses all 90 directed non-self generator pairs in
receiver-major order. This is a mapping-free candidate interaction graph, not
a recovered IEEE-39 electrical topology. A Kron-reduced physical graph remains
future work if an authoritative generator-to-bus mapping becomes available.

LG-ODE uses fixed candidate-edge messages. AT-ODE uses the identical node and
edge order and adds time-dependent attention-transport weights. LatentODE is
graph-independent and Persistence is non-learned.

## Processed Cache And Splits

The processed tensor has shape `[12852, 60, 10, 5]` and is stored at
`data/ieee39_transient/processed/ieee39_transient_v1.npz`.

- Processed SHA256: `36ca9b7e0cb0bbc7fe7126b67db086f4e89ad41886760158d2a673276e9d60c1`.
- Graph SHA256: `f8f940245a46cf3242c8af801702b1e1c013e4cbf822d57fcb7f77bfc6688e87`.
- Split seed: 2026.
- Training scenarios: 8,996.
- Validation scenarios: 1,927.
- Test scenarios: 1,929.

Splits are scenario-level, disjoint, exhaustive, and stratified only by the
available stability label. Normalization is fitted only on selected training
scenarios. Missing fault/operating metadata prevents group-aware splitting, so
unobserved scenario-family leakage remains possible.

## Tasks And Observation Masks

Interpolation uses all 60 samples as the target and evaluates deliberately
withheld entries only. Extrapolation uses samples 0-29 as context and predicts
samples 30-59; future values and timestamps never enter the encoder or
Persistence baseline.

Masks are independently sampled for every scenario and generator from scenario
ID, task, fraction, and mask seed. Counts vary by one around the requested
fraction, with at least two observations and coverage in both halves of the
legal observation domain. Mask caches are bound to the processed-cache hash.
The underlying trajectories are regularly sampled; only retained observation
events are asynchronous and irregular.

## Runtime Modes

The sole shell entrypoint is `scripts/run_publication_both_tasks.sh`.

- `RUN_MODE=smoke`: 16/8/8 scenarios, fractions 0.2 and 0.8, seed 1, one
  epoch, at most two batches.
- `RUN_MODE=development`: 512/128/128 scenarios, fractions 0.2 and 0.8,
  seed 1, 20 epochs, validation every two epochs.
- publication (default): all scenarios, fractions 0.2/0.4/0.6/0.8, seeds
  1-5, 150 epochs, validation every two epochs.

All learned models use posterior means for validation/test, validation-selected
checkpoints, and one final test evaluation. Result files record source/cache,
split, mask, graph, model, optimizer, timing, and metric provenance.

## Limitations

- The 90-edge graph is not a verified electrical topology.
- Fault location, operating condition, and clearing time are unavailable as
  explicit fields, so errors cannot be grouped by those attributes.
- The six timestamp offsets likely reflect simulation timing conditions, but
  they are preserved as timestamps and are not relabeled as clearing times.
- Recovery time is reported only when both true and predicted rotor-speed
  trajectories satisfy the declared 0.005 p.u. recovery tolerance.