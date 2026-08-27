# LG-ODE Power-Grid Protocol Audit

## Scope

The publication protocol is implemented by `run_powergrid_lgode.py` and
`scripts/run_publication_both_tasks.sh`. The legacy `run_models.py` evaluates
the test split during training and is not valid for protocol-v5 experiments.
No old protocol result is an input to this audit or to the v5 runner.

The upstream reference is ZijieH/LG-ODE (NeurIPS 2020). Its NRI decoder uses
all directed off-diagonal pairs and binary edge types. This repository's
primary power-grid adaptation intentionally uses sparse directed physical
edges. `--graph-mode all_pairs_nri` provides the upstream graph-semantics
ablation without changing the primary protocol.

## Verified Defects And Fixes

1. **Task masks were conflated.** `lib/simbench_lgode_data.py:228-321` now
   carries explicit encoder observation, training loss, interpolation
   withheld, and extrapolation future masks. Interpolation loss and primary
   metrics use withheld entries only (`lib/simbench_lgode_data.py:1250-1258`);
   extrapolation uses the complete future interval.
2. **Graph construction was implicit and previously inconsistent.**
   `run_powergrid_lgode.py:477-590` now makes `physical_sparse` and
   `all_pairs_nri` explicit. Physical edges are checked for bounds, duplicates,
   self-loops, and missing reverse directions in
   `lib/simbench_lgode_data.py:540-587`.
3. **Graph-model evaluation sampled posterior initial states.**
   `lib/latent_ode.py:583-607` and `lib/latent_ode.py:814-878` expose explicit
   `sample_z0`; training samples and validation/test use posterior means.
4. **ODE dropout reused encoder dropout.** The factory keeps encoder dropout
   but builds graph ODE dynamics with `ode_dropout=0.0`
   (`lib/powergrid_model_factory.py:751-765`). Independent dynamics contain no
   dropout.
5. **AT-ODE depended on runner monkey-patching.** Transport is constructed
   directly by the factory (`lib/powergrid_model_factory.py:813-842`) and
   consumed through the solver's stable cache interface.
6. **AT-ODE normalization cancelled temporal decay.** Incoming scaling is now
   fixed from the first transport time (`lib/attention_transport.py:1137-1275`)
   and NRI does not renormalize transport weights during RHS evaluations.
   Temporal-change diagnostics are recorded at
   `lib/attention_transport.py:1535-1590`.
7. **LG-ODE/AT-ODE initialization was not enforced on the real path.** Exact
   shared state names, shapes, dtypes, and values are checked by
   `lib/powergrid_model_factory.py:1339-1569`; the runner also records the
   shared-state hash.
8. **Interpolation summaries used full-window metrics.** The summarizer now
   requires withheld MSE/MAE for interpolation and full future MSE/MAE for
   extrapolation. A stray token that crashed the script was removed.
9. **Optimization telemetry omitted gradient norm and reconstruction term.**
   `run_powergrid_lgode.py:1195-1305` records reconstruction likelihood, KL,
   learning rate, and pre-clip gradient norm per epoch.

## Data And Evaluation Protocol

- Splits are chronological and every window is contained in one split.
- Normalization is fitted only on `[0, train_end)`
  (`lib/simbench_lgode_data.py:624-650`).
- Observation masks are deterministic from `(mask_seed, trajectory_id)` and
  independent of loader order and model.
- Per-task/seed window and mask hashes are enforced before training and saved
  with dataset SHA256, split bounds, window IDs, and edge hash.
- Validation selects checkpoints using the task-primary MSE. The selected
  checkpoint is restored and test is evaluated once.
- Interpolation is a **noncausal smoothing** task: real observations may occur
  on either side of a withheld target within the same 24-step window.
- Extrapolation is causal: only 12-step context observations enter the encoder;
  targets are the following 12 steps, and latest observation time is asserted
  to precede forecast start (`lib/latent_ode.py:930-949`).

## Graph Fidelity

The encoder event nodes contain only selected real observations. Same-bus and
physical-neighbor temporal edges are directed from earlier to later events;
`edge_same` distinguishes the two. The graph encoder jointly pools event
representations per bus. The generative NRI relation matrices use row zero of
`edge_index` as sender and row one as receiver, and aggregate into receivers.

Primary `physical_sparse` collapses electrical line types and parameters to
binary/unit adjacency. This is an intentional power-grid adaptation, not an
exact reproduction of upstream all-pairs NRI. The optional
`all_pairs_nri` mode reproduces receiver-major off-diagonal candidates with
binary physical/nonphysical labels.

## Fairness And Sensitivities

Primary LG-ODE and AT-ODE share encoder, decoder, graph ODE, prior, likelihood,
solver, tolerances, optimizer budget, dropout, dimensions, graph ordering, and
exact initial shared parameters. AT-ODE adds only transport parameters.

The independent `latentode` identifier denotes `IndependentGRULatentODE`; it is
not claimed to be the exact baseline from the upstream paper.

Optional ablations are isolated in `scripts/run_powergrid_ablations.sh`:

- `capacity_matched`: LG-ODE recognition dimension 40 and ODE hidden dimension
  48, giving 52,196 trainable parameters versus 52,788 for primary LatentODE
  (1.12% difference).
- `official_like`: augmentation 64, AdamW, weight decay `1e-3`, and three
  training samples. ODE dropout remains zero because stochastic RHS evaluations
  are numerically invalid for RK4/ODE solvers.
- `all_pairs_nri`: official graph candidate semantics while retaining the
  power-grid data protocol.

These are prespecified sensitivity analyses and must not be selected using test
performance.

## Remaining Limitations

- Interpolation is intentionally noncausal smoothing. Its posterior and AT-ODE
  transport context may use observations later than a withheld target; results
  must not be described as causal forecasting.
- Electrical edge attributes are not modeled; graph edges are binary/unit.
- Physical-unit per-feature errors are available through inverse-normalization.
  AC residual and violation metrics remain unavailable unless the dataset and
  evaluator provide the required electrical metadata; missing values are never
  replaced by zeros.
- Five paired seeds provide limited power. Paired t-tests therefore include
  normality assumptions, Wilcoxon robustness checks, paired bootstrap intervals,
  and Holm correction. Statistical significance should be interpreted
  conservatively.
- Wall-clock time is reported but not equalized. Optimization budget is
  equalized, as required for model fairness.