# Changelog

All notable changes to DiffBio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.9] - 2026-09-18

### Changed

- Requires `substrax>=0.1.11`; the lock moves it from 0.1.10 and nothing else. 0.1.11 caps jax
  below 0.11.2, whose renamed `jax.experimental.hijax.HiPrimitive` flax 0.12.9 imports at
  module load; a resolver given `substrax>=0.1.10` keeps jax 0.11.2 and picks 0.1.10 instead,
  so a fresh install of DiffBio failed on `import diffbio` until the floor moved.
- CI: the build-verification, documentation and security workflows cancel the run a newer
  push supersedes, as the test workflow already did; a contract test holds every
  push-triggered workflow to it.
- Requires `datarax>=0.1.13`; the lock moves it from 0.1.11. DiffBio builds no datarax
  pipeline of its own; its `MemorySource` subclasses inherit datarax's per-epoch order
  cache, and a pipeline a caller builds over a DiffBio source now serves batches with a
  `valid_mask` leaf and settles the epoch's final batch through `drop_last`.
- Requires `substrax>=0.1.10`; the lock moves it from 0.1.9. DiffBio uses substrax's
  optimizer, RNG, runtime and device modules, which 0.1.10 leaves as they were; it does not
  checkpoint through substrax's store, whose format 0.1.10 rewrote.

## [0.1.8] - 2026-09-17

### Changed

- Requires `avitai-artifex>=0.1.10`, which removed its `reduce_loss` helper: the multi-omics VAE
  takes the batch mean of its per-sample reconstruction error directly, and the two batch-correction
  operators' reconstruction losses are calibrax's `mse`.

## [0.1.7] - 2026-09-17

### Changed

- The classification loss is calibrax's `softmax_cross_entropy(logits, labels)`; the joint and
  minibatch training pipelines, the single-cell compute-cost benchmark and the docs call it
  directly, and `diffbio.utils.training.cross_entropy_loss` is gone. The class count is the
  logits' last axis, and the calibrax loss takes `mask`, `weights` and `reduction`.
- Benchmark classification metrics come from calibrax: `accuracy` and per-class macro
  `f1_score` with `num_classes` set to one past the largest label seen, which keeps the
  previous macro-F1 semantics (a class absent from both label sets contributes an F1 of 0).
- The Lennard-Jones and single-cell compute-cost benchmarks time their steps through
  `calibrax.profiling.time_calls`, one synchronised sample per step: the compute-cost benchmark
  reports the median step time and the molecular-dynamics benchmark divides the timed step
  count by the sum of its samples.
- `masked_value_loss` and the chromatin-guidance loss reduce through calibrax's masked `mse` and
  `mae` (an all-zero mask gives `0`); `masked_value_loss` no longer takes `epsilon`.
- Requires calibrax 0.1.8.

### Fixed

- Five documentation examples had a `from substrax.optim import OptimizerConfig` line inserted
  inside the `from diffbio.utils.training import (...)` block; the blocks are valid Python again.

## [0.1.6] - 2026-09-17

### Changed

- No library code seeds its own randomness. The fifty-odd constructors and factories that
  built an `nnx.Rngs(0)` (or a `params=0, sample=1, dropout=2` set) when handed no `rngs`
  now require it, as do the fingerprint, similarity, sequence-encoder and RNA-fold factories;
  every parameter key drawn by hand comes through `substrax.rng.key_from`, which reads the
  named stream, then NNX's `default`, and raises `MissingRngStreamError` naming the operator
  when the `nnx.Rngs` holds neither. The transformer and contextual-epigenomics operators no
  longer fabricate a `dropout` stream from a fixed key when the caller's `Rngs` lacks one.
  `tests/test_no_literal_seeds.py` reads every module and fails on a seed literal, so the
  class cannot return.
- A random or stratified split, a k-fold split and a shuffled `IndexedViewSource` draw from
  `config.seed` when it is set and otherwise from the module's `split` or `shuffle` stream; a
  splitter or view with neither raises instead of permuting from seed 0.
- Randomness that belongs to the record follows the record's key. The Solo doublet scorer
  drew its pair-sampling and reparameterisation noise from the operator's stream and the cell
  annotator ignored the key it was handed; both now split the record's key and leave the
  stream untouched, and a stochastic operator's `apply` handed no key raises. The Langevin
  integrator is declared stochastic (stream `langevin`) and draws the thermostat's random
  forces from the record's key instead of a fixed seed.
- `VAENormalizer.batch_elbo_loss(counts, library_sizes, key=None)` computes the mean ELBO
  over a batch of cells with one epsilon key per cell, split outside `jax.vmap` from `key`
  or one draw of the `sample` stream, so a batched ELBO composes with `nnx.jit` and
  `nnx.grad`; `compute_elbo_loss` keeps its single-cell contract and takes an optional key.
- `combine_scalar_losses(losses, *, balancer)` takes the `GradNormBalancer` to combine
  through, or `None` to sum, and `LossBalancingMixin.compute_balanced_loss` builds the
  balancer from the operator's own `rngs` when `config.use_gradnorm` is set, instead of a
  fresh balancer from seed 0 on every call.
- `DifferentiableSecondaryStructure` builds its artifex bond-length and bond-angle
  constraint extensions once, in `__init__` from its own `rngs`, instead of rebuilding them
  from seed 0 inside every `apply`.
- `TrainingConfig` and `JointTrainingConfig` carry their optimizer as `optimizer`, a
  `substrax.optim.OptimizerConfig` (`default_training_optimizer()`: Adam at 1e-3 with a unit
  global-norm clip; the joint default is Adam at 1e-2), and `Trainer` and `fit_jointly` build
  through `substrax.optim.create_optimizer`; the `learning_rate` and `grad_clip_norm` fields
  and `create_optax_optimizer` are gone, and substrax refuses invalid values.
- `MiniBatchConfig` carries its optimizer as `optimizer`, a `substrax.optim.OptimizerConfig`
  (default: AdamW at 1e-2 with a unit global-norm clip), and `train_minibatch` builds it with
  `substrax.optim.create_optimizer`; the `learning_rate`, `weight_decay` and `grad_clip_norm`
  fields are gone, and substrax refuses their invalid values. The benchmark helper
  `create_benchmark_optimizer` takes the model first and builds through `substrax.optim`
  instead of opifex's removed factory.
- The test session merges its XLA flag into `XLA_FLAGS` by flag name through
  `substrax.runtime.merge_xla_flags`: a flag the caller exported, such as an emulated
  device count, survives, and a different value for the same flag raises instead of being
  replaced. It used to overwrite the variable.
- Requires `substrax>=0.1.9`, `datarax>=0.1.11`, `avitai-artifex>=0.1.9`, `opifex>=0.2.7`
  and `calibrax>=0.1.6`, the latest release of each; the lock holds them.

### Removed

- `diffbio.utils.nn_utils.ensure_rngs` and `get_rng_key`; operators require `rngs`, and keys
  come from `substrax.rng.key_from`.
- `diffbio.utils.training.create_optax_optimizer`; the trainer builds through
  `substrax.optim.create_optimizer` from `TrainingConfig.optimizer`.

## [0.1.5] - 2026-09-16

### Changed

- Requires `datarax>=0.1.10`; the lock moves datarax from 0.1.7 to 0.1.10 and, through it,
  substrax from 0.1.5 to 0.1.7. datarax hands `apply` the record's PRNG key as its fourth
  argument and no longer calls `generate_random_params`, so every operator names that
  argument `key`, the four `generate_random_params` methods are gone (the simulator splits
  its six keys inside `apply`; the doublet scorers and masked-gene operators use the key
  directly), and the operators that drew from a stored `Rngs` inside `apply` draw from the
  key instead: `EncoderDecoderOperator.reparameterize` takes the key (the VAE normalizer,
  ambient removal, multi-omics VAE and metagenomic binner pass the record's; `compute_elbo_loss`
  keeps the operator's `sample` stream), `ReadDownsampler` rounds from the key, and
  `StochasticGateSelector` draws its gate noise from it. A stochastic operator handed no key
  raises instead of drawing from a fixed seed; masking genes with `mask_ratio > 0` needs a
  key. `wrap_probabilistic` takes `rngs`, which the wrapper it builds requires.
- The `_unique_id` workaround in the drug-discovery operators is gone with the attribute it
  wrapped, as are the `cacheable` config fields nothing read and the cache argument of
  `eager_reset` on the AnnData source.

## [0.1.4] - 2026-09-16

### Added

- Public `diffbio.core.soft_ops.temperature_softmax` for small masked mixtures,
  with jointly evaluated temperature/score derivative coefficients, axis support,
  empty-mask semantics, and JAX/Flax NNX transformation regressions.
- Explicit numerical range and quadratic derivative-cost contracts. Existing
  smooth sorting and quantile kernels retain their current implementations.

### Fixed

- CI runs every test module. The unit-test shards named test directories one by one, so
  `tests/benchmarks`, `tests/reductions`, `tests/scripts` and the root-level test modules
  ran in no job. The general shard now runs `tests` minus what the other shards and the
  integration job own, `tests/benchmarks` has its own shard, and
  `tests/test_ci_shards.py` fails when a test module is left out. The `test` extra gains
  `pyyaml`, which that check reads.
- `./setup.sh` on Apple Silicon syncs a `metal` extra that was never declared, so it
  failed at `uv sync`; the extra is now declared (`jax-metal`, arm64 macOS only) and
  included in `all`. Its help named a `gpu` extra renamed `cuda12` in 0.1.1. A test
  checks that every extra `setup.sh` names is declared.
- CI no longer reports success for tests it did not run. The end-to-end and performance
  jobs selected tests by markers no test carries and passed on zero tests; they are
  removed, the unit shards deselect only what the runners cannot satisfy, and coverage
  combination fails without data. `tests/test_ci_shards.py` also fails when a marker is
  deselected by every job. The benchmarks shard installs the `benchmark` extra that the
  scib-metrics bridge tests import; the Lennard-Jones benchmark tests, whose 4,096-particle
  run takes over 300 s on a CPU runner, are marked `slow`; the positioning test names
  Substrax with the other siblings.
- The combined coverage floor is checked on pull requests too, not only on pushes to main,
  and coverage is no longer uploaded to Codecov: the upload was never read back, and
  coverage.py in the Test Coverage job is the gate. `tests/test_ci_coverage.py` checks both.

## [0.1.3] - 2026-09-09

### Changed

- `eager_iter`, `eager_get_batch` and `eager_reset` are imported from `datarax.sources`,
  their public home in datarax 0.1.7; the private `datarax.sources._eager_source_ops`
  path this package used is gone there, so the floor is `datarax>=0.1.7`.

## [0.1.2] - 2026-09-09

### Changed

- The sibling floors are datarax 0.1.6, avitai-artifex 0.1.5, opifex 0.2.2 and
  calibrax 0.1.5, and `substrax` joins the runtime dependencies and the ecosystem
  packages the runtime check verifies. The declared jax floor is 0.11.1 and the flax
  floor 0.12.9, what every lock already resolved. Python 3.13 is supported.
- `ProteinExtensionConfig` is imported from its home,
  `artifex.generative_models.core.configuration`.
- `scripts/verify_gpu_setup.py` reports the device identity substrax detects
  (platform, device kind, count, kinds) and `--require-gpu` reads the kind.
- `diffbio.__version__` is read from the installed distribution's metadata instead
  of a hand-maintained string, which had stayed at 0.1.0 through the 0.1.1 release.
- The four advanced example pages that built `nnx.Optimizer` without `wrt=` and
  called `update(grads)` now match flax 0.12: `wrt=nnx.Param` and
  `update(model, grads)`.
- Publishing uses PyPI trusted publishing (OIDC) and `twine check --strict`; the
  README installs from PyPI, with `setup.sh` for a source checkout.
- CI checks that `uv.lock` matches `pyproject.toml`; the ruff hooks no longer
  receive `uv.lock`, which `identify` classifies as TOML.

### Removed

- The direct `orbax-checkpoint` dependency (nothing in this package imports it; it
  stays a transitive dependency of the siblings).
- `scripts/generate_benchmark_plots.py` and the sixteen images it rendered under
  `docs/assets/images/benchmarks/`, which no page referenced, and the unused
  `annotate_heatmap` helper.

## [0.1.1] - 2026-08-30

### Changed

- **Requires Python 3.12 or later.** jax 0.11.0 dropped 3.11, and this release
  takes that jax line.
- **The `gpu` extra is renamed `cuda12`.** JAX names its own extras for the CUDA
  major version and publishes no `gpu` extra, and this package also ships
  `metal`, which is a GPU.
- Resolves to jax 0.11.1, jaxlib 0.11.1, flax 0.12.9, optax 0.2.8 and grain
  0.2.18, and raises the sibling floors to opifex 0.2.1, avitai-artifex 0.1.4,
  calibrax 0.1.2 and datarax 0.1.5.
- `flax>=0.12.1` and `optax>=0.2.8` are now required; below those, flax lacks
  `nnx.Variable.set_value` and optax sets a jax config option removed in 0.10,
  which takes out collection for the whole suite.
- `dm-haiku>=0.0.17` is constrained. It arrives through `jax-md`, and up to
  0.0.16 it binds `jax.core.DropVar` at import, which jax removed in 0.11.0.
- `datasets` is declared in the `benchmark` extra, where
  `benchmarks/crossmodality` uses it. It had been arriving transitively through
  the sibling packages, which now keep their heavy dependencies behind extras.

### Fixed

- `top_k_mask` stays inside the `[0, 1]` bound it documents. The isotonic solver
  could finish one float32 ULP outside it; the forward value is saturated onto
  the bound while the backward pass keeps the projection's own gradient, so the
  gradients are unchanged.
- Benchmark training constructs its optimizer through the shared Opifex boundary
  rather than calling `optax.adamw` directly, preserving the optimizer exactly.

## [0.1.0] - 2026-05-02

Initial public release of DiffBio: end-to-end differentiable bioinformatics
pipelines built on JAX, Flax NNX, and the Datarax / Artifex / Opifex /
Calibrax ecosystem.

### Added

- 40+ differentiable operators across alignment, variant calling, single-cell
  analysis, drug discovery, epigenomics, multi-omics, RNA structure, protein
  structure, molecular dynamics, foundation models, and preprocessing.
- Six end-to-end pipelines: `VariantCallingPipeline`,
  `EnhancedVariantCallingPipeline`, `SingleCellPipeline`,
  `DifferentialExpressionPipeline`, `PerturbationPipeline`, and
  `PreprocessingPipeline`.
- Soft-operations primitive layer (`diffbio.core.soft_ops`) with
  straight-through and gradient-replacement variants for use inside
  differentiable bioinformatics workflows.
- Dataset sources for FASTA, BAM, AnnData, MoleculeNet, and indexed views.
- Dataset splitters for random, stratified, scaffold, Tanimoto cluster, and
  sequence-identity splits.
- Loss functions for alignment, biological regularization, single-cell
  analysis, statistical models, and metric learning.
- Training utilities (`Trainer`, `TrainingConfig`, optimizer factories,
  synthetic data generation, gradient clipping).
- Documentation site: getting-started guides, user-guide, API reference,
  examples (basic / intermediate / advanced), and contributor guides.
- Benchmark suite under `benchmarks/` with tier-based runner
  (`run_all.py --tier ci|nightly|full`) and SOTA baseline comparisons across
  single-cell, alignment, RNA structure, protein, molecular dynamics, and
  statistical domains.
- CI/CD: sharded unit tests with `pytest-xdist`, integration / e2e /
  performance jobs, coverage aggregation, security scanning, build
  verification, and documentation deployment workflows.
