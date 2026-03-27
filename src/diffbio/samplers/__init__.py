"""DiffBio samplers module.

Provides specialized sampler implementations extending datarax's SamplerModule
for bioinformatics applications.

Samplers:
    PerturbationBatchSampler: Groups cells by (cell_type, perturbation) for
        efficient batch construction in perturbation experiments.
"""
