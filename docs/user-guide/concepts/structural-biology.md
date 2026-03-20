# Structural Biology and Molecular Dynamics

Biological function depends on three-dimensional structure -- the fold of a
protein, the base-pairing pattern of an RNA, the dynamics of molecular
interactions. DiffBio provides 4 differentiable operators covering protein
secondary structure prediction, RNA folding, force field computation, and
molecular dynamics integration.

---

## Protein Secondary Structure

Proteins fold into three-dimensional shapes determined by their amino acid
sequence. Secondary structure -- the local folding pattern of the backbone --
is the first level of this structural hierarchy:

| Element | Symbol | Geometry | Stabilized By |
|---|---|---|---|
| **Alpha-helix** | H | Right-handed spiral, 3.6 residues/turn | i to i+4 hydrogen bonds |
| **Beta-strand** | E | Extended chain, forms sheets | Inter-strand hydrogen bonds |
| **Coil/loop** | C | Irregular, connecting segments | Variable |

### DSSP Algorithm

`DifferentiableSecondaryStructure` implements a differentiable version of the
DSSP algorithm (Kabsch & Sander, 1983), the standard method for assigning
secondary structure from atomic coordinates.

The algorithm proceeds in two stages:

1. **Hydrogen bond detection**: For each pair of residues, compute the
   electrostatic interaction energy using the Kabsch-Sander formula:
   $E = q_1 q_2 \left(\frac{1}{r_{ON}} + \frac{1}{r_{CH}} - \frac{1}{r_{OH}} - \frac{1}{r_{CN}}\right) \cdot f$
   where $r$ are interatomic distances and $f = 332$ kcal/mol is the
   conversion factor

2. **Pattern recognition**: Identify secondary structure from characteristic
   hydrogen bond patterns -- helices from i to i+4 bonds, strands from
   cross-chain bonds

The key innovation for differentiability is a **continuous hydrogen bond
matrix**: instead of a binary yes/no decision at the -0.5 kcal/mol threshold,
a sigmoid with learnable margin produces soft H-bond probabilities. Gradients
flow from structure assignments back through atomic coordinates.

---

## RNA Secondary Structure

RNA molecules fold into functional structures through Watson-Crick (A-U, G-C)
and wobble (G-U) base pairing. Unlike proteins, RNA secondary structure
(the pattern of base pairs) is often the most important determinant of function.

### McCaskill Partition Function

`DifferentiableRNAFold` implements the McCaskill algorithm, which computes
base pair probabilities by summing over all possible structures weighted by
their thermodynamic stability:

$$
Z = \sum_{\text{structures } S} \exp(-E(S) / RT)
$$

$$
P^{bp}(i, j) = \frac{1}{Z} \sum_{S \ni (i,j)} \exp(-E(S) / RT)
$$

The algorithm uses inside-outside dynamic programming to compute $Z$ and
$P^{bp}$ in $O(n^3)$ time without enumerating structures explicitly.

### Why Probabilities, Not a Single Structure

The minimum free energy (MFE) structure is a single point prediction that
discards uncertainty. The partition function approach returns a full probability
matrix -- $P^{bp}(i, j)$ gives the probability that positions $i$ and $j$
are paired across the thermodynamic ensemble. This is more informative and
naturally differentiable: soft probabilities produce smooth gradients, while
a single hard structure does not.

Temperature-controlled smoothing in the dynamic programming recurrence ensures
stable gradient flow through the $O(n^3)$ computation.

---

## Molecular Force Fields

Molecular dynamics simulations model the physical motion of atoms by computing
forces from an energy function. `ForceFieldOperator` provides differentiable
energy and force computation for particle systems.

### Potential Energy Functions

The total energy of a molecular system is a sum of interaction terms:

| Potential | Formula | Models |
|---|---|---|
| **Lennard-Jones** | $4\epsilon\left[(\sigma/r)^{12} - (\sigma/r)^6\right]$ | Van der Waals interactions |
| **Morse** | $\epsilon[1 - e^{-\alpha(r - r_0)}]^2$ | Covalent bonds (anharmonic) |
| **Soft sphere** | $\epsilon(\sigma/r)^n$ | Repulsive-only, for soft matter |

Each potential is parameterized by a length scale ($\sigma$), energy scale
($\epsilon$), and optional cutoff distance. All parameters are differentiable
-- gradients of the energy with respect to $\sigma$ and $\epsilon$ enable
learning force field parameters from data.

Forces are computed as the negative gradient of the energy with respect to
particle positions: $\mathbf{F}_i = -\nabla_{\mathbf{r}_i} E$. Because the
energy function is implemented in JAX, this gradient is exact and computed via
automatic differentiation.

---

## Molecular Dynamics Integration

`MDIntegratorOperator` evolves particle positions and velocities forward in
time using numerical integration of Newton's equations of motion.

Two integrator types are supported:

| Integrator | Ensemble | Method |
|---|---|---|
| **Velocity Verlet** | NVE (constant energy) | Symplectic, time-reversible |
| **Langevin** | NVT (constant temperature) | Stochastic thermostat with friction $\gamma$ and thermal energy $k_T$ |

The integrator takes positions, velocities, and a force field, then runs a
fixed number of steps $n_{\text{steps}}$ with time step $dt$. Because the
number of steps is fixed at graph construction time, the entire trajectory is
differentiable -- gradients flow from the final state back through all
integration steps to the initial conditions and force field parameters.

### Periodic Boundary Conditions

For condensed-phase simulations, the operator supports periodic boxes of
configurable size. Minimum-image convention is used for distance calculations,
with displacement functions that correctly handle periodic wrapping.

---

## Why Differentiability Matters for Structural Biology

Traditional structural biology tools compute structures or trajectories as
fixed outputs. Force field parameters are fit separately from the downstream
analysis. Structure prediction does not know what function will be evaluated.

DiffBio's differentiable operators enable:

1. **Structure-function optimization**: A loss on RNA function (e.g., binding
   affinity) propagates gradients through the folding algorithm back to
   sequence parameters, enabling sequence design for desired structures
2. **Force field fitting**: Gradients from trajectory observables (radial
   distribution function, diffusion coefficient) update force field parameters
   $\sigma$ and $\epsilon$ directly, replacing manual parameter tuning
3. **End-to-end MD pipelines**: A loss on a trajectory property (energy
   minimum, structural stability) flows back through the integrator and force
   field to initial conditions, enabling inverse design
4. **Differentiable DSSP**: Protein engineering pipelines can optimize
   backbone coordinates to maximize secondary structure content in target
   regions, with gradients from the DSSP assignment guiding the optimization

---

## Further Reading

- [Protein Operators](../operators/protein.md) -- secondary structure prediction with usage examples
- [RNA Structure Operators](../operators/rna-structure.md) -- RNA folding with usage examples
- [Molecular Dynamics Operators](../operators/molecular-dynamics.md) -- force fields and integrators
- [Protein API](../../api/operators/protein.md) -- protein structure API reference
- [RNA Structure API](../../api/operators/rna-structure.md) -- RNA folding API reference
- [Molecular Dynamics API](../../api/operators/molecular-dynamics.md) -- force field and integrator API

### References

1. Kabsch & Sander. "Dictionary of protein secondary structure: pattern
   recognition of hydrogen-bonded and geometrical features." *Biopolymers*
   22, 1983.
2. McCaskill. "The equilibrium partition function and base pair binding
   probabilities for RNA secondary structure." *Biopolymers* 29, 1990.
3. Schoenholz & Cubuk. "JAX, M.D.: A Framework for Differentiable Physics."
   *NeurIPS* 2020.
4. Matthies et al. "Differentiable partition function calculation for RNA."
   *Nucleic Acids Research* 52(3), 2024.
