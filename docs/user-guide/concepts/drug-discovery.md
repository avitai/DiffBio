# Drug Discovery

Computational drug discovery uses molecular representations and machine learning
to predict how small molecules interact with biological targets. DiffBio provides
7 differentiable operators covering molecular representation, property prediction,
ADMET profiling, and similarity search -- enabling gradient-based optimization
from molecular structure to predicted activity.

---

## Molecular Graph Representation

Small molecules are naturally represented as graphs: atoms are nodes, bonds are
edges. Each atom carries features (element type, degree, formal charge,
hybridization, aromaticity) and each bond carries features (bond type, stereo,
conjugation, ring membership).

A molecular graph $G = (V, E)$ with node features $\mathbf{x}_v \in \mathbb{R}^d$
and edge features $\mathbf{e}_{uv} \in \mathbb{R}^k$ is the standard input to
all DiffBio drug discovery operators.

This graph representation is fully differentiable -- gradients flow from
predicted properties back through atom and bond features, enabling end-to-end
learning from structure to function.

---

## Message Passing Neural Networks

Message passing neural networks (MPNNs) are the core architecture for learning
from molecular graphs. At each step, every atom aggregates information from its
neighbors:

$$
\mathbf{m}_v^{(t)} = \sum_{u \in \mathcal{N}(v)} M(\mathbf{h}_v^{(t)}, \mathbf{h}_u^{(t)}, \mathbf{e}_{uv})
$$

$$
\mathbf{h}_v^{(t+1)} = U(\mathbf{h}_v^{(t)}, \mathbf{m}_v^{(t)})
$$

After $T$ rounds of message passing, each atom's representation encodes
information about its $T$-hop neighborhood. A graph-level readout (sum or
attention pooling) produces a single molecular representation.

DiffBio implements the **directed MPNN (D-MPNN)** variant from ChemProp, where
messages pass along directed edges to avoid information shortcuts. This is the
backbone of both `MolecularPropertyPredictor` and `ADMETPredictor`.

---

## Molecular Fingerprints

Fingerprints reduce a molecule to a fixed-length vector for comparison and
retrieval. Traditional fingerprints use hashing and lose gradient information.
DiffBio provides differentiable alternatives:

| Fingerprint Type | Operator | Method | Output |
|---|---|---|---|
| **Neural graph** | `DifferentiableMolecularFingerprint` | Learned GNN convolutions + sum readout | Dense vector |
| **Circular (ECFP/Morgan)** | `CircularFingerprintOperator` | Differentiable radius-based atom environments | Bit vector (soft) |
| **MACCS keys** | `MACCSKeysOperator` | Learned pattern matching for 166 structural keys | 166-bit vector (soft) |
| **Attentive** | `AttentiveFP` | Graph attention + GRU-based molecule readout | Dense vector |

### Learned vs Classical Fingerprints

Classical fingerprints (ECFP, MACCS) encode fixed structural patterns through
hashing or predefined SMARTS patterns. They are interpretable but not optimizable
-- the same fingerprint is used regardless of the prediction task.

Learned fingerprints (neural graph, AttentiveFP) adapt their representations to
the task at hand. During training, the GNN learns which structural features are
most predictive. This task-specific adaptation typically improves accuracy for
property prediction and virtual screening.

DiffBio bridges both worlds: `CircularFingerprintOperator` and `MACCSKeysOperator`
implement classical fingerprint logic using differentiable approximations (soft
hashing, temperature-controlled pattern matching), so they maintain the
interpretability of classical fingerprints while enabling gradient flow.

---

## Property Prediction

`MolecularPropertyPredictor` implements the full ChemProp architecture for
predicting molecular properties:

1. D-MPNN encodes the molecular graph into atom representations
2. Sum readout aggregates atoms into a graph-level vector
3. Feed-forward network maps the graph vector to property predictions

The operator supports multi-task learning -- predicting multiple properties from
a single molecular representation. This is valuable when related properties share
underlying structural features.

---

## ADMET Prediction

ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties
determine whether a drug candidate can reach its target, act on it, and be
safely eliminated. These pharmacokinetic properties are the primary cause of
drug failure in clinical trials.

`ADMETPredictor` provides multi-task prediction across the 22 standard TDC
ADMET benchmark endpoints:

| Category | Endpoints | Task Type |
|---|---|---|
| **Absorption** (6) | Caco-2 permeability, HIA, P-gp inhibition, bioavailability, lipophilicity, solubility | Mixed |
| **Distribution** (3) | BBB penetration, plasma protein binding, volume of distribution | Mixed |
| **Metabolism** (6) | CYP inhibition (2C9, 2D6, 3A4), CYP substrate status | Classification |
| **Excretion** (3) | Half-life, hepatocyte clearance, microsome clearance | Regression |
| **Toxicity** (4) | LD50, hERG liability, Ames mutagenicity, drug-induced liver injury | Mixed |

The architecture shares a D-MPNN backbone across all tasks, with task-specific
output heads. Shared representation learning transfers structural knowledge
between related endpoints.

---

## Similarity Search

Virtual screening identifies molecules similar to a known active compound.
`MolecularSimilarityOperator` computes differentiable similarity between
fingerprint vectors using three metrics:

| Metric | Formula | Best For |
|---|---|---|
| **Tanimoto** | $\frac{a \cdot b}{\|a\|^2 + \|b\|^2 - a \cdot b}$ | Binary/sparse fingerprints |
| **Cosine** | $\frac{a \cdot b}{\|a\| \cdot \|b\|}$ | Dense learned fingerprints |
| **Dice** | $\frac{2(a \cdot b)}{\|a\|^2 + \|b\|^2}$ | Balanced precision/recall |

Because both the fingerprints and the similarity metric are differentiable,
gradients flow from a similarity-based loss all the way back to the molecular
graph features. This enables learning fingerprints optimized for a specific
similarity task.

---

## Why Differentiability Matters for Drug Discovery

Traditional cheminformatics pipelines apply each step independently: compute
fingerprints, then predict properties, then rank by similarity. Each step
is optimized in isolation.

DiffBio's differentiable operators enable:

1. **End-to-end property prediction**: A loss on predicted activity updates the
   MPNN, readout, and prediction head jointly -- the molecular representation
   adapts to what matters for the specific property
2. **Task-specific fingerprints**: Fingerprint generation is trained alongside
   the downstream task, rather than using fixed structural encodings
3. **Differentiable virtual screening**: Similarity thresholds and fingerprint
   weights can be optimized to maximize enrichment of active compounds
4. **Multi-task learning with gradient flow**: ADMET prediction shares
   representations across endpoints, with gradients from all tasks shaping
   the shared molecular encoder

---

## Further Reading

- [Drug Discovery Operators](../operators/drug-discovery.md) -- all 7 operators with usage examples
- [Drug Discovery API](../../api/operators/drug-discovery.md) -- full API reference
- [Data Sources](../sources.md#molnet-benchmark-datasets) -- loading MoleculeNet benchmark datasets
- [Dataset Splitters](../splitters.md) -- scaffold-based train/test splitting

### References

1. Yang et al. "Analyzing Learned Molecular Representations for Property
   Prediction." *JCIM* 59(8), 2019.
2. Xiong et al. "Pushing the Boundaries of Molecular Representation for Drug
   Discovery with the Graph Attention Mechanism." *JCIM* 60(6), 2020.
3. Swanson et al. "ADMET-AI: A machine learning ADMET platform."
   *Bioinformatics* 40(1), 2024.
4. Durant et al. "Reoptimization of MDL Keys for Use in Drug Discovery."
   *JCIM* 42(6), 2002.
