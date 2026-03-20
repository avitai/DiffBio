# Sequence Alignment

Sequence alignment arranges two or more biological sequences to identify
regions of similarity that may reflect functional, structural, or evolutionary
relationships. DiffBio provides differentiable alignment operators that enable
gradient-based learning of alignment parameters.

---

## The Alignment Problem

Given two sequences $A$ and $B$, find an arrangement that maximizes a scoring
function:

$$
\text{Score} = \sum_{\text{aligned pairs}} s(a_i, b_j) + \sum_{\text{gaps}} g(k)
$$

Where $s(a, b)$ is the substitution score and $g(k)$ is the gap penalty for a
gap of length $k$.

### Local vs Global Alignment

| Type | Algorithm | Goal | DiffBio Operator |
|---|---|---|---|
| **Local** | Smith-Waterman | Best matching subsequence | `SmoothSmithWaterman` |
| **Global** | Needleman-Wunsch | Full end-to-end alignment | — |
| **Profile** | Profile HMM | Sequence-to-model alignment | `ProfileHMMAlignment` |
| **Multiple** | Progressive | Align multiple sequences | `SoftMSA` |

DiffBio focuses on local alignment (Smith-Waterman) as the foundation.

---

## Smith-Waterman Recurrence

For sequences $A$ (length $m$) and $B$ (length $n$), the standard recurrence
fills a matrix $H$:

$$
H(i, j) = \max \begin{cases}
0 \\
H(i-1, j-1) + s(A_i, B_j) \\
H(i-1, j) + g \\
H(i, j-1) + g
\end{cases}
$$

The optimal local alignment score is $\max_{i,j} H(i,j)$.

### Why This Blocks Gradients

The `max` operation selects one branch and discards the others. Gradients only
flow through the winning branch — all other branches receive zero gradient.
This creates a hard, non-differentiable decision at every cell.

---

## Smooth Smith-Waterman

DiffBio replaces `max` with temperature-scaled `logsumexp`:

$$
H(i, j) = \tau \cdot \log \sum_{k} \exp\left(\frac{c_k}{\tau}\right)
$$

Where $c_k$ are the four candidates (0, diagonal, up, left). Gradients now
flow to all branches, weighted by their relative magnitude.

### Temperature Effect

| $\tau$ | Behavior | Use Case |
|---|---|---|
| 0.01 | Nearly identical to hard max | Final evaluation |
| 0.1 - 1.0 | Balanced accuracy and gradient flow | Training |
| 10.0+ | All branches contribute equally | Warm start / exploration |

The operator uses a scan-based implementation that fills the matrix row by row,
compatible with `jax.jit` and `jax.grad`.

---

## Scoring Matrices

The substitution score $s(a, b)$ determines how well two characters match.

### DNA Scoring

For DNA (alphabet size 4: A, C, G, T), a simple match/mismatch matrix:

$$
s(a, b) = \begin{cases}
+2 & \text{if } a = b \\
-1 & \text{if } a \neq b
\end{cases}
$$

DiffBio provides `create_dna_scoring_matrix(match, mismatch)` for this.

### Learned Scoring

Because DiffBio's scoring matrix is an `nnx.Param`, gradients from a
downstream loss update scoring values during training. This lets the alignment
learn task-specific substitution preferences — for example, penalizing
transitions less than transversions, or learning scoring from structural
similarity rather than sequence identity.

---

## Gap Penalties

Gaps represent insertions or deletions (indels). Two common models:

**Linear**: $g(k) = d \cdot k$ — each gap position costs $d$.

**Affine**: $g(k) = d_{\text{open}} + d_{\text{extend}} \cdot (k-1)$ — opening
a gap is expensive, extending it is cheaper. This better models biological
indels, which tend to occur in contiguous blocks.

Both `gap_open` and `gap_extend` are learnable parameters in DiffBio.

---

## Alignment Outputs

`SmoothSmithWaterman` returns three key outputs:

| Output Key | Shape | Description |
|---|---|---|
| `score` | scalar | Soft maximum alignment score |
| `score_matrix` | $(m+1, n+1)$ | Full DP matrix |
| `alignment_scores` | $(m, n)$ | Per-position alignment scores |

The full DP matrix can be used for traceback analysis. The per-position scores
give a soft alignment probability: higher values indicate positions that
contribute more to the optimal alignment.

---

## When to Use Differentiable Alignment

**Good fit:**

- Learning scoring matrices for non-standard alphabets or domains
- Joint optimization where alignment feeds into a downstream classifier
- Sensitivity analysis — which scoring parameters affect the result most

**Not needed:**

- Production alignment of millions of reads (BWA/minimap2 are faster)
- Standard DNA alignment with known scoring (BLOSUM62, fixed gap penalties)
- When alignment is a preprocessing step with no learnable parameters

DiffBio's alignment operators are designed for learning and analysis, not for
replacing production aligners on large-scale data.

---

## Further Reading

- [Smith-Waterman Operator](../operators/smith-waterman.md) — usage, configuration, code examples
- [Alignment API](../../api/operators/smith-waterman.md) — full API reference
- [Sequence Alignment Example](../../examples/basic/simple-alignment.md) — runnable example

### References

1. Smith & Waterman. "Identification of common molecular subsequences."
   *J. Mol. Biol.* 147(1), 1981.
2. Petti et al. "End-to-end learning of multiple sequence alignments with
   differentiable Smith-Waterman." *Bioinformatics* 39(1), 2023.
