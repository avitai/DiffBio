// MathJax configuration for DiffBio (Differentiable Bioinformatics) documentation
// Must be loaded BEFORE MathJax library

window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)'], ['$', '$']],
    displayMath: [['\\[', '\\]'], ['$$', '$$']],
    processEscapes: true,
    processEnvironments: true,
    // Bioinformatics and Differentiable Computing specific macros
    macros: {
      // Common mathematical notation
      "R": "\\mathbb{R}",
      "C": "\\mathbb{C}",
      "N": "\\mathbb{N}",
      "Z": "\\mathbb{Z}",
      "Q": "\\mathbb{Q}",

      // Vector and matrix notation
      "vec": ["\\mathbf{#1}", 1],
      "mat": ["\\mathbf{#1}", 1],
      "norm": ["\\left\\|#1\\right\\|", 1],
      "abs": ["\\left|#1\\right|", 1],

      // Statistical notation
      "mean": ["\\bar{#1}", 1],
      "var": ["\\text{Var}(#1)", 1],
      "std": ["\\text{Std}(#1)", 1],
      "cov": ["\\text{Cov}(#1, #2)", 2],
      "corr": ["\\text{Corr}(#1, #2)", 2],

      // Probability notation
      "prob": ["\\text{P}(#1)", 1],
      "expect": ["\\mathbb{E}[#1]", 1],
      "given": "\\mid",

      // Machine learning notation
      "NN": "\\mathcal{N}",
      "loss": "\\mathcal{L}",
      "params": "\\theta",
      "weights": "\\mathbf{W}",
      "bias": "\\mathbf{b}",
      "activation": "\\sigma",
      "softmax": "\\text{softmax}",
      "relu": "\\text{ReLU}",
      "gelu": "\\text{GELU}",
      "sigmoid": "\\text{sigmoid}",
      "tanh": "\\text{tanh}",

      // Sequence alignment notation
      "seq": "\\mathbf{s}",
      "query": "\\mathbf{q}",
      "target": "\\mathbf{t}",
      "align": "\\mathcal{A}",
      "score": "S",
      "gap": "g",
      "match": "m",
      "mismatch": "\\mu",
      "sw": "\\text{SW}",
      "nw": "\\text{NW}",
      "affine": "\\text{Affine}",

      // Smith-Waterman specific
      "smithwaterman": "\\text{Smith-Waterman}",
      "needlemanwunsch": "\\text{Needleman-Wunsch}",
      "substitution": "\\mathbf{S}",
      "gapopen": "g_o",
      "gapextend": "g_e",
      "traceback": "\\mathcal{T}",

      // Pileup and variant calling notation
      "pileup": "\\mathbf{P}",
      "depth": "D",
      "coverage": "C",
      "basecall": "b",
      "quality": "Q",
      "phred": "\\text{Phred}",
      "variant": "v",
      "genotype": "G",
      "allele": "a",
      "ref": "\\text{ref}",
      "alt": "\\text{alt}",

      // Differentiability notation
      "grad": "\\nabla",
      "jacobian": "\\mathbf{J}",
      "hessian": "\\mathbf{H}",
      "diff": "\\partial",
      "softthreshold": "\\sigma_\\tau",
      "ste": "\\text{STE}",
      "gumbel": "\\text{Gumbel}",
      "straightthrough": "\\text{ST}",

      // DNA/RNA sequence notation
      "dna": "\\text{DNA}",
      "rna": "\\text{RNA}",
      "adenine": "\\text{A}",
      "cytosine": "\\text{C}",
      "guanine": "\\text{G}",
      "thymine": "\\text{T}",
      "uracil": "\\text{U}",
      "nucleotide": "n",
      "basepair": "\\text{bp}",
      "kmer": "k\\text{-mer}",

      // Protein notation
      "protein": "\\mathbf{P}",
      "residue": "\\mathbf{r}",
      "aminoacid": "\\text{aa}",
      "backbone": "\\mathbf{b}",
      "sidechain": "\\mathbf{s}",
      "peptide": "\\text{peptide}",

      // Read mapping notation
      "read": "r",
      "reference": "\\mathbf{R}",
      "mapping": "M",
      "mapq": "\\text{MAPQ}",
      "cigar": "\\text{CIGAR}",
      "sam": "\\text{SAM}",
      "bam": "\\text{BAM}",

      // Variant calling notation
      "snp": "\\text{SNP}",
      "indel": "\\text{Indel}",
      "sv": "\\text{SV}",
      "cnv": "\\text{CNV}",
      "vcf": "\\text{VCF}",
      "vaf": "\\text{VAF}",
      "likelihood": "\\mathcal{L}",
      "posterior": "P(G|D)",

      // RNA-seq notation
      "rnaseq": "\\text{RNA-seq}",
      "tpm": "\\text{TPM}",
      "fpkm": "\\text{FPKM}",
      "rpkm": "\\text{RPKM}",
      "counts": "C",
      "normcounts": "\\tilde{C}",
      "expression": "E",
      "fold": "\\text{FC}",
      "pvalue": "p",
      "fdr": "\\text{FDR}",

      // Single-cell notation
      "scrna": "\\text{scRNA-seq}",
      "scatac": "\\text{scATAC-seq}",
      "cell": "c",
      "cluster": "\\mathcal{C}",
      "trajectory": "\\mathcal{T}",
      "umap": "\\text{UMAP}",
      "tsne": "\\text{t-SNE}",
      "pca": "\\text{PCA}",

      // Epigenomics notation
      "methylation": "\\text{meth}",
      "acetylation": "\\text{ac}",
      "chipseq": "\\text{ChIP-seq}",
      "atacseq": "\\text{ATAC-seq}",
      "peak": "\\mathcal{P}",
      "motif": "\\mathcal{M}",
      "pwm": "\\text{PWM}",

      // Hi-C notation
      "hic": "\\text{Hi-C}",
      "contact": "\\mathbf{C}",
      "tad": "\\text{TAD}",
      "compartment": "\\text{A/B}",
      "loop": "\\ell",

      // Assembly notation
      "contig": "c",
      "scaffold": "\\mathcal{S}",
      "n50": "\\text{N50}",
      "debruijn": "\\text{de Bruijn}",
      "overlap": "O",

      // Metagenomics notation
      "otu": "\\text{OTU}",
      "asv": "\\text{ASV}",
      "taxonomy": "\\mathcal{T}",
      "abundance": "A",
      "diversity": "\\alpha, \\beta",

      // Phylogenetics notation
      "tree": "\\mathcal{T}",
      "branch": "b",
      "clade": "\\mathcal{C}",
      "distance": "d",
      "likelihood": "L",

      // Structure notation
      "structure": "\\mathbf{X}",
      "coords": "\\mathbf{c}",
      "distances": "\\mathbf{D}",
      "angles": "\\boldsymbol{\\theta}",
      "dihedrals": "\\boldsymbol{\\phi}, \\boldsymbol{\\psi}",
      "rmsd": "\\text{RMSD}",
      "tmscore": "\\text{TM-score}",
      "lddt": "\\text{lDDT}",

      // Deep learning for bio
      "esm": "\\text{ESM}",
      "alphafold": "\\text{AlphaFold}",
      "deepvariant": "\\text{DeepVariant}",
      "dnabert": "\\text{DNABERT}",

      // Optimization notation
      "lr": "\\eta",
      "update": "\\leftarrow",
      "adam": "\\text{Adam}",
      "sgd": "\\text{SGD}",
      "adamw": "\\text{AdamW}",
      "momentum": "m",
      "epsilon": "\\varepsilon",

      // Loss functions
      "mse": "\\text{MSE}",
      "mae": "\\text{MAE}",
      "crossentropy": "\\text{CE}",
      "bce": "\\text{BCE}",
      "nll": "\\text{NLL}",
      "kldiv": "\\text{KL}",

      // JAX/Flax specific
      "jax": "\\text{JAX}",
      "flax": "\\text{Flax}",
      "datarax": "\\text{Datarax}",
      "jit": "\\text{JIT}",
      "vmap": "\\text{vmap}",
      "pmap": "\\text{pmap}",
      "scan": "\\text{scan}",
      "pytree": "\\text{PyTree}",

      // Distribution notation
      "gaussian": "\\mathcal{N}",
      "uniform": "\\mathcal{U}",
      "bernoulli": "\\text{Bernoulli}",
      "categorical": "\\text{Categorical}",
      "dirichlet": "\\text{Dirichlet}",
      "poisson": "\\text{Poisson}",
      "nbinom": "\\text{NB}",

      // Architecture notation
      "cnn": "\\text{CNN}",
      "rnn": "\\text{RNN}",
      "lstm": "\\text{LSTM}",
      "gru": "\\text{GRU}",
      "transformer": "\\text{Transformer}",
      "attention": "\\text{Attn}",
      "multihead": "\\text{MultiHead}",

      // Regularization
      "dropout": "\\text{Dropout}",
      "batchnorm": "\\text{BatchNorm}",
      "layernorm": "\\text{LayerNorm}",

      // Set notation
      "argmax": "\\operatorname*{arg\\,max}",
      "argmin": "\\operatorname*{arg\\,min}"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Support for Material theme's instant loading feature
document.addEventListener('DOMContentLoaded', function() {
  if (typeof document$ !== 'undefined') {
    document$.subscribe(() => {
      if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
        MathJax.startup.output.clearCache()
        MathJax.typesetClear()
        MathJax.texReset()
        MathJax.typesetPromise()
      }
    });
  }
});
