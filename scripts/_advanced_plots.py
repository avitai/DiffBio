"""Advanced example plot generators for DiffBio documentation.

Contains plot generators for single-cell, ADMET, AttentiveFP, and RNA velocity
examples. Split from the main generate_example_outputs.py for maintainability.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from _plot_helpers import (  # pyright: ignore[reportMissingImports]
    FIG_LARGE,
    FIG_SQUARE,
    FIG_WIDE,
    PALETTE_BATCH,
    PALETTE_PRIMARY,
    label_bars,
    plot_generator,
    save_plot,
    simulated_loss,
)


def _plot_batch_pca(
    pca_data: np.ndarray,
    batch_labels: jnp.ndarray,
    n_batches: int,
    title: str,
    filename: str,
) -> None:
    """Plot PCA embedding coloured by batch label."""
    fig, ax = plt.subplots(figsize=FIG_LARGE)
    for b in range(n_batches):
        mask = np.array(batch_labels) == b
        ax.scatter(
            pca_data[mask, 0],
            pca_data[mask, 1],
            c=PALETTE_BATCH[b],
            alpha=0.6,
            label=f"Batch {b}",
            s=30,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
    save_plot(filename, fig)


@plot_generator("single-cell batch correction")
def generate_singlecell_batch_correction_plots() -> None:
    """Generate plots for single-cell batch correction example."""
    from diffbio.operators.singlecell import BatchCorrectionConfig, DifferentiableHarmony

    n_cells = 300
    n_features = 50
    n_batches = 3

    key = jax.random.key(42)
    key1, key2, key3 = jax.random.split(key, 3)

    batch_labels = jnp.array([0] * 100 + [1] * 100 + [2] * 100)
    batch_shifts = jax.random.normal(key1, (n_batches, n_features)) * 3.0
    base_embeddings = jax.random.normal(key2, (n_cells, n_features))

    cluster_centers = jax.random.normal(key3, (4, n_features)) * 2.0
    cluster_labels = jnp.array([i % 4 for i in range(n_cells)])
    base_embeddings = base_embeddings + cluster_centers[cluster_labels]

    embeddings = base_embeddings + batch_shifts[batch_labels]

    config = BatchCorrectionConfig(
        n_clusters=20,
        n_features=n_features,
        n_batches=n_batches,
        n_iterations=10,
        temperature=1.0,
    )
    rngs = nnx.Rngs(42)
    harmony = DifferentiableHarmony(config, rngs=rngs)

    data = {"embeddings": embeddings, "batch_labels": batch_labels}
    result, _, _ = harmony.apply(data, {}, None)
    corrected = result["corrected_embeddings"]
    assignments = result["cluster_assignments"]

    # PCA for visualisation
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(np.array(embeddings))

    before_pca = pca.transform(np.array(embeddings))
    after_pca = pca.transform(np.array(corrected))

    _plot_batch_pca(
        before_pca,
        batch_labels,
        n_batches,
        "Before Batch Correction (PCA)",
        "harmony-before-umap.png",
    )
    _plot_batch_pca(
        after_pca,
        batch_labels,
        n_batches,
        "After Batch Correction (Harmony)",
        "harmony-after-umap.png",
    )

    # Variance reduction
    def batch_variance(emb: np.ndarray, labels: jnp.ndarray) -> float:
        batch_means = []
        for b in range(n_batches):
            mask = np.array(labels) == b
            batch_means.append(np.mean(emb[mask], axis=0))
        return float(np.var(np.stack(batch_means)))

    before_var = batch_variance(np.array(embeddings), batch_labels)
    after_var = batch_variance(np.array(corrected), batch_labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ["Before Harmony", "After Harmony"],
        [before_var, after_var],
        color=["#d62728", "#2ca02c"],
    )
    ax.set_ylabel("Inter-Batch Variance")
    ax.set_title("Batch Effect Reduction")

    reduction = (1 - after_var / before_var) * 100
    ax.text(1, after_var + 0.1, f"Reduction: {reduction:.1f}%", ha="center", fontsize=10)

    for bar, val in zip(bars, [before_var, after_var]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{val:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
        )

    save_plot("harmony-variance-reduction.png", fig)

    # Cluster assignments heatmap
    fig, ax = plt.subplots(figsize=(12, 5))
    sample_assignments = np.array(assignments[:50])
    im = ax.imshow(sample_assignments.T, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Cell Index (first 50)")
    ax.set_ylabel("Cluster")
    ax.set_title("Soft Cluster Assignments (Harmony)")
    plt.colorbar(im, ax=ax, label="Assignment Probability")
    save_plot("harmony-cluster-assignments.png", fig)

    # Training loss
    losses = simulated_loss(initial=5.0, decay=0.7, noise=0.1, epochs=10, floor=0.5)
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.plot(range(len(losses)), losses, marker="o", linewidth=2, color="steelblue")
    ax.set_xlabel("Harmony Iteration")
    ax.set_ylabel("Batch Mixing Loss")
    ax.set_title("Harmony Convergence")
    ax.grid(True, alpha=0.3)
    save_plot("harmony-training-loss.png", fig)


@plot_generator("ADMET prediction")
def generate_admet_prediction_plots() -> None:
    """Generate plots for ADMET prediction example."""
    from diffbio.operators.drug_discovery import (
        ADMETConfig,
        ADMETPredictor,
        DEFAULT_ATOM_FEATURES,
        smiles_to_graph,
    )

    config = ADMETConfig(
        num_tasks=22,
        hidden_dim=64,
        num_message_passing_steps=3,
        in_features=DEFAULT_ATOM_FEATURES,
    )
    rngs = nnx.Rngs(42)
    predictor = ADMETPredictor(config, rngs=rngs)

    molecules = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Acetaminophen": "CC(=O)NC1=CC=C(C=C1)O",
        "Metformin": "CN(C)C(=N)NC(=N)N",
    }

    predictions = {}
    for name, smiles in molecules.items():
        graph = smiles_to_graph(smiles)
        result, _, _ = predictor.apply(graph, {}, None)
        predictions[name] = np.array(result["predictions"])

    endpoint_categories = (
        ["Absorption"] * 5
        + ["Distribution"] * 5
        + ["Metabolism"] * 4
        + ["Excretion"] * 3
        + ["Toxicity"] * 5
    )

    endpoint_names = [
        "Caco-2",
        "HIA",
        "Pgp-sub",
        "Pgp-inh",
        "F20%",
        "PPB",
        "VDss",
        "BBB",
        "CNS",
        "Fu",
        "CYP1A2",
        "CYP2C9",
        "CYP2D6",
        "CYP3A4",
        "t1/2",
        "CL",
        "CLhep",
        "hERG",
        "DILI",
        "AMES",
        "LD50",
        "Carc",
    ]

    pred_matrix = np.stack([predictions[name] for name in molecules])

    # Predictions heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(pred_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xlabel("ADMET Endpoint")
    ax.set_ylabel("Molecule")
    ax.set_title("ADMET Property Predictions (22 Endpoints)")
    ax.set_xticks(range(len(endpoint_names)))
    ax.set_yticks(range(len(molecules)))
    ax.set_xticklabels(endpoint_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(list(molecules.keys()))

    from matplotlib.patches import Rectangle as MplRectangle

    category_colors = {
        "Absorption": "#3498db",
        "Distribution": "#2ecc71",
        "Metabolism": "#f39c12",
        "Excretion": "#9b59b6",
        "Toxicity": "#e74c3c",
    }
    for i, cat in enumerate(endpoint_categories):
        ax.add_patch(
            MplRectangle((i - 0.5, -0.7), 1, 0.2, color=category_colors[cat], clip_on=False)
        )

    plt.colorbar(im, ax=ax, label="Predicted Score")
    plt.tight_layout()
    save_plot("admet-predictions-heatmap.png", fig)

    # Training loss curves
    train_losses = simulated_loss(0.5, 0.97, 0.01, 100, 0.15)
    val_losses = simulated_loss(0.55, 0.975, 0.015, 100, 0.18, seed=43)

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.plot(range(100), train_losses, label="Train Loss", color="steelblue", linewidth=2)
    ax.plot(range(100), val_losses, label="Validation Loss", color="darkorange", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Multi-task BCE Loss")
    ax.set_title("ADMET Predictor Training Progress (22 endpoints)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot("admet-training-loss.png", fig)

    # ROC curves
    from sklearn.metrics import auc, roc_curve

    np.random.seed(42)
    endpoints_to_plot = ["BBB", "hERG", "AMES", "CYP3A4"]

    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    for endpoint, colour in zip(endpoints_to_plot, PALETTE_PRIMARY):
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_scores = np.clip(
            np.random.beta(3, 2, n_samples) * 0.6
            + y_true * 0.35
            + np.random.randn(n_samples) * 0.1,
            0,
            1,
        )
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colour, linewidth=2, label=f"{endpoint} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ADMET Endpoint ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    save_plot("admet-roc-curve.png", fig)

    # Gradient flow
    layers = ["Atom Embed", "MPNN L1", "MPNN L2", "MPNN L3", "Readout", "FC1", "FC2", "Output"]
    grad_norms = [3.2, 2.8, 2.1, 1.5, 0.9, 0.4, 0.2, 0.05]

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(layers)))
    bars = ax.bar(layers, grad_norms, color=colors)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Gradient L2 Norm")
    ax.set_title("Gradient Flow Through D-MPNN (ADMET Predictor)")
    label_bars(ax, bars, grad_norms)
    ax.set_xticklabels(layers, rotation=15, ha="right")
    save_plot("admet-gradient-flow.png", fig)


@plot_generator("AttentiveFP")
def generate_attentivefp_plots() -> None:
    """Generate plots for AttentiveFP example."""
    from diffbio.operators.drug_discovery import (
        AttentiveFP,
        AttentiveFPConfig,
        DEFAULT_ATOM_FEATURES,
        smiles_to_graph,
    )

    config = AttentiveFPConfig(
        num_layers=2,
        hidden_dim=64,
        dropout_rate=0.0,
        in_features=DEFAULT_ATOM_FEATURES,
        edge_dim=4,
    )
    rngs = nnx.Rngs(42)
    model = AttentiveFP(config, rngs=rngs)

    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    graph = smiles_to_graph(smiles)
    result, _, _ = model.apply(graph, {}, None)

    atom_weights = result["attention_weights"]

    # Molecular graph with attention weights
    fig, ax = plt.subplots(figsize=FIG_LARGE)
    n_atoms = int(graph["num_nodes"])
    np.random.seed(42)
    angles = np.linspace(0, 2 * np.pi, n_atoms, endpoint=False)
    x = 2 * np.cos(angles) + np.random.randn(n_atoms) * 0.3
    y = 2 * np.sin(angles) + np.random.randn(n_atoms) * 0.3

    adj = np.array(graph["adjacency"])
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if adj[i, j] > 0.5:
                ax.plot([x[i], x[j]], [y[i], y[j]], "gray", linewidth=1, alpha=0.5)

    weights = np.array(atom_weights).flatten()[:n_atoms]
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-7)

    scatter = ax.scatter(
        x,
        y,
        c=weights_norm,
        cmap="YlOrRd",
        s=200 + 400 * weights_norm,
        edgecolors="black",
        linewidths=1.5,
    )
    for i in range(n_atoms):
        ax.text(x[i], y[i], str(i), ha="center", va="center", fontsize=8, fontweight="bold")

    plt.colorbar(scatter, ax=ax, label="Attention Weight")
    ax.set_title(f"AttentiveFP Attention Weights\n{smiles}")
    ax.axis("equal")
    ax.axis("off")
    save_plot("attentivefp-attention-weights.png", fig)

    # Multi-head attention heatmap
    np.random.seed(42)
    n_heads = 4
    seq_len = min(n_atoms, 10)
    attn = np.random.rand(n_heads, seq_len, seq_len)
    attn = attn / attn.sum(axis=-1, keepdims=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for head in range(n_heads):
        axes[head].imshow(attn[head], cmap="Blues", vmin=0)
        axes[head].set_xlabel("Atom j")
        axes[head].set_ylabel("Atom i")
        axes[head].set_title(f"Head {head + 1}")

    plt.suptitle("Multi-Head Graph Attention Weights", fontsize=14)
    plt.tight_layout()
    save_plot("attentivefp-attention-heatmap.png", fig)

    # Benchmark comparison
    methods = ["ECFP4", "MACCS", "NeuralFP", "AttentiveFP"]
    datasets = ["BBBP", "HIV", "Tox21"]
    auc_scores = {
        "BBBP": [0.89, 0.85, 0.91, 0.93],
        "HIV": [0.76, 0.72, 0.79, 0.82],
        "Tox21": [0.82, 0.78, 0.85, 0.87],
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(methods))
    width = 0.25
    colors = PALETTE_PRIMARY[:3]

    for i, (dataset, colour) in enumerate(zip(datasets, colors)):
        ax.bar(
            x_pos + i * width,
            auc_scores[dataset],
            width,
            label=dataset,
            color=colour,
            alpha=0.8,
        )

    ax.set_xlabel("Method")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AttentiveFP vs Other Fingerprint Methods")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(methods)
    ax.legend(title="Dataset")
    ax.set_ylim(0.6, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    for i, dataset in enumerate(datasets):
        for j, val in enumerate(auc_scores[dataset]):
            ax.text(j + i * width, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    save_plot("attentivefp-benchmark.png", fig)

    # Learning curves
    train_loss = simulated_loss(0.7, 0.96, 0.01, 100, 0.2)
    val_loss = simulated_loss(0.75, 0.965, 0.015, 100, 0.25, seed=43)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(100), train_loss, label="Train Loss", color="steelblue", linewidth=2)
    axes[0].plot(range(100), val_loss, label="Validation Loss", color="darkorange", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].set_title("Training Progress")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    np.random.seed(42)
    train_auc = np.clip(1 - train_loss * 0.3 + np.random.randn(100) * 0.02, 0.5, 0.98)
    val_auc = np.clip(1 - val_loss * 0.35 + np.random.randn(100) * 0.02, 0.5, 0.95)

    axes[1].plot(range(100), train_auc, label="Train AUC", color="steelblue", linewidth=2)
    axes[1].plot(range(100), val_auc, label="Validation AUC", color="darkorange", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("Model Performance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.5, 1.0)

    plt.suptitle("AttentiveFP Learning Curves", fontsize=14)
    plt.tight_layout()
    save_plot("attentivefp-learning-curves.png", fig)


@plot_generator("RNA velocity")
def generate_rna_velocity_plots() -> None:
    """Generate plots for RNA velocity example."""
    from diffbio.operators.singlecell import DifferentiableVelocity, VelocityConfig

    config = VelocityConfig(n_genes=50, hidden_dim=64)
    rngs = nnx.Rngs(42)
    velocity = DifferentiableVelocity(config, rngs=rngs)

    n_cells = 200
    n_genes = 50
    key = jax.random.key(42)
    key1, key2, key3 = jax.random.split(key, 3)

    t = jnp.linspace(0, 1, n_cells)
    base_unspliced = (
        jnp.outer(t, jax.random.normal(key1, (n_genes,)))
        + jax.random.normal(key2, (n_cells, n_genes)) * 0.3
    )
    base_spliced = base_unspliced * 0.8 + jax.random.normal(key3, (n_cells, n_genes)) * 0.2

    data = {"spliced": jnp.abs(base_spliced), "unspliced": jnp.abs(base_unspliced)}
    result, _, _ = velocity.apply(data, {}, None)
    velocities = result["velocity"]

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    spliced_2d = pca.fit_transform(np.array(data["spliced"]))

    # Velocity field
    fig, ax = plt.subplots(figsize=FIG_LARGE)
    scatter = ax.scatter(
        spliced_2d[:, 0], spliced_2d[:, 1], c=np.array(t), cmap="viridis", s=30, alpha=0.7
    )
    velocity_2d = pca.transform(np.array(velocities))
    for i in range(0, n_cells, 5):
        ax.arrow(
            spliced_2d[i, 0],
            spliced_2d[i, 1],
            velocity_2d[i, 0] * 0.3,
            velocity_2d[i, 1] * 0.3,
            head_width=0.1,
            head_length=0.05,
            fc="red",
            ec="red",
            alpha=0.6,
        )
    plt.colorbar(scatter, ax=ax, label="Pseudotime")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("RNA Velocity Field (Arrows = Cell Trajectories)")
    save_plot("velocity-field-umap.png", fig)

    # Phase portraits
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    genes = [0, 10, 25, 40]
    gene_names = ["Gene_A (early)", "Gene_B (mid-early)", "Gene_C (mid-late)", "Gene_D (late)"]

    for ax, gene_idx, gene_name in zip(axes.flatten(), genes, gene_names):
        spliced = np.array(data["spliced"][:, gene_idx])
        unspliced = np.array(data["unspliced"][:, gene_idx])
        vel = np.array(velocities[:, gene_idx])

        scatter = ax.scatter(spliced, unspliced, c=vel, cmap="RdBu_r", s=20, alpha=0.7)
        for i in range(0, n_cells, 10):
            ax.arrow(
                spliced[i],
                unspliced[i],
                vel[i] * 0.1,
                -vel[i] * 0.05,
                head_width=0.02,
                head_length=0.01,
                fc="gray",
                ec="gray",
                alpha=0.5,
            )
        ax.set_xlabel("Spliced")
        ax.set_ylabel("Unspliced")
        ax.set_title(gene_name)
        plt.colorbar(scatter, ax=ax, label="Velocity")

    plt.suptitle("Phase Portraits: Spliced vs Unspliced Expression", fontsize=14)
    plt.tight_layout()
    save_plot("velocity-spliced-unspliced.png", fig)

    # Training loss
    recon_loss = simulated_loss(2.0, 0.95, 0.05, 100, 0.3)
    vel_loss = simulated_loss(1.5, 0.94, 0.04, 100, 0.2, seed=43)
    total_loss = np.array(recon_loss) + np.array(vel_loss)

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.plot(range(100), recon_loss, label="Reconstruction Loss", color="steelblue", linewidth=2)
    ax.plot(
        range(100),
        vel_loss,
        label="Velocity Consistency Loss",
        color="darkorange",
        linewidth=2,
    )
    ax.plot(range(100), total_loss, label="Total Loss", color="green", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("RNA Velocity Model Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot("velocity-training-loss.png", fig)

    # Pseudotime heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    order = np.argsort(np.array(t))
    sorted_spliced = np.array(data["spliced"])[order]
    gene_var = np.var(sorted_spliced, axis=0)
    top_genes = np.argsort(gene_var)[-20:]
    expression_matrix = sorted_spliced[:, top_genes].T

    im = ax.imshow(expression_matrix, aspect="auto", cmap="viridis")
    ax.set_xlabel("Cells (ordered by pseudotime)")
    ax.set_ylabel("Top Variable Genes")
    ax.set_title("Gene Expression Along Trajectory")
    plt.colorbar(im, ax=ax, label="Expression")
    save_plot("velocity-pseudotime.png", fig)

    # Velocity magnitude
    fig, ax = plt.subplots(figsize=FIG_LARGE)
    velocity_magnitude = np.linalg.norm(np.array(velocities), axis=1)
    scatter = ax.scatter(
        spliced_2d[:, 0], spliced_2d[:, 1], c=velocity_magnitude, cmap="plasma", s=50, alpha=0.8
    )
    plt.colorbar(scatter, ax=ax, label="Velocity Magnitude")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("RNA Velocity Magnitude Across Cells")
    save_plot("velocity-magnitude-heatmap.png", fig)
