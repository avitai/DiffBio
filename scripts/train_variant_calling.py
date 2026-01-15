"""Script to train variant calling pipeline and output actual results."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax

from diffbio.pipelines import create_cnn_variant_pipeline
from diffbio.utils.training import create_realistic_training_data

# Set random seed for reproducibility
np.random.seed(42)

print("Creating CNN pipeline...")
pipeline = create_cnn_variant_pipeline(
    reference_length=100,
    num_classes=3,
    quality_threshold=20.0,
    pileup_window_size=21,
    cnn_hidden_channels=[32, 64],
    cnn_fc_dims=[64, 32],
    seed=42,
)

print("Generating training data...")
train_inputs, train_targets = create_realistic_training_data(
    num_samples=500,
    num_reads=30,
    read_length=50,
    reference_length=100,
    variant_rate=0.05,
    heterozygous_rate=0.5,
    error_rate=0.01,
    seed=42,
)

# Split into train/val
val_split = 400
train_inputs, val_inputs = train_inputs[:val_split], train_inputs[val_split:]
train_targets, val_targets = train_targets[:val_split], train_targets[val_split:]

print(f"Training samples: {len(train_inputs)}")
print(f"Validation samples: {len(val_inputs)}")


def evaluate(pipeline, inputs, targets):
    """Evaluate pipeline on a dataset."""
    pipeline.eval_mode()

    all_preds = []
    all_labels = []

    for inp, tgt in zip(inputs, targets):
        result, _, _ = pipeline.apply(inp, {}, None)
        preds = jnp.argmax(result["probabilities"], axis=-1)
        all_preds.append(preds)
        all_labels.append(tgt["labels"])

    preds = jnp.concatenate(all_preds)
    labels = jnp.concatenate(all_labels)

    # Variant detection metrics
    true_variants = labels > 0
    pred_variants = preds > 0

    tp = (pred_variants & true_variants).sum()
    fp = (pred_variants & ~true_variants).sum()
    fn = (~pred_variants & true_variants).sum()
    tn = (~pred_variants & ~true_variants).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": float((preds == labels).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# Evaluate untrained model
print("\nUNTRAINED MODEL PERFORMANCE:")
untrained_metrics = evaluate(pipeline, val_inputs, val_targets)
print(f"  Precision: {untrained_metrics['precision']:.4f}")
print(f"  Recall:    {untrained_metrics['recall']:.4f}")
print(f"  F1 Score:  {untrained_metrics['f1']:.4f}")


# Class-weighted cross-entropy loss
def weighted_cross_entropy_loss(logits, labels, num_classes=3):
    """Cross-entropy with class weights to handle imbalance."""
    class_weights = jnp.array([1.0, 20.0, 20.0])
    one_hot = jax.nn.one_hot(labels, num_classes)
    log_probs = jax.nn.log_softmax(logits)
    weighted_loss = -jnp.sum(one_hot * log_probs * class_weights, axis=-1)
    return jnp.mean(weighted_loss)


# Create optimizer
optimizer = nnx.Optimizer(pipeline, optax.adam(learning_rate=3e-3), wrt=nnx.Param)

# Training loop
loss_history = []
print("\nTraining variant calling pipeline...")

for epoch in range(100):
    pipeline.train_mode()
    epoch_losses = []

    for inp, tgt in zip(train_inputs, train_targets):

        def compute_loss(model):
            result, _, _ = model.apply(inp, {}, None)
            return weighted_cross_entropy_loss(result["logits"], tgt["labels"])

        loss, grads = nnx.value_and_grad(compute_loss)(pipeline)
        optimizer.update(pipeline, grads)
        epoch_losses.append(float(loss))

    avg_loss = np.mean(epoch_losses)
    loss_history.append(avg_loss)

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: loss = {avg_loss:.4f}")

print(f"\nFinal loss: {loss_history[-1]:.4f}")

# Evaluate trained model
print("\nTRAINED MODEL PERFORMANCE:")
trained_metrics = evaluate(pipeline, val_inputs, val_targets)
print(f"  Precision: {trained_metrics['precision']:.4f}")
print(f"  Recall:    {trained_metrics['recall']:.4f}")
print(f"  F1 Score:  {trained_metrics['f1']:.4f}")

# Display comparison
print("\n" + "=" * 65)
print("VARIANT CALLING PERFORMANCE: UNTRAINED vs TRAINED")
print("=" * 65)
print(f"\n{'Metric':<20} {'Untrained':>15} {'Trained':>15} {'Change':>15}")
print("-" * 65)
prec_change = trained_metrics["precision"] - untrained_metrics["precision"]
recall_change = trained_metrics["recall"] - untrained_metrics["recall"]
f1_change = trained_metrics["f1"] - untrained_metrics["f1"]
acc_change = trained_metrics["accuracy"] - untrained_metrics["accuracy"]

print(
    f"{'Precision':<20} {untrained_metrics['precision']:>15.4f} "
    f"{trained_metrics['precision']:>15.4f} {prec_change:>+15.4f}"
)
print(
    f"{'Recall':<20} {untrained_metrics['recall']:>15.4f} "
    f"{trained_metrics['recall']:>15.4f} {recall_change:>+15.4f}"
)
print(
    f"{'F1 Score':<20} {untrained_metrics['f1']:>15.4f} "
    f"{trained_metrics['f1']:>15.4f} {f1_change:>+15.4f}"
)
print(
    f"{'Accuracy':<20} {untrained_metrics['accuracy']:>15.4f} "
    f"{trained_metrics['accuracy']:>15.4f} {acc_change:>+15.4f}"
)

print("\nConfusion Matrix (Trained Model):")
print(f"  True Positives:  {trained_metrics['tp']}")
print(f"  False Positives: {trained_metrics['fp']}")
print(f"  False Negatives: {trained_metrics['fn']}")
print(f"  True Negatives:  {trained_metrics['tn']}")

# Sample analysis
print("\n\n=== Sample Analysis ===")
result, _, _ = pipeline.apply(val_inputs[0], {}, None)
probs = result["probabilities"]
preds = jnp.argmax(probs, axis=-1)
true_labels = val_targets[0]["labels"]

print("Sample analysis:")
print(f"  True variants: {int((true_labels > 0).sum())}")
print(f"  Predicted variants: {int((preds > 0).sum())}")

variant_positions = jnp.where(true_labels > 0)[0]
print(f"\nTrue variant positions: {list(variant_positions)}")
print(f"Predictions at those positions: {list(preds[variant_positions])}")
conf_at_variants = probs[variant_positions].max(axis=-1)
print(f"Confidence at those positions: {[f'{c:.4f}' for c in conf_at_variants]}")

# Learned parameters
print("\n\n=== Learned Parameters ===")
print(f"Learned quality threshold: {float(pipeline.quality_filter.threshold.value):.2f}")
print(f"Pileup temperature: {float(pipeline.pileup.temperature.value):.4f}")

# Production inference stats
print("\n\n=== Production Inference ===")
confidence = probs.max(axis=-1)
variant_mask = preds > 0
high_conf_variants = variant_mask & (confidence > 0.8)
print(f"Total predicted variants: {int(variant_mask.sum())}")
print(f"High-confidence variants (>80%): {int(high_conf_variants.sum())}")
