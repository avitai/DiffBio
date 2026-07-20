"""Case study C extraction: real GIAB HG002 chr20 -> labeled pileup-window tensors.

DeepVariant-style candidate generation on REAL data: scan chr20 for columns with a
non-reference allele fraction above threshold (candidate variant sites), extract a
pileup window (reads pre-placed at genomic columns, positions=0, read_length=window),
and label each candidate by membership in the GIAB v4.2.1 truth VCF. The task the
downstream CNN must solve is exactly DeepVariant's: separate true variants from
candidate look-alikes (sequencing errors / mapping artifacts) using the pileup image.

Disjoint train / test genomic regions so the split is a real held-out locus, never a
random shuffle. No synthetic data: every read, quality, and label comes from the BAM
and the truth set.
"""

import os

import sys

import numpy as np
import pysam  # noqa: PLC0415


_DATA = os.environ.get("DIFFBIO_DATA_ROOT", "/mnt/ssd2/Data")
BAM = f"{_DATA}/giab/HG002.chr20.bam"
REF = f"{_DATA}/giab/chr20.fa"
TRUTH = f"{_DATA}/giab/truth.vcf.gz"
OUT = f"{_DATA}/giab/variant_windows.npz"

CHROM = "chr20"
WINDOW = 41
HALF = WINDOW // 2
MAX_READS = 50
MIN_DEPTH = 12
MIN_ALT_FRAC = 0.10
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

TRAIN_REGION = (2_000_000, 22_000_000)
TEST_REGION = (30_000_000, 40_000_000)
PER_SPLIT = {"train": 8000, "test": 2500}  # balanced cap (half positive, half negative)


def load_truth(chrom: str, start: int, stop: int) -> dict[int, int]:
    """Map 0-based truth positions to class (1=SNP, 2=indel) for PASS genotyped variants."""
    truth: dict[int, int] = {}
    vcf = pysam.VariantFile(TRUTH)
    for rec in vcf.fetch(chrom, start, stop):
        if rec.filter.keys() not in ([], ["PASS"]):
            continue
        alt = rec.alts[0] if rec.alts else None
        if alt is None or rec.ref is None:
            continue
        gt = rec.samples[0].get("GT") if rec.samples else None
        if gt is not None and all(a in (0, None) for a in gt):
            continue  # homozygous reference -> not a variant
        is_snp = len(rec.ref) == 1 and len(alt) == 1
        truth[rec.pos - 1] = 1 if is_snp else 2
    return truth


def find_candidates(bam: pysam.AlignmentFile, ref: pysam.FastaFile, start: int, stop: int):
    """Yield 0-based candidate positions with non-ref allele fraction >= MIN_ALT_FRAC."""
    for chunk_start in range(start, stop, 1_000_000):
        chunk_stop = min(chunk_start + 1_000_000, stop)
        counts = np.asarray(
            bam.count_coverage(CHROM, chunk_start, chunk_stop, quality_threshold=0)
        )  # (4, L) A/C/G/T; keep low-quality mismatches as hard error-negatives
        depth = counts.sum(0)
        refseq = ref.fetch(CHROM, chunk_start, chunk_stop).upper()
        for i in range(chunk_stop - chunk_start):
            if depth[i] < MIN_DEPTH:
                continue
            ref_base = refseq[i]
            ref_idx = BASE_TO_IDX.get(ref_base)
            if ref_idx is None:
                continue
            alt_frac = 1.0 - counts[ref_idx, i] / depth[i]
            if alt_frac >= MIN_ALT_FRAC:
                yield chunk_start + i


def extract_window(bam: pysam.AlignmentFile, ref: pysam.FastaFile, pos0: int) -> dict | None:
    """Build the operator-input tensors for a single candidate window around pos0."""
    win_start, win_stop = pos0 - HALF, pos0 + HALF + 1
    if win_start < 0:
        return None
    reads = np.zeros((MAX_READS, WINDOW, 4), np.float32)
    quals = np.zeros((MAX_READS, WINDOW), np.float32)
    mapqs = np.zeros((MAX_READS,), np.float32)
    strands = np.zeros((MAX_READS,), np.float32)
    r = 0
    for read in bam.fetch(CHROM, win_start, win_stop):
        if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate:
            continue
        seq = read.query_sequence
        if seq is None:
            continue
        if r >= MAX_READS:
            break
        qual = read.query_qualities
        for qpos, rpos in read.get_aligned_pairs(matches_only=True):
            col = rpos - win_start
            if 0 <= col < WINDOW:
                base_idx = BASE_TO_IDX.get(seq[qpos].upper())
                if base_idx is not None:
                    reads[r, col, base_idx] = 1.0
                    quals[r, col] = float(qual[qpos]) if qual is not None else 30.0
        mapqs[r] = float(read.mapping_quality)
        strands[r] = 1.0 if read.is_reverse else 0.0
        r += 1
    if r < 3:
        return None  # too few reads to form a pileup
    ref_onehot = np.zeros((WINDOW, 4), np.float32)
    for i, base in enumerate(ref.fetch(CHROM, win_start, win_stop).upper()):
        base_idx = BASE_TO_IDX.get(base)
        if base_idx is not None:
            ref_onehot[i, base_idx] = 1.0
    return {
        "reads": reads,
        "reference": ref_onehot,
        "base_qualities": quals,
        "mapping_qualities": mapqs,
        "strands": strands,
        "positions": np.zeros((MAX_READS,), np.int32),
    }


def build_split(split: str, region: tuple[int, int], rng: np.random.Generator) -> dict:
    """Collect a balanced set of positive/negative candidate windows for a region."""
    bam = pysam.AlignmentFile(BAM)
    ref = pysam.FastaFile(REF)
    truth = load_truth(CHROM, *region)
    print(f"[{split}] truth variants in region: {len(truth)}", flush=True)

    cands = list(find_candidates(bam, ref, *region))
    rng.shuffle(cands)
    print(f"[{split}] raw candidates: {len(cands)}", flush=True)

    cap = PER_SPLIT[split] // 2
    pos_windows, neg_windows, labels = [], [], []
    for pos0 in cands:
        is_pos = pos0 in truth
        bucket = pos_windows if is_pos else neg_windows
        if len(bucket) >= cap:
            if len(pos_windows) >= cap and len(neg_windows) >= cap:
                break
            continue
        window = extract_window(bam, ref, pos0)
        if window is None:
            continue
        bucket.append(window)
        labels.append(1 if is_pos else 0)
    windows = pos_windows + neg_windows
    labels = [1] * len(pos_windows) + [0] * len(neg_windows)
    order = rng.permutation(len(windows))
    print(f"[{split}] kept pos={len(pos_windows)} neg={len(neg_windows)}", flush=True)

    stacked = {key: np.stack([windows[i][key] for i in order]) for key in windows[0]}
    stacked["labels"] = np.asarray([labels[i] for i in order], np.int32)
    return stacked


def main() -> None:
    """Extract train and test splits and save them to a single npz."""
    rng = np.random.default_rng(0)
    train = build_split("train", TRAIN_REGION, rng)
    test = build_split("test", TEST_REGION, rng)
    payload = {f"train_{k}": v for k, v in train.items()}
    payload.update({f"test_{k}": v for k, v in test.items()})
    np.savez(OUT, **payload)
    print(
        f"saved {OUT}: train {train['labels'].shape[0]} "
        f"(pos {int(train['labels'].sum())}) test {test['labels'].shape[0]} "
        f"(pos {int(test['labels'].sum())})",
        flush=True,
    )
    print("EXTRACT DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
