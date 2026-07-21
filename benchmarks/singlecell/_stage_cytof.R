#!/usr/bin/env Rscript
# Stage HDCytoData CyTOF gating datasets (Samusik_all, Levine_32dim) to flat CSVs.
#
# Emits one CSV per dataset: marker columns + population_id (gated label) + sample_id
# (donor). Consumed by benchmarks/singlecell/_prep_cytof.py, which converts to the .npz
# schema used by cytof_gating.py. Marker orientation and raw-vs-transformed values are
# reported to stdout so the Python side can be validated against them.
#
# Env:
#   R_CYTOF_LIB  personal R library to install into (created if missing)
#   CYTOF_OUT    output directory for the CSVs

lib <- Sys.getenv("R_CYTOF_LIB", unset = file.path(path.expand("~"), ".R", "cytof-lib"))
out <- Sys.getenv("CYTOF_OUT", unset = file.path(path.expand("~"), "cytof-staging"))
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
dir.create(out, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 3600)

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", lib = lib)
}
if (!requireNamespace("HDCytoData", quietly = TRUE)) {
  BiocManager::install("HDCytoData", update = FALSE, ask = FALSE, lib = lib)
}

suppressPackageStartupMessages({
  library(HDCytoData)
  library(SummarizedExperiment)
})

export_dataset <- function(se, name, path) {
  rd <- as.data.frame(rowData(se))
  cd <- as.data.frame(colData(se))
  # Keep only the 'type' lineage markers used for gating; drop 'none' channels
  # (Time, Cell_length, DNA, beads, viability) and any 'state' markers.
  is_type <- if ("marker_class" %in% colnames(cd)) cd$marker_class == "type" else rep(TRUE, ncol(se))
  expr <- assay(se)[, is_type, drop = FALSE]
  cat(sprintf("[%s] all channels %d -> type markers %d\n", name, ncol(se), ncol(expr)))
  cat(sprintf("[%s] type-marker value range: [%.3f, %.3f] (raw intensities are ~0..1e4)\n",
              name, min(expr), max(expr)))
  cat(sprintf("[%s] markers: %s\n", name, paste(colnames(expr), collapse = ", ")))
  cat(sprintf("[%s] populations: %s\n", name,
              paste(sort(unique(as.character(rd$population_id))), collapse = " | ")))
  df <- as.data.frame(expr)
  colnames(df) <- make.names(colnames(df), unique = TRUE)
  label_col <- if ("population_id" %in% colnames(rd)) "population_id" else colnames(rd)[1]
  donor_col <- if ("sample_id" %in% colnames(rd)) "sample_id" else
    if ("patient_id" %in% colnames(rd)) "patient_id" else colnames(rd)[1]
  df$population_id <- as.character(rd[[label_col]])
  df$sample_id <- as.character(rd[[donor_col]])
  cat(sprintf("[%s] populations: %d | donors: %d\n", name,
              length(unique(df$population_id)), length(unique(df$sample_id))))
  write.csv(df, path, row.names = FALSE)
  cat(sprintf("[%s] wrote %s\n", name, path))
}

export_dataset(Samusik_all_SE(), "samusik", file.path(out, "samusik.csv"))
export_dataset(Levine_32dim_SE(), "levine32", file.path(out, "levine32.csv"))
cat("STAGED OK\n")
