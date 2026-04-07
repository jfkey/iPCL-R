#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Compare evaluation metrics across multiple models.

Usage:
    python -m flow.evaluation.eval_metric_compare \
        --csv  baselineSmall=/mnt/local_data1/liweiguo/experiments/model_size_comparison/work_dir/Small-DecimalWordLevel/stage_evaluation/metrics/evaluation_metrics.csv \
             baselineMedium=/mnt/local_data1/liweiguo/experiments/model_size_comparison/work_dir/Medium-DecimalWordLevel/stage_evaluation/metrics/evaluation_metrics.csv \
             baselineLarge=/mnt/local_data1/liweiguo/experiments/model_size_comparison/work_dir/Large-DecimalWordLevel/stage_evaluation/metrics/evaluation_metrics.csv \
             ourLargeE13=/mnt/local_data1/liujunfeng/exp/Large-GeoPE/stage_evaluation/model_wope-checkpoint-145041/metrics/evaluation_metrics.csv \
             ourLargeE10=/mnt/local_data1/liujunfeng/exp/Large-GeoPE/stage_evaluation/model_wope-checkpoint-111570/metrics/evaluation_metrics.csv \
             ourMedium=/mnt/local_data1/liujunfeng/exp/Medium-GeoPE/stage_evaluation/model_wope_check/metrics/evaluation_metrics.csv \
             ourMedium40=/mnt/local_data1/liujunfeng/exp/Medium-GeoPE/stage_evaluation/model_wope40/metrics/evaluation_metrics.csv \
             ourwogeope=/mnt/local_data1/liujunfeng/exp/Medium-GeoPE/stage_evaluation/model_wogeope40/metrics/evaluation_metrics.csv \
             ourwo_bias=/mnt/local_data1/liujunfeng/exp/Medium-GeoPE/stage_evaluation/model_wope_wo_bias/metrics/evaluation_metrics.csv \
             ourwo_v=/mnt/local_data1/liujunfeng/exp/Medium-GeoPE/stage_evaluation/model_wope_wo_v/metrics/evaluation_metrics.csv \
        --output /mnt/local_data1/liujunfeng/exp/Large-GeoPE/eval_compare.csv

        
        python -m flow.evaluation.eval_metric_compare \
        --csv  baselineMedium=/mnt/local_data1/liweiguo/experiments/model_size_comparison/work_dir/Medium-DecimalWordLevel/stage_evaluation/metrics/evaluation_metrics.csv \
             newBaseline=/mnt/local_data1/liujunfeng/exp/Medium-Refine/stage_evaluation/check_baseline_from40/metrics/evaluation_metrics.csv \
             onlygeope=/mnt/local_data1/liujunfeng/exp/Medium-Refine/stage_evaluation/recheck_baseline/metrics/evaluation_metrics.csv \
             newslefcross=/mnt/local_data1/liujunfeng/exp/Medium-Refine/stage_evaluation/recheck_geope_cross_epoch_80/metrics/evaluation_metrics.csv \
             newslefcrossnoise=/mnt/local_data1/liujunfeng/exp/Medium-Refine/stage_evaluation/recheck_geope_cross_noise_epoch_80/metrics/evaluation_metrics.csv \
             newslefcross2=/mnt/local_data1/liujunfeng/exp/Medium-Refine/stage_evaluation/recheck_geope_cross_epoch_96/metrics/evaluation_metrics.csv \
             newslefcrossnoise2=/mnt/local_data1/liujunfeng/exp/Medium-Refine/stage_evaluation/recheck_geope_cross_noise_epoch_96/metrics/evaluation_metrics.csv \
             
             
             
    # Or use glob pattern:
    python -m flow.evaluation.eval_metric_compare \
        --glob "/mnt/local_data1/liujunfeng/exp/Medium-GeoPE/stage_evaluation/*/metrics/evaluation_metrics.csv" \
        [--output /path/to/output_dir]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_metrics(csv_paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    """Load evaluation_metrics.csv files into a dict of DataFrames."""
    dfs = {}
    for name, path in csv_paths.items():
        path = Path(path)
        if not path.exists():
            logging.warning(f"File not found, skipping: {path}")
            continue
        df = pd.read_csv(path)
        if len(df) < 5:
            logging.warning(f"Dataset '{name}' has only {len(df)} samples, skipping")
            continue
        dfs[name] = df
        logging.info(f"Loaded '{name}': {len(df)} samples from {path}")
    return dfs


def compute_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Compute summary statistics from a single evaluation DataFrame."""
    total = len(df)
    s = {}

    # --- Evaluation Summary ---
    s["total_samples"] = total
    s["Perfect Sequence Match (%)"] = df["is_perfect_match"].sum() / total * 100
    s["Branch Structure Match (%)"] = df["is_branch_struct_match"].sum() / total * 100
    s["Leaf Set Match (%)"] = df["is_leaf_set_match"].sum() / total * 100
    s["All Loads Connected (%)"] = df["is_connected_all_loads"].sum() / total * 100

    s["Avg Leaf Accuracy (%)"] = df["leaf_accuracy"].mean() * 100
    s["Avg Leaf Precision (%)"] = df["leaf_precision"].mean() * 100
    s["Avg Leaf Recall (%)"] = df["leaf_recall"].mean() * 100
    s["Avg Leaf IoU (%)"] = df["leaf_iou"].mean() * 100

    s["Avg Edge Accuracy (%)"] = df["edge_accuracy"].mean() * 100
    s["Avg Edge Precision (%)"] = df["edge_precision"].mean() * 100
    s["Avg Edge Recall (%)"] = df["edge_recall"].mean() * 100
    s["Avg Edge IoU (%)"] = df["edge_iou"].mean() * 100

    # NLP metrics
    s["Avg ROUGE-L F1"] = df["rougeL_f"].mean()
    s["Avg BLEU-1"] = df["bleu_1"].mean()
    s["Avg BLEU-2"] = df["bleu_2"].mean()
    s["Avg BLEU-4"] = df["bleu_4"].mean()

    # Physical metrics
    via_pred = df["num_vias_pred"].sum()
    via_true = df["num_vias_true"].sum()
    wl_pred = df["wirelength_pred"].sum()
    wl_true = df["wirelength_true"].sum()
    # delay_pred = df["max_elmore_delay_pred"].sum()
    # delay_true = df["max_elmore_delay_true"].sum()

    s["Via (Pred/GT) (%)"] = via_pred / via_true * 100 if via_true > 0 else 0.0
    s["Wirelength (Pred/GT) (%)"] = wl_pred / wl_true * 100 if wl_true > 0 else 0.0
    s["Avg Via Ratio"] = df["via_ratio"].mean()
    s["Avg Wirelength Ratio"] = df["wirelength_ratio"].mean()
    # s["Max Elmore Delay (Pred/GT) (%)"] = (
    #     delay_pred / delay_true * 100
    #     if delay_true > 0
    #     else (100.0 if delay_pred == 0 else 0.0)
    # )
    s["RED Similarity Score"] = df["red_similarity_score"].mean()

    return s


def compare(dfs: Dict[str, pd.DataFrame], output_dir: Path | None = None):
    """Compare metrics across models and print a summary table."""
    all_stats = {}
    for name, df in dfs.items():
        all_stats[name] = compute_stats(df)

    # Build comparison DataFrame: rows=metrics, columns=models
    comparison = pd.DataFrame(all_stats)

    # Print to console
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    print(comparison.to_string())
    print("=" * 100 + "\n")

    # Save to file
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "model_comparison.csv"
        comparison.to_csv(csv_path)
        logging.info(f"Comparison table saved to: {csv_path}")


def parse_csv_args(csv_args: list[str]) -> Dict[str, Path]:
    """Parse name=path pairs from --csv arguments."""
    result = {}
    for arg in csv_args:
        if "=" in arg:
            name, path = arg.split("=", 1)
        else:
            # Use parent directory name as model name
            p = Path(arg)
            name = p.parent.parent.name  # e.g. .../model_wope/metrics/eval.csv -> model_wope
            path = arg
        result[name] = Path(path)
    return result


def glob_csv_paths(pattern: str) -> Dict[str, Path]:
    """Resolve glob pattern to name->path dict, using grandparent dir as name."""
    import glob

    paths = sorted(glob.glob(pattern))
    result = {}
    for p in paths:
        p = Path(p)
        name = p.parent.parent.name  # e.g. .../model_wope/metrics/eval.csv -> model_wope
        result[name] = p
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation metrics across models"
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        help="Metric CSV files as name=path pairs (or just paths)",
    )
    parser.add_argument(
        "--glob",
        type=str,
        help="Glob pattern to find evaluation_metrics.csv files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for comparison results",
    )

    args = parser.parse_args()

    if not args.csv and not args.glob:
        parser.error("Provide either --csv or --glob")

    csv_paths = {}
    if args.glob:
        csv_paths.update(glob_csv_paths(args.glob))
    if args.csv:
        csv_paths.update(parse_csv_args(args.csv))

    if not csv_paths:
        logging.error("No CSV files found")
        sys.exit(1)

    dfs = load_metrics(csv_paths)
    if not dfs:
        logging.error("No valid DataFrames loaded")
        sys.exit(1)

    compare(dfs, output_dir=Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
