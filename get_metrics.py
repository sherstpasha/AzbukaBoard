# -*- coding: utf-8 -*-
# AzbukaBoard — Unified Evaluation Script
# python get_metrics.py --submission "C:\Users\USER\EasyOcrAzbukaBoard\orig_cyrillic_cyrillic_g1.csv" --data-root "C:\shared\ocr_datasets"

import yaml
import csv
import os
import argparse
from evaluate import load
import unicodedata
import sys


# ---------------- NORMALIZATION ----------------

def normalize_text(text, rules):
    if rules.get("normalize_unicode"):
        text = unicodedata.normalize(rules["normalize_unicode"], text)

    if rules.get("lowercase"):
        text = text.lower()

    return text.strip()


# ---------------- LOAD GROUND TRUTH ----------------

def load_gt(csv_path, image_col, text_col, encoding, normalize_rules, has_header=True):
    gt = {}
    with open(csv_path, encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        
        if has_header:
            next(reader, None)

        for row in reader:
            if len(row) <= max(image_col, text_col):
                continue
            image = row[image_col].strip()
            text = normalize_text(row[text_col], normalize_rules)
            gt[image] = text

    return gt


# ---------------- LOAD PREDICTIONS ----------------

def load_predictions(submission_path, normalize_rules):
    pred = {}

    with open(submission_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            if len(row) < 2:
                continue
            image = row[0].strip()
            text = normalize_text(row[1], normalize_rules)
            pred[image] = text

    return pred


# ---------------- MAIN EVALUATION ----------------

def evaluate_datasets(config_path, submission_path, data_root):
    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cer_metric = load("cer")
    wer_metric = load("wer")

    # prediction normalization
    pred_normalize = {
        "lowercase": True,
        "normalize_unicode": "NFC",
    }

    predictions = load_predictions(submission_path, pred_normalize)

    results = {}

    # evaluate on all datasets in config
    for dataset_name, ds in config["datasets"].items():
        print(f"\nEvaluating dataset: {dataset_name}")

        csv_path = os.path.join(data_root, ds["csv"])

        if not os.path.exists(csv_path):
            print(f"[WARN] GT CSV not found -> skipping dataset: {csv_path}")
            continue

        gt = load_gt(
            csv_path,
            ds["image_column"],
            ds["text_column"],
            ds["encoding"],
            ds["normalize"],
            has_header=ds["has_header"]
        )

        y_true, y_pred = [], []

        for image, true_text in gt.items():
            if image in predictions:
                y_true.append(true_text)
                y_pred.append(predictions[image])
            else:
                print(f"[WARN] Missing prediction for: {image}")

        if not y_true:
            print(f"[ERROR] No matching predictions for dataset '{dataset_name}'")
            continue

        cer = cer_metric.compute(predictions=y_pred, references=y_true)
        wer = wer_metric.compute(predictions=y_pred, references=y_true)
        accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

        results[dataset_name] = {
            "cer": cer,
            "wer": wer,
            "accuracy": accuracy
        }

    # --------------------- OUTPUT -----------------------

    model_name = os.path.basename(submission_path).replace(".csv", "")

    print("\n====================================================")
    print(f"                 MODEL: {model_name}")
    print("====================================================")

    print(f"{'DATASET':35} {'CER':>10} {'WER':>10} {'ACC':>10}")
    print("-" * 70)

    for name, r in results.items():
        print(f"{name:35} {r['cer']:.4f}    {r['wer']:.4f}    {r['accuracy']:.4f}")

    print("-" * 70)

    # NOW: Markdown rows per dataset (for dataset-specific leaderboards)
    print("\n=== MARKDOWN ROWS PER DATASET ===")

    for name, r in results.items():
        md_row = f"| {model_name} | {r['cer']:.4f} | {r['wer']:.4f} | {r['accuracy']:.4f} |"
        print(f"{name}: {md_row}")

    print("\n[OK] Evaluation complete.\n")

    return results


# ---------------- CLI ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR predictions using AzbukaBoard datasets."
    )
    parser.add_argument(
        "--config",
        required=False,
        default=None,
        help="Path to config.yaml (optional, defaults to ./config.yaml)"
    )
    parser.add_argument(
        "--submission",
        required=True,
        help="Path to submission.csv (predictions)"
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Base directory where dataset folders reside"
    )

    args = parser.parse_args()

    # default config
    if args.config is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.config = os.path.join(script_dir, "config.yaml")
        print(f"[INFO] Using default config: {args.config}")
    else:
        print(f"[INFO] Using config: {args.config}")

    if not os.path.exists(args.config):
        print(f"[ERROR] config not found: {args.config}")
        sys.exit(1)

    if not os.path.exists(args.submission):
        print(f"[ERROR] submission not found: {args.submission}")
        sys.exit(1)

    if not os.path.isdir(args.data_root):
        print(f"[ERROR] invalid data_root: {args.data_root}")
        sys.exit(1)

    evaluate_datasets(args.config, args.submission, args.data_root)


if __name__ == "__main__":
    main()
