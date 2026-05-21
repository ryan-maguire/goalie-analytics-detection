"""
Train a binary per-second shot classifier on HockeyAI YOLO features.

Inputs:
  - Feature TSVs from util/extract_yolo_features.py, one per video
  - GT CSVs (data/ground_truth/gt_<hudl>.csv) with "Shots" action rows

For each second of each video, label = 1 iff that second falls inside
a GT "Shots" window for the target team only. (target_filter parity
with cv_seg: opponent shots are NOT positives.)

Model: logistic regression (sklearn). Evaluated via leave-one-video-out
cross-validation. Per-fold metrics + aggregate ROC AUC + F1 at a tuned
threshold.

Output:
  - Trained model joblib (per fold + aggregate)
  - Per-second probability TSV for each video (`<vID>_probs.tsv`),
    suitable for converting to window predictions and comparing to
    cv_seg's motion-based output

Usage:
    python3 util/train_shot_classifier.py \\
        --features-dir data/output/yolo_features \\
        --gt-dir data/ground_truth \\
        --customers-file data/customers/all.json \\
        --out-dir data/output/shot_classifier
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


# vID -> hudl id mapping is the eval script's canonical table (eval/
# eval_cv_seg_output.py:VID_TO_HUDL). Import to stay in sync.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.eval_cv_seg_output import VID_TO_HUDL  # noqa: E402


def _load_vid_to_target_team(customer_paths: list[str]) -> dict[str, str]:
    """Returns {vID: targetGoalieTeamName}."""
    out: dict[str, str] = {}
    for p in customer_paths:
        with open(p) as f:
            cfg = json.load(f)
        for rec in cfg:
            vid = str(rec.get("vID", "")).strip()
            target = rec.get("targetGoalieTeamName") or ""
            if vid:
                out[vid] = target
    return out


def _load_gt_shot_seconds(gt_csv: str, target_team: str) -> set[int]:
    """Set of seconds inside any GT 'Shots' window for the target team."""
    secs: set[int] = set()
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            if row.get("action", "").strip().lower() != "shots":
                continue
            team = row.get("team", "").strip()
            if target_team and team != target_team:
                continue
            try:
                s = int(float(row["start"])); e = int(float(row["end"]))
            except (ValueError, KeyError):
                continue
            for t in range(s, e + 1):
                secs.add(t)
    return secs


def _load_features(tsv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (times, X, feature_names) — X is per-second feature matrix."""
    with open(tsv_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        cols = [c for c in header if c != "t"]
        times, X = [], []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            t = float(parts[0])
            row = []
            for v in parts[1:]:
                if v == "":
                    row.append(0.0)
                else:
                    try:
                        row.append(float(v))
                    except ValueError:
                        row.append(0.0)
            times.append(int(t))
            X.append(row)
    return np.asarray(times), np.asarray(X, dtype=float), cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True,
                    help="dir containing <vID>.tsv from extract_yolo_features.py")
    ap.add_argument("--gt-dir",       required=True)
    ap.add_argument("--customers",    nargs="+", required=True,
                    help="one or more customer JSON files (for vID->hudl mapping)")
    ap.add_argument("--out-dir",      required=True)
    ap.add_argument("--target-team",  default=None,
                    help="if set, only count shots for this team string in GT")
    args = ap.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    except ImportError:
        print("ERROR: scikit-learn required. pip install scikit-learn",
              file=sys.stderr); return 2
    try:
        import joblib
    except ImportError:
        print("ERROR: joblib required. pip install joblib",
              file=sys.stderr); return 2

    os.makedirs(args.out_dir, exist_ok=True)
    target_team_of = _load_vid_to_target_team(args.customers)
    print(f"Loaded target-team for {len(target_team_of)} vIDs; "
          f"hudl table has {len(VID_TO_HUDL)} entries", file=sys.stderr)

    # Collect per-video (X, y, times) — keyed by vID
    data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
    for tsv in sorted(Path(args.features_dir).glob("*.tsv")):
        vid = tsv.stem
        if vid not in VID_TO_HUDL:
            print(f"  skip {vid}: not in VID_TO_HUDL", file=sys.stderr); continue
        hudl = VID_TO_HUDL[vid]
        target = args.target_team or target_team_of.get(vid, "")
        gt_csv = os.path.join(args.gt_dir, f"gt_{hudl}.csv")
        if not os.path.exists(gt_csv):
            print(f"  skip {vid}: no GT at {gt_csv}", file=sys.stderr); continue
        gt_secs = _load_gt_shot_seconds(gt_csv, target)
        times, X, cols = _load_features(str(tsv))
        y = np.asarray([1 if int(t) in gt_secs else 0 for t in times], dtype=int)
        data[vid] = (X, y, times, cols)
        print(f"  {vid}: n={len(y)}  pos={y.sum()} ({y.mean():.1%})  "
              f"target_team={target!r}", file=sys.stderr)

    if len(data) < 2:
        print("ERROR: need >=2 videos for LOO CV", file=sys.stderr); return 2
    vids = sorted(data.keys())
    cols = next(iter(data.values()))[3]
    print(f"Features: {cols}", file=sys.stderr)

    fold_results = []
    all_probs: dict[str, np.ndarray] = {}
    for held_out in vids:
        X_train = np.vstack([data[v][0] for v in vids if v != held_out])
        y_train = np.concatenate([data[v][1] for v in vids if v != held_out])
        X_test, y_test, times_test, _ = data[held_out]

        scaler = StandardScaler().fit(X_train)
        Xtr = scaler.transform(X_train)
        Xte = scaler.transform(X_test)

        # class_weight balanced because positives are ~5-10% of seconds
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(Xtr, y_train)
        probs = clf.predict_proba(Xte)[:, 1]
        try:
            auc = roc_auc_score(y_test, probs) if y_test.sum() > 0 else float("nan")
        except ValueError:
            auc = float("nan")
        # F1 at threshold 0.5 (sanity baseline)
        preds = (probs >= 0.5).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )
        fold_results.append({
            "vID": held_out, "n": len(y_test), "pos": int(y_test.sum()),
            "auc": float(auc), "p": float(p), "r": float(r), "f1": float(f),
        })
        all_probs[held_out] = probs
        print(f"  fold {held_out}: AUC={auc:.3f}  "
              f"P={p:.3f} R={r:.3f} F1@0.5={f:.3f}", file=sys.stderr)

        # Write per-second probabilities for this held-out video
        out_tsv = os.path.join(args.out_dir, f"{held_out}_probs.tsv")
        with open(out_tsv, "w") as f:
            f.write("t\tprob\tlabel\n")
            for ti, pr, yi in zip(times_test, probs, y_test):
                f.write(f"{int(ti)}\t{pr:.4f}\t{int(yi)}\n")

    # Aggregate
    aucs = [r["auc"] for r in fold_results if not np.isnan(r["auc"])]
    print(f"\n=== Leave-one-video-out summary ===")
    print(f"  mean AUC: {np.mean(aucs):.3f} (std {np.std(aucs):.3f})")
    print(f"  per-fold:")
    for r in fold_results:
        print(f"    {r['vID']}: AUC={r['auc']:.3f}  P={r['p']:.3f}  R={r['r']:.3f}  F1@0.5={r['f1']:.3f}")

    # Train final model on all data (for downstream use)
    X_all = np.vstack([data[v][0] for v in vids])
    y_all = np.concatenate([data[v][1] for v in vids])
    scaler = StandardScaler().fit(X_all)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(scaler.transform(X_all), y_all)
    final_path = os.path.join(args.out_dir, "model_final.joblib")
    joblib.dump({"scaler": scaler, "clf": clf, "feature_cols": cols}, final_path)
    print(f"  final model: {final_path}")
    # Coefficient summary — which features matter most?
    coefs = sorted(zip(cols, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    print(f"  top feature weights:")
    for name, w in coefs[:8]:
        print(f"    {name:>22}  {w:+.3f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
