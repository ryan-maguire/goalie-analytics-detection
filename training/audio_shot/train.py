"""Train the audio-only shot detector. Mirrors temporal_shot/train.py
but on audio features."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO       = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import AudioShotDataset, N_FEATURES   # noqa: E402
from model   import AudioShotHead, count_params    # noqa: E402


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(arg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-vids", nargs="+", required=True)
    ap.add_argument("--features-dir", type=Path,
                    default=REPO / "data" / "output" / "audio_features")
    ap.add_argument("--gt-dir",       type=Path,
                    default=REPO / "data" / "ground_truth")
    ap.add_argument("--out-dir",      type=Path, required=True)
    ap.add_argument("--seq-len",      type=int, default=32)
    ap.add_argument("--stride",       type=int, default=4)
    ap.add_argument("--hidden",       type=int, default=64)
    ap.add_argument("--n-layers",     type=int, default=2)
    ap.add_argument("--dropout",      type=float, default=0.25)
    ap.add_argument("--epochs",       type=int, default=50)
    ap.add_argument("--batch",        type=int, default=64)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--val-frac",     type=float, default=0.10)
    ap.add_argument("--seed",         type=int, default=7)
    ap.add_argument("--device",       default="cpu",
                    help="audio model is tiny — CPU is fastest")
    ap.add_argument("--label-pad",    type=int, default=1)
    ap.add_argument("--window-diff",  type=int, default=8)
    ap.add_argument("--patience",     type=int, default=8,
                    help="early stop after this many epochs of no val_loss "
                         "improvement (set 0 to disable). Audio is small/fast "
                         "so default 8 is more generous than B's 5.")
    ap.add_argument("--min-epochs",   type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"device: {device}", file=sys.stderr)

    train_frac_end = 1.0 - args.val_frac
    print(f"\nloading audio sequences from {len(args.train_vids)} videos "
          f"(train=(0,{train_frac_end:.2f}) / val=({train_frac_end:.2f},1) "
          f"per video)…", file=sys.stderr)
    train_ds = AudioShotDataset(
        args.train_vids, args.features_dir, args.gt_dir,
        seq_len=args.seq_len, stride=args.stride,
        window_diff=args.window_diff, label_pad=args.label_pad,
        time_window=(0.0, train_frac_end),
    )
    val_ds = AudioShotDataset(
        args.train_vids, args.features_dir, args.gt_dir,
        seq_len=args.seq_len, stride=args.stride,
        window_diff=args.window_diff, label_pad=args.label_pad,
        time_window=(train_frac_end, 1.0),
    )
    print(f"  train: {len(train_ds)} sequences", file=sys.stderr)
    for vid, st in train_ds.vid_stats.items():
        print(f"    {vid}: window={st['window']} pos_secs={st['pos_s']} "
              f"n_seq={st['n_seq']}", file=sys.stderr)
    print(f"  val:   {len(val_ds)} sequences", file=sys.stderr)
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("ERROR: empty split", file=sys.stderr)
        return 2

    total = sum(int(y.numel()) for _, y in train_ds)
    pos   = sum(int(y.sum().item()) for _, y in train_ds)
    pos_w = max(1.0, (total - pos) / max(1, pos))
    print(f"  pos={pos}  neg={total - pos}  pos_weight={pos_w:.2f}",
          file=sys.stderr)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)

    model = AudioShotHead(in_features=N_FEATURES, hidden=args.hidden,
                           n_layers=args.n_layers, dropout=args.dropout).to(device)
    print(f"model: {count_params(model)} params", file=sys.stderr)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_rows = []
    best_val = float("inf")
    best_path = args.out_dir / "best.pt"
    no_improve_count = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        for step_i, (x, y) in enumerate(train_loader):
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tot_loss += loss.item() * x.size(0)
        train_loss = tot_loss / len(train_ds)

        model.eval()
        v_loss = 0.0; tp = fp = fn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                logits = model(x)
                v_loss += criterion(logits, y).item() * x.size(0)
                pred = (torch.sigmoid(logits) > 0.5).float()
                tp += int(((pred == 1) & (y == 1)).sum().item())
                fp += int(((pred == 1) & (y == 0)).sum().item())
                fn += int(((pred == 0) & (y == 1)).sum().item())
        v_loss /= max(1, len(val_ds))
        vP = tp / max(1, tp + fp); vR = tp / max(1, tp + fn)
        vF1 = 2 * vP * vR / (vP + vR) if (vP + vR) else 0.0
        log_rows.append({"epoch": epoch, "train_loss": train_loss,
                          "val_loss": v_loss, "val_p": vP,
                          "val_r": vR, "val_f1": vF1})
        print(f"  epoch {epoch:>3}  train={train_loss:.4f}  val={v_loss:.4f}  "
              f"val-per-tick P={vP:.3f} R={vR:.3f} F1={vF1:.3f}",
              file=sys.stderr)
        if v_loss < best_val:
            best_val = v_loss
            no_improve_count = 0
            args_dict = {k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()}
            torch.save({
                "model_state": model.state_dict(),
                "args":        args_dict,
                "epoch":       epoch,
                "val_loss":    v_loss,
                "val_f1":      vF1,
                "n_features":  N_FEATURES,
            }, best_path)
        else:
            no_improve_count += 1
            if (args.patience > 0 and epoch >= args.min_epochs
                    and no_improve_count >= args.patience):
                print(f"  early stop at epoch {epoch} "
                      f"(no improvement in {args.patience} epochs)",
                      file=sys.stderr)
                break

    (args.out_dir / "train_log.json").write_text(json.dumps({
        "rows": log_rows, "best_val": best_val,
        "elapsed_s": time.time() - t0,
    }, indent=2))
    print(f"\nsaved → {best_path}  (best val={best_val:.4f}, "
          f"{(time.time()-t0)/60:.1f} min)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
