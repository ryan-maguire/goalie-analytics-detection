"""Audio-only sliding-window inference. Mirrors temporal_shot/infer.py."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO       = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import load_features, N_FEATURES  # noqa: E402
from model   import AudioShotHead              # noqa: E402


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(arg)


def predict_video(feats: np.ndarray, model, device,
                   seq_len: int = 32, stride: int = 4,
                   batch: int = 256) -> np.ndarray:
    T = feats.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.float32)
    pad_n = max(0, seq_len - T)
    if pad_n > 0:
        feats = np.concatenate(
            [feats, np.zeros((pad_n, feats.shape[1]), dtype=feats.dtype)],
            axis=0)
    T_pad = feats.shape[0]
    starts = list(range(0, max(1, T_pad - seq_len + 1), stride))
    if starts[-1] + seq_len < T_pad:
        starts.append(T_pad - seq_len)

    sum_p = np.zeros(T_pad, dtype=np.float32)
    cnt   = np.zeros(T_pad, dtype=np.float32)
    feats_t = torch.from_numpy(feats).float()
    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            chunk_starts = starts[i:i + batch]
            chunk = torch.stack([feats_t[s:s + seq_len] for s in chunk_starts])
            chunk = chunk.to(device)
            logits = model(chunk)
            probs  = torch.sigmoid(logits).cpu().numpy()
            for j, s in enumerate(chunk_starts):
                sum_p[s:s + seq_len] += probs[j]
                cnt[s:s + seq_len]   += 1
    avg = sum_p / np.maximum(cnt, 1e-6)
    return avg[:T].astype(np.float32)


def write_probs_tsv(out_path: Path, probs: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("t\tshot_max_conf\tshot_n\tgoal_max_conf\n")
        for t, p in enumerate(probs):
            f.write(f"{t}\t{p:.4f}\t{int(p > 0.5)}\t0.0\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",      required=True, type=Path)
    ap.add_argument("--vIDs",         nargs="+", required=True)
    ap.add_argument("--features-dir", type=Path,
                    default=REPO / "data" / "output" / "audio_features")
    ap.add_argument("--out-dir",      type=Path, required=True)
    ap.add_argument("--seq-len",      type=int, default=32)
    ap.add_argument("--stride",       type=int, default=4)
    ap.add_argument("--batch",        type=int, default=256)
    ap.add_argument("--device",       default="cpu")
    ap.add_argument("--hidden",       type=int, default=64)
    ap.add_argument("--n-layers",     type=int, default=2)
    ap.add_argument("--dropout",      type=float, default=0.25)
    args = ap.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    hidden   = saved.get("hidden",   args.hidden)
    n_layers = saved.get("n_layers", args.n_layers)
    dropout  = saved.get("dropout",  args.dropout)
    seq_len  = saved.get("seq_len",  args.seq_len)

    model = AudioShotHead(in_features=N_FEATURES, hidden=hidden,
                           n_layers=n_layers, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded {args.weights}  (epoch {ckpt.get('epoch')}, "
          f"val_loss={ckpt.get('val_loss'):.4f})", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for vid in args.vIDs:
        feats = load_features(args.features_dir / f"{vid}.tsv")
        if feats.shape[0] == 0:
            print(f"  [skip] no features for {vid}", file=sys.stderr)
            continue
        probs = predict_video(feats, model, device,
                               seq_len=seq_len, stride=args.stride,
                               batch=args.batch)
        write_probs_tsv(args.out_dir / f"{vid}.tsv", probs)
        n_hot = int((probs > 0.5).sum())
        print(f"  {vid}: {feats.shape[0]} secs → {n_hot} hot-secs (thr 0.5)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
