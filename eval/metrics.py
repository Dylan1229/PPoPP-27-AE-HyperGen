#!/usr/bin/env python
"""
PSNR / SSIM / LPIPS between a reference video and a test video.

For cache comparisons the reference is the *no-cache* output of the SAME model
(measures how much caching changed the output; see README.md, "Metrics").

Usage:
    python metrics.py --ref ref.mp4 --test test.mp4            # prints: psnr,ssim,lpips
    python metrics.py --ref r.mp4 --test t.mp4 --csv           # just "psnr,ssim,lpips"

Env: $ENV_ROOT/metrics (env/setup_envs.sh metrics)
"""
import argparse, sys
import numpy as np


def read_frames(path):
    import imageio.v3 as iio
    return [np.asarray(f)[..., :3] for f in iio.imiter(path, plugin="pyav")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--csv", action="store_true", help="print only 'psnr,ssim,lpips'")
    a = ap.parse_args()

    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn
    import torch, lpips

    ref, test = read_frames(a.ref), read_frames(a.test)
    n = min(len(ref), len(test))
    if n == 0:
        print("ERROR: no frames", file=sys.stderr); sys.exit(1)
    if len(ref) != len(test):
        print(f"WARN: frame count {len(ref)} vs {len(test)}, using first {n}", file=sys.stderr)

    loss = lpips.LPIPS(net="alex")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    loss = loss.to(dev)

    ps, ss, lp = [], [], []
    for i in range(n):
        r, t = ref[i], test[i]
        if r.shape != t.shape:  # resize test to ref if needed
            from skimage.transform import resize
            t = (resize(t, r.shape, preserve_range=True)).astype(np.uint8)
        ps.append(psnr_fn(r, t, data_range=255))
        ss.append(ssim_fn(r, t, channel_axis=2, data_range=255))
        def to_t(x):
            return (torch.from_numpy(x).float().permute(2, 0, 1)[None] / 127.5 - 1).to(dev)
        with torch.no_grad():
            lp.append(float(loss(to_t(r), to_t(t)).item()))

    P, S, L = float(np.mean(ps)), float(np.mean(ss)), float(np.mean(lp))
    if a.csv:
        print(f"{P:.3f},{S:.3f},{L:.3f}")
    else:
        print(f"PSNR={P:.3f}  SSIM={S:.3f}  LPIPS={L:.3f}  (over {n} frames)")


if __name__ == "__main__":
    main()
