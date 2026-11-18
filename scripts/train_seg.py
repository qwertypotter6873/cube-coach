"""연속 추적용 인식 모델 학습: 베이스라인(PixelMLP)과 제안 모델(CubeSegNet).

실행: .venv/bin/python scripts/train_seg.py [--epochs 14]
결과: models/cube_seg_net.{pt,onnx}, models/pixel_mlp.{pt,onnx}, reports/train_seg.json
"""

from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import f1_score
from torch import nn

from cubecoach.models import n_params
from cubecoach.render3d import N_CLASSES
from cubecoach.segnet import CubeSegNet, PixelMLP, export_onnx, seg_loss

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / os.environ.get("SYNTH_DIR", "synth3d_v2")  # v1 은 synth3d
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def load(split: str) -> dict[str, np.ndarray]:
    return {k: np.load(DATA / f"{split}_{k}.npy", mmap_mode="r") for k in ("img", "seg", "heat", "centers", "vis", "colors")}


def to_input(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img)).to(DEVICE).permute(0, 3, 1, 2).float().div(255)


@torch.no_grad()
def infer(model: nn.Module, imgs: np.ndarray, bs: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """분류 예측 (N, H, W)과 중심 지도 (N, H, W)."""
    model.eval()
    cls, heat = [], []
    for i in range(0, len(imgs), bs):
        out = model(to_input(imgs[i : i + bs]))
        cls.append(out[:, :N_CLASSES].argmax(1).cpu().numpy().astype(np.uint8))
        heat.append(torch.sigmoid(out[:, N_CLASSES]).cpu().numpy())
    return np.concatenate(cls), np.concatenate(heat)


def sticker_targets(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """중심이 가려지지 않은 스티커들의 (이미지 번호, 행, 열, 색)."""
    n_lab = d["seg"].shape[1]
    c = np.floor(np.asarray(d["centers"])).astype(int)
    inside = (c >= 0).all(-1) & (c < n_lab).all(-1) & (np.asarray(d["vis"]) >= 0.5)
    img_i, k = np.nonzero(inside)
    y, x = c[img_i, k, 1], c[img_i, k, 0]
    color = np.asarray(d["colors"])[img_i, k]
    keep = np.asarray(d["seg"])[img_i, y, x] == color
    return img_i[keep], y[keep], x[keep], color[keep]


def metrics(cls: np.ndarray, heat: np.ndarray, d: dict[str, np.ndarray]) -> dict[str, float]:
    seg = np.asarray(d["seg"])
    cm = np.bincount(seg.ravel().astype(int) * N_CLASSES + cls.ravel(), minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)
    iou = np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1) - np.diag(cm), 1)
    i, y, x, color = sticker_targets(d)
    pred = cls[i, y, x]
    pad = np.pad(heat, ((0, 0), (1, 1), (1, 1)))
    near = np.max([pad[i, y + dy, x + dx] for dy in range(3) for dx in range(3)], axis=0)
    return {
        "pixel_acc": float(np.diag(cm).sum() / cm.sum()),
        "miou": float(iou.mean()),
        "sticker_acc": float((pred == color).mean()),
        "sticker_f1": float(f1_score(color, pred, labels=list(range(6)), average="macro", zero_division=0)),
        "center_recall": float((near > 0.3).mean()),
        "n_stickers": int(len(color)),
    }


def hsv_reference(d: dict[str, np.ndarray]) -> dict[str, float]:
    """학습 없는 참고선: 스티커 중심 칸의 평균색을 HSV 임계값으로 분류."""
    i, y, x, color = sticker_targets(d)
    n_lab = d["seg"].shape[1]
    pred = np.empty(len(color), dtype=np.int64)
    for s in range(0, len(d["img"]), 256):
        small = np.stack([cv2.resize(im, (n_lab, n_lab), interpolation=cv2.INTER_AREA) for im in d["img"][s : s + 256]])
        hsv = cv2.cvtColor((small.reshape(-1, n_lab, 3) / 255).astype(np.float32), cv2.COLOR_RGB2HSV).reshape(-1, n_lab, n_lab, 3)
        m = (i >= s) & (i < s + 256)
        h, sat, val = hsv[i[m] - s, y[m], x[m]].T
        p = np.full(m.sum(), 5)
        p[h < 180] = 2
        p[h < 75] = 3
        p[h < 38] = 4
        p[(h < 12) | (h >= 330)] = 1
        p[(sat < 0.25) & (val > 0.45)] = 0
        pred[m] = p
    return {"sticker_acc": float((pred == color).mean()),
            "sticker_f1": float(f1_score(color, pred, labels=list(range(6)), average="macro", zero_division=0))}


def train(model: nn.Module, name: str, tr: dict, va: dict, epochs: int, lr: float, bs: int) -> list[dict]:
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = len(tr["img"]) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    rng = np.random.default_rng(SEED)
    log = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(tr["img"]))
        tot_ce = tot_bce = 0.0
        t0 = time.time()
        for s in range(steps):
            idx = np.sort(perm[s * bs : (s + 1) * bs])
            out = model(to_input(tr["img"][idx]))
            seg = torch.from_numpy(tr["seg"][idx].astype(np.int64)).to(DEVICE)
            heat = torch.from_numpy(tr["heat"][idx]).to(DEVICE).float().div(255)
            ce, bce = seg_loss(out, seg, heat)
            opt.zero_grad()
            (ce + 2 * bce).backward()
            opt.step()
            sched.step()
            tot_ce += ce.item()
            tot_bce += bce.item()
        m = metrics(*infer(model, va["img"]), va)
        log.append({"epoch": ep, "ce": tot_ce / steps, "bce": tot_bce / steps, **{f"val_{k}": v for k, v in m.items()}})
        print(f"  [{name}] epoch {ep:2d}/{epochs}  ce {tot_ce / steps:.4f}  bce {tot_bce / steps:.4f}  "
              f"val mIoU {m['miou']:.4f}  스티커 정확도 {m['sticker_acc']:.4f}  중심 재현율 {m['center_recall']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return log


def export(model: nn.Module, path: Path) -> dict[str, float]:
    # 앱과 같은 모양(입력 크기를 열어 둠, 192·256 모두 확인)으로 내보낸다: cubecoach.segnet.export_onnx
    model = model.cpu().eval()
    diff = export_onnx(model, path)
    dummy = torch.rand(1, 3, 192, 192)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    times = []
    for i in range(120):
        t = time.perf_counter()
        sess.run(None, {"image": dummy.numpy()})
        if i >= 20:
            times.append((time.perf_counter() - t) * 1000)
    return {"params": n_params(model), "onnx_kb": path.stat().st_size / 1024, "ms_cpu1": float(np.median(times)), "onnx_max_diff": diff}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--only", choices=["all", "cube"], default="all", help="cube: 제안 모델만 (실험용)")
    ap.add_argument("--width", type=int, default=24, help="CubeSegNet 폭 (기본 24 = 지금 앱 모델)")
    ap.add_argument("--tag", default="", help="실험 꼬리표: 있으면 models/exp/<이름>_<꼬리표>.* 와 reports/train_seg_<꼬리표>.json 에 저장 (앱 모델을 덮어쓰지 않는다)")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    tr, va, te = load("train"), load("val"), load("test")
    print(f"train {len(tr['img'])} / val {len(va['img'])} / test {len(te['img'])}, 장치 {DEVICE}", flush=True)
    out_dir = ROOT / "models" / ("exp" if args.tag else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""
    results: dict[str, dict] = {"HSV 규칙 (학습 없음)": {"test": hsv_reference(te)}}
    if args.tag:
        results["설정"] = {"width": args.width, "epochs": args.epochs, "bs": args.bs, "data": str(DATA.name)}

    runs = [("CubeSegNet (제안)", CubeSegNet(width=args.width), args.epochs, 4e-3, "cube_seg_net")]
    if args.only == "all":
        runs.insert(0, ("PixelMLP (칸 평균색만, 베이스라인)", PixelMLP(), 4, 5e-3, "pixel_mlp"))
    for name, model, epochs, lr, fname in runs:
        log = train(model, name.split()[0], tr, va, epochs, lr, args.bs)
        test = metrics(*infer(model, te["img"]), te)
        torch.save(model.state_dict(), out_dir / f"{fname}{sfx}.pt")
        results[name] = {"test": test, "log": log, **export(model, out_dir / f"{fname}{sfx}.onnx")}
        print(f"  [{name}] test: {test}", flush=True)

    (ROOT / "reports" / f"train_seg{sfx}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("\n=== 테스트셋(합성) 결과 ===")
    for name, r in results.items():
        if "test" not in r:
            continue
        t = r["test"]
        extra = f"  mIoU {t['miou']:.4f}  중심 재현율 {t['center_recall']:.4f}  {r['params']:,}개  {r['onnx_kb']:.0f}KB  {r['ms_cpu1']:.1f}ms" if "miou" in t else ""
        print(f"{name:<36} 스티커 정확도 {t['sticker_acc']:.4f}  F1 {t['sticker_f1']:.4f}{extra}")


if __name__ == "__main__":
    main()
