"""실제 촬영 프레임에서 인식 모델들을 비교한다 (정답: scripts/label_real.py 로 찍은 칸 중심·색).

지표
- 칸 색 정확도: 정답 칸 중심 자리에서 모델이 고른 분류가 정답 색과 같은 비율 (손·몸체로 고르면 틀림)
- 칸 색 macro-F1: 여섯 색 각각의 F1 평균 (빨강·주황처럼 드문 쌍의 실수를 드러낸다)
- 중심 재현율: 정답 칸 중심 근처(3x3 칸)에 중심 지도 값이 0.3을 넘는 비율
- 배경 오검출률: 큐브에서 먼 곳(정답 점에서 칸 간격 1.2배 이상)을 스티커 색으로 읽은 비율
  (합성만으로 학습한 v1이 플리스를 파랑으로 읽던 문제를 재는 지표)

실행: .venv/bin/python scripts/eval_real.py
결과: reports/eval_real.json, reports/eval_real_confusion.png, reports/eval_real_examples.png
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from cubecoach.render3d import N_CLASSES  # noqa: E402
from cubecoach.segnet import CubeSegNet, PixelMLP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "real" / "label_frames"
NAMES = ["흰", "빨", "초", "노", "주", "파", "몸체", "그 밖"]
PAL = np.array([[244, 244, 240], [214, 35, 58], [4, 168, 70], [233, 224, 44], [250, 106, 44], [4, 100, 188], [70, 70, 70], [0, 0, 0]], np.uint8)
plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]


def load(cls, path: Path):
    m = cls()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


def hsv_rule(img: np.ndarray, n: int) -> np.ndarray:
    """학습 없는 참고선 (train_seg.hsv_reference 와 같은 규칙). '그 밖'이 없어 배경도 색으로 읽는다."""
    small = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    h, s, v = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).transpose(2, 0, 1)
    p = np.full((n, n), 5)
    p[h < 180] = 2
    p[h < 75] = 3
    p[h < 38] = 4
    p[(h < 12) | (h >= 330)] = 1
    p[(s < 0.25) & (v > 0.45)] = 0
    return p


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "data" / "real" / "labels.json"))
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()
    OUT = Path(args.out)
    frames = json.loads((DIR / "frames.json").read_text())
    labels = json.loads(Path(args.labels).read_text())
    models = {
        "HSV 규칙 (학습 없음)": None,
        "PixelMLP v2 (베이스라인)": load(PixelMLP, ROOT / "models" / "pixel_mlp.pt"),
        "CubeSegNet v1 (합성만)": load(CubeSegNet, ROOT / "models" / "v1" / "cube_seg_net.pt"),
        "CubeSegNet v2 (실제 맞춤 합성)": load(CubeSegNet, ROOT / "models" / "cube_seg_net.pt"),
    }
    used = [f for f in frames if f["file"] in labels and not labels[f["file"]]["skip"] and labels[f["file"]]["points"]]
    print(f"정답 있는 프레임 {len(used)}장, 칸 {sum(len(labels[f['file']]['points']) for f in used)}개")
    out: dict[str, dict] = {}
    preds: dict[str, list] = {}
    for name, model in models.items():
        y_true, y_pred, near_heat = [], [], []
        fp = bg = 0
        per_frame = []
        for f in used:
            img = cv2.cvtColor(cv2.imread(str(DIR / f["file"])), cv2.COLOR_BGR2RGB)
            x = cv2.resize(img, (192, 192), interpolation=cv2.INTER_AREA)
            if model is None:
                cls, heat = hsv_rule(x, 48), None
            else:
                with torch.no_grad():
                    o = model(torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255)[0]
                cls = o[:N_CLASSES].argmax(0).numpy()
                heat = torch.sigmoid(o[N_CLASSES]).numpy()
            n = cls.shape[0]
            pts = np.array([[p["x"], p["y"]] for p in labels[f["file"]]["points"]])
            col = np.array([p["c"] for p in labels[f["file"]]["points"]])
            ij = np.clip((pts * n).astype(int), 0, n - 1)
            pr = cls[ij[:, 1], ij[:, 0]]
            y_true += col.tolist()
            y_pred += pr.tolist()
            if heat is not None:
                pad = np.pad(heat, 1)
                near_heat += [pad[i + dy, j + dx] for (j, i) in ij for dy in range(3) for dx in range(3)]
            # 배경: 정답 점에서 칸 간격의 1.2배보다 먼 곳
            d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)) + np.eye(len(pts)) * 9
            spacing = float(np.median(d.min(1))) if len(pts) > 1 else 0.1
            gy, gx = (np.mgrid[0:n, 0:n] + 0.5) / n
            far = np.sqrt(((np.stack([gx, gy], -1)[:, :, None] - pts[None, None]) ** 2).sum(-1)).min(-1) > 1.2 * spacing
            fp += int(((cls < 6) & far).sum())
            bg += int(far.sum())
            per_frame.append({"file": f["file"], "acc": float((pr == col).mean()), "cls": cls})
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        r = {
            "sticker_acc": float((y_true == y_pred).mean()),
            "sticker_f1": float(f1_score(y_true, y_pred, labels=list(range(6)), average="macro", zero_division=0)),
            "background_fp": fp / max(bg, 1),
            "n_stickers": int(len(y_true)),
        }
        if near_heat:
            r["center_recall"] = float((np.array(near_heat).reshape(-1, 9).max(1) > 0.3).mean())
        out[name] = r
        preds[name] = per_frame
        print(f"{name:<28} 칸 색 정확도 {r['sticker_acc']:.3f}  F1 {r['sticker_f1']:.3f}  "
              f"배경 오검출 {r['background_fp']:.3f}" + (f"  중심 재현율 {r['center_recall']:.3f}" if "center_recall" in r else ""))
        if name.startswith("CubeSegNet v2"):
            cm = confusion_matrix(y_true, y_pred, labels=list(range(8)))[:6]
            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
            for i in range(6):
                for j in range(8):
                    if cm[i, j]:
                        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9, color="white" if cm[i, j] > cm[i].sum() / 2 else "black")
            ax.set_xticks(range(8), NAMES)
            ax.set_yticks(range(6), NAMES[:6])
            ax.set_xlabel("모델이 읽은 것")
            ax.set_ylabel("정답 색")
            ax.set_title("실제 프레임 칸 색 혼동행렬 (CubeSegNet v2)")
            fig.tight_layout()
            fig.savefig(OUT / "eval_real_confusion.png", dpi=150)
    (OUT / "eval_real.json").write_text(json.dumps({"frames": len(used), "results": out}, ensure_ascii=False, indent=2))

    # 오류 사례: v2가 가장 많이 틀린 프레임 4장과 가장 잘 맞힌 2장 (원본 / v1 / v2)
    v1, v2 = preds["CubeSegNet v1 (합성만)"], preds["CubeSegNet v2 (실제 맞춤 합성)"]
    order = np.argsort([p["acc"] for p in v2])
    pick = list(order[:4]) + list(order[-2:])
    rows = []
    for i in pick:
        img = cv2.cvtColor(cv2.imread(str(DIR / v2[i]["file"])), cv2.COLOR_BGR2RGB)
        tile = lambda c: cv2.resize(PAL[c], (240, 240), interpolation=cv2.INTER_NEAREST)
        im = cv2.resize(img, (240, 240), interpolation=cv2.INTER_AREA)
        for p in labels[v2[i]["file"]]["points"]:
            cv2.circle(im, (int(p["x"] * 240), int(p["y"] * 240)), 5, tuple(int(v) for v in PAL[p["c"]]), -1)
            cv2.circle(im, (int(p["x"] * 240), int(p["y"] * 240)), 5, (0, 0, 0), 1)
        row = np.hstack([im, tile(v1[i]["cls"]), tile(v2[i]["cls"])])
        cv2.putText(row, f"v1 {v1[i]['acc']:.2f}", (248, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(row, f"v2 {v2[i]['acc']:.2f}", (488, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        rows.append(row)
    grid = np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows), 2)])
    cv2.imwrite(str(OUT / "eval_real_examples.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
