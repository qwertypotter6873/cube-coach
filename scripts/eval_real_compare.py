"""실제 프레임 정답으로 모델·입력 설정을 견준다 (모델 버전 고르기용, 기본은 검증 세트).

eval_real.py 와 같은 지표(정답 칸 중심 자리의 분류 = 정답 색)를 쓰고, 기준(지금 앱 모델·기본 입력)과의 차이를
프레임 단위로 짝지어 부트스트랩한다(프레임을 복원 추출 4,000번). 채택 규칙: 95% 구간의 아래 끝이 0보다 크다.

실행: .venv/bin/python scripts/eval_real_compare.py --set val \\
        --model v2=models/cube_seg_net.pt:24 --model w32=models/exp/cube_seg_net_w32.pt:32 [--zoom 1.0 0.8] [--size 192 256]
  --set ""  : 시험셋(data/real/label_frames, labels.json) — 버전을 고른 뒤 마지막에 한 번만 쓴다
  --zoom z  : 라벨 프레임(큐브가 절반쯤 차게 자른 정사각)의 가운데 z 배만 잘라 넣는다 (z<1 이면 큐브가 더 크게)
  --size s  : 모델 입력 한 변 (기본 192, 모델은 합성곱만이라 크기를 바꿀 수 있다)
출력: reports/eval_real_compare_<set>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score

from cubecoach.render3d import N_CLASSES
from cubecoach.segnet import CubeSegNet, maps_to

ROOT = Path(__file__).resolve().parents[1]


def frames_of(name: str):
    sfx = f"_{name}" if name else ""
    d = ROOT / "data" / "real" / f"label_frames{sfx}"
    labels = json.loads((ROOT / "data" / "real" / f"labels{sfx}.json").read_text())
    frames = json.loads((d / "frames.json").read_text())
    used = [f for f in frames if f["file"] in labels and not labels[f["file"]]["skip"] and labels[f["file"]]["points"]]
    return d, labels, used


def predict(model, img: np.ndarray, zoom: float, size: int, resample: int = 0) -> np.ndarray:
    h = img.shape[0]
    if zoom != 1.0:
        side = int(round(h * zoom))
        if zoom < 1.0:
            o = (h - side) // 2
            img = img[o:o + side, o:o + side]
        else:
            pad = (side - h) // 2
            img = cv2.copyMakeBorder(img, pad, side - h - pad, pad, side - h - pad, cv2.BORDER_CONSTANT, value=0)
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        o = model(torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255)[0]
    if resample:  # 앱처럼 지도를 resample x resample 로 넓이 평균해 줄인 뒤 읽는다
        logp, heat = maps_to(torch.log_softmax(o[:N_CLASSES], 0).numpy(), torch.sigmoid(o[N_CLASSES]).numpy(), resample)
        return logp.argmax(0)
    return o[:N_CLASSES].argmax(0).numpy()


def correct_cells(cls: np.ndarray, pts: np.ndarray, zoom: float) -> np.ndarray:
    n = cls.shape[0]
    q = (pts - 0.5) / zoom + 0.5  # 잘라 넣은 그림에서의 정답 점 자리
    ok = (q >= 0).all(1) & (q < 1).all(1)
    ij = np.clip((q * n).astype(int), 0, n - 1)
    pr = cls[ij[:, 1], ij[:, 0]]
    return np.where(ok, pr, 7)  # 잘려 나간 점은 '그 밖'으로 읽은 것으로 친다


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="val")
    ap.add_argument("--model", action="append", required=True, help="이름=가중치경로:폭 (첫 번째가 기준)")
    ap.add_argument("--zoom", type=float, nargs="+", default=[1.0])
    ap.add_argument("--size", type=int, nargs="+", default=[192])
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--resample", type=int, default=0, help="지도를 이 크기로 줄인 뒤 읽는다 (앱의 256 입력 + 48 지도 흉내)")
    args = ap.parse_args()
    d, labels, used = frames_of(args.set)
    imgs = [cv2.cvtColor(cv2.imread(str(d / f["file"])), cv2.COLOR_BGR2RGB) for f in used]
    pts = [np.array([[p["x"], p["y"]] for p in labels[f["file"]]["points"]]) for f in used]
    cols = [np.array([p["c"] for p in labels[f["file"]]["points"]]) for f in used]
    print(f"세트 '{args.set or '시험'}': 프레임 {len(used)}장, 칸 {sum(len(c) for c in cols)}개", flush=True)

    rows = {}
    for spec in args.model:
        name, rest = spec.split("=", 1)
        path, width = rest.rsplit(":", 1)
        m = CubeSegNet(width=int(width))
        m.load_state_dict(torch.load(ROOT / path, map_location="cpu", weights_only=True))
        m.eval()
        for zoom in args.zoom:
            for size in args.size:
                key = f"{name} zoom{zoom:g} size{size}" + (f"→{args.resample}" if args.resample and size != 192 else "")
                hits, preds = [], []
                for img, p, c in zip(imgs, pts, cols):
                    pr = correct_cells(predict(m, img, zoom, size, args.resample if size != 192 else 0), p, zoom)
                    hits.append(pr == c)
                    preds.append(pr)
                yt, yp = np.concatenate(cols), np.concatenate(preds)
                rows[key] = {"hits": hits, "acc": float((yt == yp).mean()),
                             "f1": float(f1_score(yt, yp, labels=list(range(6)), average="macro", zero_division=0)),
                             "missed_as_other_or_body": float(np.isin(yp, [6, 7]).mean())}
                print(f"{key:<28} 칸 색 정확도 {rows[key]['acc']:.3f}  F1 {rows[key]['f1']:.3f}  "
                      f"몸체·그 밖으로 놓침 {rows[key]['missed_as_other_or_body']:.3f}", flush=True)

    base_key = next(iter(rows))
    rng = np.random.default_rng(0)
    nf = len(used)
    idx = rng.integers(0, nf, size=(args.boot, nf))
    out = {"세트": args.set or "시험", "프레임": nf, "칸": int(sum(len(c) for c in cols)), "기준": base_key, "결과": {}}
    b_hits = rows[base_key]["hits"]
    for key, r in rows.items():
        res = {k: v for k, v in r.items() if k != "hits"}
        if key != base_key:
            # 프레임별 (맞힌 칸 수 차이, 칸 수) → 복원 추출한 프레임 묶음에서 정확도 차이
            dh = np.array([h.sum() - b.sum() for h, b in zip(r["hits"], b_hits)], float)
            nc = np.array([len(b) for b in b_hits], float)
            diffs = dh[idx].sum(1) / nc[idx].sum(1)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            res.update({"차이": float(dh.sum() / nc.sum()), "차이_95%": [float(lo), float(hi)],
                        "차이가_0_이하일_비율": float((diffs <= 0).mean()), "채택_조건(아래 끝 > 0)": bool(lo > 0)})
            print(f"  {key} - 기준: {res['차이']:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]  채택 조건 {'충족' if lo > 0 else '미충족'}")
        out["결과"][key] = res
    path = ROOT / "reports" / f"eval_real_compare_{args.set or 'test'}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(path)


if __name__ == "__main__":
    main()
