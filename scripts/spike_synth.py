"""합성 데이터만으로 하는 사전 검증.

묻는 것은 세 가지다.
1. 후보 모델들이 실제로 얼마나 가벼운가 (가중치 수, ONNX 크기, CPU 추론 시간)
2. 면 전체를 보는 모델이 칸 하나만 보는 방식보다 나은가
3. 학습 → ONNX 내보내기 → onnxruntime 추론까지 파이프라인이 끝까지 도는가

여기서 나온 정확도는 합성 이미지 기준이라 실제 성능이 아니다.
최종 수치는 직접 촬영한 테스트셋으로 다시 낸다.

실행: PYTHONPATH=src .venv/bin/python scripts/spike_synth.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch import nn

from cubecoach.models import (
    FaceGridNet,
    StickerCNN,
    cell_patches,
    color_features,
    hsv_rule,
    mobilenet_grid,
    n_params,
)
from cubecoach.synth import make_dataset

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "synth"
SEED = 42
SPLITS = {  # 이름: (장수, 시드, 난이도)
    "train": (12000, 1000, 1.0),
    "val": (1500, 2000, 1.0),
    "test": (2000, 3000, 1.0),
    "test_hard": (2000, 4000, 1.6),  # 학습 때보다 조명·정렬이 나쁜 조건
}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _chunk(args: tuple[int, int, float]) -> tuple[np.ndarray, np.ndarray]:
    return make_dataset(*args[:2], difficulty=args[2])


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    n, seed, diff = SPLITS[name]
    path = CACHE / f"{name}.npz"
    if path.exists():
        d = np.load(path)
        return d["x"], d["y"]
    jobs = [(n // 8, seed + i, diff) for i in range(8)]
    with ProcessPoolExecutor(8) as ex:
        parts = list(ex.map(_chunk, jobs))
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y)
    return x, y


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 → (N, 3, H, W) uint8. 실수 변환은 배치를 장치로 올릴 때 한다."""
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


def to_device(x: torch.Tensor) -> torch.Tensor:
    return x.to(DEVICE).float().div(255)


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """pred, y: (N, 9)."""
    return {
        "sticker_acc": float((pred == y).mean()),
        "macro_f1": float(f1_score(y.ravel(), pred.ravel(), average="macro")),
        "face_exact": float((pred == y).all(axis=1).mean()),
    }


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(model(to_device(x[i : i + bs])).argmax(1).cpu())
    return torch.cat(out).numpy()


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, xv: torch.Tensor, yv: np.ndarray,
          epochs: int, lr: float, bs: int, name: str) -> list[dict]:
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(x) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(SEED)
    log = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x), generator=gen)
        total, t0 = 0.0, time.time()
        for i in range(len(x) // bs):
            idx = perm[i * bs : (i + 1) * bs]
            loss = loss_fn(model(to_device(x[idx])), y[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        val_acc = float((predict(model, xv).reshape(yv.shape) == yv).mean())
        log.append({"epoch": ep, "loss": total / (len(x) // bs), "val_sticker_acc": val_acc})
        print(f"  [{name}] epoch {ep:2d}/{epochs}  loss {log[-1]['loss']:.4f}  val acc {val_acc:.4f}  ({time.time() - t0:.1f}s)")
    return log


def export_and_time(model: nn.Module, shape: tuple[int, ...], path: Path) -> dict[str, float]:
    """ONNX로 내보내고, PyTorch와 출력이 같은지 확인한 뒤, CPU 1스레드 추론 시간을 잰다."""
    model = model.cpu().eval()
    dummy = torch.rand(shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (dummy,), str(path), input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    inp = dummy.numpy()
    got = sess.run(None, {"image": inp})[0]
    diff = float(np.abs(got - model(dummy).detach().numpy()).max())
    for _ in range(20):
        sess.run(None, {"image": inp})
    times = []
    for _ in range(200):
        t = time.perf_counter()
        sess.run(None, {"image": inp})
        times.append((time.perf_counter() - t) * 1000)
    return {"onnx_kb": path.stat().st_size / 1024, "ms_per_face_cpu1": float(np.median(times)), "onnx_max_diff": diff}


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()
    data = {k: load_split(k) for k in SPLITS}
    print(f"데이터 준비 {time.time() - t0:.1f}s:", {k: v[0].shape[0] for k, v in data.items()})
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    results: dict[str, dict] = {}
    tests = ("test", "test_hard")

    # 0) 학습 없는 참고선: HSV 임계값
    results["HSV 규칙 (학습 없음)"] = {
        t: metrics(np.stack([hsv_rule(im) for im in data[t][0]]), data[t][1]) for t in tests
    }

    # 1) 고전 ML 베이스라인: 칸 평균색 → kNN
    feats = np.concatenate([color_features(im) for im in xtr[:4000]])
    knn = KNeighborsClassifier(n_neighbors=7).fit(feats, ytr[:4000].ravel())
    results["kNN (칸 평균색)"] = {
        t: metrics(knn.predict(np.concatenate([color_features(im) for im in data[t][0]])).reshape(-1, 9), data[t][1])
        for t in tests
    }

    # 2) 베이스라인 CNN: 칸 하나만 본다
    sticker = StickerCNN()
    ptr, pva = to_tensor(cell_patches(xtr)), to_tensor(cell_patches(xva))
    log_s = train(sticker, ptr, torch.from_numpy(ytr.ravel()), pva, yva.ravel(), epochs=6, lr=3e-3, bs=512, name="StickerCNN")
    results["StickerCNN (칸 단위, scratch)"] = {
        t: metrics(predict(sticker, to_tensor(cell_patches(data[t][0]))).reshape(-1, 9), data[t][1]) for t in tests
    }
    results["StickerCNN (칸 단위, scratch)"]["log"] = log_s
    del ptr, pva

    # 3) 제안: 면 전체를 본다
    ttr, tva = to_tensor(xtr), to_tensor(xva)
    ytr_grid = torch.from_numpy(ytr.reshape(-1, 3, 3))
    grid = FaceGridNet()
    log_g = train(grid, ttr, ytr_grid, tva, yva.reshape(-1, 3, 3), epochs=20, lr=4e-3, bs=128, name="FaceGridNet")
    results["FaceGridNet (면 단위, scratch)"] = {
        t: metrics(predict(grid, to_tensor(data[t][0])).reshape(-1, 9), data[t][1]) for t in tests
    }
    results["FaceGridNet (면 단위, scratch)"]["log"] = log_g

    # 4) 전이학습 비교군
    try:
        mnet = mobilenet_grid(pretrained=True)
        log_m = train(mnet, ttr, ytr_grid, tva, yva.reshape(-1, 3, 3), epochs=8, lr=1.5e-3, bs=128, name="MobileNetV3-S")
        results["MobileNetV3-Small (면 단위, 전이학습)"] = {
            t: metrics(predict(mnet, to_tensor(data[t][0])).reshape(-1, 9), data[t][1]) for t in tests
        }
        results["MobileNetV3-Small (면 단위, 전이학습)"]["log"] = log_m
    except Exception as e:  # 사전학습 가중치를 못 받으면 건너뛴다
        print("MobileNetV3 건너뜀:", repr(e))
        mnet = None

    # 크기와 속도. StickerCNN은 한 면에 9칸을 한꺼번에 넣는 시간이다.
    sizes = {
        "StickerCNN (칸 단위, scratch)": (sticker, (9, 3, 32, 32), "sticker_cnn.onnx"),
        "FaceGridNet (면 단위, scratch)": (grid, (1, 3, 96, 96), "face_grid_net.onnx"),
    }
    if mnet is not None:
        sizes["MobileNetV3-Small (면 단위, 전이학습)"] = (mnet, (1, 3, 96, 96), "mobilenet_grid.onnx")
    for name, (model, shape, fname) in sizes.items():
        results[name]["params"] = n_params(model)
        results[name].update(export_and_time(model, shape, ROOT / "models" / "spike" / fname))

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "spike_synth.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n=== 합성 데이터 사전 검증 결과 ===")
    print(f"{'모델':<40}{'가중치':>9}{'ONNX':>9}{'ms/면':>8} | {'칸 정확도':>8}{'F1':>7}{'면 일치':>8} | {'(어려움) 칸':>10}{'면 일치':>8}")
    for name, r in results.items():
        p = f"{r['params']:,}" if "params" in r else "-"
        kb = f"{r['onnx_kb']:.0f}KB" if "onnx_kb" in r else "-"
        ms = f"{r['ms_per_face_cpu1']:.2f}" if "ms_per_face_cpu1" in r else "-"
        a, h = r["test"], r["test_hard"]
        print(f"{name:<40}{p:>9}{kb:>9}{ms:>8} | {a['sticker_acc']:>8.4f}{a['macro_f1']:>7.4f}{a['face_exact']:>8.4f} | {h['sticker_acc']:>10.4f}{h['face_exact']:>8.4f}")
    print(f"\n전체 {time.time() - t0:.0f}s, 장치 {DEVICE}")


if __name__ == "__main__":
    main()
