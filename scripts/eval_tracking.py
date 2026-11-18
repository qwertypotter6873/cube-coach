"""합성 영상으로 연속 추적을 평가한다: 인식 모델 → 추적기 → 회전 판정.

실행: .venv/bin/python scripts/eval_tracking.py [--seqs 12] [--moves 20] [--no-fingers] [--video 0]
결과: reports/eval_tracking.json  (+ --video N 이면 N번 영상을 reports/track_N.mp4 로 저장)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from cubecoach.cube import MOVES
from cubecoach.geometry import CENTERS, NORMALS, rotation_angle, rotvec
from cubecoach.render3d import N_CLASSES, real_bg, render
from cubecoach.segnet import CubeSegNet, load_segnet
from cubecoach.simseq import simulate
from cubecoach.synth import PALETTE
from cubecoach.tracker import Event, Tracker

ROOT = Path(__file__).resolve().parents[1]


def _closure(gens: list[np.ndarray], compose) -> list[np.ndarray]:
    """생성원들로 만들 수 있는 모든 원소 (compose(None, None)이 항등원)."""
    out: list[np.ndarray] = []
    frontier = [compose(None, None)]
    while frontier:
        g = frontier.pop()
        if any(np.array_equal(g, h) for h in out):
            continue
        out.append(g)
        frontier += [compose(g, x) for x in gens]
    return out


# 큐브를 통째로 돌리는 24가지: 칸 순열과, 같은 순서의 회전 행렬
ROT_PERMS = _closure([MOVES["x"], MOVES["y"]], lambda g, x: np.arange(54) if g is None else g[x])
ROT_MATS = _closure(
    [rotvec(np.array([np.pi / 2, 0, 0])), rotvec(np.array([0, np.pi / 2, 0]))],
    lambda g, x: np.eye(3) if g is None else np.round(g @ x),
)


def merge(events: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """같은 층을 연달아 돌린 것은 하나로 합친다 (90도 두 번 = 180도 한 번)."""
    out: list[tuple[int, int]] = []
    for c, k in events:
        if out and out[-1][0] == c:
            k = (out.pop()[1] + k) % 4
            if k == 0:
                continue
        out.append((c, k))
    return out


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def overlay(img: np.ndarray, tr: Tracker, roi, text: str) -> np.ndarray:
    """추적 중인 자세로 54칸 중심을 영상 위에 찍는다 (기대 색으로)."""
    big = cv2.resize(img, (384, 384), interpolation=cv2.INTER_NEAREST)
    if tr.R is not None:
        Xc = CENTERS @ tr.R.T + tr.T
        nc = NORMALS @ tr.R.T
        facing = -(nc * Xc).sum(1) / np.linalg.norm(Xc, axis=1)
        px = ((tr.f * Xc[:, :2] / Xc[:, 2:3] - np.array(roi[:2])) / roi[2] + 0.5) * 384
        for i in np.nonzero(facing > 0.12)[0]:
            col = tuple(int(v * 255) for v in PALETTE[tr.colors[i]])
            cv2.circle(big, (int(px[i, 0]), int(px[i, 1])), 5, (0, 0, 0), -1)
            cv2.circle(big, (int(px[i, 0]), int(px[i, 1])), 3, col, -1)
    cv2.putText(big, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(big, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return big


class TsTracker:
    """브라우저용 TS 추적기(web/src/vision/tracker.ts)를 node 로 돌려 파이썬 Tracker 와 같은 모양으로 쓴다."""

    def __init__(self, colors, f: float = 0.95, seed: int = 0, cover: bool = False, settle: int = 0,
                 fast: bool = False, bank: int = 2500, follow: list[str] | None = None, aspect: float = 1.0,
                 opts: dict | None = None) -> None:
        import base64, json as _json, subprocess
        self._b64, self._json = base64.b64encode, _json
        self.p = subprocess.Popen(["node", "scripts/track_bridge.ts"], cwd=ROOT / "web", stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self._apply(self._ask({"init": [int(c) for c in colors], "f": f, "seed": seed, "cover": cover,
                               "settle": settle, "fast": fast, "bank": bank, "follow": follow or [], "aspect": aspect, "opts": opts or {}}))
        self.ms: list[float] = []

    def _ask(self, msg: dict) -> dict:
        self.p.stdin.write(self._json.dumps(msg) + "\n")
        self.p.stdin.flush()
        return self._json.loads(self.p.stdout.readline())

    def _apply(self, r: dict) -> None:
        self.lost_roi = tuple(r["lostRoi"]) if r.get("lostRoi") else None
        # 놓쳤을 때는 추적기가 정한 찾을 영역 (앱과 같다; 합성 영상은 정사각형이라 기본은 (0, 0, 1))
        self._roi = tuple(r["roi"]) if r["roi"] else (self.lost_roi or (0.0, 0.0, 1.0))
        self.moving = r.get("moving", False)
        self.R = None if r["R"] is None else np.array(r["R"]).reshape(3, 3)
        self.T = None if r.get("T") is None else np.array(r["T"])
        self.diag = r.get("diag") or {}
        self.colors = np.array(r["colors"])
        self.quality, self.layer, self.phi = r["quality"], r["layer"], r["phi"]
        self.events = [Event(e[0], e[1], e[2]) for e in r["events"]]
        self.event_info = [e[3] if len(e) > 3 else None for e in r["events"]]
        self.coverage, self.relocks = r.get("coverage", 1.0), r.get("relocks", 0)
        self.local_relocks = r.get("localRelocks", 0)
        self.inits = r.get("inits", 0)
        self.rejected_turns = r.get("rejectedTurns", 0)
        self.off_rejected = r.get("offRejected", 0)
        self.reverts = r.get("reverts", 0)
        self.view = r.get("view")
        if "ms" in r:
            self.ms.append(r["ms"])

    def roi(self):
        return self._roi

    def update(self, logp, heat, roi):
        n = heat.shape[0]
        b = lambda a: self._b64(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()
        before = len(self.events)
        self._apply(self._ask({"n": n, "logp": b(logp), "heat": b(heat), "roi": [float(v) for v in roi]}))
        return self.events[before:]

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=12)
    ap.add_argument("--moves", type=int, default=20)
    ap.add_argument("--no-fingers", action="store_true")
    ap.add_argument("--any-view", action="store_true", help="한 면만 보이는 자세까지 포함 (어려운 조건)")
    ap.add_argument("--video", type=int, default=-1)
    ap.add_argument("--turn-frames", type=int, nargs=2, default=[5, 11], help="90도 회전 하나의 프레임 수 범위 [최소, 최대)")
    ap.add_argument("--pause", type=int, nargs=2, default=[3, 10], help="회전 사이 멈춤 프레임 수 범위 [최소, 최대)")
    ap.add_argument("--no-regrip", action="store_true", help="큐브 통째 회전·고쳐 쥐기 없이 연속으로만 돌린다")
    ap.add_argument("--tag", default="", help="결과 파일 이름에 붙일 꼬리표")
    ap.add_argument("--oracle", action="store_true", help="인식 모델 대신 정답 지도를 넣는다 (추적기만 따로 검증)")
    ap.add_argument("--weights", default=str(ROOT / "models" / "cube_seg_net.pt"))
    ap.add_argument("--ts", action="store_true", help="파이썬 추적기 대신 브라우저용 TS 추적기를 돌린다 (이식 검증)")
    ap.add_argument("--fast", action="store_true", help="(TS) 큐브를 옮길 때 빨리 따라가기 켜기")
    ap.add_argument("--bank", type=int, default=2500, help="(TS) 전체 탐색 회전 뱅크 크기 (앱은 1500)")
    ap.add_argument("--opts", default="", help='(TS) 추적기 설정 JSON, 예 {"turnVsBody": 8}')
    args = ap.parse_args()
    if real_bg()[0] is None and os.environ.get("ALLOW_NO_REAL_BG") != "1":
        # 합성 영상의 배경도 v2 렌더러가 실제 촬영 배경 조각으로 그린다. 없으면 보고한 것과 다른 영상이 된다.
        raise SystemExit("data/real/bg_patches.npy 가 없습니다 (scripts/extract_real_bg.py, 촬영 영상 필요). "
                         "합성 배경만으로 재려면 ALLOW_NO_REAL_BG=1.")

    torch.set_num_threads(2)
    model = CubeSegNet() if args.oracle else load_segnet(args.weights)  # 폭은 가중치에서 알아낸다
    model.eval()

    rows = []
    for s in range(args.seqs):
        rng = np.random.default_rng(5000 + s)
        noise = np.random.default_rng(9000 + s)
        tracker: Tracker | None = None
        gt, det, pose_err, t_net, t_trk = [], [], [], [], []
        writer = None
        if s == args.video:
            writer = cv2.VideoWriter(str(ROOT / "reports" / f"track_{s}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (384, 384))
        last = None
        sim = simulate(rng, args.moves, fingers=not args.no_fingers, any_view=args.any_view,
                       turn_frames=tuple(args.turn_frames), pause_frames=tuple(args.pause), regrip=not args.no_regrip)
        for i, step in enumerate(sim):
            if tracker is None:
                tracker = (TsTracker(step.scene.colors, f=0.95, seed=s, fast=args.fast, bank=args.bank, opts=json.loads(args.opts) if args.opts else None)
                           if args.ts else Tracker(step.scene.colors, f=0.95, seed=s))
            roi = tracker.roi()
            fr = render(step.scene, roi, noise)
            t0 = time.perf_counter()
            if args.oracle:
                prob = np.full((N_CLASSES,) + fr.seg.shape, 0.03 / (N_CLASSES - 1), dtype=np.float32)
                np.put_along_axis(prob, fr.seg[None].astype(np.int64), 0.97, axis=0)
                logp, heat = np.log(prob), fr.heat
            else:
                with torch.no_grad():
                    out = model(torch.from_numpy(fr.img).permute(2, 0, 1)[None].float().div(255))[0]
                logp = torch.log_softmax(out[:N_CLASSES], 0).numpy()
                heat = torch.sigmoid(out[N_CLASSES]).numpy()
            t1 = time.perf_counter()
            events = tracker.update(logp, heat, roi)
            t2 = time.perf_counter()
            t_net.append(t1 - t0)
            t_trk.append(t2 - t1)
            gt += [(c, k, i) for c, k in step.done]
            if tracker.R is not None:
                pose_err.append(min(rotation_angle(step.scene.R, tracker.R @ G) for G in ROT_MATS))
            else:
                pose_err.append(180.0)
            if writer is not None:
                txt = f"f{i} q={tracker.quality:.2f} gt={len(gt)} det={len(tracker.events)}"
                writer.write(cv2.cvtColor(overlay(fr.img, tracker, roi, txt), cv2.COLOR_RGB2BGR))
            last = step
        if writer is not None:
            writer.release()

        if hasattr(tracker, "close"):
            tracker.close()
        det = [(e.center, e.quarter_turns, e.frame - 1) for e in tracker.events]  # 사후 삽입까지 반영된 순서
        g, d = merge([(c, k) for c, k, _ in gt]), merge([(c, k) for c, k, _ in det])
        final_ok = any(np.array_equal(tracker.colors[r], last.scene.colors) for r in ROT_PERMS)
        delays = [fd - fg for (_, _, fg), (_, _, fd) in zip(gt, det)] if [x[:2] for x in gt] == [x[:2] for x in det] else []
        err = np.array(pose_err)
        rows.append({
            "seq": s, "frames": len(err), "moves": len(g), "detected": len(d),
            "exact": g == d, "move_acc": 1 - edit_distance(g, d) / max(len(g), 1), "final_state_ok": bool(final_ok),
            "lock_ratio": float((err < 12).mean()), "median_pose_err": float(np.median(err)),
            "mean_delay": float(np.mean(delays)) if delays else None,
            "ms_net": float(np.median(t_net) * 1000), "ms_tracker": float(np.median(t_trk) * 1000),
        })
        r = rows[-1]
        print(f"영상 {s:2d}: {r['frames']}프레임  회전 {r['moves']}개 중 판정 {r['detected']}개  정확도 {r['move_acc']:.2f}  "
              f"완전일치 {r['exact']}  최종상태 {r['final_state_ok']}  자세유지 {r['lock_ratio']:.2f}  "
              f"지연 {r['mean_delay']}  추적 {r['ms_tracker']:.1f}ms  모델 {r['ms_net']:.1f}ms", flush=True)

    summary = {
        "sequences": len(rows),
        "move_acc": float(np.mean([r["move_acc"] for r in rows])),
        "exact_rate": float(np.mean([r["exact"] for r in rows])),
        "final_state_rate": float(np.mean([r["final_state_ok"] for r in rows])),
        "lock_ratio": float(np.mean([r["lock_ratio"] for r in rows])),
        "mean_delay_frames": float(np.mean([r["mean_delay"] for r in rows if r["mean_delay"] is not None] or [np.nan])),
        "ms_tracker": float(np.median([r["ms_tracker"] for r in rows])),
        "ms_net": float(np.median([r["ms_net"] for r in rows])),
        "fingers": not args.no_fingers, "any_view": args.any_view, "oracle": args.oracle,
        "turn_frames": args.turn_frames, "pause": args.pause,
        "frames_per_move": float(np.mean([r["frames"] / max(r["moves"], 1) for r in rows])),
    }
    out = ROOT / "reports" / f"eval_tracking{'_' + args.tag if args.tag else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print("\n요약:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
