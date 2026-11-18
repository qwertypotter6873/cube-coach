"""입력 256 실험 갈래(추적 단계 입력 256, 지도는 넓이 평균으로 48x48, 면 읽기 192)를 지금 기본(입력 192)과 견준다.

README 5.2·5.4절의 입력 256 수치를 모은다. 출력: reports/track_real_input256_summary.json
  개발 영상(sexy_move.mov): A·B 구간과 들고 있기는 시드 15개(track_real_pf_*_{g10,r256m48}), 앱 흐름 7.7~90초 시드 10개
                           (track_real_gt2_app{gate,r256m48}), 나머지 구간 시드 5개(track_real_gt2_{gate,r256}_*),
                           샘플 mp4 앱 흐름 7.7~46초 시드 10개(track_real_mp4app_{g10,r256m48})
  시험 영상(latest·latest2): track_real_<영상>_{gate,r256}_<구간>_s0~4 (r256 은 시험셋 결과를 본 뒤 실험 갈래로 돌렸다,
                           scripts/summarize_latest.py 머리말)
  실제 프레임: reports/eval_real_compare_val_resample48.json(검증), reports/eval_real_compare_test.json(시험, 한 번)
실행: .venv/bin/python scripts/summarize_input256.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from track_video import LATEST2_L, LATEST2_R, LATEST_L, LATEST_R, SEXY_MOVE_GT, lcs  # noqa: E402

R = ROOT / "reports"
A, B = SEXY_MOVE_GT["A"][3], SEXY_MOVE_GT["B"][3]


def load(name: str) -> dict:
    return json.loads((R / f"track_real_{name}.json").read_text())


def paired(a: list[float], b: list[float]) -> dict:
    d = np.array(b) - np.array(a)
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
    return {"차이_평균": round(float(d.mean()), 3), "시드_짝_95%": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
            "256이_나은_시드": int((d > 0).sum()), "같은_시드": int((d == 0).sum()), "못한_시드": int((d < 0).sum())}


def accs(names: list[str]) -> dict:
    rs = [load(n) for n in names]
    a = [r["move_acc"] for r in rs]
    return {"평균": round(st.mean(a), 3), "중앙값": round(st.median(a), 3), "시드별": [round(x, 3) for x in a],
            "끝_상태_일치": sum(bool(r["final_state_ok"]) for r in rs), "입력": sorted({(r.get("res", 192), r.get("map_n", 0)) for r in rs})}


def inside(names: list[str], wins: list[tuple]) -> dict:
    """이어서 잰 구간: 동작 창 안 판정만 정답과 짝짓고(최장 공통 부분열), 창 밖 판정은 따로 센다."""
    m, o = [], []
    for n in names:
        ev = [(e[0], e[1], e[2]) for e in load(n)["events"]]
        ins = [[(c, k) for c, k, t in ev if t0 <= t <= t1] for t0, t1, _ in wins]
        m.append(sum(lcs(x, g) for x, (_, _, g) in zip(ins, wins)))
        o.append(len(ev) - sum(len(x) for x in ins))
    return {"창_안_정답과_짝지은_회전(시드별)": m, "평균": round(st.mean(m), 2), "창_밖_판정(시드별)": o, "창_밖_합": sum(o)}


def false(names: list[str]) -> dict:
    f = [load(n)["false_moves"] for n in names]
    return {"합": sum(f), "시드별": f}


def sexy_windows(names: list[str]) -> dict:
    hold, am, bm = [], [], []
    for n in names:
        ev = load(n)["events"]
        w = lambda a, b: [(e[0], e[1]) for e in ev if a <= e[2] < b]
        hold.append(len([e for e in ev if 8.2 <= e[2] < 19.5]))
        am.append(lcs(w(19.5, 27.5), A))
        bm.append(lcs(w(30.5, 90.0), B))
    return {"들고_있기_판정(8.2~19.5초)": hold, "A_짝지은_회전(24수)": am, "B_짝지은_회전": bm}


def main() -> None:
    out: dict = {"설명": __doc__.strip(), "개발_영상": {}, "시험_영상": {}, "실제_프레임": {}}
    dev = out["개발_영상"]
    for seg in ("A", "B"):
        g = [load(f"pf_{seg}_g10_s{s}")["move_acc"] for s in range(15)]
        r = [load(f"pf_{seg}_r256m48_s{s}")["move_acc"] for s in range(15)]
        dev[f"{seg}(시드 15개)"] = {"192": round(st.mean(g), 3), "256": round(st.mean(r), 3), **paired(g, r)}
    for tag, key in (("g10", "192"), ("r256m48", "256")):
        raw = [len([e for e in load(f"pf_hold_{tag}_s{s}")["events"] if 8.2 <= e[2] <= 19.5]) for s in range(15)]
        dev.setdefault("들고_있기(7.7~19.5초 시작, 시드 15개)", {})[key] = {"판정_평균": round(st.mean(raw), 2), "시드별": raw}
    g = [load(f"gt2_appgate_s{s}")["move_acc"] for s in range(10)]
    r = [load(f"gt2_appr256m48_s{s}")["move_acc"] for s in range(10)]
    dev["앱_흐름(7.7~90초, 시드 10개)"] = {"192": accs([f"gt2_appgate_s{s}" for s in range(10)]) | sexy_windows([f"gt2_appgate_s{s}" for s in range(10)]),
                                     "256": accs([f"gt2_appr256m48_s{s}" for s in range(10)]) | sexy_windows([f"gt2_appr256m48_s{s}" for s in range(10)]),
                                     **paired(g, r)}
    dev["샘플_mp4_앱_흐름(7.7~46초, 시드 10개)"] = {k: sexy_windows([f"mp4app_{t}_s{s}" for s in range(10)]) for k, t in (("192", "g10"), ("256", "r256m48"))}
    for seg in ("AB", "all"):
        dev[f"{seg}(시드 5개)"] = {k: accs([f"gt2_{t}_{seg}_s{s}" for s in range(5)]) for k, t in (("192", "gate"), ("256", "r256"))}
    for seg in ("Z1b", "Z1", "Z2"):
        dev[f"가짜_{seg}(시드 5개)"] = {k: false([f"gt2_{t}_{seg}_s{s}" for s in range(5)]) for k, t in (("192", "gate"), ("256", "r256"))}

    conf = {"latest": ({"R": (52.5, 72.8, LATEST_R), "L": (85.0, 96.8, LATEST_L)}, ["R", "L"], [("ZR", ["R"]), ("all", ["R", "L"])], []),
            "latest2": ({"R": (64.0, 97.5, LATEST2_R), "L": (104.0, 134.5, LATEST2_L)}, ["R", "L", "RL"], [("RL", ["R", "L"]), ("all", ["R", "L"])],
                        [("afterscan", ["R", "L"]), ("app", ["R", "L"])])}
    for vid, (win, segs, cont, extra) in conf.items():
        v = out["시험_영상"][vid] = {}
        for seg in segs:
            v[seg] = {k: accs([f"{vid}_{t}_{seg}_s{s}" for s in range(5)]) for k, t in (("192", "gate"), ("256", "r256"))}
        for seg, parts in cont + extra:
            v[seg + "_제자리"] = {k: inside([f"{vid}_{t}_{seg}_s{s}" for s in range(5)], [win[p] for p in parts]) for k, t in (("192", "gate"), ("256", "r256"))}
        for seg in ("Z1", "Z1b", "Z2", "Z3"):
            v["가짜_" + seg] = {k: false([f"{vid}_{t}_{seg}_s{s}" for s in range(5)]) for k, t in (("192", "gate"), ("256", "r256"))}
        v["정답_없는_구간"] = {k: {"판정_회전_수": [load(f"{vid}_{t}_free_s{s}")["detected_moves"] for s in range(5)],
                                 "추적_비율": [round(load(f"{vid}_{t}_free_s{s}")["tracked_ratio"], 3) for s in range(5)]}
                             for k, t in (("192", "gate"), ("256", "r256"))}

    for name, f in (("검증(20장)", "eval_real_compare_val_resample48.json"), ("시험(40장, 한 번)", "eval_real_compare_test.json")):
        res = json.loads((R / f).read_text())["결과"]
        out["실제_프레임"][name] = {k: {kk: vv for kk, vv in x.items() if kk in ("acc", "missed_as_other_or_body", "차이", "차이_95%")} for k, x in res.items()}

    (R / "track_real_input256_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["시험_영상"], ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
