import { test } from "node:test";
import assert from "node:assert/strict";
import fx from "./fixtures/tracker_score.json" with { type: "json" };
import { Tracker, prepare, UNKNOWN } from "../src/vision/tracker.ts";
import { validState } from "../src/vision/scan.ts";

const maps = { n: fx.n, logp: Float32Array.from(fx.logp), heat: Float32Array.from(fx.heat) };
const N = fx.layers.length;
const hyps = { n: N, R: Float64Array.from(fx.Rs), T: Float64Array.from(fx.Ts), layer: Int32Array.from(fx.layers), phi: Float64Array.from(fx.phis) };
const roi = fx.roi as [number, number, number];

test("봉우리 찾기가 파이썬과 같다", () => {
  const p = prepare(maps);
  assert.equal(p.peaks.n, fx.peaks.col.length);
  const got = Array.from({ length: p.peaks.n }, (_, i) => [p.peaks.xy[2 * i], p.peaks.xy[2 * i + 1], p.peaks.col[i]]);
  const want = fx.peaks.xy.map((xy, i) => [xy[0], xy[1], fx.peaks.col[i]]);
  const key = (a: number[]) => a.map((v) => v.toFixed(3)).join();
  assert.deepEqual(got.map(key).sort(), want.map(key).sort());
});

test("점수 함수가 파이썬과 같은 값을 낸다 (가설 24개)", () => {
  const tr = new Tracker(Uint8Array.from(fx.colors), fx.f);
  const s = tr.score(hyps, prepare(maps), roi);
  for (let i = 0; i < N; i++) assert.ok(Math.abs(s[i] - fx.score[i]) < 1e-3, `가설 ${i}: ${s[i]} vs ${fx.score[i]}`);
});

test("상태 여러 개를 한꺼번에 견줄 때도 같다", () => {
  const tr = new Tracker(Uint8Array.from(fx.colors), fx.f);
  const states = fx.states.map((s) => Uint8Array.from(s));
  const s = tr.score(hyps, prepare(maps), roi, states);
  for (let i = 0; i < N; i++)
    for (let m = 0; m < states.length; m++) assert.ok(Math.abs(s[i * states.length + m] - fx.score_states[i][m]) < 1e-3);
});

test("일치율(quality)이 파이썬과 같다", () => {
  const tr = new Tracker(Uint8Array.from(fx.colors), fx.f);
  tr.R = Float64Array.from(fx.Rs.slice(0, 9));
  tr.Tv = Float64Array.from(fx.Ts.slice(0, 3));
  assert.ok(Math.abs(tr.qualityOf(prepare(maps), roi) - fx.quality0) < 1e-9);
});

test("색을 모르는 칸은 '스티커가 있는가'로만 따진다", () => {
  const known = new Tracker(Uint8Array.from(fx.colors), fx.f);
  const unknown = Uint8Array.from(fx.colors).map((c, i) => (i % 9 === 4 ? c : UNKNOWN));
  const tr = new Tracker(unknown, fx.f);
  const p = prepare(maps);
  const a = known.score(hyps, p, roi), b = tr.score(hyps, p, roi);
  // 모르는 칸은 어떤 색이든 받아들이므로 점수가 아는 경우보다 낮을 수 없다
  for (let i = 0; i < N; i++) assert.ok(b[i] >= a[i] - 1e-9);
});

test("상태 유효성 검사가 파이썬 validate 와 같다", () => {
  for (const c of fx.validate) assert.equal(validState(Uint8Array.from(c.state)), c.valid, c.state.join(""));
});
