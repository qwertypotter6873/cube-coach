import { test } from "node:test";
import assert from "node:assert/strict";
import { Tracker, rotvec, mulberry32 } from "../src/vision/tracker.ts";
import { hintArc, stickers, inLayer, stickyView } from "../src/vision/hint.ts";
import type { Pose } from "../src/vision/hint.ts";
import { PoseSmoother, OneEuro, angleBetween, matToQuat, quatToMat } from "../src/vision/smooth.ts";

const F = 0.85;
// 위에서 U면을 내려다보는 자세: 큐브 x → 카메라 x(오른쪽), 큐브 z(앞) → 카메라 y(아래), 큐브 y(위) → 카메라 -z(카메라 쪽)
const TOP: Pose = { R: Float64Array.of(1, 0, 0, 0, 0, 1, 0, -1, 0), T: Float64Array.of(0, 0, 15), layer: -1, phi: 0 };
// 화면 기준 회전 방향: 중심 c에서 본 p → q 움직임의 외적 (화면 y가 아래이므로 +면 시계 방향)
const turnSign = (c: [number, number], p: [number, number], q: [number, number]) =>
  Math.sign((p[0] - c[0]) * (q[1] - p[1]) - (p[1] - c[1]) * (q[0] - p[0]));

test("층 회전 규약: U를 시계 방향(+)으로 돌리면 앞쪽 칸이 화면 왼쪽으로 간다 (위에서 볼 때)", () => {
  const k = 0 * 9 + 7; // U면의 앞쪽 가운데 칸 (큐브 좌표 (0, 1.5, 1))
  const a = stickers(TOP, F), b = stickers({ ...TOP, layer: 0, phi: 0.2 }, F);
  assert.ok(a.v[k] > a.v[4], "앞쪽 칸은 화면 아래쪽");
  assert.ok(b.u[k] < a.u[k], "시계 방향이면 왼쪽으로");
});

test("표시용 투영이 추적기 투영과 같다 (층 회전 중 포함)", () => {
  const rand = mulberry32(7);
  const tr = new Tracker(new Uint8Array(54), F);
  for (let n = 0; n < 5; n++) {
    const R = rotvec(rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1);
    const T = Float64Array.of(rand() - 0.5, rand() - 0.5, 12 + rand() * 6);
    const layer = n % 2 ? Math.floor(rand() * 6) : -1;
    const phi = layer >= 0 ? rand() - 0.5 : 0;
    tr.R = R; tr.Tv = T; tr.layer = layer; tr.phi = phi;
    const want = tr.overlay([0, 0, 1]);
    const got = stickers({ R, T, layer, phi }, F);
    for (const p of want) {
      assert.ok(Math.abs(got.u[p.k] - (p.x - 0.5)) < 1e-9 && Math.abs(got.v[p.k] - (p.y - 0.5)) < 1e-9, `칸 ${p.k}`);
      assert.ok(Math.abs(got.w[p.k] - p.w) < 1e-9);
    }
  }
});

test("힌트 화살표가 실제 층이 도는 방향과 같다 (시계·반시계, 면 위·옆 띠)", () => {
  const rand = mulberry32(3);
  let faceCases = 0, bandCases = 0;
  for (let n = 0; n < 60; n++) {
    const R = rotvec((rand() * 2 - 1) * 2, (rand() * 2 - 1) * 2, (rand() * 2 - 1) * 2);
    const pose: Pose = { R, T: Float64Array.of(0, 0, 15), layer: -1, phi: 0 };
    for (let L = 0; L < 6; L++) {
      for (const k of [1, 3] as const) {
        const arc = hintArc(pose, F, L, k);
        if (!arc) continue;
        // 그 층의 보이는 칸이 phi 를 조금 돌릴 때 움직이는 방향 (k=1 → +, k=3 → -)
        const d = k === 1 ? 0.05 : -0.05;
        const a = stickers(pose, F), b = stickers({ ...pose, layer: L, phi: d }, F);
        const c = [a.u[L * 9 + 4], a.v[L * 9 + 4]] as [number, number];
        // 층 축이 화면에 찍히는 점 (면 가운데 칸이 안 보이면 반대쪽 가운데와의 중점 = 큐브 중심 쪽)
        let want = 0;
        for (let s = 0; s < 54; s++) {
          if (!inLayer(L, s) || s === L * 9 + 4 || a.w[s] < 0.5) continue;
          want += turnSign(c, [a.u[s], a.v[s]], [b.u[s], b.v[s]]);
        }
        if (!arc.onFace) continue; // 옆 띠는 아래에서 따로 본다
        let got = 0;
        for (let i = 0; i + 1 < arc.pts.length; i++) got += turnSign(c, arc.pts[i], arc.pts[i + 1]);
        assert.equal(Math.sign(got), Math.sign(want), `층 ${L} k=${k}`);
        faceCases++;
      }
    }
    // 옆 띠: 화살표 위 점들이 '층의 칸들이 실제로 움직이는 쪽'으로 진행하는가 (가장 가까운 칸의 이동 방향과 내적)
    for (let L = 0; L < 6; L++) {
      const arc = hintArc(pose, F, L, 1);
      if (!arc || arc.onFace) continue;
      const a = stickers(pose, F), b = stickers({ ...pose, layer: L, phi: 0.05 }, F);
      let agree = 0;
      for (let i = 0; i + 1 < arc.pts.length; i++) {
        const [x, y] = arc.pts[i];
        let best = -1, bd = Infinity;
        for (let s = 0; s < 54; s++) {
          if (!inLayer(L, s) || a.w[s] < 0.3) continue;
          const dd = (a.u[s] - x) ** 2 + (a.v[s] - y) ** 2;
          if (dd < bd) { bd = dd; best = s; }
        }
        if (best < 0) continue;
        const mv = [b.u[best] - a.u[best], b.v[best] - a.v[best]];
        const step = [arc.pts[i + 1][0] - x, arc.pts[i + 1][1] - y];
        agree += Math.sign(mv[0] * step[0] + mv[1] * step[1]);
      }
      assert.ok(agree > 0, `옆 띠 층 ${L}: ${agree}`);
      bandCases++;
    }
  }
  assert.ok(faceCases > 50 && bandCases > 10, `면 ${faceCases} 띠 ${bandCases}`);
});

test("U 시계 방향 힌트 화살표: 위에서 보면 화면 기준 시계 방향", () => {
  const arc = hintArc(TOP, F, 0, 1)!;
  assert.ok(arc.onFace);
  const c: [number, number] = [0, 0];
  let s = 0;
  for (let i = 0; i + 1 < arc.pts.length; i++) s += turnSign(c, arc.pts[i], arc.pts[i + 1]);
  assert.ok(s > 0);
  const ccw = hintArc(TOP, F, 0, 3)!;
  let s2 = 0;
  for (let i = 0; i + 1 < ccw.pts.length; i++) s2 += turnSign(c, ccw.pts[i], ccw.pts[i + 1]);
  assert.ok(s2 < 0);
  assert.equal(hintArc(TOP, F, 0, 2)!.heads, 2);
});

test("사원수 변환이 왕복한다", () => {
  const rand = mulberry32(11);
  for (let n = 0; n < 20; n++) {
    const R = rotvec((rand() * 2 - 1) * 3, (rand() * 2 - 1) * 3, (rand() * 2 - 1) * 3);
    assert.ok(angleBetween(R, quatToMat(matToQuat(R))) < 1e-4);
  }
});

test("표시 평활: 가만히 있을 때 떨림을 줄이고, 옮기면 몇 프레임 안에 따라간다", () => {
  const rand = mulberry32(5);
  const sm = new PoseSmoother();
  const R0 = rotvec(0.3, 0.2, 0.1);
  let rawJ = 0, outJ = 0, prevRaw = 0, prevOut = 0;
  for (let i = 0; i < 90; i++) {
    const nx = (rand() - 0.5) * 0.2; // 조각 단위 떨림 ±0.1
    const R = rotvec(0.3 + (rand() - 0.5) * 0.03, 0.2, 0.1);
    sm.update(R, [nx, 0, 15], i / 30);
    if (i > 10) {
      rawJ += Math.abs(nx - prevRaw);
      outJ += Math.abs(sm.T![0] - prevOut);
    }
    prevRaw = nx;
    prevOut = sm.T![0];
  }
  assert.ok(outJ < rawJ * 0.45, `떨림 ${outJ.toFixed(3)} vs ${rawJ.toFixed(3)}`);
  assert.ok(angleBetween(sm.R!, R0) < 1.5);
  // 한 프레임에 0.8조각씩 옮기면 (빠르게 옮기기) 표시가 1조각 넘게 뒤처지지 않는다
  let x = 0;
  for (let i = 90; i < 110; i++) {
    x += 0.8;
    sm.update(R0, [x, 0, 15], i / 30);
  }
  assert.ok(x - sm.T![0] < 1.0, `뒤처짐 ${(x - sm.T![0]).toFixed(2)}`);
  // 바로 옮기기
  sm.update(R0, [40, 3, 15], 111 / 30, true);
  assert.deepEqual(Array.from(sm.T!), [40, 3, 15]);
});

test("One-Euro: 처음 값은 그대로, 같은 값이 이어지면 그 값에 머문다", () => {
  const f = new OneEuro(1, 1, 0.5);
  assert.equal(f.filter([2], 0)[0], 2);
  for (let i = 1; i < 30; i++) f.filter([2], i / 30);
  assert.ok(Math.abs(f.filter([2], 1)[0] - 2) < 1e-9);
});

test("보는 방향: 45도로 비스듬히 들고 흔들려도 바뀌지 않고, 90도 돌려 잡으면 바뀐다", () => {
  // 위에서 비스듬히: 위 면(U)과 카메라 쪽 면(F)이 위쪽 점수로 비기는 자세 근처
  const base = (deg: number) => {
    const t = rotvec((deg * Math.PI) / 180, 0, 0);
    const B = Float64Array.of(1, 0, 0, 0, -1, 0, 0, 0, -1); // 큐브 위 → 화면 위, 큐브 앞 → 카메라 쪽
    const R = new Float64Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) R[i * 3 + j] += t[i * 3 + k] * B[k * 3 + j];
    return R;
  };
  let v = stickyView(base(40), null);
  const first = v.up;
  for (const d of [44, 47, 50, 43, 52, 46, 41, 55]) {
    v = stickyView(base(d), v);
    assert.equal(v.up, first, `${d}도에서 바뀌면 안 된다`);
  }
  v = stickyView(base(95), v); // 통째로 돌려 잡음
  assert.notEqual(v.up, first);
});
