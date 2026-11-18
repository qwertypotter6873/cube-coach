/**
 * 카메라 화면 위 실제 큐브에 겹쳐 그릴 것들의 위치 계산 (그리기는 camera.ts).
 *  - 주어진 자세로 54칸 중심을 화면에 투영 (표시용 평활 자세를 쓴다)
 *  - 다음에 돌릴 층의 회전 방향 화살표 (곡선)
 *
 * 좌표: 화면 좌표는 프레임 너비 기준, 프레임 중심이 0 (추적기 Roi 와 같다).
 * 회전 방향 규약은 추적기와 같다: 층 L을 바깥에서 봐서 시계 방향으로 phi 돌리면
 * 그 층의 점은 rotvec(-phi * AXES[L]) 만큼 돈다 (tracker.ts project()).
 */

import Tb from "./tables.json" with { type: "json" };
import { rotvec } from "./tracker.ts";

const CENTERS = Float64Array.from(Tb.centers.flat());
const NORMALS = Float64Array.from(Tb.normals.flat());
const AXES = Float64Array.from(Tb.axes.flat());
const LAYER = Uint8Array.from(Tb.layer.flat());

export interface Pose {
  R: ArrayLike<number>; // 3x3 행 우선
  T: ArrayLike<number>; // 3
  layer: number; // 돌아가는 층 (-1이면 없음)
  phi: number; // 그 층의 각도 (바깥에서 봐서 시계 방향 +)
}

/** 층 L에 속한 칸인가 */
export function inLayer(L: number, k: number): boolean {
  return LAYER[L * 54 + k] === 1;
}

/** 큐브 좌표의 점 p(와 바깥 방향 n)를 자세로 옮겨 화면 좌표 (u, v)와 보이는 정도(0~1)를 낸다 */
function toScreen(pose: Pose, f: number, p: ArrayLike<number>, n: ArrayLike<number>): { u: number; v: number; facing: number } {
  const r = pose.R, t = pose.T;
  const X = r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0];
  const Y = r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1];
  const Z = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2];
  const nx = r[0] * n[0] + r[1] * n[1] + r[2] * n[2];
  const ny = r[3] * n[0] + r[4] * n[1] + r[5] * n[2];
  const nz = r[6] * n[0] + r[7] * n[1] + r[8] * n[2];
  return { u: (f * X) / Z, v: (f * Y) / Z, facing: -(nx * X + ny * Y + nz * Z) / Math.hypot(X, Y, Z) };
}

const apply3 = (M: Float64Array, p: ArrayLike<number>): number[] => [
  M[0] * p[0] + M[1] * p[1] + M[2] * p[2],
  M[3] * p[0] + M[4] * p[1] + M[5] * p[2],
  M[6] * p[0] + M[7] * p[1] + M[8] * p[2],
];

/** 층 L의 점을 시계 방향 phi만큼 돌리는 행렬 (추적기와 같은 규약) */
export function layerRot(L: number, phi: number): Float64Array {
  return rotvec(-phi * AXES[L * 3], -phi * AXES[L * 3 + 1], -phi * AXES[L * 3 + 2]);
}

/** 54칸 중심의 화면 좌표와 보이는 정도 (추적기 project()와 같은 식, w = 보이는 정도 0~1) */
export function stickers(pose: Pose, f: number): { u: Float64Array; v: Float64Array; w: Float64Array } {
  const u = new Float64Array(54), v = new Float64Array(54), w = new Float64Array(54);
  const M = pose.layer >= 0 ? layerRot(pose.layer, pose.phi) : null;
  for (let k = 0; k < 54; k++) {
    let c: ArrayLike<number> = CENTERS.subarray(k * 3, k * 3 + 3);
    let n: ArrayLike<number> = NORMALS.subarray(k * 3, k * 3 + 3);
    if (M && LAYER[pose.layer * 54 + k]) {
      c = apply3(M, c);
      n = apply3(M, n);
    }
    const s = toScreen(pose, f, c, n);
    u[k] = s.u;
    v[k] = s.v;
    w[k] = Math.min(1, Math.max(0, (s.facing - 0.12) / 0.25));
  }
  return { u, v, w };
}

export interface Arc {
  /** 화면 좌표 점들 (회전 방향 순서) */
  pts: [number, number][];
  /** 면 위에 그렸나(true), 층 옆 띠에 그렸나(false) */
  onFace: boolean;
  /** 화살표 머리를 몇 개 그릴지 (180도 회전이면 2) */
  heads: number;
}

/**
 * 층 L을 k번(시계 방향 90도 단위, 3 = 반시계) 돌리라는 화살표.
 * 면이 보이면 면 위에 270도 곡선, 안 보이면 그 층 옆면 띠를 따라 보이는 만큼 곡선.
 * 곡선의 점은 '층의 점이 돌아가는 방향' 순서다.
 */
export function hintArc(pose: Pose, f: number, L: number, k: 1 | 2 | 3): Arc | null {
  const ax = AXES.subarray(L * 3, L * 3 + 3);
  const dir = k === 3 ? -1 : 1; // phi가 커지는 쪽이 시계 방향
  const face = toScreen(pose, f, CENTERS.subarray((L * 9 + 4) * 3, (L * 9 + 4) * 3 + 3), ax);
  // 면 위 곡선의 출발점: 면 가운데에서 모서리 칸 쪽 (화면에서 가장 위쪽으로 보이는 것을 골라 곡선의 빈틈을 위에 둔다)
  const center = [ax[0] * 1.5, ax[1] * 1.5, ax[2] * 1.5];
  let e1: number[] = [0, 0, 0];
  let best = Infinity;
  for (const j of [1, 3, 5, 7]) {
    const kk = L * 9 + j;
    const d = [CENTERS[kk * 3] - center[0], CENTERS[kk * 3 + 1] - center[1], CENTERS[kk * 3 + 2] - center[2]];
    const s = toScreen(pose, f, CENTERS.subarray(kk * 3, kk * 3 + 3), ax);
    if (s.v < best) { best = s.v; e1 = d; }
  }
  const heads = k === 2 ? 2 : 1;
  if (face.facing > 0.2) {
    // 면 위: 빈틈(위쪽)에서 시작해 회전 방향으로 270도
    const pts: [number, number][] = [];
    const off = [ax[0] * 1.58, ax[1] * 1.58, ax[2] * 1.58];
    const r = 1.05;
    for (let a = 45; a <= 315; a += 6) {
      const M = layerRot(L, dir * a * (Math.PI / 180));
      const q = apply3(M, e1);
      const p = [off[0] + q[0] * r, off[1] + q[1] * r, off[2] + q[2] * r];
      const s = toScreen(pose, f, p, ax);
      pts.push([s.u, s.v]);
    }
    return { pts, onFace: true, heads };
  }
  // 옆 띠: 층 두께 가운데 높이에서 큐브 바깥을 한 바퀴 돌며 보이는 구간만.
  // 그 층의 칸이 거의 안 보이면(정면에서 본 뒷면 층 등) 화살표를 그리지 않는다.
  const st = stickers(pose, f);
  let seen = 0;
  for (let s = 0; s < 54; s++) if (LAYER[L * 54 + s] && st.w[s] >= 0.3) seen++;
  if (seen < 2) return null;
  const mid = [ax[0] * 1.0, ax[1] * 1.0, ax[2] * 1.0];
  const N = 72;
  const vis: boolean[] = [];
  const P: [number, number][] = [];
  for (let i = 0; i < N; i++) {
    const M = layerRot(L, dir * i * (2 * Math.PI / N));
    const d = apply3(M, e1);
    const p = [mid[0] + d[0] * 1.8, mid[1] + d[1] * 1.8, mid[2] + d[2] * 1.8];
    const s = toScreen(pose, f, p, d);
    vis.push(s.facing > 0.15);
    P.push([s.u, s.v]);
  }
  // 가장 긴 연속 보이는 구간 (한 바퀴 이어짐)
  let bestStart = -1, bestLen = 0;
  for (let i = 0; i < N; i++) {
    if (!vis[i] || vis[(i - 1 + N) % N]) continue;
    let len = 0;
    while (len < N && vis[(i + len) % N]) len++;
    if (len > bestLen) { bestLen = len; bestStart = i; }
  }
  if (bestStart < 0 && vis.every(Boolean)) { bestStart = 0; bestLen = N; }
  if (bestLen < 6) return null;
  const trim = Math.floor(bestLen * 0.12);
  const pts: [number, number][] = [];
  for (let j = trim; j < bestLen - trim; j++) pts.push(P[(bestStart + j) % N]);
  return { pts, onFace: false, heads };
}

/**
 * 면(가운데 칸 자리) 6개가 화면 위쪽·사람 쪽을 얼마나 향하는지 (-1~1).
 * 카메라 y는 아래, z는 보는 방향이고 사람은 카메라 맞은편에 있으므로 사람 쪽 = 카메라 z + (tracker.view()와 같은 규약).
 */
export function faceDirs(R: ArrayLike<number>): { up: number[]; away: number[] } {
  const up: number[] = [], away: number[] = [];
  for (let s = 0; s < 6; s++) {
    const nx = AXES[s * 3], ny = AXES[s * 3 + 1], nz = AXES[s * 3 + 2];
    up.push(-(R[3] * nx + R[4] * ny + R[5] * nz));
    away.push(R[6] * nx + R[7] * ny + R[8] * nz);
  }
  return { up, away };
}

/**
 * 보는 방향(위 면·사람 쪽 면의 자리)을 히스테리시스로 고른다. 큐브를 45도쯤 비스듬히 들면 두 면이 거의 같은 점수라
 * 자세가 조금만 흔들려도 답이 왔다 갔다 한다. 지금 고른 면보다 margin 넘게 더 향하는 면이 있을 때만 바꾼다.
 */
export function stickyView(R: ArrayLike<number>, prev: { up: number; front: number } | null, margin = 0.5): { up: number; front: number } {
  const { up, away } = faceDirs(R);
  const best = (a: number[], keep: number) => {
    let b = 0;
    for (let s = 1; s < 6; s++) if (a[s] > a[b]) b = s;
    return keep >= 0 && a[b] - a[keep] < margin ? keep : b;
  };
  let u = best(up, prev ? prev.up : -1);
  let f = best(away, prev ? prev.front : -1);
  // 위와 사람 쪽이 같은 면이거나 맞은편이면 히스테리시스 없이 다시 고른다
  if (u === f || u % 3 === f % 3) {
    u = best(up, -1);
    f = best(away, -1);
  }
  return { up: u, front: f };
}
