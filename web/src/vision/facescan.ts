/**
 * 처음 상태 읽기: 한 면씩 정면으로 보여 주면 3x3 격자를 읽고, 여섯 면을 모아 상태를 만든다.
 * 파이썬 src/cubecoach/facescan.py 와 같은 규칙 (실제 영상에서 사람이 읽은 상태와 54칸 모두 일치함을 확인).
 *
 * 1. 면 읽기: 큐브로 보이는 칸(스티커·몸체)이 이어진 덩어리 가운데 가장 큰 것만 큐브로 본다
 *    (뒤쪽 책장이 스티커로 읽혀도 가운데 칸·관심영역이 끌려가지 않게). 그 안의 중심 지도 봉우리 가운데
 *    덩어리 한가운데에 가장 가까운 것을 가운데 칸으로 보고, 가까운 이웃으로 격자 방향·간격을 잡아
 *    9칸 자리를 예측한다. 9자리 모두에 봉우리가 있으면 인정.
 *    칸 색은 둘레 3x3 지도 칸의 다수결 (흰 센터의 파란 로고처럼 한가운데만 다른 경우를 견디도록).
 * 2. 모으기: 가운데 색마다 9칸의 색을 프레임에 걸쳐 투표하고 (돌려 보여 줘도 방향을 맞춰서 더한다),
 *    9칸 통째(회전 무시)로 같은 격자가 몇 번 나왔는지도 센다. 기울여 든 면에서는 두 면에 걸친 격자처럼
 *    틀린 읽기가 몇 프레임씩 같은 모양으로 나와서, 칸 다수결만 쓰면 드물게 맞게 읽힌 면이 덮인다.
 * 3. 맞추기: 면마다 후보 격자(통째로 NEED번 이상 나온 것 많은 순 3개 + 칸 다수결)를 두고, 1등끼리 →
 *    한 면만 다른 후보로 바꾼 조합 순으로, 각 면을 0/90/180/270도 돌린 4^6 = 4,096가지 가운데
 *    실제로 존재할 수 있는 조합이 하나뿐인지 본다. 사람이 면을 어느 방향으로 들고 보여 줬는지 몰라도 된다.
 */

import type { Maps, Roi } from "./tracker.ts";
import { validState } from "./scan.ts";

const STICKER = 6;
const OTHER = 7;
const PEAK_MIN = 0.35;
const MATCH = 0.38;
/** 면 하나에 같은 격자(회전 무시)가 이만큼 나와야 '읽었다'고 보고 맞추기 후보로 쓴다 */
export const NEED = 3;
const ROT = [6, 3, 0, 7, 4, 1, 8, 5, 2]; // 3x3 격자를 시계 방향으로 90도
const rot = (g: number[]) => ROT.map((i) => g[i]);
const MIRROR = [2, 1, 0, 5, 4, 3, 8, 7, 6]; // 좌우반전된 화면에서 읽은 격자를 되돌린다
const mirror = (g: number[]) => MIRROR.map((i) => g[i]);

export interface ScanResult {
  state: Uint8Array;
  /** 카메라 화면이 좌우반전돼 있었다 (거울상으로 읽어야만 실제 큐브가 됐다) */
  mirrored: boolean;
}

function classes(maps: Maps): Uint8Array {
  const { n, logp } = maps;
  const nn = n * n;
  const cls = new Uint8Array(nn);
  for (let p = 0; p < nn; p++) {
    let b = 0;
    for (let c = 1; c < 8; c++) if (logp[c * nn + p] > logp[b * nn + p]) b = c;
    cls[p] = b;
  }
  return cls;
}

/** 큐브로 보이는 칸(그 밖이 아닌 칸)이 8방향으로 이어진 덩어리 가운데 가장 큰 것 (파이썬 cube_blob 과 같음) */
function cubeBlob(cls: Uint8Array, n: number): Uint8Array {
  const lab = new Int32Array(n * n).fill(-1);
  const stack: number[] = [];
  let best = -1, bestSize = 0, k = 0;
  for (let p0 = 0; p0 < n * n; p0++) {
    if (cls[p0] === OTHER || lab[p0] >= 0) continue;
    let size = 0;
    lab[p0] = k;
    stack.push(p0);
    while (stack.length) {
      const p = stack.pop()!;
      size++;
      const y = Math.floor(p / n), x = p % n;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy, xx = x + dx;
          if (yy < 0 || xx < 0 || yy >= n || xx >= n) continue;
          const q = yy * n + xx;
          if (cls[q] !== OTHER && lab[q] < 0) { lab[q] = k; stack.push(q); }
        }
    }
    if (size > bestSize) { best = k; bestSize = size; }
    k++;
  }
  const out = new Uint8Array(n * n);
  for (let p = 0; p < n * n; p++) out[p] = lab[p] === best && best >= 0 ? 1 : 0;
  return out;
}

function peaks(maps: Maps, cls: Uint8Array): [number, number][] {
  const { n, heat } = maps;
  const H = (y: number, x: number) => (y < 0 || x < 0 || y >= n || x >= n ? 0 : heat[y * n + x]);
  const out: [number, number][] = [];
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const h = heat[y * n + x];
      if (h <= PEAK_MIN || cls[y * n + x] >= STICKER) continue;
      let ok = true, sw = 0, ox = 0, oy = 0;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const v = H(y + dy, x + dx);
          if (v > h) ok = false;
          sw += v; ox += v * dx; oy += v * dy;
        }
      if (ok) out.push([x + 0.5 + ox / sw, y + 0.5 + oy / sw]);
    }
  return out;
}

function cellColor(cls: Uint8Array, n: number, x: number, y: number): number {
  const cnt = new Array(6).fill(0);
  const i = Math.floor(y), j = Math.floor(x);
  for (let a = Math.max(i - 1, 0); a <= Math.min(i + 1, n - 1); a++)
    for (let b = Math.max(j - 1, 0); b <= Math.min(j + 1, n - 1); b++) if (cls[a * n + b] < STICKER) cnt[cls[a * n + b]]++;
  const best = cnt.indexOf(Math.max(...cnt));
  return cnt[best] ? best : -1;
}

/** 정면으로 보이는 면 하나의 9칸 색 (화면에서 보이는 그대로, 행 우선). 못 읽으면 null. */
export function readFace(maps: Maps): number[] | null {
  const n = maps.n;
  const cls = classes(maps);
  const blob = cubeBlob(cls, n);
  const inBlob = (q: [number, number]) => {
    const j = Math.min(n - 1, Math.max(0, Math.floor(q[0]))), i = Math.min(n - 1, Math.max(0, Math.floor(q[1])));
    return blob[i * n + j] === 1;
  };
  const P = peaks(maps, cls).filter(inBlob); // 가장 큰 덩어리 안의 봉우리만
  if (P.length < 9) return null;
  let cnt = 0, sx = 0, sy = 0;
  for (let p = 0; p < n * n; p++) if (blob[p]) { cnt++; sx += p % n; sy += Math.floor(p / n); }
  if (cnt < 20) return null;
  const cx = sx / cnt + 0.5, cy = sy / cnt + 0.5;
  const d2 = (a: [number, number], b: [number, number]) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  let c = P[0];
  for (const p of P) if (d2(p, [cx, cy]) < d2(c, [cx, cy])) c = p;
  const near = P.filter((p) => d2(p, c) > 1e-12).sort((a, b) => d2(a, c) - d2(b, c));
  if (near.length < 4) return null;
  const spacing = near.slice(0, 4).reduce((s, p) => s + Math.sqrt(d2(p, c)), 0) / 4;
  if (spacing < 1.5) return null;
  let ang = Math.atan2(near[0][1] - c[1], near[0][0] - c[0]);
  ang = ((((ang + Math.PI / 4) % (Math.PI / 2)) + Math.PI / 2) % (Math.PI / 2)) - Math.PI / 4;
  const u = [Math.cos(ang) * spacing, Math.sin(ang) * spacing];
  const v = [-u[1], u[0]]; // 화면 y가 아래쪽: 오른쪽을 시계 방향으로 90도 = 아래쪽
  const out: number[] = [];
  for (const r of [-1, 0, 1])
    for (const q of [-1, 0, 1]) {
      const p: [number, number] = [c[0] + q * u[0] + r * v[0], c[1] + q * u[1] + r * v[1]];
      let best = P[0];
      for (const s of P) if (d2(s, p) < d2(best, p)) best = s;
      if (Math.sqrt(d2(best, p)) > MATCH * spacing) return null;
      const col = cellColor(cls, n, best[0], best[1]);
      if (col < 0) return null;
      out.push(col);
    }
  return out;
}

/** 큐브로 보이는 곳(가장 큰 덩어리)을 따라가는 다음 관심영역 (큐브가 영역의 절반쯤 차게) */
export function followRoi(maps: Maps, roi: Roi): Roi | null {
  const n = maps.n;
  const blob = cubeBlob(classes(maps), n);
  const xs: number[] = [], ys: number[] = [];
  for (let p = 0; p < n * n; p++) if (blob[p]) { xs.push(p % n); ys.push(Math.floor(p / n)); }
  if (xs.length < 30) return null;
  const pct = (a: number[], q: number) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };
  const x0 = pct(xs, 0.05) / n, x1 = pct(xs, 0.95) / n, y0 = pct(ys, 0.05) / n, y1 = pct(ys, 0.95) / n;
  const side = Math.max(x1 - x0, y1 - y0) * 2.0;
  return [roi[0] + ((x0 + x1) / 2 - 0.5) * roi[2], roi[1] + ((y0 + y1) / 2 - 0.5) * roi[2], Math.min(1.2, Math.max(0.15, side * roi[2]))];
}

/** 표준 배색: 슬롯(U R F D L B)의 가운데 색 */
const SLOT_COLOR = [0, 1, 2, 3, 4, 5];

/** 격자를 회전과 상관없이 같은 것끼리 묶는 열쇠 (네 방향 가운데 사전순 최소, 파이썬 _canon 과 같음) */
function canon(g: number[]): string {
  let cur = g, best = g.join("");
  for (let k = 0; k < 3; k++) {
    cur = rot(cur);
    const s = cur.join("");
    if (s < best) best = s;
  }
  return best;
}

export class FaceScanner {
  votes = Array.from({ length: 6 }, () => Array.from({ length: 9 }, () => new Array(6).fill(0)));
  frames = new Array(6).fill(0);
  /** 가운데 색마다 9칸 통째로 본 격자(회전 무시)와 나온 횟수. 처음 본 방향 그대로 둔다. */
  seen = Array.from({ length: 6 }, () => new Map<string, { grid: number[]; n: number }>());
  /**
   * 마지막 solve()에서, 면마다 가장 많이 나온 격자로 맞춰 보니 실제로 존재할 수 있는 서로 다른 상태가 둘 이상이었다.
   * 면을 하나씩 따로 읽으면 면끼리 어떻게 붙어 있는지는 모르므로, 다 맞춘 상태에 가까운 큐브(예: R 한 번과 L' 한 번)는
   * 여섯 면이 서로 돌린 모양이라 면만으로는 가릴 수 없다. 이때는 다른 후보로 억지로 맞추지 않고 멈춘다.
   */
  ambiguous = false;

  private grid(center: number): number[] {
    return this.votes[center].map((v) => v.indexOf(Math.max(...v)));
  }

  /** 읽은 면 하나를 넣고 가운데 색을 돌려준다. 같은 면은 지난번 방향에 맞춰 돌려서 더한다. */
  add(face: number[]): number {
    const center = face[4];
    let g = face;
    if (this.frames[center]) {
      const ref = this.grid(center);
      let best = -1, bg = g, cur = g;
      for (let k = 0; k < 4; k++) {
        const s = cur.reduce((a, c, i) => a + (c === ref[i] ? 1 : 0), 0);
        if (s > best) { best = s; bg = cur; }
        cur = rot(cur);
      }
      g = bg;
    }
    g.forEach((c, i) => this.votes[center][i][c]++);
    this.frames[center]++;
    const key = canon(face);
    const e = this.seen[center].get(key);
    if (e) e.n++;
    else this.seen[center].set(key, { grid: [...face], n: 1 });
    return center;
  }

  /** 그 면에서 가장 많이 나온 격자(회전 무시)의 횟수 */
  agree(center: number): number {
    let m = 0;
    for (const e of this.seen[center].values()) m = Math.max(m, e.n);
    return m;
  }

  /** 가장 덜 확실한 면의 가운데 색 (같은 격자가 가장 적게 나온 면). 다시 보여 달라고 할 때 쓴다. */
  weakest(): number {
    let b = 0;
    for (let c = 1; c < 6; c++) {
      const a = this.agree(c), ab = this.agree(b);
      if (a < ab || (a === ab && this.frames[c] < this.frames[b])) b = c;
    }
    return b;
  }

  /** 여섯 면 모두 같은 격자가 need번 이상 나왔다 (가운데 색만 맞는 틀린 읽기로는 차지 않는다) */
  ready(need = NEED): boolean {
    return [0, 1, 2, 3, 4, 5].every((c) => this.agree(c) >= need);
  }

  /** 면마다 후보 격자: 통째로 NEED번 이상 나온 격자(많은 순 3개) + 칸 다수결 격자 */
  private candidates(): number[][][] {
    return [0, 1, 2, 3, 4, 5].map((c) => {
      const whole = [...this.seen[c].values()].sort((a, b) => b.n - a.n).filter((e) => e.n >= NEED).slice(0, 3).map((e) => e.grid);
      const cell = this.grid(c);
      const keys = new Set(whole.map(canon));
      if (!keys.has(canon(cell))) whole.push(cell);
      return whole;
    });
  }

  /**
   * 여섯 면을 맞춘 상태. 맞는 조합이 하나로 정해지지 않으면 null.
   * 보이는 그대로 맞지 않으면 거울상으로도 맞춰 본다. 거울상 큐브는 실제로 존재할 수 없으므로
   * (꼭짓점 조각의 색 순서가 뒤집힌다) 둘 중 한쪽만 맞고, 그것으로 카메라의 좌우반전 여부까지 알 수 있다.
   */
  solve(): ScanResult | null {
    this.ambiguous = false;
    const state = this.solveGrids(false);
    if (state) return { state, mirrored: false };
    if (this.ambiguous) return null;
    const m = this.solveGrids(true);
    return m ? { state: m, mirrored: true } : null;
  }

  private solveGrids(mirrored: boolean): Uint8Array | null {
    const fix = (g: number[]) => (mirrored ? mirror(g) : g);
    // 면 후보: 1등끼리 → 한 면만 2·3등(또는 칸 다수결)으로 바꾼 조합
    const cands = this.candidates().map((l) => l.map(fix));
    const top = cands.map((l) => l[0]);
    const firstTry = this.orient(top);
    if (firstTry === "ambiguous") {
      this.ambiguous = true;
      return null;
    }
    if (firstTry) return firstTry;
    for (let f = 0; f < 6; f++)
      for (const g of cands[f].slice(1)) {
        const r = this.orient(top.map((x, i) => (i === f ? g : x)));
        if (r instanceof Uint8Array) return r;
      }
    // 칸 다수결 격자에서 자신 없는 칸을 둘째 후보 색으로 바꿔 본다
    const grids = [0, 1, 2, 3, 4, 5].map((c) => fix(this.grid(c)));
    const first = this.orient(grids);
    if (first instanceof Uint8Array) return first;
    // 자신 없는 칸(가운데 제외) 8개 가운데 1~2개를 둘째 후보 색으로 바꿔 본다
    const doubt: { m: number; f: number; i: number; alt: number }[] = [];
    for (let f = 0; f < 6; f++)
      for (let i = 0; i < 9; i++) {
        if (i === 4) continue;
        const v = this.votes[f][i];
        const order = [0, 1, 2, 3, 4, 5].sort((a, b) => v[b] - v[a]);
        doubt.push({ m: (v[order[0]] + 1) / (v[order[1]] + 1), f, i: mirrored ? MIRROR[i] : i, alt: order[1] });
      }
    doubt.sort((a, b) => a.m - b.m);
    const pool = doubt.slice(0, 8);
    const tryWith = (idx: number[]) => {
      const g = grids.map((x) => [...x]);
      for (const j of idx) g[pool[j].f][pool[j].i] = pool[j].alt;
      const r = this.orient(g);
      return r instanceof Uint8Array ? r : null;
    };
    for (let a = 0; a < pool.length; a++) { const r = tryWith([a]); if (r) return r; }
    for (let a = 0; a < pool.length; a++) for (let b = a + 1; b < pool.length; b++) { const r = tryWith([a, b]); if (r) return r; }
    return null;
  }

  /** 4^6 방향 조합이 만드는 유효한 상태가 (같은 상태는 하나로 세어) 정확히 하나면 그것, 둘 이상이면 "ambiguous" */
  private orient(grids: number[][]): Uint8Array | "ambiguous" | null {
    const count = new Array(6).fill(0);
    for (const g of grids) for (const c of g) count[c]++;
    if (count.some((k) => k !== 9)) return null; // 색마다 9칸이 아니면 어느 방향으로 돌려도 안 맞는다
    const faces = SLOT_COLOR.map((c) => {
      const r = [grids[c]];
      for (let k = 0; k < 3; k++) r.push(rot(r[r.length - 1]));
      return r;
    });
    let found: Uint8Array | null = null;
    const s = new Uint8Array(54);
    for (let code = 0; code < 4096; code++) {
      for (let slot = 0; slot < 6; slot++) {
        const g = faces[slot][(code >> (2 * slot)) & 3];
        for (let i = 0; i < 9; i++) s[slot * 9 + i] = g[i];
      }
      if (validState(s)) {
        if (!found) found = Uint8Array.from(s);
        else if (found.some((v, i) => v !== s[i])) return "ambiguous"; // 서로 다른 상태 둘 이상: 정해지지 않음
        // 같은 상태면 넘어간다 (다 맞춘 면처럼 돌려도 같은 면이 있으면 여러 방향 조합이 같은 상태를 만든다)
      }
    }
    return found;
  }
}
