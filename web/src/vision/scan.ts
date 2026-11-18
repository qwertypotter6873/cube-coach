/**
 * 처음 상태 읽기(스캔)와 유효성 검사.
 *
 * 스캔은 따로 "면을 하나씩 정해진 방향으로 보여 주세요" 단계를 두지 않는다.
 * 추적기에 가운데 칸만 알려 주고(나머지는 색을 모르는 칸) 사람이 큐브를 굴리는 동안
 * 자세를 잡게 한 뒤, 보이는 칸마다 모델이 읽은 색을 표로 모은다. 54칸이 다 모이면
 * 가장 많이 나온 색으로 상태를 만들고, 실제로 존재할 수 있는 큐브인지 검사한다.
 * 맞지 않으면 가장 자신 없는 칸들을 둘째 후보 색으로 바꿔 보며 맞는 조합을 찾는다
 * (빨강·주황처럼 헷갈리는 쌍을 여기서 바로잡는다).
 */

import T from "./tables.json" with { type: "json" };
import { UNKNOWN } from "./tracker.ts";

const CORNERS: number[][] = T.corners;
const EDGES: number[][] = T.edges;
const slotOf = (i: number) => Math.floor(i / 9);
const UD = (slot: number) => slot === 0 || slot === 3;

function parity(p: number[]): number {
  const seen = new Array(p.length).fill(false);
  let par = 0;
  for (let i = 0; i < p.length; i++) {
    if (seen[i]) continue;
    let j = i, n = 0;
    while (!seen[j]) { seen[j] = true; j = p[j]; n++; }
    par ^= (n - 1) & 1;
  }
  return par;
}

/** 실제로 존재할 수 있는(풀 수 있는) 상태인가. 파이썬 cube.validate 와 같은 규칙. */
export function validState(s: Uint8Array): boolean {
  const count = new Array(6).fill(0);
  for (const c of s) {
    if (c > 5) return false;
    count[c]++;
  }
  if (count.some((n) => n !== 9)) return false;
  const toSlot = new Array(6).fill(-1);
  for (let slot = 0; slot < 6; slot++) {
    if (toSlot[s[slot * 9 + 4]] >= 0) return false;
    toSlot[s[slot * 9 + 4]] = slot;
  }
  const f = Array.from(s, (c) => toSlot[c]); // 칸마다 "그 색이 원래 있어야 할 면"
  const key = (a: number[]) => [...a].sort().join();

  const cornerHomes = CORNERS.map((p) => p.map(slotOf));
  const cornerKeys = cornerHomes.map(key);
  const cperm: number[] = [];
  let twist = 0;
  for (const p of CORNERS) {
    const cols = p.map((i) => f[i]);
    const j = cornerKeys.indexOf(key(cols));
    if (new Set(cols).size !== 3 || j < 0) return false;
    const h = cornerHomes[j];
    const rots = [h, [h[1], h[2], h[0]], [h[2], h[0], h[1]]];
    if (!rots.some((r) => r.every((v, i) => v === cols[i]))) return false; // 거울상 조각
    cperm.push(j);
    const ref = p.findIndex((i) => UD(slotOf(i)));
    const ud = cols.findIndex((c) => UD(c));
    twist += (((ud - ref) % 3) + 3) % 3;
  }
  if (new Set(cperm).size !== 8) return false;

  const edgeKeys = EDGES.map((p) => key(p.map(slotOf)));
  const rank = (slot: number) => (UD(slot) ? 0 : slot === 2 || slot === 5 ? 1 : 2);
  const eperm: number[] = [];
  let flip = 0;
  for (const p of EDGES) {
    const cols = p.map((i) => f[i]);
    const j = edgeKeys.indexOf(key(cols));
    if (new Set(cols).size !== 2 || j < 0) return false;
    eperm.push(j);
    const refSlot = rank(slotOf(p[0])) <= rank(slotOf(p[1])) ? 0 : 1;
    const refCol = rank(cols[0]) <= rank(cols[1]) ? 0 : 1;
    flip += refSlot !== refCol ? 1 : 0;
  }
  if (new Set(eperm).size !== 12) return false;
  return twist % 3 === 0 && flip % 2 === 0 && parity(cperm) === parity(eperm);
}

/** 가운데 칸만 아는 상태 (표준 배색: 위 흰, 오른쪽 빨, 앞 초, 아래 노, 왼쪽 주, 뒤 파) */
export function centersOnly(): Uint8Array {
  const s = new Uint8Array(54).fill(UNKNOWN);
  for (let slot = 0; slot < 6; slot++) s[slot * 9 + 4] = [0, 1, 2, 3, 4, 5][slot];
  return s;
}

export interface Reading {
  k: number; // 칸 번호
  w: number; // 정면에 가까울수록 1
  cls: number; // 모델이 읽은 분류 (0~5 색, 6 몸체, 7 그 밖)
  p: number; // 그 분류의 확률
}

export const MIN_VOTE = 2.5; // 칸 하나에 이만큼 쌓이면 "읽었다"
const CONFIDENT = 3.0; // 1등이 2등의 이 배 이상이면 추적기에 알려 준다

export class Scanner {
  votes = new Float32Array(54 * 6);

  add(readings: readonly Reading[]): void {
    for (const r of readings) if (r.cls < 6 && r.k % 9 !== 4) this.votes[r.k * 6 + r.cls] += r.w * r.p;
  }

  private ranked(k: number): [number, number, number, number] {
    let a = 0, b = 1;
    if (this.votes[k * 6 + b] > this.votes[k * 6 + a]) [a, b] = [b, a];
    for (let c = 2; c < 6; c++) {
      const v = this.votes[k * 6 + c];
      if (v > this.votes[k * 6 + a]) { b = a; a = c; }
      else if (v > this.votes[k * 6 + b]) b = c;
    }
    return [a, this.votes[k * 6 + a], b, this.votes[k * 6 + b]];
  }

  total(k: number): number {
    let t = 0;
    for (let c = 0; c < 6; c++) t += this.votes[k * 6 + c];
    return t;
  }

  /** 읽은 칸 수 (가운데 6칸 제외, 최대 48) */
  done(): number {
    let n = 0;
    for (let k = 0; k < 54; k++) if (k % 9 !== 4 && this.total(k) >= MIN_VOTE) n++;
    return n;
  }

  /** 면(슬롯)마다 읽은 칸 수 0~8 */
  perSlot(): number[] {
    return [0, 1, 2, 3, 4, 5].map((slot) => {
      let n = 0;
      for (let j = 0; j < 9; j++) if (j !== 4 && this.total(slot * 9 + j) >= MIN_VOTE) n++;
      return n;
    });
  }

  /** 추적기에 알려 줄 상태: 자신 있게 읽은 칸만 색을 채운다 */
  partial(): Uint8Array {
    const s = centersOnly();
    for (let k = 0; k < 54; k++) {
      if (k % 9 === 4 || this.total(k) < MIN_VOTE) continue;
      const [a, va, , vb] = this.ranked(k);
      if (va >= CONFIDENT * Math.max(vb, 0.05)) s[k] = a;
    }
    return s;
  }

  /** 다 읽었으면 유효한 상태를, 아니면 null. 맞지 않으면 자신 없는 칸부터 둘째 후보로 바꿔 본다. */
  result(): Uint8Array | null {
    if (this.done() < 48) return null;
    const s = centersOnly();
    const doubt: { k: number; alt: number; margin: number }[] = [];
    for (let k = 0; k < 54; k++) {
      if (k % 9 === 4) continue;
      const [a, va, b, vb] = this.ranked(k);
      s[k] = a;
      doubt.push({ k, alt: b, margin: va / Math.max(vb, 1e-3) });
    }
    if (validState(s)) return s;
    doubt.sort((x, y) => x.margin - y.margin);
    const pool = doubt.slice(0, 12);
    // 한 칸, 두 칸, 세 칸 바꾸기 (12 + 66 + 220 가지)
    const tryFix = (idx: number[]): Uint8Array | null => {
      const t = Uint8Array.from(s);
      for (const i of idx) t[pool[i].k] = pool[i].alt;
      return validState(t) ? t : null;
    };
    for (let i = 0; i < pool.length; i++) { const r = tryFix([i]); if (r) return r; }
    for (let i = 0; i < pool.length; i++) for (let j = i + 1; j < pool.length; j++) { const r = tryFix([i, j]); if (r) return r; }
    for (let i = 0; i < pool.length; i++)
      for (let j = i + 1; j < pool.length; j++)
        for (let l = j + 1; l < pool.length; l++) { const r = tryFix([i, j, l]); if (r) return r; }
    return null;
  }

  /** 다 읽었는데 맞는 조합이 없을 때: 가장 자신 없는 칸들의 표를 지워 다시 보게 한다 */
  forgetWeakest(n = 6): void {
    const order = Array.from({ length: 54 }, (_, k) => k).filter((k) => k % 9 !== 4)
      .sort((x, y) => { const a = this.ranked(x), b = this.ranked(y); return a[1] / Math.max(a[3], 1e-3) - b[1] / Math.max(b[3], 1e-3); });
    for (const k of order.slice(0, n)) this.votes.fill(0, k * 6, k * 6 + 6);
  }
}
