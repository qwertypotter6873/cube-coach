import { test } from "node:test";
import assert from "node:assert/strict";
import { SOLVED, applyColorMoves, translate, invertAlg, slotOfColor, viewColors, FACELET_POS, SLOT_NORMAL } from "../src/coach/cube.ts";
import type { State } from "../src/coach/cube.ts";
import { analyze, OCLL, OLL_EDGES, PLL, BEGINNER_ALGS, YELLOW } from "../src/coach/method.ts";

const LL_VIEW = { up: YELLOW, front: 2 };
const apply = (s: State, alg: string) => applyColorMoves(s, translate(s, alg, LL_VIEW));

/** 윗면 상태를 보는 방향·윗면 돌리기와 무관한 열쇠로: 옆면 윗줄 스티커의 "가운데 색과의 차이"와 노랑 여부 */
function llKey(s: State, withPerm: boolean): string {
  const cycle = [2, 1, 5, 4].map((f) => viewColors(s, { up: YELLOW, front: f }).F); // 앞→오른쪽→뒤→왼쪽
  return keyByGeometry(s, cycle, withPerm);
}

function keyByGeometry(s: State, cycle: number[], withPerm: boolean): string {
  const ny = SLOT_NORMAL[slotOfColor(s, YELLOW)];
  const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a: readonly number[], b: readonly number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const stickers: number[][] = []; // 앞·오른·뒤·왼 순서로, 각 면 윗줄 왼→오 의 값
  for (const c of cycle) {
    const slot = slotOfColor(s, c);
    const nf = SLOT_NORMAL[slot];
    const right = cross(ny, nf);
    const cells = [0, 1, 2, 3, 4, 5, 6, 7, 8].map((j) => slot * 9 + j).filter((i) => dot(FACELET_POS[i], ny) === 1);
    cells.sort((a, b) => dot(FACELET_POS[a], right) - dot(FACELET_POS[b], right));
    stickers.push(cells.map((i) => (s[i] === YELLOW ? 9 : withPerm ? (cycle.indexOf(s[i]) - cycle.indexOf(c) + 4) % 4 : 0)));
  }
  let best = "";
  for (let r = 0; r < 4; r++) {
    for (let a = 0; a < 4; a++) {
      const rot = [0, 1, 2, 3].map((i) => stickers[(i + r) % 4]);
      const k = rot.map((row) => row.map((v) => (v === 9 ? "y" : String((v + a) % 4))).join("")).join("|");
      if (!best || k < best) best = k;
    }
  }
  return best;
}

test("모든 공식은 1·2층을 건드리지 않는다", () => {
  const all = [...OCLL, ...OLL_EDGES, ...PLL, ...Object.values(BEGINNER_ALGS)];
  for (const { name, alg } of all) {
    const a = analyze(apply(SOLVED, alg));
    assert.ok(a.f2l, `${name}: 1·2층이 깨짐`);
    assert.ok(!a.solved || name === "?", `${name}: 아무것도 안 바뀜`);
  }
});

test("PLL은 윗면 방향을 유지하고, 21가지 경우를 모두 덮는다", () => {
  const keys = new Map<string, string>();
  for (const { name, alg } of PLL) {
    const a = analyze(apply(SOLVED, alg));
    assert.ok(a.oriented, `${name}: 윗면 방향이 깨짐`);
    const key = llKey(apply(SOLVED, invertAlg(alg)), true);
    assert.ok(!keys.has(key), `${name}: ${keys.get(key)}와 같은 경우`);
    keys.set(key, name);
  }
  assert.equal(keys.size, 21);
});

test("OCLL 7가지는 모서리 방향을 유지하고 서로 다른 경우다", () => {
  const keys = new Set<string>();
  for (const { name, alg } of OCLL) {
    const a = analyze(apply(SOLVED, alg));
    assert.equal(a.yEdgesUp, 4, `${name}: 모서리 방향이 깨짐`);
    assert.ok(a.yCornersUp < 4, `${name}: 꼭짓점 방향이 그대로`);
    keys.add(llKey(apply(SOLVED, invertAlg(alg)), false));
  }
  assert.equal(keys.size, 7);
});
