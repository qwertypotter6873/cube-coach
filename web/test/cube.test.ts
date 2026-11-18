import { test } from "node:test";
import assert from "node:assert/strict";
import {
  SOLVED, applySlotAlg, isSolved, translate, applyColorMoves, viewColors, viewWithRight,
  EDGES, CORNERS, invertAlg, randomScramble, mulberry32, ALL_VIEWS, simplify,
} from "../src/coach/cube.ts";

test("알려진 공식의 반복 횟수 (파이썬 엔진과 같은 표)", () => {
  for (const [alg, order] of [["R U R' U'", 6], ["R U", 105], ["R U R' U' R' F R2 U' R' U' R U R' F'", 2]] as const) {
    let s = SOLVED;
    for (let i = 1; i <= order; i++) {
      s = applySlotAlg(s, alg);
      assert.equal(isSolved(s), i === order, `${alg} ${i}번째`);
    }
  }
});

test("조각 수", () => {
  assert.equal(EDGES.length, 12);
  assert.equal(CORNERS.length, 8);
});

test("보는 방향: 표준 배색에서 흰 위 초록 앞이면 오른쪽은 빨강", () => {
  const vc = viewColors(SOLVED, { up: 0, front: 2 });
  assert.deepEqual(vc, { U: 0, F: 2, R: 1, D: 3, B: 5, L: 4 });
  // 흰 아래, 오른쪽 빨강이면 앞은 파랑
  assert.equal(viewWithRight(SOLVED, 3, 1).front, 5);
  assert.equal(ALL_VIEWS.length, 24);
});

test("공식 번역: 어느 방향에서 보든 공식과 그 역은 원래대로", () => {
  const rand = mulberry32(1);
  for (let t = 0; t < 50; t++) {
    const s = applySlotAlg(SOLVED, randomScramble(20, rand) + " y x2");
    const v = ALL_VIEWS[Math.floor(rand() * 24)];
    const alg = "R U R' F' L2 D B'";
    const s2 = applyColorMoves(applyColorMoves(s, translate(s, alg, v)), translate(s, invertAlg(alg), v));
    assert.deepEqual(s2, s);
  }
});

test("번역은 통째 회전과 맞물린다: (흰 위, 초록 앞)의 R 은 큐브를 돌린 뒤에도 빨강 면", () => {
  const s = applySlotAlg(SOLVED, "y");
  const m = translate(s, "R", { up: 0, front: 2 })[0];
  assert.equal(m.color, 1);
});

test("같은 면 합치기", () => {
  const ms = simplify([{ color: 1, k: 1 }, { color: 1, k: 1 }, { color: 2, k: 1 }, { color: 2, k: 3 }] as const);
  assert.deepEqual(ms, [{ color: 1, k: 2 }]);
});
