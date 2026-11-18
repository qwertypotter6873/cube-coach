import { test } from "node:test";
import assert from "node:assert/strict";
import { solveWithPlanner, scrambled } from "./student.ts";
import { applySlotAlg } from "../src/coach/cube.ts";

for (const method of ["beginner", "cfop4", "cfop"] as const) {
  test(`${method}: 섞인 큐브 60개를 모두 맞춘다`, () => {
    const fails: number[] = [];
    let total = 0;
    for (let seed = 1; seed <= 60; seed++) {
      const log = solveWithPlanner(scrambled(seed), method);
      if (!log.solved) fails.push(seed);
      total += Object.values(log.moves).reduce((a, b) => a + (b ?? 0), 0);
    }
    assert.deepEqual(fails, [], `못 맞춘 시드: ${fails}`);
    console.log(`${method}: 평균 ${(total / 60).toFixed(1)}수`);
  });
}

test("큐브를 통째로 돌려 들고 있어도(가운데 색 배치가 달라도) 맞춘다", () => {
  for (let seed = 100; seed < 110; seed++) {
    const s = applySlotAlg(scrambled(seed), ["x", "y'", "z2", "x' y"][seed % 4]);
    assert.ok(solveWithPlanner(s, "beginner").solved);
  }
});
