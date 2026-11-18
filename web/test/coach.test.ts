import { test } from "node:test";
import assert from "node:assert/strict";
import { Coach } from "../src/coach/coach.ts";
import type { Utterance } from "../src/coach/coach.ts";
import { ALL_VIEWS, SOLVED, isSolved, mulberry32, applyColorMoves, rotateView, translate } from "../src/coach/cube.ts";
import type { ColorMove, View } from "../src/coach/cube.ts";
import { OCLL, PLL, planNext } from "../src/coach/method.ts";
import { Drill } from "../src/coach/drill.ts";
import { chunkSymbols, chunkText, holdText, moveSymbol, symbolSay } from "../src/coach/speech.ts";
import { drive, scrambled } from "./student.ts";

const bad = (u: Utterance[]) => u.filter((x) => !x.text || /undefined|NaN|\[object|\$\{/.test(x.text));

/** 코치가 시키는 대로 하는 학생. errRate 확률로 엉뚱한 면을 돌리고, 반 바퀴는 가끔 90도씩 두 번 나눠 돌린다. */
function run(seed: number, method: "beginner" | "cfop4" | "cfop", errRate: number, level: "guided" | "hint" = "guided") {
  const rand = mulberry32(seed * 7 + 1);
  const coach = new Coach(scrambled(seed), { method, level });
  const said: Utterance[] = [...coach.start(0)];
  let t = 0;
  for (let i = 0; i < 3000 && !isSolved(coach.s); i++) {
    t += 700;
    let m = coach.expected();
    if (!m) break;
    if (rand() < errRate) m = { color: Math.floor(rand() * 6), k: (1 + Math.floor(rand() * 3)) as 1 | 2 | 3 };
    if (m.k === 2 && rand() < 0.5) {
      const q = { color: m.color, k: (rand() < 0.5 ? 1 : 3) as 1 | 3 };
      said.push(...coach.onMove(q, t));
      t += 300;
      said.push(...coach.onMove(q, t));
    } else said.push(...coach.onMove(m, t));
  }
  return { coach, said };
}

for (const method of ["beginner", "cfop4", "cfop"] as const) {
  test(`${method}: 시키는 대로 하면 다 맞춰지고, 이상한 문장이 없다`, () => {
    for (let seed = 1; seed <= 15; seed++) {
      const { coach, said } = run(seed, method, 0);
      assert.ok(isSolved(coach.s), `시드 ${seed} 못 맞춤`);
      assert.deepEqual(bad(said), []);
      assert.ok(said.at(-1)!.kind === "summary", "끝나면 정리해 준다");
    }
  });
}

test("10% 확률로 실수해도 되돌리기 안내를 따라 결국 맞춘다", () => {
  for (let seed = 20; seed < 35; seed++) {
    const { coach, said } = run(seed, "beginner", 0.1);
    assert.ok(isSolved(coach.s), `시드 ${seed} 못 맞춤`);
    assert.ok(said.some((u) => u.kind === "fix"), "되돌리기 안내가 나왔어야 함");
    assert.deepEqual(bad(said), []);
  }
});

test("반대 방향으로 돌리면 '반대 방향' 안내, 다른 면이면 되돌리기 안내", () => {
  const coach = new Coach(scrambled(3), { method: "beginner", level: "guided" });
  coach.start(0);
  const exp = coach.expected()!;
  if (exp.k !== 2) {
    const u = coach.onMove({ color: exp.color, k: (4 - exp.k) as 1 | 3 }, 1000);
    assert.match(u[0].text, /반대 방향/);
  }
  const coach2 = new Coach(scrambled(3), { method: "beginner", level: "guided" });
  coach2.start(0);
  const e2 = coach2.expected()!;
  const other = (e2.color + 1) % 6 === 3 ? (e2.color + 2) % 6 : (e2.color + 1) % 6;
  const u2 = coach2.onMove({ color: other, k: 1 }, 1000);
  assert.match(u2[0].text, /되돌릴게요/);
  const fix = coach2.expected()!;
  assert.deepEqual(fix, { color: other, k: 3 });
});

test("observe: 혼자 풀면 단계별 시간을 기록하고 끝에 정리한다", () => {
  let s = scrambled(5);
  const coach = new Coach(s, { method: "cfop4", level: "observe" });
  coach.start(0);
  let t = 0;
  const said: Utterance[] = [];
  while (!isSolved(s)) {
    const step = planNext(s, "cfop4")!;
    for (const m of step.moves) {
      t += 400;
      said.push(...coach.onMove({ color: m.color, k: m.k } as ColorMove, t));
    }
    s = applyColorMoves(s, step.moves);
  }
  assert.deepEqual(Object.keys(coach.splits), ["cross", "f2l", "oll", "pll"]);
  const v = Object.values(coach.splits) as number[];
  assert.ok(v.every((x, i) => i === 0 || x >= v[i - 1]), "시간 순서");
  assert.equal(said.at(-1)!.kind, "summary");
});

test("hint 수준: 막히면 힌트가 점점 자세해지다가 한 수씩 불러 준다", () => {
  const coach = new Coach(scrambled(8), { method: "beginner", level: "hint", hintAfterMs: 5000, taughtBasics: true });
  coach.start(0);
  // 시작 안내를 다 말한 뒤부터 시간을 센다
  const h1 = coach.tick(40000);
  const h2 = coach.tick(46000);
  const h3 = coach.tick(52000);
  assert.ok(h1.length && h2.length && h3.length);
  assert.equal(coach.level, "guided");
  assert.ok(h3.some((u) => u.kind === "move"));
});

test("회전 도중 개입: 반대 방향이나 엉뚱한 면이면 다 돌리기 전에 말린다", () => {
  const coach = new Coach(scrambled(11), { method: "beginner", level: "guided" });
  coach.start(0);
  const exp = coach.expected()!;
  const dir = exp.k === 3 ? 1 : -1; // 기대와 반대 방향 (반 바퀴면 아무 방향이나 괜찮으므로 이 경우는 건너뜀)
  if (exp.k !== 2) {
    assert.deepEqual(coach.onTurning(exp.color, dir * 0.2, 100), [], "작게 흔들린 것은 무시");
    const w = coach.onTurning(exp.color, dir * 0.7, 200);
    assert.match(w[0].text, /반대 방향/);
    assert.deepEqual(coach.onTurning(exp.color, dir * 1.0, 300), [], "한 회전에 한 번만");
  }
  const coach2 = new Coach(scrambled(11), { method: "beginner", level: "guided" });
  coach2.start(0);
  const e2 = coach2.expected()!;
  const wrong = [0, 1, 2, 3, 4, 5].find((c) => c !== e2.color && c !== [3, 4, 5, 0, 1, 2][e2.color])!;
  const w2 = coach2.onTurning(wrong, 0.8, 100);
  assert.match(w2[0].text, /잠깐요/);
  assert.equal(w2[0].kind, "fix");
  // 바른 방향으로 돌리는 중이면 조용하다
  const coach3 = new Coach(scrambled(11), { method: "beginner", level: "guided" });
  coach3.start(0);
  const e3 = coach3.expected()!;
  assert.deepEqual(coach3.onTurning(e3.color, (e3.k === 3 ? -1 : 1) * 0.9, 100), []);
});

// --- 기준 면 안내 -------------------------------------------------------------

const START_POSE: View = { up: 3, front: 2 }; // 노랑 위, 초록 앞

/** 노랑 위·초록 앞으로 들고 시작해 첫 수까지 둔 코치 (첫 수 전에는 시작 기준으로 잡기를 기다리므로) */
function guided(seed: number, method: "beginner" | "cfop" = "beginner") {
  const coach = new Coach(scrambled(seed), { method, level: "guided", taughtBasics: true });
  coach.pose = START_POSE;
  const said = [...coach.start(0)];
  said.push(...coach.onMove(coach.expected()!, 100));
  return { coach, said };
}

/** 지시의 기호가 달라지고, 공식이 스스로 돌리는 방향과는 겹치지 않는 '돌려 잡기' 하나 */
function regrip(coach: Coach): View {
  const exp = coach.expected()!;
  const now = coach.pose!;
  for (const axis of ["y", "x", "z"]) {
    for (const k of [1, 3]) {
      const v = rotateView(coach.s, now, axis, k);
      if (moveSymbol(coach.s, v, exp) === moveSymbol(coach.s, now, exp)) continue;
      if (coach.currentStep!.moves.some((m) => m.viewBefore.up === v.up && m.viewBefore.front === v.front)) continue;
      return v;
    }
  }
  throw new Error("돌려 잡을 방향을 못 찾음");
}

/** 조건에 맞는 덩어리가 나올 때까지 계획대로 풀어 본다 */
function findStep(method: "beginner" | "cfop", ok: (step: NonNullable<ReturnType<typeof planNext>>) => boolean) {
  for (let seed = 1; seed <= 60; seed++) {
    let s = scrambled(seed);
    for (let i = 0; i < 80 && !isSolved(s); i++) {
      const step = planNext(s, method)!;
      if (ok(step)) return { s, step };
      s = applyColorMoves(s, step.moves);
    }
  }
  throw new Error("그런 덩어리를 못 찾음");
}

test("시작할 때 기준 면과 기호 읽는 법을 알려 준다 (처음은 자세히, 다음부터는 한 줄)", () => {
  const begin = (method: "beginner" | "cfop", level: "guided" | "hint" | "observe", taughtBasics: boolean) => {
    const c = new Coach(scrambled(2), { method, level, taughtBasics });
    c.pose = START_POSE; // main.ts 처럼 노랑 위, 초록 앞으로 들고 시작
    return c.start(0);
  };
  const first = begin("beginner", "guided", false);
  const intro = first.filter((u) => u.kind === "intro").map((u) => u.text);
  assert.ok(intro[0].includes("화면 속 큐브가 아니라") && intro[0].includes("손에 든 큐브"), "화면이 아니라 손에 든 큐브 기준이라고 말한다");
  assert.ok(intro.some((x) => x.includes("나를 보는 면이 앞면 에프") && x.includes("위가 윗면 유")), "나를 보는 면이 F, 위가 U");
  assert.ok(intro.some((x) => ["오른쪽이 알", "왼쪽이 엘", "아래가 디", "뒤가 비"].every((l) => x.includes(l))), "R L D B 가 어느 면인지");
  assert.ok(intro.some((x) => x.includes("시계 방향") && x.includes("프라임") && x.includes("투")), "기호·프라임·2 읽는 법");
  // 기준 면의 색: 첫 지시 바로 앞에서 잡는 법으로 말한다 (카메라가 아직 자세를 못 읽었을 수 있어 "지금 ~예요"가 아니다)
  const firstMove = first.findIndex((u) => u.kind === "move");
  assert.ok(firstMove > 0, "첫 지시가 나온다");
  assert.equal(first[firstMove - 1].text, holdText(START_POSE), "첫 지시 바로 앞에서 기준 면을 색으로 말한다");
  assert.notEqual(first[firstMove - 1].kind, "hold", "바로 뒤 지시가 지우지 않게 '잡기 지시'가 아닌 종류로 둔다");

  const again = begin("beginner", "guided", true);
  assert.equal(again.filter((u) => u.kind === "intro" && u.text.includes("손에 든 큐브")).length, 1, "두 번째부터는 한 줄만");
  assert.ok(!again.some((u) => u.text.includes("구십 도")), "기호 설명은 되풀이하지 않는다");
  assert.ok(again.some((u) => u.text === holdText(START_POSE)), "기준 면 색은 매번 말한다");
  const cfop = begin("cfop", "guided", false);
  assert.ok(cfop.filter((u) => u.kind === "intro").length < first.filter((u) => u.kind === "intro").length, "CFOP는 더 짧게");
  assert.ok(cfop.some((u) => u.text.includes("시계 방향") && u.text.includes("프라임") && u.text.includes("투")), "CFOP도 시계 방향·프라임·투는 짚는다");
  assert.deepEqual(begin("beginner", "observe", false).filter((u) => u.kind === "intro"), [], "혼자 풀기에서는 설명하지 않는다");
});

test("처음 기준으로 잡을 방향이 지금과 다르면, 그렇게 잡을 때까지 기다렸다가 첫 수를 부른다", () => {
  const coach = new Coach(scrambled(2), { method: "beginner", level: "guided", taughtBasics: true });
  coach.pose = { up: 0, front: 1 }; // 흰색이 위: 초급 십자는 노랑을 위로 잡는다
  const said = coach.start(0);
  const hold = said.find((u) => u.kind === "hold");
  assert.ok(hold && /^노랑이 위/.test(hold.text), "노랑을 위로 잡게 한다");
  assert.deepEqual(said.filter((u) => u.kind === "move"), [], "잡기 전에는 첫 수를 부르지 않는다");
  assert.equal(coach.grip?.up, 3, "화면에 띄울 잡는 법 (노랑 위)");
  const back = coach.onPose(coach.grip!, 1000);
  assert.ok(back.some((u) => u.kind === "move"), "맞게 잡으면 첫 수를 부른다");
  assert.equal(coach.grip, null);
});

test("지시는 언제나 기호와 쉬운 말을 함께 말한다", () => {
  const sym = /(^|[\s,.])(유|디|알|엘|에프|비)( 더블)?( 투| 프라임)?[,.]/;
  for (const seed of [3, 9, 21]) {
    const { said } = run(seed, "beginner", 0.1);
    for (const u of said.filter((x) => x.kind === "move" || x.kind === "fix")) {
      assert.ok(sym.test(u.text) || /섹시무브|한 번 더|잠깐|흐트러졌어요|괜찮아요/.test(u.text), `기호가 빠진 안내: ${u.text}`);
    }
  }
});

test("들고 있는 방향이 바뀌면 기준을 알리고, 같은 지시를 새 기준의 기호로 다시 말한다", () => {
  const { coach, said } = guided(4);
  const exp = coach.expected()!;
  const before = moveSymbol(coach.s, coach.pose!, exp);
  assert.ok(said.some((u) => u.kind === "move"));
  const v = regrip(coach);
  assert.deepEqual(coach.onPose(v, 1000), [], "자세가 바뀐 그 순간에는 말하지 않는다");
  assert.deepEqual(coach.tick(1400), [], "흔들림일 수 있으니 자리 잡을 때까지 기다린다");
  const out = coach.tick(2200);
  assert.equal(out[0].kind, "status");
  assert.match(out[0].text, /^이제 .*(앞|위)/);
  const move = out.find((u) => u.kind === "move");
  assert.ok(move, "기다리던 지시를 다시 말한다");
  assert.deepEqual(coach.expected(), exp, "돌려 잡아도 해야 할 회전은 그대로");
  const after = moveSymbol(coach.s, v, exp);
  assert.notEqual(after, before);
  assert.equal(move!.text.startsWith(symbolSay(coach.s, v, exp)), true, `새 기준 기호(${after})로 불러야 한다: ${move!.text}`);
  assert.equal(coach.reference().front, v.front, "기준 면 getter도 함께 바뀐다");
});

test("자세가 왔다 갔다 하면 같은 알림을 되풀이하지 않는다", () => {
  const { coach } = guided(4);
  const a = coach.pose!;
  const b = regrip(coach);
  coach.onPose(b, 1000);
  const n1 = coach.tick(2200);
  coach.onPose(a, 2400);
  const n2 = coach.tick(3600);
  coach.onPose(b, 3800);
  const n3 = coach.tick(5000);
  const notices = (us: Utterance[]) => us.filter((u) => u.kind === "status").length;
  assert.equal(notices(n1), 1);
  assert.equal(notices(n2), 1);
  assert.equal(notices(n3), 0, "조금 전에 알린 기준은 또 알리지 않는다");
  assert.ok(n3.some((u) => u.kind === "move"), "알림은 생략해도 지시는 새 기준으로 다시 말한다");
});

test("층을 돌리는 도중에 자세가 바뀌면, 다 돌릴 때까지 알림을 미룬다", () => {
  const { coach } = guided(4);
  const exp = coach.expected()!;
  const v = regrip(coach);
  coach.onPose(v, 1000);
  assert.deepEqual(coach.onTurning(exp.color, exp.k === 3 ? -0.2 : 0.2, 2000), [], "조금 돌린 것은 말리지 않는다");
  assert.deepEqual(coach.tick(2100), [], "돌리는 중에는 미룬다");
  assert.ok(coach.tick(3000).some((u) => u.kind === "status"), "다 돌린 뒤에 알린다");
});

test("공식이 큐브를 통째로 돌리는 수(두 줄 돌리기) 때문에 바뀐 자세는 알리지 않는다", () => {
  const { s, step } = findStep("cfop", (st) => !!st.view && st.moves.some((m) => /^[a-z]/.test(m.token)));
  const coach = new Coach(s, { method: "cfop", level: "guided", verbosity: "full", taughtBasics: true });
  coach.pose = step.view!;
  const wide = step.moves.find((m) => /^[a-z]/.test(m.token))!;
  assert.match(symbolSay(s, wide.viewBefore, wide), /더블/, "두 줄 돌리기 기호는 '더블'로 읽는다");
  assert.match(chunkText(s, step.view!, step.moves), /더블/, "덩어리를 한 번에 부를 때도 마찬가지");
  const said: Utterance[] = [...coach.start(0)];
  let t = 0;
  for (let i = 0; i < step.moves.length; i++) {
    const m = step.moves[i];
    t += 500;
    said.push(...coach.onMove({ color: m.color, k: m.k }, t));
    const next = step.moves[i + 1];
    if (!next) break;
    t += 100;
    said.push(...coach.onPose(next.viewBefore, t)); // 공식이 큐브를 돌려 놓은 만큼 추적기가 알려 준다
    t += 1500;
    const late = coach.tick(t);
    said.push(...late);
    assert.deepEqual(late.filter((u) => u.kind === "status"), [], "공식이 돌린 것은 알릴 일이 아니다");
  }
});

test("첫 수 전에 정해 둔 잡는 법과 다르게 잡으면, 기준 알림 없이 그 잡는 법을 기다린다", () => {
  const { s, step } = findStep("beginner", (st) => !!st.view);
  const coach = new Coach(s, { method: "beginner", level: "guided", taughtBasics: true, repeatAfterMs: 4000 });
  coach.pose = step.view!;
  coach.start(0);
  const v = regrip(coach);
  assert.deepEqual(coach.onPose(v, 1000), []);
  assert.deepEqual(coach.tick(2200), [], "첫 수 전에는 바뀐 기준을 알리거나 다른 기호로 다시 부르지 않는다");
  assert.deepEqual(coach.grip, step.view, "화면에는 잡는 법을 띄운다");
  const again = coach.tick(60000);
  assert.ok(again.some((u) => u.kind === "hold" && u.text === holdText(step.view!)), "오래 그대로면 잡는 법을 한 번 더");
  const back = coach.onPose(step.view!, 61000);
  assert.ok(back.some((u) => u.kind === "move"), "맞게 잡으면 다시 불러 준다");
});

test("카메라: 처음 자세가 짐작(노랑 위·초록 앞)과 달라도 '이제 …' 없이 기준대로 잡기를 기다리고, 잡으면 첫 수를 다시 부른다", () => {
  const coach = new Coach(scrambled(2), { method: "beginner", level: "guided", taughtBasics: true });
  coach.pose = START_POSE; // main.ts 는 시작할 때 자세를 노랑 위·초록 앞으로 둔다 (카메라가 읽은 값이 아니다)
  const said = coach.start(0);
  const first = said.find((u) => u.kind === "move")!;
  assert.deepEqual(coach.onPose({ up: 0, front: 1 }, 300), [], "추적기가 실제 자세(흰색 위)를 알려 줘도");
  assert.deepEqual(coach.tick(1500), [], "바뀐 기준을 알리거나 다른 기호로 부르지 않는다");
  assert.deepEqual(coach.grip, START_POSE, "화면에는 잡는 법을 띄운다");
  const back = coach.onPose(START_POSE, 3000);
  assert.ok(back.some((u) => u.kind === "move" && u.text === first.text), "맞게 잡으면 첫 수를 같은 기호로 다시");
  assert.deepEqual(coach.tick(4500), [], "잡은 뒤에 기준 알림이 또 나오지 않는다");
});

test("시작 안내를 말하는 동안에는 재촉하지 않는다", () => {
  const coach = new Coach(scrambled(6), { method: "beginner", level: "guided", taughtBasics: false, repeatAfterMs: 4000 });
  coach.pose = START_POSE;
  coach.start(0);
  assert.deepEqual(coach.tick(5000), [], "아직 설명을 말하는 중");
  assert.ok(coach.tick(90000).some((u) => u.text.startsWith("다시,")), "다 말하고도 가만히 있으면 다시 불러 준다");
});

test("반대로 돌린 뒤 '두 번' 안내를 받고 한 번 돌리면, 진행으로 보고 '한 번 더'라고 한다", () => {
  for (let seed = 1; seed < 40; seed++) {
    const coach = new Coach(scrambled(seed), { method: "beginner", level: "guided" });
    coach.start(0);
    const exp = coach.expected()!;
    if (exp.k === 2) continue;
    const wrong = coach.onMove({ color: exp.color, k: (4 - exp.k) as 1 | 3 }, 100);
    assert.match(wrong[0].text, /반대 방향/);
    assert.equal(coach.expected()!.k, 2);
    const half = coach.onMove({ color: exp.color, k: exp.k }, 200);
    assert.match(half[0].text, /한 번 더/);
    const fin = coach.onMove({ color: exp.color, k: exp.k }, 300);
    assert.ok(!fin.some((u) => u.kind === "fix"), "다 바로잡았으면 바로잡기 안내가 없어야 함");
    return;
  }
});

// --- 실제 흐름(카메라·키보드)을 흉내 내는 학생 --------------------------------------

test("카메라·키보드 학생이 끝까지 풀고, 지시 기호는 언제나 그 순간 손에 든 방향의 기호다", () => {
  for (const method of ["beginner", "cfop4", "cfop"] as const) {
    for (const mode of ["camera", "keyboard"] as const) {
      for (const seed of [3, 9, 14]) {
        const d = drive({ seed, method, mode, regripEvery: 5, errRate: 0.05 });
        assert.ok(isSolved(d.coach.s), `${method}/${mode}/${seed} 못 맞춤`);
        assert.deepEqual(d.problems, [], `${method}/${mode}/${seed}`);
        assert.deepEqual(bad(d.said), []);
        assert.ok(d.said.some((u) => u.kind === "status" && u.text.startsWith("이제 ")), "돌려 잡으면 바뀐 기준을 알린다");
      }
    }
  }
});

test("공식을 한 번에 부르는 수준과 힌트 수준에서도 기호가 든 방향과 맞는다 (두 줄 돌리기 공식 포함)", () => {
  for (const mode of ["camera", "keyboard"] as const) {
    for (const seed of [3, 9]) {
      for (const [level, verbosity] of [["guided", "short"], ["hint", undefined]] as const) {
        const d = drive({ seed, method: "cfop4", mode, level, verbosity, regripEvery: 7 });
        assert.ok(isSolved(d.coach.s));
        assert.deepEqual(d.problems, [], `${mode}/${seed}/${level}`);
      }
    }
  }
});

test("두 줄 돌리기로 큐브가 함께 돌면 기준이 바뀌었다고 알리지 않는다 (공식을 한 번에 부른 경우)", () => {
  const lines: { t: number; kind: string; text: string }[] = [];
  const d = drive({ seed: 3, method: "cfop4", level: "guided", verbosity: "short", mode: "camera", log: (t, u) => lines.push({ t, ...u }) });
  assert.deepEqual(d.problems, []);
  const at = lines.findIndex((l) => l.kind === "move" && l.text.startsWith("알 더블 오른쪽 두 줄 위로"));
  assert.ok(at > 0, "T 공식을 알 더블로 시작해 한 번에 부른다");
  const end = lines.findIndex((l, i) => i > at && l.kind === "stage");
  assert.deepEqual(lines.slice(at + 1, end).filter((l) => l.kind === "status" || l.kind === "move"), [], "공식 도중 기준 알림이나 다시 부르기가 없다");
});

test("한 번에 부르는 공식 기호는, 다르게 들고 있어도 두 줄 돌리기 뒤까지 실제 돌릴 층과 맞는다", () => {
  const algs = [OCLL.find((a) => a.name === "T")!, OCLL.find((a) => a.name === "나비")!, PLL.find((a) => a.name === "V")!, PLL.find((a) => a.name === "E")!];
  for (const alg of algs) {
    const ms = translate(SOLVED, alg.alg, { up: 3, front: 2 });
    for (const p of ALL_VIEWS) {
      const syms = chunkSymbols(SOLVED, p, ms).join(" ");
      const again = translate(SOLVED, syms, p).map((m) => [m.color, m.k]);
      assert.deepEqual(again, ms.map((m) => [m.color, m.k]), `${alg.name} / 든 방향 ${p.up},${p.front}: ${syms}`);
    }
  }
});

test("잡는 방향이 정해지지 않은 단계(십자)에서 돌려 잡으면, 어느 방향이든 새 기준 기호로 다시 말한다", () => {
  let checked = 0;
  for (let seed = 1; seed <= 30 && checked < 3; seed++) {
    const { coach } = guided(seed);
    const step = coach.currentStep;
    const pm = coach.pendingMove();
    if (!step || step.stage !== "cross" || step.view || !pm) continue;
    const exp = coach.expected()!;
    // 예전에는 이 수의 공식 기준 방향(노랑 위, 그 모서리 색 앞)으로 돌려 잡으면 아무 말도 없었다
    const v = pm.viewBefore;
    if (moveSymbol(coach.s, v, exp) === moveSymbol(coach.s, coach.pose!, exp)) continue;
    coach.onPose(v, 1000);
    const out = coach.tick(2200);
    assert.ok(out.some((u) => u.kind === "status"), `시드 ${seed}: 바뀐 기준을 알린다`);
    const move = out.find((u) => u.kind === "move");
    assert.ok(move?.text.startsWith(symbolSay(coach.s, v, exp) + ","), `시드 ${seed}: 새 기준 기호로 다시 말한다 (${move?.text})`);
    checked++;
  }
  assert.ok(checked > 0, "시험할 장면을 못 찾음");
});

test("공식 도중 돌려 잡으면 잡던 방향으로 돌아오게 하고, 돌아오면 같은 수를 원래 기호로 다시 부른다", () => {
  const { s, step } = findStep("beginner", (st) => st.alg === "R U R' U'");
  const coach = new Coach(s, { method: "beginner", level: "guided", taughtBasics: true });
  coach.pose = step.view!;
  coach.start(0);
  const m0 = coach.expected()!;
  coach.onMove(m0, 500); // 알
  const exp = coach.expected()!;
  const said = moveSymbol(s, step.view!, exp);
  assert.equal(said, "U");
  const v = rotateView(coach.s, step.view!, "y", 1);
  coach.onPose(v, 1000);
  const out = coach.tick(2200);
  assert.ok(out.some((u) => u.kind === "hold" && u.text === holdText(step.view!)), "잡던 방향으로 돌아오게 한다");
  assert.deepEqual(out.filter((u) => u.kind === "move"), [], "돌아올 때까지 다음 수를 다른 기호로 부르지 않는다");
  assert.deepEqual(coach.grip, step.view, "화면에 띄울 잡는 법");
  const back = coach.onPose(step.view!, 3000);
  assert.ok(back.some((u) => u.kind === "move" && u.text.startsWith("유,")), "돌아오면 같은 수를 원래 기호(U)로");
  assert.equal(coach.grip, null);
});

test("잡아 달라고 했는데 가만히 있으면, 잡는 법을 한 번 더 말하고 그래도 그대로면 지금 방향으로 지시한다", () => {
  const coach = new Coach(scrambled(2), { method: "beginner", level: "guided", taughtBasics: true, repeatAfterMs: 4000 });
  coach.pose = { up: 0, front: 1 };
  coach.start(0);
  const r1 = coach.tick(60000);
  assert.equal(r1.length, 1);
  assert.equal(r1[0].kind, "hold", "먼저 잡는 법을 다시");
  const r2 = coach.tick(65000);
  assert.ok(r2.some((u) => u.kind === "move" && u.text.startsWith("다시, ")), "그다음엔 지금 방향으로 지시");
  assert.equal(coach.grip, null, "더는 잡기를 기다리지 않는다");
  assert.deepEqual(coach.tick(90000), [], "같은 재촉을 끝없이 하지 않는다");
});

test("공식 연습: 쥔 방향이 바뀌면 바뀐 기준과 준비 동작을 새 기호로 다시 부르고, 흔들리면 알림은 생략한다", () => {
  const d = new Drill(SOLVED, PLL.slice(0, 2), "pll", mulberry32(5));
  d.pose = { up: 3, front: 2 };
  d.start(0);
  const exp = d.expected()!;
  const a = d.pose;
  const b = rotateView(d.s, a, "y", 1);
  const o1 = d.onPose(b, 1000);
  assert.equal(o1[0].kind, "status");
  assert.ok(o1[1].text.startsWith(symbolSay(d.s, b, exp)), "새 기준 기호로 다시 부른다");
  const o2 = d.onPose(a, 1500);
  const o3 = d.onPose(b, 2000);
  assert.equal(o2[0].kind, "status");
  assert.ok(!o3.some((u) => u.kind === "status"), "조금 전에 알린 기준은 또 알리지 않는다");
  assert.ok(o3.some((u) => u.kind === "move"), "동작은 다시 부른다");
});

test("fixing: 엉뚱한 면을 되돌리거나 반대로 돌린 것을 마저 돌리는 동안만 true", () => {
  const coach = new Coach(scrambled(3), { method: "beginner", level: "guided", taughtBasics: true });
  coach.start(0);
  const exp = coach.expected()!;
  assert.equal(coach.fixing, false);
  const other = [0, 1, 2, 3, 4, 5].find((c) => c !== exp.color && c !== [3, 4, 5, 0, 1, 2][exp.color])!;
  coach.onMove({ color: other, k: 1 }, 100);
  assert.equal(coach.fixing, true, "되돌리는 중");
  assert.equal(coach.pendingMove(), null);
  coach.onMove({ color: other, k: 3 }, 200);
  assert.equal(coach.fixing, false, "되돌렸으면 끝");
  assert.deepEqual(coach.expected(), exp);
  if (exp.k !== 2) {
    coach.onMove({ color: exp.color, k: (4 - exp.k) as 1 | 3 }, 300); // 반대로
    assert.equal(coach.fixing, true, "반대로 돌린 것을 마저 돌리는 중");
    coach.onMove({ color: exp.color, k: 2 }, 400);
    assert.equal(coach.fixing, false);
  }
});

// --- 끝난 풀이는 한 번만 기록한다 -------------------------------------------------------

/** 혼자 풀기: 계획대로 혼자 다 푼다 */
function solveObserve(seed: number) {
  let s = scrambled(seed);
  const coach = new Coach(s, { method: "cfop4", level: "observe" });
  const said: Utterance[] = [...coach.start(0)];
  let t = 0;
  while (!isSolved(s)) {
    const step = planNext(s, "cfop4")!;
    for (const m of step.moves) {
      t += 400;
      said.push(...coach.onMove({ color: m.color, k: m.k }, t));
    }
    s = applyColorMoves(s, step.moves);
  }
  return { coach, said };
}

test("다 맞춘 뒤 새로 시작하지 않고 한 수 돌렸다 되돌려도, 완성 정리·회전·단계 시간이 다시 생기지 않는다", () => {
  const solved = [
    { level: "guided", ...run(5, "beginner", 0) },
    { level: "hint", ...run(5, "beginner", 0, "hint") },
    { level: "observe", ...solveObserve(5) },
  ];
  for (const { level, coach, said } of solved) {
    assert.equal(said.filter((u) => u.kind === "summary").length, 1, `${level}: 완성 정리는 한 번`);
    assert.equal(coach.closed, true, `${level}: 다 맞추면 풀이가 닫힌다`);
    const moves = coach.moves.length;
    const splits = { ...coach.splits };
    const steps = coach.steps.length;
    const t = coach.moves.at(-1)!.t;
    assert.deepEqual(coach.onMove({ color: 2, k: 1 }, t + 1000), [], `${level}: 한 수 돌려도 아무 말도 하지 않는다`);
    assert.ok(!isSolved(coach.s), `${level}: 큐브 상태는 따라간다`);
    assert.deepEqual(coach.onMove({ color: 2, k: 3 }, t + 1500), [], `${level}: 되돌려도 완성 정리를 다시 하지 않는다`);
    assert.ok(isSolved(coach.s));
    assert.equal(coach.moves.length, moves, `${level}: 기록할 회전 수는 그대로`);
    assert.deepEqual(coach.splits, splits, `${level}: 단계 시간은 그대로 (splits.undefined 가 생기지 않는다)`);
    assert.equal(coach.steps.length, steps);
    assert.equal(coach.currentStage, "done", `${level}: 화면은 '완성'에 머문다 (다시 하려면 시작 버튼)`);
    assert.deepEqual(coach.tick(t + 60000), []);
  }
});

test("맞춰진 큐브로 시작했으면 섞는 회전을 풀이로 기록하지 않는다 (단계 시간·완성 정리 없음)", () => {
  for (const level of ["guided", "hint", "observe"] as const) {
    const coach = new Coach(SOLVED, { method: "beginner", level, taughtBasics: true });
    assert.match(coach.start(0)[0].text, /다 맞춰진 큐브/);
    assert.equal(coach.closed, true, level);
    assert.deepEqual(coach.onMove({ color: 2, k: 1 }, 1000), [], level);
    assert.deepEqual(coach.onMove({ color: 2, k: 3 }, 1500), [], level);
    assert.deepEqual(coach.moves, [], level);
    assert.deepEqual(coach.splits, {}, level);
  }
});

// --- 힌트만 수준: 화면에 보여 줄 공식 이름과 기호 -----------------------------------------

/** F2L까지 맞고 OLL 안티수네만 남은 큐브 (다 맞춘 큐브에 수네를 하면 안티수네로 풀린다) */
const antiSune = () => applyColorMoves(SOLVED, translate(SOLVED, "R U R' U R U2 R'", START_POSE));

test("힌트만: 화면의 공식 기호는 코치가 불러 준 만큼만 보이고, 공식 이름은 말한 뒤에 보인다", () => {
  const coach = new Coach(antiSune(), { method: "cfop4", level: "hint", hintAfterMs: 5000, taughtBasics: true });
  coach.pose = START_POSE;
  const said = coach.start(0);
  const step = coach.currentStep!;
  assert.equal(step.caseName, "안티수네");
  assert.ok(said.some((u) => u.kind === "goal" && u.text.includes("안티수네")), "힌트만 수준의 목표 문장이 공식 이름을 말한다");
  assert.deepEqual(coach.revealed(), { caseName: true, moves: 0 }, "처음에는 기호를 보여 주지 않는다");
  const h1 = coach.tick(40000);
  assert.ok(h1.some((u) => u.text === "안티수네를 써 보세요."), "첫 힌트");
  assert.equal(coach.revealed().moves, 0, "첫 힌트(공식 이름)까지도 기호는 숨긴다");
  const h2 = coach.tick(46000);
  assert.ok(h2.some((u) => u.text.startsWith("처음은 ")), "둘째 힌트");
  assert.equal(coach.revealed().moves, 3, "둘째 힌트로 불러 준 세 수까지 보인다");
  coach.onMove(coach.expected()!, 47000);
  assert.equal(coach.revealed().moves, 3, "불러 준 수는 한 수 둔 뒤에도 보인다");
  coach.tick(53000);
  assert.equal(coach.level, "guided", "셋째 힌트부터는 한 수씩 불러 준다");
  assert.deepEqual(coach.revealed(), { caseName: true, moves: step.moves.length }, "그때부터는 다 보인다");
});

test("힌트만: 다른 길로 가서 계획을 새로 세우면, 아직 불러 주지 않은 새 계획의 기호는 숨긴다", () => {
  const coach = new Coach(antiSune(), { method: "cfop4", level: "hint", hintAfterMs: 5000, taughtBasics: true });
  coach.pose = START_POSE;
  coach.start(0);
  coach.tick(40000);
  coach.tick(46000);
  assert.equal(coach.revealed().moves, 3);
  // 윗면(노랑)을 돌렸다: 공식의 첫 수가 아니다. 같은 단계 안에서 말없이 계획만 다시 세운다.
  assert.deepEqual(coach.onMove({ color: 3, k: 1 }, 47000), []);
  assert.equal(coach.currentStage, "oll");
  assert.equal(coach.revealed().moves, 0, "새 계획의 기호는 아직 불러 주지 않았다");
  assert.equal(coach.revealed().caseName, coach.currentStep!.caseName === "안티수네", "공식 이름은 앞서 말한 이름과 같을 때만");
});

test("따라 하기는 공식 이름과 기호를 처음부터 다 보여 준다 (가리는 것은 힌트만 수준뿐)", () => {
  const coach = new Coach(antiSune(), { method: "cfop4", level: "guided", taughtBasics: true });
  coach.pose = START_POSE;
  coach.start(0);
  assert.deepEqual(coach.revealed(), { caseName: true, moves: coach.currentStep!.moves.length });
});
