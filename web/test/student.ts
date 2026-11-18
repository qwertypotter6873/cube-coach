/** 가상 학생: 코치의 계획을 그대로 따라 하며 섞인 큐브를 맞춘다. 테스트와 통계에 함께 쓴다. */
import { SOLVED, applySlotAlg, applyColorMoves, randomScramble, mulberry32, isSolved } from "../src/coach/cube.ts";
import type { State } from "../src/coach/cube.ts";
import { planNext, analyze, stageOf } from "../src/coach/method.ts";
import type { Method, StageId } from "../src/coach/method.ts";
import { Coach } from "../src/coach/coach.ts";
import type { Level, Utterance } from "../src/coach/coach.ts";
import { letterOf, rotateView } from "../src/coach/cube.ts";
import type { ColorMove, View } from "../src/coach/cube.ts";
import { moveSymbol, sameView, symbolSay } from "../src/coach/speech.ts";

export interface SolveLog {
  solved: boolean;
  steps: number;
  moves: Partial<Record<StageId, number>>;
  ms: number;
}

export function solveWithPlanner(s: State, method: Method, maxSteps = 120): SolveLog {
  const t0 = performance.now();
  const moves: Partial<Record<StageId, number>> = {};
  let steps = 0;
  while (!isSolved(s) && steps < maxSteps) {
    const stage = stageOf(analyze(s), method);
    const step = planNext(s, method);
    if (!step) break;
    s = applyColorMoves(s, step.moves);
    moves[stage] = (moves[stage] ?? 0) + step.moves.length;
    steps++;
  }
  return { solved: isSolved(s), steps, moves, ms: performance.now() - t0 };
}

export function scrambled(seed: number): State {
  return applySlotAlg(SOLVED, randomScramble(25, mulberry32(seed)));
}

// --- 카메라·키보드 흐름을 흉내 내는 학생 (main.ts 처럼 자세를 알려 준다) ---------------------------

export interface DriveOptions {
  seed: number;
  method: Method;
  level?: Level;
  /** camera: 두 줄 돌리기를 하면 큐브가 함께 돌아 새 자세가 들어온다. keyboard: 자세는 화살표로만 바뀐다. */
  mode?: "camera" | "keyboard";
  /** 엉뚱한 면을 돌릴 확률 */
  errRate?: number;
  /** 몇 수마다 큐브를 통째로 돌려 잡는지 (0이면 안 함) */
  regripEvery?: number;
  taughtBasics?: boolean;
  verbosity?: "full" | "short";
  /** 대본용: 말과 사람의 행동을 차례로 받는다 */
  log?: (t: number, line: { kind: string; text: string }) => void;
}

export interface Drive {
  coach: Coach;
  said: Utterance[];
  /** 사람이 실제로 든 방향과 맞지 않는 기호로 지시한 경우 등 */
  problems: string[];
}

const SYM = /(?:^|[\s,.])((?:유|디|알|엘|에프|비)(?: 더블)?(?: 투| 프라임)?)(?=[\s,.])/;
/** 두 줄 돌리기는 반대쪽 층(color)과 함께 큐브가 통째로 돈다: 그 층을 바깥에서 볼 때 반시계 방향으로 k번 */
const CCW_FROM: Record<string, [string, number]> = { R: ["x", -1], L: ["x", 1], U: ["y", -1], D: ["y", 1], F: ["z", -1], B: ["z", 1] };

/**
 * 시키는 대로 하는 학생. main.ts 처럼 처음에 노랑 위·초록 앞으로 자세를 알려 주고,
 * 잡아 달라는 말을 들으면 그렇게 잡고, 500ms마다 tick 을 보낸다.
 * 코치가 말한 기호가 그 순간 학생이 실제로 든 방향의 기호와 같은지 매번 확인한다.
 */
export function drive(o: DriveOptions): Drive {
  const rand = mulberry32(o.seed * 7 + 3);
  const coach = new Coach(scrambled(o.seed), { method: o.method, level: o.level ?? "guided", taughtBasics: o.taughtBasics ?? true, verbosity: o.verbosity });
  let pose: View = { up: 3, front: 2 };
  coach.pose = pose;
  const said: Utterance[] = [];
  const problems: string[] = [];
  let t = 0;
  const act = (text: string) => o.log?.(t, { kind: "사람", text });
  let heardWide = false; // 마지막 지시가 두 줄 돌리기("더블")를 말했는가
  const hear = (us: Utterance[]) => {
    for (const u of us) {
      said.push(u);
      o.log?.(t, u);
      const exp = coach.expected();
      if (!exp) continue;
      if (u.kind === "move" || u.kind === "fix" || u.kind === "hint") {
        heardWide = u.text.includes("더블");
        const m = SYM.exec(u.text);
        // 한 수씩 부를 때는 두 줄 돌리기도 한 층(반대쪽 면)으로 부르고, 한 번에 부를 때는 "알 더블"로 부른다
        const pm = coach.pendingMove();
        const ok = [symbolSay(coach.s, pose, exp), ...(pm && pm.color === exp.color ? [symbolSay(coach.s, pose, { ...pm, k: exp.k })] : [])];
        if (m && !ok.includes(m[1])) problems.push(`${(t / 1000).toFixed(1)}s 든 방향은 ${ok.join("/")}인데 "${u.text}"`);
      }
      if (u.kind === "move" && u.text.includes("섹시무브") && moveSymbol(coach.s, pose, exp) !== "R") {
        problems.push(`${(t / 1000).toFixed(1)}s 다르게 들고 있는데 묶음 이름으로 불렀다: "${u.text}"`);
      }
    }
  };
  const tickFor = (ms: number) => {
    for (let end = t + ms; t < end; ) {
      t += 500;
      hear(coach.tick(t));
    }
  };
  hear(coach.start(0));
  t = Math.max(t, 0);
  for (let i = 0; i < 3000 && !isSolved(coach.s); i++) {
    tickFor(1000);
    // 잡아 달라고 하면 그렇게 잡는다
    const g = coach.grip;
    if (g && !sameView(g, pose)) {
      pose = g;
      act("말대로 잡는다");
      t += 300;
      hear(coach.onPose(pose, t));
      tickFor(500);
    }
    if (o.regripEvery && i && i % o.regripEvery === 0) {
      const axis = "xyz"[Math.floor(rand() * 3)];
      pose = rotateView(coach.s, pose, axis, rand() < 0.5 ? 1 : 3);
      act("큐브를 통째로 돌려 잡는다");
      t += 200;
      hear(coach.onPose(pose, t));
      tickFor(1500);
      continue;
    }
    let m: ColorMove | null = coach.expected();
    if (!m) break;
    const am = coach.pendingMove();
    let wide = heardWide && !!am && /^[a-z]|w/.test(am.token) && am.color === m.color && am.k === m.k;
    if (rand() < (o.errRate ?? 0)) {
      m = { color: Math.floor(rand() * 6), k: (1 + Math.floor(rand() * 3)) as 1 | 2 | 3 };
      wide = false;
    }
    const turned = wide && o.mode !== "keyboard";
    if (turned) {
      // 두 줄을 돌리면 큐브가 통째로 함께 돈다: 다 돌린 순간 이미 새 방향으로 들고 있다
      const [axis, sign] = CCW_FROM[letterOf(coach.s, pose, m.color)];
      pose = rotateView(coach.s, pose, axis, sign * m.k);
    }
    t += 200;
    hear(coach.onMove(m, t));
    if (turned) {
      t += 200; // 카메라는 자세가 몇 프레임 안정된 뒤에 알려 준다
      hear(coach.onPose(pose, t));
    }
  }
  return { coach, said, problems };
}
