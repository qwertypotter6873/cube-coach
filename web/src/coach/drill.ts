/**
 * 공식 연습 모드: 약한 공식을 골라 반복시킨다.
 *
 * 한 번의 연습
 *  1. 준비: 코치가 "그 상황을 만드는 동작"(공식을 거꾸로 + 윗면 무작위 돌리기)을 불러 준다.
 *  2. 풀기: 사람이 상황을 알아보고 공식으로 푼다. 코치는 지켜보며
 *     반응 시간(준비 끝 → 첫 수)과 실행 시간(첫 수 → 다 풀림)을 잰다.
 *  3. 결과와 함께 그 상황의 이름("정답은 안티수네였어요.")을 알려 주고 다음 공식으로.
 *     알아보기 연습이므로 준비 동작 전·풀기 전에는 이름을 말하지 않는다.
 * 풀다가 엉키면, 맞춰 주는 동작을 불러 준 다음 다시 이어 간다.
 *
 * 1·2층이 맞은 큐브에서 시작해야 한다 (OLL·PLL 연습).
 */

import type { ColorMove, State, View } from "./cube.ts";
import { applyColorMove, inverse, invertAlg, simplify, translate } from "./cube.ts";
import type { Utterance } from "./coach.ts";
import type { Alg } from "./method.ts";
import { analyze, planNext, YELLOW } from "./method.ts";
import { josa, moveSay, reframeText, sameView, secondsWord } from "./speech.ts";

export interface DrillResult {
  name: string;
  reactMs: number;
  execMs: number;
  moves: number;
  clean: boolean; // 한 번에 풀었는가
}

const say = (text: string, kind: Utterance["kind"], priority = 5): Utterance => ({ text, kind, priority });
/** 같은 기준을 이 시간 안에 또 알리지는 않는다 (ms) */
const NOTICE_MS = 6000;
const U = (k: number): ColorMove => ({ color: YELLOW, k: (((k % 4) + 4) % 4) as 1 | 2 | 3 });

export class Drill {
  s: State;
  pose: View | null = null;
  readonly results: DrillResult[] = [];
  phase: "idle" | "setup" | "solve" | "restore" = "idle";
  current: Alg | null = null;

  private cases: Alg[];
  private kind: "oll" | "pll";
  private rand: () => number;
  private queue: ColorMove[] = []; // 불러 줄 동작 (준비·복구)
  private fixes: ColorMove[] = [];
  private partial = 0;
  private readyAt = 0;
  private firstAt: number | null = null;
  private solveMoves = 0;
  private turn = 0;
  private noticed = new Map<string, number>(); // 기준(위,앞) → 마지막으로 알린 시각
  private correcting = false; // 반대로 돌린 동작을 마저 돌리는 중

  constructor(s: State, cases: Alg[], kind: "oll" | "pll", rand: () => number = Math.random) {
    this.s = s;
    this.cases = cases;
    this.kind = kind;
    this.rand = rand;
  }

  /** 바로잡는 중인가 (잘못 돌린 면을 되돌리거나, 반대로 돌린 동작을 마저 돌리는 중) */
  get fixing(): boolean {
    return this.fixes.length > 0 || this.correcting;
  }

  /** 지금 사람이 해야 할 동작 */
  expected(): ColorMove | null {
    if (this.fixes.length) return this.fixes[0];
    if (this.phase !== "setup" && this.phase !== "restore") return null;
    const m = this.queue[0];
    if (!m) return null;
    return { color: m.color, k: ((m.k - this.partial + 4) % 4) as 1 | 2 | 3 };
  }

  /**
   * 쥔 방향이 바뀌었다: 불러 주던 동작이 있으면 바뀐 기준을 알리고 새 기준의 기호와 말로 다시 불러 준다.
   * 흔들려서 왔다 갔다 하면 같은 기준 알림은 NOTICE_MS 안에 되풀이하지 않는다 (동작은 새 기준으로 다시 부른다).
   */
  onPose(v: View, t = 0): Utterance[] {
    const was = this.pose;
    this.pose = v;
    if (sameView(was, v) || (this.phase !== "setup" && this.phase !== "restore")) return [];
    const key = `${v.up},${v.front}`;
    const fresh = t - (this.noticed.get(key) ?? -Infinity) > NOTICE_MS;
    if (fresh) this.noticed.set(key, t);
    return [...(was && fresh ? [say(reframeText(was, v), "status", 2)] : []), ...this.cue()];
  }

  start(t: number): Utterance[] {
    if (!this.ready()) {
      this.phase = "restore";
      return [say("연습은 맞춘 큐브에서 해요. 먼저 맞춰 드릴게요.", "status"), ...this.restore(t)];
    }
    return this.next(t);
  }

  /** 다음 연습을 준비할 수 있는 상태인가: 공식을 거꾸로 했을 때 그 공식 하나로 풀리는 상황이 되어야 한다 */
  private ready(): boolean {
    const a = analyze(this.s);
    return a.f2l && (this.kind === "oll" ? a.oriented : a.auf >= 0);
  }

  onMove(m: ColorMove, t: number): Utterance[] {
    this.s = applyColorMove(this.s, m);
    if (this.phase === "solve") return this.onSolveMove(t);
    if (this.phase !== "setup" && this.phase !== "restore") return [];
    // 불러 준 동작을 따라 하는 중
    if (this.fixes.length) {
      const f = this.fixes[0];
      if (m.color === f.color) {
        const rem = (f.k - m.k + 4) % 4;
        if (rem) this.fixes[0] = { color: f.color, k: rem as 1 | 2 | 3 };
        else this.fixes.shift();
      } else this.fixes = simplify([inverse(m), ...this.fixes]);
      if (this.fixes.length > 3) {
        this.fixes = [];
        this.phase = "restore";
        return [say("괜찮아요, 다시 맞춰 드릴게요.", "fix"), ...this.restore(t)];
      }
      return this.fixes.length ? [say(`${moveSay(this.s, this.view(), this.fixes[0], "short")}.`, "fix")] : this.cue();
    }
    const exp = this.queue[0];
    if (m.color === exp.color) {
      const done = (this.partial + m.k) % 4;
      const dist = (x: number) => Math.min(x, 4 - x);
      if (done === exp.k) {
        this.queue.shift();
        this.partial = 0;
        this.correcting = false;
      } else if (dist((exp.k - done + 4) % 4) < dist((exp.k - this.partial + 4) % 4)) {
        this.partial = done; // 반 바퀴의 절반
        return [say("한 번 더요.", "move")];
      } else {
        this.partial = done;
        this.correcting = true;
        return [say(`반대 방향이에요. ${moveSay(this.s, this.view(), this.expected()!, "short")}.`, "fix")];
      }
    } else {
      this.fixes = [inverse(m)];
      return [say(`앗, 되돌려 주세요. ${moveSay(this.s, this.view(), this.fixes[0], "short")}.`, "fix")];
    }
    if (this.queue.length) return this.cue();
    if (this.phase === "restore") {
      const more = this.restore(t);
      return more.length ? more : this.next(t);
    }
    // 준비 끝: 이제 사람이 푼다
    this.phase = "solve";
    this.readyAt = t;
    this.firstAt = null;
    this.solveMoves = 0;
    return [say("준비 끝. 시작!", "goal", 6)];
  }

  /**
   * 카메라 추적기가 놓쳤던 회전을 뒤늦게 찾아 지난 수들 사이에 끼워 넣었을 때 (coach.ts Coach.resync 와 같다).
   * 상태를 추적기 상태 s로 바꾼다. onMove 대신 부른다.
   *  - 풀기: 끼워 넣은 회전도 푸는 수로 센다 (다 풀었는지·엉켰는지는 onMove 와 똑같이 본다).
   *  - 준비·복구: 불러 주던 동작은 틀린 상태를 기준으로 한 것이라 버리고, 지금 상태에서 다시 맞춘 뒤 다음 연습으로.
   */
  resync(s: State, m: ColorMove, t: number): Utterance[] {
    void m;
    this.s = s;
    if (this.phase === "solve") return this.onSolveMove(t);
    if (this.phase !== "setup" && this.phase !== "restore") return [];
    this.fixes = [];
    this.partial = 0;
    this.correcting = false;
    this.phase = "restore";
    const more = this.restore(t);
    return more.length ? [say("괜찮아요, 다시 맞춰 드릴게요.", "fix"), ...more] : this.next(t);
  }

  private onSolveMove(t: number): Utterance[] {
    this.firstAt ??= t;
    this.solveMoves++;
    const a = analyze(this.s);
    const goal = this.kind === "oll" ? a.oriented && a.f2l : a.auf >= 0;
    if (goal) {
      const r: DrillResult = {
        name: this.current!.name,
        reactMs: this.firstAt - this.readyAt,
        execMs: t - this.firstAt,
        moves: this.solveMoves,
        clean: this.solveMoves <= this.current!.alg.split(" ").length + 2,
      };
      this.results.push(r);
      // 알아보기 연습이므로 공식 이름은 다 푼 뒤에야 알려 준다
      const msg = `${this.answer()} ${secondsWord((r.reactMs + r.execMs) / 1000)}. 알아보는 데 ${secondsWord(r.reactMs / 1000)} 걸렸어요.`;
      return [say(msg, "summary", 7), ...this.next(t)];
    }
    // 공식을 하는 도중에는 1·2층이 잠깐 깨지는 게 정상이다. 수가 너무 많아졌을 때만 엉킨 것으로 본다.
    const limit = this.current!.alg.split(" ").length * 2 + 6;
    if (this.solveMoves > limit) {
      this.phase = "restore";
      return [say(`엉켰네요. ${this.answer()} 맞춰 드릴 테니 다시 해 봐요.`, "fix"), ...this.restore(t)];
    }
    if (!a.f2l || this.ready()) return [];
    // 1·2층은 맞았는데 목표는 아니다: 공식을 끝냈는데 다른 상황이 됐다
    // (다른 공식을 쓰는 사람도 있으니 넉넉하게 본다)
    if (this.solveMoves >= this.current!.alg.split(" ").length + 4) {
      this.phase = "restore";
      return [say(`다른 상황이 됐네요. ${this.answer()} 맞춰 드릴게요.`, "fix"), ...this.restore(t)];
    }
    return [];
  }

  /** 다음 연습: 공식을 고르고 준비 동작을 불러 주기 시작한다 */
  private next(t: number): Utterance[] {
    void t;
    const alg = this.cases[this.turn++ % this.cases.length];
    this.current = alg;
    const view: View = { up: YELLOW, front: [2, 1, 5, 4][Math.floor(this.rand() * 4)] };
    const pre = Math.floor(this.rand() * 4);
    const post = Math.floor(this.rand() * 4);
    // 공식을 거꾸로 하면 그 공식이 푸는 상황이 된다. 앞뒤로 윗면을 돌려 알아보기 연습이 되게 한다.
    let moves: ColorMove[] = [];
    if (pre) moves.push(U(pre));
    moves.push(...translate(this.s, invertAlg(alg.alg), view).map((m) => ({ color: m.color, k: m.k })));
    if (post) moves.push(U(post));
    moves = simplify(moves);
    this.queue = moves;
    this.partial = 0;
    this.fixes = [];
    this.phase = "setup";
    // 이름은 말하지 않는다: 준비가 끝난 상황을 보고 스스로 알아보는 연습이다 (이름은 다 푼 뒤 answer()로)
    return [say("다음 연습이에요. 준비 동작을 불러 드릴게요.", "goal"), ...this.cue()];
  }

  /** 방금 연습한 상황의 이름 (다 풀었거나 엉켜서 그만둘 때만 말한다) */
  private answer(): string {
    return `정답은 ${josa.ieosseoyo(this.current!.name)}.`;
  }

  /** 엉켰을 때: OLL 연습이면 윗면 방향까지, PLL 연습이면 다 맞을 때까지 맞추는 동작을 불러 준다 */
  private restore(t: number): Utterance[] {
    void t;
    if (this.ready()) {
      this.queue = [];
      return [];
    }
    const step = planNext(this.s, "cfop");
    if (!step) return [];
    this.queue = step.moves.map((m) => ({ color: m.color, k: m.k }));
    this.partial = 0;
    return this.cue();
  }

  private cue(): Utterance[] {
    const m = this.expected();
    return m ? [say(`${moveSay(this.s, this.view(), m, "short")}.`, "move")] : [];
  }

  private view(): View {
    return this.pose ?? { up: YELLOW, front: 2 };
  }
}
