/**
 * 코치: 추적기가 알려 주는 회전과 자세를 받아, 무엇을 말할지 정한다.
 *
 * 수준
 *  - guided : 처음 배우는 사람. 한 수씩 불러 주고, 틀리면 바로 되돌리게 한다.
 *  - hint   : 연습하는 사람. 목표만 말하고 지켜보다가, 막히면 점점 자세한 힌트를 준다.
 *  - observe: 혼자 푸는 사람. 말하지 않고 단계별 시간만 잰 뒤 끝나면 정리해 준다.
 *
 * 입력
 *  - onMove(회전, 시각): 추적기가 확정한 회전 (가운데 색 기준)
 *  - onPose(보는 방향, 시각): 사람이 지금 큐브를 어떻게 들고 있는지
 *  - tick(시각): 시간이 흐름 (막힘 감지, 바뀐 자세가 자리 잡았는지)
 *
 * 기준 면: 모든 지시는 사람이 손에 든 큐브를 자기 눈으로 보는 방향(pose) 기준이다.
 * 시작할 때 "노랑이 위, 초록이 앞에 오게 잡아 주세요"로 기준을 정하고, 들고 있는 방향이 바뀌면
 * "이제 파랑이 앞이에요"라고 알린 뒤 같은 지시를 새 기준의 기호로 다시 말한다.
 * 출력: 말할 문장 목록 (Utterance). 실제로 언제 소리 낼지는 음성 대기열이 정한다.
 */

import type { AlgMove, Color, ColorMove, State, View } from "./cube.ts";
import { applyColorMove, inverse, rebaseView, simplify, slotOfColor, viewColors, SLOT_NORMAL } from "./cube.ts";
import type { Method, StageId, Step } from "./method.ts";
import { STAGES, analyze, planNext, stageOf } from "./method.ts";
import { basicsTaught, noteBasicsTaught } from "./progress.ts";
import type { Verbosity } from "./speech.ts";
import {
  Praise,
  STAGE_NAME,
  chunkSymbols,
  chunkText,
  countWord,
  halfDoneText,
  holdText,
  josa,
  moveSay,
  moveSymbol,
  orientationLines,
  reframeText,
  sameView,
  secondsWord,
  sinoWord,
  stageDoneText,
  stageIntro,
  stepGoal,
  wrongDirectionText,
  wrongFaceText,
} from "./speech.ts";

export type Level = "guided" | "hint" | "observe";

export interface Utterance {
  text: string;
  kind: "status" | "intro" | "goal" | "hold" | "move" | "praise" | "fix" | "stage" | "hint" | "summary";
  /** 높을수록 급하다: 말하던 낮은 순위 문장을 끊고 말해도 된다 */
  priority: number;
}

interface Current {
  step: Step;
  moves: Step["moves"]; // 순서를 바꿔 받아들일 수 있게 복사해 둔다
  idx: number; // 다음에 둘 수의 번호
  partial: number; // 지금 수에서 이미 돌린 양 (90도 단위)
  announcedAt: number;
  firstMoveAt: number | null;
  mistakes: number;
  silentUntil: number; // hint 수준: 이 번호의 수까지는 말하지 않고 지켜본다
  named: boolean; // 이 덩어리의 공식 이름을 말해 주었는가 (목표 문장이나 첫 힌트로)
  shown: number; // 앞에서부터 몇 수의 기호를 말해 주었는가 (hint 수준의 둘째 힌트 "처음은 …")
}

export interface StepRecord {
  stage: StageId;
  caseName?: string;
  alg?: string;
  /** 안내를 듣고 첫 수를 두기까지 (인식·반응 시간) */
  reactMs: number;
  /** 첫 수부터 덩어리를 끝내기까지 */
  execMs: number;
  moves: number;
  mistakes: number;
}

export interface MoveRecord {
  t: number;
  move: ColorMove;
  stage: StageId;
  ok: boolean;
}

export interface CoachOptions {
  method: Method;
  level: Level;
  verbosity?: Verbosity;
  /** guided 수준에서 안 움직이면 다시 말해 주기까지 (ms) */
  repeatAfterMs?: number;
  /** hint 수준에서 다음 힌트로 넘어가기까지 (ms) */
  hintAfterMs?: number;
  /** 기준 면·기호 설명을 이미 들은 사람인가 (없으면 기기에 남은 기록을 본다) */
  taughtBasics?: boolean;
  /** 자세가 이만큼 그대로여야 바뀐 것으로 본다 (ms) — 흔들림에 말이 따라 나오지 않게 */
  poseSettleMs?: number;
  /** 같은 기준을 이 시간 안에 또 알리지는 않는다 (ms) */
  poseNoticeMs?: number;
}

const P = { status: 2, intro: 2, goal: 3, hold: 4, move: 5, praise: 1, fix: 8, stage: 6, hint: 5, summary: 7 } as const;
const say = (text: string, kind: Utterance["kind"]): Utterance => ({ text, kind, priority: P[kind] });
/** 이만큼 말하는 데 걸린다고 어림잡는다 — 아직 말하는 중인데 "다시," 하고 또 부르지 않게 */
const speakMs = (us: readonly Utterance[]) => us.reduce((a, u) => a + 300 + u.text.length * 130, 0);

const SEXY = "R U R' U'";
const ORDINAL = ["첫 번째", "두 번째", "세 번째", "네 번째", "다섯 번째", "여섯 번째"];
/** 이 횟수만큼 맞게 해 본 동작은 짧게 부른다 (가르칠 때처럼 도움을 점점 줄인다) */
const FAMILIAR_AFTER = 4;
/** 실수 없이 이만큼 해 본 묶음은 묶음 이름만 부른다 */
const CHUNK_AFTER = 2;
/** 이만큼 돌아가면 의도가 분명하다고 보고 말린다 (라디안) */
const WARN_ANGLE = (28 * Math.PI) / 180;

/** 90도 단위 회전량 x(0~3)가 원래대로부터 몇 번 떨어져 있는가 (3 = 반대로 한 번 = 1) */
const quarterDist = (x: number) => Math.min(x, 4 - x);

const sameAxis = (s: State, a: number, b: number) => {
  const na = SLOT_NORMAL[slotOfColor(s, a)];
  const nb = SLOT_NORMAL[slotOfColor(s, b)];
  return na[0] === -nb[0] && na[1] === -nb[1] && na[2] === -nb[2];
};

export class Coach {
  s: State;
  readonly method: Method;
  level: Level;
  verbosity: Verbosity;
  pose: View | null = null;
  readonly moves: MoveRecord[] = [];
  readonly steps: StepRecord[] = [];
  /** 단계가 처음 끝난 시각 (풀이 시작 기준 ms) */
  readonly splits: Partial<Record<StageId, number>> = {};

  private cur: Current | null = null;
  private fixes: ColorMove[] = [];
  private introduced = new Set<StageId>();
  private praise = new Praise();
  private stage: StageId;
  private t0: number | null = null;
  private lastActivity = 0;
  private hintLevel = 0;
  private repeats = 0; // 가만히 있을 때 다시 말해 준 정도 (1: 잡는 법을 한 번 더, 2: 지시를 한 번 더)
  private holdAsked: View | null = null; // 잡아 달라고 해 두고 기다리는 방향
  private anchor: View | null = null; // 시작할 때 기준으로 잡아 달라고 한 방향 (첫 수를 두기 전까지만)
  private familiar = new Map<string, number>(); // 동작 기호(지금 방향 기준) → 맞게 한 횟수
  private chunksDone = 0; // 실수 없이 끝낸 섹시무브 수
  private repMistake = false;
  private explainedSexy = false;
  private turnWarned = false; // 이번 회전에서 이미 말렸는가
  private correcting = false; // 반대로(또는 더) 돌려서 같은 면을 남은 만큼 마저 돌리는 중
  private finished = new Set<StageId>(); // 한 번이라도 끝낸 단계
  private poseAt = 0; // 지금 자세를 처음 본 시각
  private spokenView: View | null = null; // 사람에게 마지막으로 알린 기준 (지시를 말한 방향, 잡아 달라고 한 방향)
  private noticed = new Map<string, number>(); // 기준(위,앞) → 마지막으로 알린 시각
  private lastTurnAt = -Infinity; // 층이 돌아가는 중인지 보려고
  private algPose: View | null = null; // 공식의 두 줄 돌리기로 큐브가 함께 돌면 이렇게 보일 것이다
  private chunkSpoken = false; // 이 덩어리를 한 번에 불러 주었는가 (그러면 두 줄 돌리기 뒤 방향까지 이미 말한 셈)
  private clock = 0; // 마지막으로 받은 시각
  private ended = false; // 이 풀이는 끝났다 (다 맞췄거나, 맞춰진 큐브로 시작했다)
  private readonly repeatAfterMs: number;
  private readonly hintAfterMs: number;
  private readonly taughtBasics: boolean;
  private readonly poseSettleMs: number;
  private readonly poseNoticeMs: number;

  constructor(s: State, opts: CoachOptions) {
    this.s = s;
    this.method = opts.method;
    this.level = opts.level;
    this.verbosity = opts.verbosity ?? (opts.level === "guided" ? "full" : "short");
    this.repeatAfterMs = opts.repeatAfterMs ?? 9000;
    this.hintAfterMs = opts.hintAfterMs ?? 12000;
    this.taughtBasics = opts.taughtBasics ?? basicsTaught();
    this.poseSettleMs = opts.poseSettleMs ?? 900;
    this.poseNoticeMs = opts.poseNoticeMs ?? 6000;
    this.stage = stageOf(analyze(s), this.method);
  }

  get currentStage(): StageId {
    return this.stage;
  }

  /** 지금 말의 기준이 되는 면들의 가운데 색 (화면에 "앞 초록 / 위 노랑"처럼 보여 주라고) */
  reference(): { up: Color; front: Color; right: Color } {
    const v = this.view();
    return { up: v.up, front: v.front, right: viewColors(this.s, v).R };
  }

  /** 잡아 달라고 하고 기다리는 방향 (화면에 잡는 법을 띄우라고). 이미 그렇게 잡았거나 기다리는 게 없으면 null */
  get grip(): View | null {
    return this.holdAsked && !sameView(this.pose, this.holdAsked) ? this.holdAsked : null;
  }

  get currentStep(): Step | null {
    return this.cur?.step ?? null;
  }

  /**
   * 이 풀이가 끝났는가: 다 맞췄거나(완성 정리를 말했다), 맞춰진 큐브로 시작했다.
   * 끝난 풀이는 새로 시작하기 전까지 회전을 기록하지 않고 아무 말도 하지 않는다 (기록이 한 번만 남게).
   */
  get closed(): boolean {
    return this.ended;
  }

  /**
   * 화면에 지금 덩어리의 공식 이름과 기호를 어디까지 보여도 되는가 — 음성으로 말해 준 만큼만.
   *  - hint: 공식 이름은 목표 문장이나 첫 힌트로 말한 뒤에, 기호는 둘째 힌트("처음은 …")로 불러 준 수까지.
   *    셋째 힌트부터는 한 수씩 불러 주는 guided 로 바뀌므로 다 보인다.
   *  - 그 밖의 수준: 다 (guided 는 한 수씩 불러 준다).
   * moves: 앞에서부터 몇 수의 기호를 보여도 되는가.
   */
  revealed(): { caseName: boolean; moves: number } {
    const c = this.cur;
    if (!c) return { caseName: false, moves: 0 };
    if (this.level === "hint") return { caseName: c.named, moves: c.shown };
    return { caseName: true, moves: c.moves.length };
  }

  /**
   * 바로잡는 중인가: 잘못 돌린 면을 되돌리는 중이거나, 반대로 돌린 면을 남은 만큼 마저 돌리는 중.
   * 이때 expected()는 공식의 다음 수가 아니라 바로잡을 회전이다 (화면에 "이것부터"라고 띄우라고).
   */
  get fixing(): boolean {
    return this.fixes.length > 0 || (this.correcting && !!this.cur);
  }

  /** 지금 기다리는 공식의 한 수 (기호, 두 줄 돌리기인지, 공식 기준 방향). 되돌리는 중이거나 없으면 null */
  pendingMove(): AlgMove | null {
    if (this.fixes.length || !this.cur) return null;
    return this.cur.moves[this.cur.idx] ?? null;
  }

  /** 다음에 사람이 해야 하는 회전 (되돌리기가 먼저) */
  expected(): ColorMove | null {
    if (this.fixes.length) return this.fixes[0];
    if (!this.cur) return null;
    const m = this.cur.moves[this.cur.idx];
    if (!m) return null;
    const need = (m.k - this.cur.partial + 4) % 4;
    return { color: m.color, k: need as 1 | 2 | 3 };
  }

  /** 시작: 기준 면과 기호를 알려 주고, 지금 어디까지 되어 있는지 말한 뒤 첫 안내를 한다 */
  start(t = 0): Utterance[] {
    this.clock = t;
    this.lastActivity = t;
    this.poseAt = t;
    const a = analyze(this.s);
    if (a.solved) {
      this.ended = true; // 풀 것이 없다: 섞는 회전을 풀이로 기록하지 않는다 (다시 시작하면 새 코치가 맡는다)
      return [say("다 맞춰진 큐브예요. 섞은 다음 시작해 주세요.", "status")];
    }
    const order = STAGES[this.method];
    const idx = order.indexOf(this.stage);
    const out: Utterance[] = [];
    if (idx > 0) out.push(say(`${STAGE_NAME[order[idx - 1]]}까지 되어 있네요. ${STAGE_NAME[this.stage]}부터 할게요.`, "status"));
    if (this.level === "observe") return [...out, say("준비되면 시작하세요. 단계별 시간을 재 둘게요.", "status")];
    for (const line of orientationLines(this.method, this.taughtBasics)) out.push(say(line, "intro"));
    if (!this.taughtBasics) noteBasicsTaught();
    const all = [...out, ...this.announceNext(t, true)];
    this.lastActivity = t + speakMs(all); // 말이 다 끝나기 전에 "다시," 가 나오지 않게
    return all;
  }

  /**
   * 사람이 큐브를 쥔 방향이 바뀌었다 (추적기가 네 프레임 안정되면 알려 준다).
   * 바로 알리지는 않는다: 자세가 자리 잡을 때까지 기다렸다가 tick에서 알린다 (흔들림 억제).
   * 다만 기다리던 자세(잡아 달라고 한 방향, 공식의 두 줄 돌리기로 바뀐 방향)이면 바로 지시를 한다.
   */
  onPose(v: View, t: number): Utterance[] {
    this.clock = t;
    const was = this.pose;
    this.pose = v;
    if (sameView(was, v)) return [];
    this.poseAt = t;
    if (!this.cur || this.level === "observe") return [];
    // 공식의 두 줄 돌리기로 큐브가 함께 돈 것: 알릴 일이 아니다. 지시만 새 방향으로 한다.
    if (this.algPose && sameView(v, this.algPose)) {
      this.algPose = null;
      if (sameView(this.holdAsked, v)) this.holdAsked = null;
      const before = this.spokenView;
      this.spokenView = v;
      if (this.level !== "guided" || !before || sameView(before, v)) return [];
      this.lastActivity = t;
      return this.instruction();
    }
    // 첫 수 전에 시작 기준과 다르게 들고 있다: 아직 잡는 중으로 본다 (main.ts 의 처음 자세는 짐작일 수 있다).
    // 바뀐 기준을 알리거나 다시 부르지 않고 기준 방향을 기다린다. 오래 그대로면 재촉이 잡는 법을 한 번 더 말한다.
    if (this.anchor && !sameView(v, this.anchor) && this.level === "guided" && !this.fixes.length) {
      this.holdAsked = this.anchor;
      return [];
    }
    // 잡아 달라고 한 방향으로 잡았다: 기다리던 지시를 말한다
    if (this.holdAsked && sameView(v, this.holdAsked) && this.level === "guided") {
      this.holdAsked = null;
      this.spokenView = v;
      this.lastActivity = t;
      this.repeats = 0;
      return [say("좋아요.", "praise"), ...this.instruction()];
    }
    return [];
  }

  /**
   * 층이 돌아가는 도중 (추적기가 매 프레임 알려 준다). angle은 라디안, 바깥에서 볼 때 시계 방향이 +.
   * 엉뚱한 면이나 반대 방향으로 어느 정도 돌아가면, 다 돌리기 전에 말린다.
   */
  onTurning(turnColor: number, angle: number, t: number): Utterance[] {
    this.clock = t;
    this.lastActivity = t;
    this.lastTurnAt = t;
    if (this.turnWarned || this.level !== "guided" || Math.abs(angle) < WARN_ANGLE) return [];
    const exp = this.expected();
    if (!exp) return [];
    const view = this.view();
    if (turnColor !== exp.color) {
      // 마주 보는 면을 먼저 돌리는 것은 괜찮다 (순서만 바뀜)
      const next = this.cur?.moves[this.cur.idx + 1];
      if (!this.fixes.length && next && next.color === turnColor && sameAxis(this.s, exp.color, turnColor)) return [];
      this.turnWarned = true;
      this.told(view);
      return [say(`잠깐요, 그 면 말고요. ${moveSay(this.s, view, exp, "short")}.`, "fix")];
    }
    const cw = angle > 0;
    if ((exp.k === 1 && !cw) || (exp.k === 3 && cw)) {
      this.turnWarned = true;
      return [say("잠깐, 반대 방향이에요!", "fix")];
    }
    return [];
  }

  /** 돌리다 말고 원래대로 되돌려 놓았을 때 */
  onTurnCancel(): void {
    this.turnWarned = false;
  }

  onMove(m: ColorMove, t: number): Utterance[] {
    if (this.ended) {
      // 끝난 풀이: 큐브 상태만 따라가고, 회전·단계 시간·완성 정리는 다시 만들지 않는다
      this.clock = t;
      this.s = applyColorMove(this.s, m);
      return [];
    }
    if (this.t0 === null) this.t0 = t;
    this.clock = t;
    this.lastActivity = t;
    this.repeats = 0;
    this.turnWarned = false;
    this.anchor = null;
    if (this.level === "hint") this.spokenView = null; // 힌트 뒤에 움직였으면 그 힌트를 다시 말할 일은 없다
    this.s = applyColorMove(this.s, m);
    const before = this.stage;
    const a = analyze(this.s);
    const now = stageOf(a, this.method);

    if (this.level === "observe" || !this.cur) {
      this.moves.push({ t, move: m, stage: before, ok: true });
      return this.observeProgress(before, now, t);
    }

    // 1) 되돌리기 중
    if (this.fixes.length) {
      const f = this.fixes[0];
      if (m.color === f.color) {
        const rem = (f.k - m.k + 4) % 4;
        if (rem === 0) this.fixes.shift();
        else this.fixes[0] = { color: f.color, k: rem as 1 | 2 | 3 };
      } else {
        this.fixes = simplify([inverse(m), ...this.fixes]);
      }
      this.moves.push({ t, move: m, stage: before, ok: false });
      if (this.fixes.length > 3) return this.abandon(t);
      if (this.fixes.length) return this.instruction();
      return this.level === "guided" ? this.moveUtterances() : [];
    }

    // 2) 지금 덩어리의 다음 수와 비교
    const c = this.cur;
    const exp = c.moves[c.idx];
    if (c.firstMoveAt === null) c.firstMoveAt = t;
    if (m.color === exp.color) {
      const done = (c.partial + m.k) % 4;
      if (done === exp.k) {
        this.correcting = false;
        const sym = moveSymbol(this.s, this.view(), exp);
        this.familiar.set(sym, (this.familiar.get(sym) ?? 0) + 1);
        c.idx++;
        c.partial = 0;
        this.moves.push({ t, move: m, stage: before, ok: true });
        if (c.step.alg === SEXY && c.idx % 4 === 0) {
          if (!this.repMistake) this.chunksDone++;
          this.repMistake = false;
        }
        this.algPose = null;
        const nxt = c.moves[c.idx];
        if (nxt && !sameView(exp.viewBefore, nxt.viewBefore)) {
          // 두 줄 돌리기(r, d)로 공식 기준 방향이 바뀐다. 한 수씩 부를 때는 한 층만 돌리라고 했으니 큐브가 돌지 않지만,
          // 두 줄로 돌린 사람은 큐브가 통째로 함께 돈다: 그때 들어올 자세를 적어 둔다 (알릴 일이 아니므로).
          const from = this.view();
          this.algPose = rebaseView(this.s, exp.viewBefore, nxt.viewBefore, from);
          // 덩어리를 한 번에 불렀으면("알 더블 …") 돈 뒤의 방향까지 이미 말한 셈이다
          if (this.chunkSpoken && sameView(this.spokenView, from)) this.spokenView = this.algPose;
        }
      } else if (quarterDist((exp.k - done + 4) % 4) < quarterDist((exp.k - c.partial + 4) % 4)) {
        // 남은 양이 줄었다: 반 바퀴의 절반이거나, 반대로 돌린 것을 바로잡는 중
        c.partial = done;
        this.moves.push({ t, move: m, stage: before, ok: true });
        return this.level === "guided" ? [say(halfDoneText(), "move")] : [];
      } else {
        // 반대로 돌렸거나 더 돌렸다: 같은 면을 남은 만큼 더 돌리면 된다
        c.partial = done;
        c.mistakes++;
        this.correcting = true;
        this.repMistake = true;
        this.moves.push({ t, move: m, stage: before, ok: false });
        if (this.level === "hint" && c.idx < c.silentUntil) return this.replanSilently(t);
        return [say(wrongDirectionText(this.s, this.told(this.view()), this.expected()!, this.verbosity), "fix")];
      }
    } else {
      const next = c.moves[c.idx + 1];
      if (next && next.color === m.color && next.k === m.k && c.partial === 0 && sameAxis(this.s, exp.color, m.color)) {
        // 마주 보는 두 면은 어느 쪽을 먼저 돌려도 같다
        [c.moves[c.idx], c.moves[c.idx + 1]] = [next, exp];
        c.idx++;
        this.moves.push({ t, move: m, stage: before, ok: true });
      } else {
        c.mistakes++;
        this.repMistake = true;
        this.moves.push({ t, move: m, stage: before, ok: false });
        if (this.level === "hint") return this.replanSilently(t);
        this.fixes = [inverse(m)];
        return [say(wrongFaceText(this.s, this.told(this.view()), m, this.fixes[0], this.verbosity), "fix")];
      }
    }

    if (c.idx >= c.moves.length) return this.finishStep(t);
    if (this.level === "guided" && this.verbosity === "full") return this.moveUtterances();
    return [];
  }

  /**
   * 카메라 추적기가 놓쳤던 회전 m을 뒤늦게 찾아 지난 수들 사이에 끼워 넣었을 때 (tracker.ts decide, back > 0).
   * 회전은 순서를 바꾸면 결과가 달라서 m을 지금 상태 위에 얹으면 추적기와 어긋난다: 상태를 추적기 상태 s로 바꾼다.
   * 그사이 틀린 상태를 기준으로 세운 되돌리기와 지금 덩어리는 버리고, 지금 상태에서 다시 계획한다.
   * onMove 대신 부른다 (둘 다 부르면 한 수가 두 번 기록된다). 끼워 넣은 수는 실수로 세지 않는다.
   */
  resync(s: State, m: ColorMove, t: number): Utterance[] {
    this.clock = t;
    this.s = s;
    if (this.ended) return []; // 끝난 풀이: 상태만 따라간다 (onMove 와 같다)
    if (this.t0 === null) this.t0 = t;
    this.lastActivity = t;
    this.repeats = 0;
    this.turnWarned = false;
    this.anchor = null;
    if (this.level === "hint") this.spokenView = null;
    const before = this.stage;
    this.moves.push({ t, move: m, stage: before, ok: true });
    if (this.level === "observe" || !this.cur) return this.observeProgress(before, stageOf(analyze(s), this.method), t);
    this.fixes = [];
    this.correcting = false;
    this.algPose = null;
    const step = planNext(s, this.method);
    if (!step) {
      this.markSplits(before, "done", t);
      return this.finish(t);
    }
    // 힌트 수준이거나 단계가 바뀌었으면 힌트 수준의 다시 계획하기와 같다 (단계 완성·흐트러짐을 알리고 새로 안내)
    if (this.level === "hint" || step.stage !== before) return this.replanSilently(t);
    // 한 수씩 부르는 수준, 같은 단계: 새 덩어리를 지금 상태에서 바로 안내한다
    this.cur = null;
    return this.announceNext(t);
  }

  /** 시간이 흐를 때: 기준이 바뀌었으면 다시 맞춰 말하고, 막혔으면 다시 말하거나 힌트를 준다 */
  tick(t: number): Utterance[] {
    this.clock = t;
    if (!this.cur || this.level === "observe") return [];
    const reframed = this.reframe(t);
    if (reframed.length) return reframed;
    const idle = t - this.lastActivity;
    if (this.level === "guided") {
      if (idle <= this.repeatAfterMs) return [];
      // 잡아 달라고 한 방향을 아직 안 잡았으면 잡는 법을 먼저 한 번 더, 그래도 그대로면 지금 방향으로 지시를 한다
      if (this.holdAsked && !sameView(this.pose, this.holdAsked) && this.repeats === 0) {
        this.repeats = 1;
        this.lastActivity = t;
        return [say(holdText(this.holdAsked), "hold")];
      }
      if (this.repeats >= 2) return [];
      this.repeats = 2;
      return this.instruction("full", "다시, ");
    }
    if (idle > this.hintAfterMs) {
      this.lastActivity = t;
      return this.escalateHint();
    }
    return [];
  }

  // --- 내부 ---------------------------------------------------------------

  private view(): View {
    if (this.pose) return this.pose;
    const m = this.cur?.moves[this.cur.idx] ?? this.cur?.moves[this.cur.moves.length - 1];
    return m?.viewBefore ?? this.cur?.step.view ?? { up: 3, front: 2 };
  }

  /** 이 방향을 기준으로 말했다고 적어 둔다. 잡아 달라고 한 방향과 다르게 말했으면 기다리기를 그만둔다. */
  private told(view: View): View {
    this.spokenView = view;
    if (this.holdAsked && !sameView(this.holdAsked, view)) this.holdAsked = null;
    if (this.anchor && !sameView(this.anchor, view)) this.anchor = null;
    return view;
  }

  /** 바뀐 기준을 알리는 한 줄. 같은 기준을 조금 전에 알렸으면 생략한다 (흔들려서 왔다 갔다 하는 경우). */
  private noticeText(prev: View, now: View): string | null {
    const key = `${now.up},${now.front}`;
    if (this.clock - (this.noticed.get(key) ?? -Infinity) <= this.poseNoticeMs) return null;
    this.noticed.set(key, this.clock);
    return reframeText(prev, now);
  }

  /** 지시를 말할 방향을 정해 둔다. 알려 둔 기준과 다르면 바뀐 기준을 알리는 한 줄을 돌려준다. */
  private noteFor(view: View): string | null {
    const prev = this.spokenView;
    this.told(view);
    if (!this.pose || !prev || sameView(prev, view)) return null;
    return this.noticeText(prev, view);
  }

  /** 지금 둘 수를 위해 잡고 있어야 하는 방향 (단계가 잡는 방향을 정해 둔 경우만). 두 줄 돌리기 뒤에는 그만큼 돈 방향. */
  private gripTarget(): View | null {
    const c = this.cur;
    if (!c?.step.view || !c.moves.length) return null;
    const m = c.moves[Math.min(c.idx, c.moves.length - 1)];
    return rebaseView(this.s, c.moves[0].viewBefore, m.viewBefore, c.step.view);
  }

  /**
   * 들고 있는 방향이 알려 둔 기준과 달라졌으면 다시 맞춰 말한다.
   * 자세가 자리 잡고(poseSettleMs) 층을 돌리는 중이 아닐 때만 — 흔들리거나 돌리는 도중에는 미룬다.
   *  - 단계가 잡는 방향을 정해 두었고 그렇게 잡고 하던 중이면: 그 방향으로 돌아오게 한다.
   *  - 아니면: 바뀐 기준을 알리고, 같은 지시를 새 기준의 기호와 말로 다시 한다.
   */
  private reframe(t: number): Utterance[] {
    const c = this.cur;
    const now = this.pose;
    if (!c || !now || !this.spokenView || sameView(now, this.spokenView)) return [];
    if (t - this.poseAt < this.poseSettleMs || t - this.lastTurnAt < 700) return [];
    if (this.level !== "guided") {
      // 힌트 수준: 기호를 넣어 힌트를 준 뒤라면 새 기준으로 다시 말해 준다
      const prev = this.spokenView;
      if (this.hintLevel !== 2) {
        this.spokenView = now;
        return [];
      }
      const note = this.noticeText(prev, now);
      return [...(note ? [say(note, "status")] : []), ...this.firstMovesHint()];
    }
    if (this.holdAsked && !this.fixes.length) return []; // 잡아 달라고 한 방향을 기다리는 중 (되풀이는 재촉이 맡는다)
    const target = this.gripTarget();
    let out: Utterance[];
    if (target && !this.fixes.length && sameView(this.spokenView, target)) {
      const note = this.noticeText(this.spokenView, now);
      out = [...(note ? [say(note, "status")] : []), say(holdText(target), "hold")];
      this.holdAsked = target;
    } else {
      out = this.instruction("full");
    }
    this.repeats = 0;
    this.lastActivity = t + speakMs(out);
    return out;
  }

  /** 기다리는 지시를 지금 방향으로 (되돌리기가 먼저). prefix는 지시 문장 앞에 붙인다 ("다시, "). */
  private instruction(v: Verbosity = this.verbosity, prefix = ""): Utterance[] {
    if (!this.fixes.length) return this.moveUtterances(v, prefix, true);
    const view = this.view();
    const note = this.noteFor(view);
    const text = `${prefix}${moveSay(this.s, view, this.fixes[0], "full")}.`;
    return [say(note ? `${note} ${text}` : text, "fix")];
  }

  private announceNext(t: number, first = false): Utterance[] {
    const step = planNext(this.s, this.method);
    if (!step) return this.finish(t);
    const out: Utterance[] = [];
    if (!this.introduced.has(step.stage)) {
      this.introduced.add(step.stage);
      const intro = stageIntro(step.stage, this.method, analyze(this.s), this.verbosity);
      if (intro) out.push(say(intro, "intro"));
    }
    this.stage = step.stage;
    const goal = stepGoal(this.s, step, this.verbosity);
    this.cur = {
      step,
      moves: [...step.moves],
      idx: 0,
      partial: 0,
      announcedAt: t,
      firstMoveAt: null,
      mistakes: 0,
      silentUntil: this.level === "hint" ? step.moves.length : 0,
      named: !!step.caseName && goal.includes(step.caseName), // 공식 이름은 목표 문장이 말해 준다 ("안티수네.")
      shown: 0,
    };
    this.hintLevel = 0;
    this.repMistake = false;
    this.holdAsked = null;
    this.anchor = null;
    this.correcting = false;
    this.algPose = null;
    this.chunkSpoken = false;
    out.push(say(goal, "goal"));
    if (step.alg === SEXY && !this.explainedSexy && this.verbosity === "full") {
      this.explainedSexy = true;
      out.push(say("섹시무브는 알, 유, 알 프라임, 유 프라임이에요. 오른쪽 면 위로, 윗면 왼쪽으로, 오른쪽 면 아래로, 윗면 오른쪽으로. 네 동작을 한 묶음으로 기억해 두세요.", "intro"));
    }
    if (this.level === "hint") {
      out.push(say("해 보세요.", "hint"));
      return out;
    }
    // 잡는 방향: 단계가 정해 둔 방향. 시작할 때는 정해 둔 방향이 없어도 기준이 될 방향을 색으로 말해 둔다.
    const grip = step.view ?? (first ? this.firstGrip(step) : undefined);
    if (grip && (first || !sameView(this.pose, grip))) {
      const wait = !!this.pose && !sameView(this.pose, grip);
      // 기다리지 않을 때(이미 그렇게 들고 있음)는 '상태'로 둔다: 바로 뒤 지시가 '지시'를 지우는 음성 대기열에서 살아남게
      out.push(say(holdText(grip), wait ? "hold" : "status"));
      this.told(grip);
      if (first) this.anchor = grip;
      if (wait) {
        this.holdAsked = grip;
        return out; // 맞게 잡으면 첫 수를 말한다 (onPose)
      }
    }
    return [...out, ...this.moveUtterances()];
  }

  /** 시작할 때 기준으로 삼을 방향: 첫 수의 기준 방향과 위 색이 같으면 지금 든 그대로, 아니면 첫 수의 기준 방향 */
  private firstGrip(step: Step): View | undefined {
    const a0 = step.moves[0]?.viewBefore;
    if (!a0) return undefined;
    return this.pose && this.pose.up === a0.up ? this.pose : a0;
  }

  /**
   * 지금 둘 수 (full: 한 수씩 / short: 덩어리 전체). 짧게 말해도 기호는 남긴다.
   * 알려 둔 기준과 지금 방향이 다르면 바뀐 기준을 먼저 한 줄로 알린다.
   * force: 익숙해서 말을 아끼는 자리(섹시무브 도중 등)에서도 이번 수를 꼭 말한다 (기준이 바뀌어 다시 말할 때).
   */
  private moveUtterances(v: Verbosity = this.verbosity, prefix = "", force = false): Utterance[] {
    const c = this.cur;
    if (!c || c.idx >= c.moves.length) return [];
    const view = this.view();
    const body = this.moveBody(c, view, v, force);
    if (!body.length) return [];
    if (prefix) body[0] = { ...body[0], text: prefix + body[0].text };
    const note = this.noteFor(view);
    return note ? [say(note, "status"), ...body] : body;
  }

  private moveBody(c: Current, view: View, v: Verbosity, force: boolean): Utterance[] {
    if (v === "short" && c.idx === 0 && c.partial === 0) {
      this.chunkSpoken = true;
      return [say(`${chunkText(this.s, view, c.moves)}.`, "move")];
    }
    if (v === "short" && !force) return [];
    const exp = this.expected()!;
    const style: Verbosity = (this.familiar.get(moveSymbol(this.s, view, exp)) ?? 0) >= FAMILIAR_AFTER ? "short" : "full";
    if (c.step.alg === SEXY) {
      // 섹시무브 반복: 몇 번째인지 세어 주고, 익숙해지면 묶음 이름만 부른다
      const rep = Math.floor(c.idx / 4);
      const first = c.idx % 4 === 0 && c.partial === 0;
      const count = (c.step.repeat ?? 1) > 1 ? `${ORDINAL[rep]}. ` : "";
      // 묶음 이름은 지금 든 방향에서 정말 R U R' U' 일 때만 (다르게 들고 있으면 한 수씩 부른다)
      const here = chunkSymbols(this.s, view, c.moves.slice(rep * 4, rep * 4 + 4)).join(" ") === SEXY;
      if (this.chunksDone >= CHUNK_AFTER && here) {
        if (first) return [say(`${count}섹시무브.`, "move")];
        if (!force) return [];
      }
      return [say(`${first ? count : ""}${moveSay(this.s, view, exp, style)}.`, "move")];
    }
    const left = c.moves.length - c.idx;
    const tail = left === 1 && c.moves.length >= 6 ? " 마지막이에요." : "";
    return [say(`${moveSay(this.s, view, exp, style)}.${tail}`, "move")];
  }

  private finishStep(t: number): Utterance[] {
    const c = this.cur!;
    this.steps.push({
      stage: c.step.stage,
      caseName: c.step.caseName,
      alg: c.step.alg,
      reactMs: (c.firstMoveAt ?? t) - c.announcedAt,
      execMs: t - (c.firstMoveAt ?? t),
      moves: c.moves.length,
      mistakes: c.mistakes,
    });
    const out: Utterance[] = [];
    const now = stageOf(analyze(this.s), this.method);
    const order = STAGES[this.method];
    if (now !== c.step.stage && (now === "done" || order.indexOf(now) > order.indexOf(c.step.stage))) {
      this.markSplits(c.step.stage, now, t);
      if (now !== "done") out.push(this.stageDone(c.step.stage, now)); // 다 끝났으면 정리 문장이 대신한다
    } else if (c.step.kind !== "auf" && this.level === "guided" && c.mistakes === 0 && c.moves.length >= 4) {
      out.push(say(this.praise.next(), "praise"));
    }
    this.cur = null;
    return [...out, ...this.announceNext(t)];
  }

  private stageDone(from: StageId, to: StageId): Utterance {
    const order = STAGES[this.method];
    const again = order.slice(order.indexOf(from), to === "done" ? order.length : order.indexOf(to)).every((st) => this.finished.has(st));
    for (let i = order.indexOf(from); i < (to === "done" ? order.length : order.indexOf(to)); i++) this.finished.add(order[i]);
    return again ? say("좋아요, 다시 맞췄어요.", "stage") : say(stageDoneText(from, to), "stage");
  }

  private markSplits(from: StageId, to: StageId, t: number): void {
    const order = STAGES[this.method];
    const end = to === "done" ? order.length : order.indexOf(to);
    for (let i = order.indexOf(from); i < end; i++) if (this.splits[order[i]] === undefined) this.splits[order[i]] = t - (this.t0 ?? t);
  }

  /** observe 수준 (그리고 덩어리 없이 움직일 때): 단계가 넘어가는지만 본다 */
  private observeProgress(before: StageId, now: StageId, t: number): Utterance[] {
    const order = STAGES[this.method];
    if (now === before) return [];
    this.stage = now;
    if (now === "done") {
      this.markSplits(before, now, t);
      return this.finish(t);
    }
    if (order.indexOf(now) > order.indexOf(before)) {
      this.markSplits(before, now, t);
      // 미리 만든 음성은 "흰색 십자 완성!"처럼 느낌표 문장이다 (stageDoneText 와 같은 첫 문장). 마침표로 쓰면 읽을 클립이 없다.
      return [say(`${STAGE_NAME[before]} 완성!`, "stage")];
    }
    return [];
  }

  private replanSilently(t: number): Utterance[] {
    // 힌트 수준에서는 다른 길로 가도 막지 않는다: 지금 상태에서 계획만 새로 세워 둔다
    this.fixes = [];
    const prevHint = this.hintLevel;
    const step = planNext(this.s, this.method);
    if (!step) return this.finish(t);
    if (step.stage !== this.stage) {
      const order = STAGES[this.method];
      const out: Utterance[] = [];
      if (order.indexOf(step.stage) > order.indexOf(this.stage)) {
        this.markSplits(this.stage, step.stage, t);
        out.push(this.stageDone(this.stage, step.stage));
      } else {
        // 실수로 앞 단계가 흐트러졌다
        out.push(say(`${josa.iGa(STAGE_NAME[step.stage])} 흐트러졌어요. 먼저 다시 맞춰요.`, "fix"));
      }
      this.cur = null;
      return [...out, ...this.announceNext(t)];
    }
    // 새 계획은 아직 말하지 않았다: 기호는 다음 힌트에서 말한 뒤에야 화면에 보인다.
    // 공식 이름은 앞서 말한 이름과 같을 때만 그대로 둔다 (윗면을 한 번 돌려 맞춘 것처럼 같은 공식을 다른 방향에서 할 때).
    const named = !!this.cur?.named && this.cur.step.caseName === step.caseName;
    this.cur = { step, moves: [...step.moves], idx: 0, partial: 0, announcedAt: t, firstMoveAt: null, mistakes: 0, silentUntil: step.moves.length, named, shown: 0 };
    this.hintLevel = prevHint;
    return [];
  }

  /** 두 번째 힌트: 처음 몇 수를 지금 방향의 기호로 (방향이 바뀌면 다시 말할 수 있게 기준을 적어 둔다) */
  private firstMovesHint(): Utterance[] {
    const c = this.cur!;
    const view = this.told(this.view());
    const first = c.moves.slice(c.idx, c.idx + Math.min(3, c.moves.length - c.idx));
    c.shown = Math.max(c.shown, c.idx + first.length);
    return [say(`처음은 ${chunkText(this.s, view, first)}.`, "hint")];
  }

  private escalateHint(): Utterance[] {
    const c = this.cur!;
    this.hintLevel++;
    if (this.hintLevel === 1) {
      const out: Utterance[] = [];
      if (c.step.view && !sameView(this.pose, c.step.view)) out.push(say(holdText(c.step.view), "hold"));
      if (c.step.alg) {
        c.named = true;
        out.push(say(`${josa.eulReul(c.step.caseName ?? (c.step.alg === "R U R' U'" ? "섹시무브" : "공식"))} 써 보세요.`, "hint"));
      }
      else out.push(say(`${countWord(c.moves.length - c.idx)} 수면 돼요.`, "hint"));
      return out;
    }
    if (this.hintLevel === 2) return this.firstMovesHint();
    // 3단계: 한 수씩 불러 준다
    c.silentUntil = 0;
    this.level = "guided";
    this.verbosity = "full";
    return [say("같이 해 볼게요.", "hint"), ...this.moveUtterances("full")];
  }

  private abandon(t: number): Utterance[] {
    this.fixes = [];
    this.cur = null;
    return [say("괜찮아요. 지금 상태에서 다시 안내할게요.", "fix"), ...this.announceNext(t)];
  }

  private finish(t: number): Utterance[] {
    this.cur = null;
    this.stage = "done";
    this.ended = true; // 완성 정리는 한 번만: 이 뒤의 회전은 새로 시작할 때까지 기록하지 않는다
    const total = this.t0 === null ? 0 : (t - this.t0) / 1000;
    const n = this.moves.length;
    const order = STAGES[this.method];
    let slowest: StageId | null = null;
    let worst = -1;
    let prev = 0;
    for (const st of order) {
      const end = this.splits[st];
      if (end === undefined) continue;
      if (end - prev > worst) {
        worst = end - prev;
        slowest = st;
      }
      prev = end;
    }
    const parts = [`완성! 총 ${durationWord(total)}, ${countWord(Math.min(n, 99))}${n > 99 ? " 수 넘게" : " 수"} 걸렸어요.`];
    if (slowest && total > 0) parts.push(`가장 오래 걸린 단계는 ${josa.ieosseoyo(STAGE_NAME[slowest])}.`);
    return [say(parts.join(" "), "summary")];
  }
}

export function durationWord(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec - m * 60);
  if (m === 0) return secondsWord(s);
  return `${sinoWord(m)} 분${s ? " " + secondsWord(s) : ""}`;
}
