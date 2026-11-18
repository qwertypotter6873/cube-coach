/**
 * 한국어 안내 문장.
 *
 * 원칙
 * - 사람이 지금 들고 있는 방향 기준으로 말한다 ("오른쪽 면을 위로"). 색은 보조로만 쓴다.
 * - 지시는 늘 "기호 + 쉬운 말"로 한다 ("알, 오른쪽 면을 위로 올려 주세요"). 짧게 말할 때도 기호는 남긴다.
 * - 숫자도 기호도 읽는 그대로 적는다 ("세 번", "여덟 수", "알 프라임") — 음성 합성기가
 *   "3번"을 "삼 번"으로, R'를 엉뚱하게 읽지 않게. 화면에 쓰는 기호(R')는 moveSymbol이 따로 만든다.
 * - 같은 칭찬을 계속 반복하지 않는다.
 * - 말 길이는 수준에 따라: full(처음 배우는 사람) / short(익숙한 사람).
 */

import type { AlgMove, Color, ColorMove, Letter, State, View } from "./cube.ts";
import { COLOR_KO, letterOf, rebaseView } from "./cube.ts";
import type { Analysis, Method, StageId, Step } from "./method.ts";
import { stageProgress } from "./method.ts";

export type Verbosity = "full" | "short";

const NUM = ["영", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉", "열"];
const SINO = ["영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"];

/** 1~99 고유어 수 관형형 (한, 두, 세 … 열한, 스무) */
export function countWord(n: number): string {
  if (n <= 10) return NUM[n];
  const tens = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"];
  const t = Math.floor(n / 10);
  const o = n % 10;
  if (o === 0) return t === 2 ? "스무" : tens[t];
  return tens[t] + NUM[o];
}

/** 0~999 한자어 수 (일, 십이, 백오) */
export function sinoWord(n: number): string {
  n = Math.round(n);
  if (n === 0) return "영";
  const h = Math.floor(n / 100);
  const t = Math.floor((n % 100) / 10);
  const o = n % 10;
  return (h ? (h > 1 ? SINO[h] : "") + "백" : "") + (t ? (t > 1 ? SINO[t] : "") + "십" : "") + (o ? SINO[o] : "");
}

/** 초 단위 시간 읽기 */
export function secondsWord(sec: number): string {
  return `${sinoWord(sec)} 초`;
}

export const color = (c: Color) => COLOR_KO[c];

/** 받침 있는 낱말 뒤의 조사 (색 이름은 모두 받침이 있다) */
const iGa = (w: string) => (hasBatchim(w) ? `${w}이` : `${w}가`);
const eulReul = (w: string) => (hasBatchim(w) ? `${w}을` : `${w}를`);
const euroRo = (w: string) => {
  const last = w.charCodeAt(w.length - 1) - 0xac00;
  const jong = last >= 0 && last < 11172 ? last % 28 : 0;
  return jong === 0 || jong === 8 ? `${w}로` : `${w}으로`; // 받침 없음·ㄹ받침 → 로
};
function hasBatchim(w: string): boolean {
  const ch = w[w.length - 1];
  if (/[A-Za-z]/.test(ch)) return /[LMNRlmnr]/.test(ch); // 엘, 엠, 엔, 알
  const code = ch.charCodeAt(0) - 0xac00;
  return code >= 0 && code < 11172 && code % 28 !== 0;
}
const gwaWa = (w: string) => (hasBatchim(w) ? `${w}과` : `${w}와`);
const ieyo = (w: string) => (hasBatchim(w) ? `${w}이에요` : `${w}예요`);
const ieosseoyo = (w: string) => (hasBatchim(w) ? `${w}이었어요` : `${w}였어요`);

// --- 회전 한 수 --------------------------------------------------------------

const FACE_KO: Record<Letter, string> = { U: "윗면", D: "아랫면", R: "오른쪽 면", L: "왼쪽 면", F: "앞면", B: "뒷면" };
const WIDE_KO: Record<Letter, string> = { U: "위 두 줄", D: "아래 두 줄", R: "오른쪽 두 줄", L: "왼쪽 두 줄", F: "앞 두 줄", B: "뒤 두 줄" };
const OPP: Record<Letter, Letter> = { U: "D", D: "U", R: "L", L: "R", F: "B", B: "F" };
/** 기호를 소리 내어 읽는 꼴. 화면에는 R, U'처럼 쓰지만 말할 때는 이렇게 읽는다. */
const LETTER_SAY: Record<Letter, string> = { U: "유", D: "디", R: "알", L: "엘", F: "에프", B: "비" };

/**
 * [시계 방향 한 번, 반시계 한 번]의 방향 말과 동사. 모두 앞에서 보는 사람 손 기준이다.
 * 뒷면은 보이지 않으니 "뒷면 윗줄을 왼쪽으로"처럼 윗줄이 가는 쪽으로 말한다 (B = 윗줄이 왼쪽으로).
 */
const DIR: Record<Letter, [[string, string], [string, string]]> = {
  R: [["위로", "올려"], ["아래로", "내려"]],
  L: [["아래로", "내려"], ["위로", "올려"]],
  U: [["왼쪽으로", "돌려"], ["오른쪽으로", "돌려"]],
  D: [["오른쪽으로", "돌려"], ["왼쪽으로", "돌려"]],
  F: [["시계 방향으로", "돌려"], ["반시계 방향으로", "돌려"]],
  B: [["왼쪽으로", "돌려"], ["오른쪽으로", "돌려"]],
};
/** 한 번 돌릴 때 부르는 대상 (뒷면만 "뒷면 윗줄") */
const TURN_KO: Record<Letter, string> = { ...FACE_KO, B: "뒷면 윗줄" };
/** 짧게 말할 때의 방향 (뒷면만 줄여 부른다) */
const DIR_SHORT: Record<Letter, [string, string]> = {
  R: ["위로", "아래로"],
  L: ["아래로", "위로"],
  U: ["왼쪽으로", "오른쪽으로"],
  D: ["오른쪽으로", "왼쪽으로"],
  F: ["시계 방향으로", "반시계 방향으로"],
  B: ["윗줄 왼쪽으로", "윗줄 오른쪽으로"],
};

/** 지금 들고 있는 방향 view에서 이 회전이 어느 면인지 */
export function moveLetter(s: State, view: View, m: AlgMove | ColorMove): { letter: Letter; wide: boolean } {
  const token = "token" in m ? m.token : "";
  const wide = /^[urfdlb]|w/.test(token);
  const l = letterOf(s, view, m.color);
  return { letter: wide ? OPP[l] : l, wide };
}

/**
 * 쉬운 말로 한 수. 면 이름은 짧게 말할 때도 "면"을 붙여 둔다
 * ("왼쪽 위로"는 왼쪽 면인지 왼쪽 위 모서리인지 헷갈린다).
 */
export function moveText(s: State, view: View, m: AlgMove | ColorMove, v: Verbosity = "full"): string {
  const { letter, wide } = moveLetter(s, view, m);
  const name = wide ? WIDE_KO[letter] : FACE_KO[letter];
  if (m.k === 2) return v === "full" ? `${eulReul(name)} 두 번 돌려 주세요` : `${name} 두 번`;
  if (v === "short") return `${name} ${DIR_SHORT[letter][m.k === 1 ? 0 : 1]}`;
  const [dir, verb] = DIR[letter][m.k === 1 ? 0 : 1];
  const what = wide ? name : TURN_KO[letter];
  return `${eulReul(what)} ${dir} ${verb} 주세요`;
}

/** 공식 기호 (화면 표시용): 지금 방향 기준 */
export function moveSymbol(s: State, view: View, m: AlgMove | ColorMove): string {
  const { letter, wide } = moveLetter(s, view, m);
  return (wide ? letter.toLowerCase() : letter) + ["", "", "2", "'"][m.k];
}

/** 기호를 읽는 꼴 (알, 유 프라임, 알 투, 두 줄은 알 더블) */
export function symbolSay(s: State, view: View, m: AlgMove | ColorMove): string {
  const { letter, wide } = moveLetter(s, view, m);
  return LETTER_SAY[letter] + (wide ? " 더블" : "") + ["", "", " 투", " 프라임"][m.k];
}

/** 말할 때 쓰는 한 수: 기호를 먼저 부르고 쉬운 말을 덧붙인다 */
export function moveSay(s: State, view: View, m: AlgMove | ColorMove, v: Verbosity = "full", sep = ", "): string {
  return symbolSay(s, view, m) + sep + moveText(s, view, m, v);
}

/** 여러 수를 한 번에 부를 때 (익숙한 사람용): 이름 붙은 묶음은 이름으로 */
const TRIGGERS: [string, string][] = [
  ["R U R' U'", "섹시무브"],
  ["R' F R F'", "슬레지해머"],
];

/**
 * 여러 수를 차례로 둘 때 수마다 사람이 보게 될 방향. 지금 view로 들고 시작한다고 본다.
 * 두 줄 돌리기(r, d)는 큐브를 통째로 함께 돌리므로 그 뒤 수는 돌아간 방향에서 부른다.
 * 공식 기준 방향이 바뀐 만큼(ms[0] → 그 수) 사람이 든 방향도 똑같이 돌린다.
 */
function viewsAlong(s: State, view: View, ms: readonly AlgMove[]): View[] {
  const base = ms[0]?.viewBefore;
  return ms.map((m) => (base && m.viewBefore ? rebaseView(s, base, m.viewBefore, view) : view));
}

/** 여러 수의 기호 (화면 표시용, 지금 방향 기준): ["r", "U", "R'", …] */
export function chunkSymbols(s: State, view: View, ms: readonly AlgMove[]): string[] {
  const vs = viewsAlong(s, view, ms);
  return ms.map((m, i) => moveSymbol(s, vs[i], m));
}

export function chunkText(s: State, view: View, ms: readonly AlgMove[]): string {
  const vs = viewsAlong(s, view, ms);
  const syms = chunkSymbols(s, view, ms).join(" ");
  for (const [alg, name] of TRIGGERS) {
    const n = alg.split(" ").length;
    if (syms === alg) return name;
    const parts = syms.split(" ");
    if (parts.length % n === 0 && parts.length > n) {
      const reps = parts.length / n;
      if (Array.from({ length: reps }, () => alg).join(" ") === syms) return `${name} ${countWord(reps)} 번`;
    }
  }
  return ms.map((m, i) => moveSay(s, vs[i], m, "short", " ")).join(", ");
}

// --- 방향 잡기 ---------------------------------------------------------------

export function holdText(v: View): string {
  return `${iGa(color(v.up))} 위, ${iGa(color(v.front))} 앞에 오게 잡아 주세요.`;
}

export function sameView(a: View | null | undefined, b: View | null | undefined): boolean {
  return !!a && !!b && a.up === b.up && a.front === b.front;
}

// --- 기준 면 안내 -------------------------------------------------------------

/**
 * 시작할 때: 무엇을 기준으로 말하는지, 기호가 어느 면인지, 프라임·투가 무엇인지.
 * 여기에는 색을 넣지 않는다. 색은 곧이어 "노랑이 위, 초록이 앞에 오게 잡아 주세요"(holdText)로 말한다
 * — 카메라가 아직 자세를 못 읽었을 수 있어서 "지금 ~예요"가 아니라 잡아 달라고 한다.
 */
const BASIS = "안내는 화면 속 큐브가 아니라, 손에 든 큐브를 내 눈으로 보는 방향 기준이에요.";
const FACE_LETTERS = "나를 보는 면이 앞면 에프, 위가 윗면 유, 오른쪽이 알, 왼쪽이 엘, 아래가 디, 뒤가 비예요.";
const SYMBOL_RULE =
  "알처럼 기호만 부르면 그 면을 정면에서 봤을 때 시계 방향으로 한 번, 그러니까 구십 도 돌리는 거예요. 알 프라임은 반대 방향으로 한 번, 알 투는 두 번이에요.";
const CFOP_RULE = "나를 보는 면이 에프, 위가 유예요. 기호만 부르면 그 면을 정면에서 봤을 때 시계 방향, 프라임은 반대 방향, 투는 두 번이에요.";
const BASIS_AGAIN = "안내는 손에 든 큐브를 내 눈으로 보는 방향 기준이에요. 나를 보는 면이 에프예요.";

/** 시작 안내 문장들. 처음 쓰는 사람에게만 자세히(초급은 더 자세히), 다음부터는 한 줄. */
export function orientationLines(method: Method, taught: boolean): string[] {
  if (taught) return [BASIS_AGAIN];
  if (method === "beginner") return [BASIS, FACE_LETTERS, SYMBOL_RULE];
  return [BASIS, CFOP_RULE];
}

/** 들고 있는 방향이 바뀌었을 때 한 줄. 바뀐 쪽만 말한다. 둘 다 바뀌면 잡는 법(holdText)처럼 위부터. */
export function reframeText(prev: View | null, now: View): string {
  const f = `${iGa(color(now.front))} 앞`;
  const u = `${iGa(color(now.up))} 위`;
  if (prev && prev.front === now.front) return `이제 ${u}예요.`;
  if (prev && prev.up === now.up) return `이제 ${f}이에요.`;
  return `이제 ${u}, ${f}이에요.`;
}

// --- 단계 소개 (처음 들어설 때 한 번) -----------------------------------------

export const STAGE_NAME: Record<StageId, string> = {
  cross: "흰색 십자",
  corners: "1층 꼭짓점",
  middle: "2층 모서리",
  yCross: "노란 십자",
  yFace: "노란 면",
  llCorners: "3층 꼭짓점 자리",
  llEdges: "3층 모서리 자리",
  f2l: "F2L",
  oll: "OLL",
  pll: "PLL",
  done: "완성",
};

const BEGINNER_INTRO: Partial<Record<StageId, string>> = {
  cross: "1단계는 흰색 십자예요. 노란색을 위로, 흰색을 아래로 잡고, 흰색 모서리 네 개를 옆면 색까지 맞춰 아래에 십자로 모읍니다.",
  corners: "2단계, 1층 꼭짓점이에요. 흰색이 붙은 꼭짓점을 들어갈 자리 바로 위로 옮긴 다음, 섹시무브를 반복하면 쏙 들어가요.",
  middle: "3단계, 2층 모서리예요. 노란색이 없는 위층 모서리를 옆면 색에 맞춘 다음, 오른쪽이나 왼쪽 공식으로 내려 넣습니다.",
  yCross: "4단계, 노란 십자예요. 공식 하나로 점, ㄱ자, 일자를 거쳐 노란 십자를 만듭니다.",
  yFace: "5단계, 노란 면이에요. 수네 공식을 반복해서 윗면을 모두 노랗게 만듭니다.",
  llCorners: "6단계, 3층 꼭짓점이에요. 같은 색 꼭짓점 두 개를 헤드라이트라고 불러요. 공식으로 네 꼭짓점을 제자리에 보냅니다.",
  llEdges: "마지막 7단계, 3층 모서리예요. 공식 한두 번이면 끝납니다.",
};

const CFOP_INTRO: Partial<Record<StageId, string>> = {
  cross: "십자부터 할게요. 흰색 십자를 아래에 한 번에 만들어 봅니다.",
  f2l: "F2L이에요. 흰색 꼭짓점과 옆 모서리를 한 쌍으로 묶어 슬롯에 넣습니다. 노랑을 위로, 넣을 슬롯을 오른쪽 앞에 두세요.",
  oll: "OLL, 윗면을 노랗게 만드는 단계예요. 모서리 먼저, 그다음 꼭짓점.",
  pll: "PLL, 마지막 층 자리를 맞추는 단계예요.",
};

export function stageIntro(stage: StageId, method: Method, a: Analysis, v: Verbosity): string {
  const intro = (method === "beginner" ? BEGINNER_INTRO : CFOP_INTRO)[stage] ?? "";
  const p = stageProgress(a, stage);
  const status = p.total === 4 && p.done > 0 ? ` 지금 네 개 중 ${NUM[p.done]} 개가 되어 있어요.` : "";
  return v === "full" ? intro + status : `${STAGE_NAME[stage]} 단계.${status}`;
}

// --- 덩어리(Step) 설명 --------------------------------------------------------

export function stepGoal(s: State, step: Step, v: Verbosity): string {
  const it = step.intent;
  const n = step.moves.length;
  switch (it.type) {
    case "crossEdge":
      return `흰색과 ${color(it.colors[1])} 모서리를 ${color(it.colors[1])} 면 아래에 맞출게요.`;
    case "crossAll":
      return `십자는 ${countWord(n)} 수면 됩니다.`;
    case "cornerOut":
      return `흰색, ${color(it.colors[1])}, ${color(it.colors[2])} 꼭짓점이 아래층 다른 자리에 끼어 있어요. 섹시무브 한 번으로 먼저 꺼낼게요.`;
    case "cornerAbove":
      return `흰색, ${color(it.colors[1])}, ${color(it.colors[2])} 꼭짓점을 들어갈 자리 위로 옮길게요.`;
    case "cornerInsert":
      return v === "full"
        ? `${gwaWa(color(it.colors[1]))} ${color(it.colors[2])} 사이 자리를 오른쪽 앞에 두고, 섹시무브 ${countWord(step.repeat ?? 1)} 번이면 들어가요.`
        : `섹시무브 ${countWord(step.repeat ?? 1)} 번.`;
    case "edgeAlign":
      return `${color(it.colors[0])}, ${color(it.colors[1])} 모서리를 같은 색 면 위로 옮길게요.`;
    case "edgeInsert":
      return `${color(it.colors[0])}, ${color(it.colors[1])} 모서리를 ${it.side === "right" ? "오른쪽" : "왼쪽"}으로 내려 넣을게요.`;
    case "edgeOut":
      return `${color(it.colors[0])}, ${color(it.colors[1])} 모서리가 2층에 잘못 들어가 있어요. 오른쪽 넣기 공식으로 꺼낼게요.`;
    case "pair":
      return `${color(it.colors[0])}, ${color(it.colors[1])} 쌍을 넣을게요. ${countWord(n)} 수예요.`;
    case "llAlg":
      return llText(step, v);
    case "auf":
      return it.final ? "마지막으로 윗면만 맞추면 끝이에요." : "공식을 하기 전에 윗면 위치를 맞출게요.";
  }
}

function llText(step: Step, v: Verbosity): string {
  const it = step.intent as { type: "llAlg"; shape: string };
  const name = step.caseName ?? "";
  if (step.stage === "yCross") return `지금은 ${it.shape} 모양이에요. 노란 십자 공식을 할게요.`;
  if (step.stage === "yFace") return `지금은 ${it.shape} 모양이에요. 수네를 할게요.`;
  if (step.stage === "llCorners") return "꼭짓점 자리 바꾸기 공식이에요.";
  if (step.stage === "llEdges") return "모서리 돌리기 공식이에요.";
  if (step.stage === "oll") return v === "full" ? `${it.shape} 모양이에요. ${name} 공식을 할게요.` : `${name}.`;
  return v === "full" ? `PLL ${name} 공식이에요.` : `${name}.`;
}

// --- 피드백 ------------------------------------------------------------------

const PRAISE = ["좋아요.", "잘했어요.", "네, 맞아요.", "좋습니다.", "깔끔해요."];

export class Praise {
  private i = 0;
  next(): string {
    return PRAISE[this.i++ % PRAISE.length];
  }
}

export function stageDoneText(stage: StageId, nextStage: StageId): string {
  if (nextStage === "done") return "완성! 정말 잘했어요.";
  return `${STAGE_NAME[stage]} 완성! 다음은 ${ieyo(STAGE_NAME[nextStage])}.`;
}

export function wrongDirectionText(s: State, view: View, fix: ColorMove, v: Verbosity): string {
  return `반대 방향이에요. ${moveSay(s, view, fix, v)}.`;
}

export function wrongFaceText(s: State, view: View, done: ColorMove, fix: ColorMove, v: Verbosity): string {
  const { letter } = moveLetter(s, view, done);
  return `앗, 방금은 ${eulReul(FACE_KO[letter])} 돌렸어요. 먼저 되돌릴게요. ${moveSay(s, view, fix, v)}.`;
}

export function halfDoneText(): string {
  return "한 번 더요.";
}

export const josa = { euroRo, iGa, eulReul, gwaWa, ieyo, ieosseoyo, hasBatchim };
