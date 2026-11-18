/**
 * 음성 대기열: 코치가 만든 문장을 언제, 무엇부터 소리 낼지 정한다.
 *
 * 실시간 코칭에서 중요한 규칙
 * 1. 낡은 지시는 버린다. 사람이 이미 다음 수로 넘어갔는데 지난 지시를 끝까지 읽으면 방해만 된다.
 *    새 "지시(move/hold)"가 오면 앞서 받아 둔 채 아직 읽지 않은 지시는 지우고, 읽고 있던 지시도 끊는다.
 *    한 번에 같이 들어온 말(예: "노랑이 위… 잡아 주세요." + 첫 동작)은 코치가 차례대로 하라는 것이라 지우지 않는다.
 * 2. 바로잡기(fix)는 무엇보다 급하다. 말하던 것을 끊고 맨 앞에서 말한다.
 *    바로잡기도 낡는다: 사람이 이미 되돌렸으면 "되돌리세요"는 방해다. 그래서
 *    - 코치가 시키던 바로잡기가 끝나면(setFixing(false)) 새 말이 없어도 남은 바로잡기를 지우고 읽던 것도 끊는다.
 *    - 바로잡는 중에 새 지시가 오면("한 번 더") 앞서 받아 둔 바로잡기도 지우고 읽던 것도 끊는다.
 * 3. 칭찬은 그 뒤에 새 말이 밀려 들어왔으면 생략한다 (같이 들어온 다음 지시 앞에서는 말한다: "좋아요. 알, …").
 *
 * 소리를 내는 방법(엔진)은 바꿔 끼울 수 있다.
 *  - ClipEngine: 미리 합성해 둔 음성 파일을 바로 재생 (지연 거의 없음).
 *  - WebSpeechEngine: 브라우저 내장 음성 합성 (모델 없이 어디서나, 음질은 기기마다 다름).
 *
 * 목소리는 한 번에 하나만. 클립을 쓰는 동안에는 문장마다 브라우저 음성을 섞지 않는다
 * (섞으면 대사마다 목소리가 바뀌어 들린다). 클립 목록 자체를 못 불러왔을 때만 전부 브라우저 음성으로 간다.
 */

import type { Utterance } from "./coach.ts";

export interface SpeechEngine {
  /** 다 읽거나 끊기면 끝나는 약속 */
  speak(text: string): Promise<void>;
  cancel(): void;
  /** 곧 읽을 문장을 미리 준비한다 (파일 받기 등). 없어도 된다. */
  prepare?(text: string): void;
}

const INSTRUCTION = new Set<Utterance["kind"]>(["move", "hold"]);
const INTERRUPTIBLE = new Set<Utterance["kind"]>(["move", "hold", "praise", "goal", "intro", "hint", "status"]);

/** 대기열의 한 줄: 같이 들어온 묶음 번호를 붙여 둔다 (같은 묶음은 코치가 차례대로 말하라고 낸 것) */
interface Queued {
  u: Utterance;
  batch: number;
}

export class VoiceQueue {
  private queue: Queued[] = [];
  private speaking: Utterance | null = null;
  private engine: SpeechEngine;
  private batch = 0;
  private fixing = false; // 코치가 바로잡기를 시키는 중인가 (setFixing 으로 알려 준다)
  enabled = true;
  onSpeak: ((u: Utterance) => void) | null = null;

  constructor(engine: SpeechEngine) {
    this.engine = engine;
  }

  setEngine(engine: SpeechEngine): void {
    this.cancelAll();
    this.engine = engine;
  }

  push(us: readonly Utterance[]): void {
    if (!us.length) return;
    const b = ++this.batch;
    // "낡았다"는 것은 앞선 묶음에서 온 것만이다. 같은 묶음 안의 잡기 → 동작, 칭찬 → 다음 지시는 다 말한다.
    const old = (q: Queued) => q.batch < b;
    let interrupt = false;
    for (const u of us) {
      if (u.kind === "fix") {
        // 바로잡기: 쌓여 있던 지시·칭찬·이전 바로잡기는 의미가 없다
        this.queue = this.queue.filter((q) => !(old(q) && (INSTRUCTION.has(q.u.kind) || q.u.kind === "praise" || q.u.kind === "fix")));
        // 맨 앞에 두되, 같은 묶음에서 먼저 앞에 선 바로잡기 뒤에 선다 ("괜찮아요…" → "…흐트러졌어요." 차례 유지)
        const at = this.queue.findIndex((q) => q.batch !== b);
        this.queue.splice(at < 0 ? this.queue.length : at, 0, { u, batch: b });
        if (this.speaking && INTERRUPTIBLE.has(this.speaking.kind)) interrupt = true;
        if (this.speaking?.kind === "fix") interrupt = true;
        continue;
      }
      if (INSTRUCTION.has(u.kind)) {
        // 새 지시: 앞서 받아 두고 아직 안 읽은 지시는 낡았다.
        // 바로잡는 중이면 앞선 바로잡기 말도 그렇다 (반대로 돌린 것을 반만 바로잡아 "한 번 더"가 온 경우 등).
        // 바로잡기가 끝난 경우는 setFixing(false)가 이미 지웠다. 바로잡기가 아닌 안내("엉켰네요. …")는 남긴다.
        const stale = (k: Utterance["kind"]) => INSTRUCTION.has(k) || (this.fixing && k === "fix");
        this.queue = this.queue.filter((q) => !(old(q) && stale(q.u.kind)));
        if (this.speaking && stale(this.speaking.kind)) interrupt = true;
      }
      if (u.kind === "stage" || u.kind === "summary") {
        this.queue = this.queue.filter((q) => !(old(q) && q.u.kind === "praise"));
        if (this.speaking && (INSTRUCTION.has(this.speaking.kind) || this.speaking.kind === "praise")) interrupt = true;
      }
      this.queue.push({ u, batch: b });
    }
    if (this.enabled) for (const q of this.queue) this.engine.prepare?.(q.u.text);
    if (interrupt) this.engine.cancel();
    void this.pump();
  }

  /**
   * 코치가 지금 바로잡기를 시키는 중인지 알려 준다 (코치에게서 말을 받을 때마다, 받은 말이 없어도).
   * 시키던 바로잡기가 끝났으면 — 되돌렸거나, 반대로 돌린 것을 마저 돌렸거나, 코치가 그만두었으면 —
   * 남은 바로잡기 말은 낡았다: 아직 안 읽은 것은 지우고, 읽고 있던 것은 끊는다.
   * 그 순간 코치가 아무 말도 하지 않을 수 있어서(익숙한 섹시무브 도중 등) push 와 따로 둔다.
   * 같이 들어온 새 말("괜찮아요. …")을 지우지 않게 push 보다 먼저 부른다.
   */
  setFixing(on: boolean): void {
    if (this.fixing && !on) {
      this.queue = this.queue.filter((q) => q.u.kind !== "fix");
      if (this.speaking?.kind === "fix") this.engine.cancel(); // 끊기만 한다: 다음 말은 pump 가 잇는다
    }
    this.fixing = on;
  }

  cancelAll(): void {
    this.queue = [];
    this.fixing = false;
    this.engine.cancel();
  }

  private async pump(): Promise<void> {
    if (this.speaking) return;
    while (this.queue.length) {
      const { u, batch } = this.queue.shift()!;
      // 칭찬 뒤에 새로 들어온 말이 밀려 있으면 칭찬은 건너뛴다 (같이 들어온 다음 지시는 기다려 준다)
      if (u.kind === "praise" && this.queue.some((q) => q.batch > batch)) continue;
      this.speaking = u;
      this.onSpeak?.(u);
      try {
        if (this.enabled) await this.engine.speak(u.text);
      } catch {
        /* 재생 실패는 무시하고 다음 문장으로 */
      }
      this.speaking = null;
    }
  }
}

/** 브라우저 내장 음성 합성 */
export class WebSpeechEngine implements SpeechEngine {
  rate = 1.08;
  private voice: SpeechSynthesisVoice | null = null;

  constructor() {
    const pick = () => {
      const ko = speechSynthesis.getVoices().filter((v) => v.lang.toLowerCase().startsWith("ko"));
      // 신경망 음성이 있으면 우선 (이름에 Natural/Neural/Google 등이 붙는 경우가 많다)
      this.voice = ko.find((v) => /natural|neural|google|online/i.test(v.name)) ?? ko.find((v) => /yuna|유나/i.test(v.name)) ?? ko[0] ?? null;
    };
    pick();
    speechSynthesis.addEventListener?.("voiceschanged", pick);
  }

  get voiceName(): string {
    return this.voice?.name ?? "(기본)";
  }

  speak(text: string): Promise<void> {
    return new Promise((resolve) => {
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "ko-KR";
      u.rate = this.rate;
      if (this.voice) u.voice = this.voice;
      // 일부 브라우저는 끝났다는 신호(onend)를 빠뜨린다: 글자 수로 어림한 시간이 지나면 넘어간다
      const guard = setTimeout(resolve, 1500 + text.length * 180);
      const done = () => {
        clearTimeout(guard);
        resolve();
      };
      u.onend = done;
      u.onerror = done;
      speechSynthesis.speak(u);
    });
  }

  cancel(): void {
    speechSynthesis.cancel();
  }
}

// --- 문장을 클립으로 풀기 --------------------------------------------------------
//
// 시간·수처럼 끝없이 바뀌는 숫자가 든 문장은 통째로 미리 만들 수 없다.
// 그래서 문장을 "조각"으로 나눠 이어 재생한다. 끊겨 들리지 않게 자르는 자리는
//   (1) 문장 끝(. ! ?)  (2) 쉼표  (3) 숫자 말을 통째로 품은 토막
// 만 쓴다. 조각 하나는 "숫자 + 단위 + 그 뒤 마침까지"를 다 담는다 (예: "삼 초 빨랐어요.").
// 조각 소리는 이어질 자리에서 끝을 내려 읽지 않게, 합성할 때 쉼표를 붙여 만든다 (ttsTextFor).
// 클립은 앞뒤 무음을 잘라 두었으므로(scripts/tts_build_clips.py) 쉼은 여기서 넣는다.

/** 이어 재생할 클립 한 조각: 찾을 열쇠 글과, 뒤에 둘 쉼(ms). 음수면 그만큼 겹쳐 붙인다. */
export interface Clip {
  text: string;
  gap: number;
}

// 쉼 길이는 같은 문장을 통째로 합성했을 때 모델이 두는 쉼에 맞췄다 (F2·M1 평균, 클립 끝의 여백 포함).
export const GAP = {
  /** 마침표·물음표로 끝난 문장 뒤 */
  end: 150,
  /** 느낌표 뒤 ("완성!" 다음은 한 박자 쉰다) */
  excl: 280,
  /** 쉼표 자리 */
  comma: 130,
  /** 한 말 토막 안에서 이어 붙이는 자리 ("평균보다 | 삼 초 빨랐어요.") — 클립 끝 여백만큼 겹친다 */
  phrase: -30,
  /** 따로 들어온 대사와 대사 사이 */
  utter: 300,
};

const clip = (text: string, gap = 0): Clip => ({ text, gap });
const endGap = (piece: string): number => (/!$/.test(piece) ? GAP.excl : GAP.end);

/** 문장 단위로 쪼개기 ("완성! 총 ... 걸렸어요. 가장 ..." → 세 문장) */
function sentences(text: string): string[] {
  return text.split(/(?<=[.!?])\s+/).filter((x) => x.length > 0);
}

/**
 * 숫자가 든 문장을 조각으로 나누는 본. 문장 하나(마침표까지)만 받는다.
 * 돌려주는 것은 후보 여러 개 — 조각이 적은 쪽이 앞이고, 앞에서부터 클립이 다 있는 것을 쓴다.
 * 조각이 하나뿐인 후보는 "이 문장 자체가 숫자 집안"이라는 뜻이다 (합성 목록을 만들 때 쓴다).
 */
const TEMPLATES: ((s: string) => Clip[][])[] = [
  // 완성 정리: "총 일 분 이십삼 초, 여든일곱 수 걸렸어요."
  (s) => {
    const m = /^총 (.+?), (.+? 수(?: 넘게)?) 걸렸어요\.$/.exec(s);
    if (!m) return [];
    const [, dur, count] = m;
    const out = [[clip(`총 ${dur}`, GAP.comma), clip(`${count} 걸렸어요.`)]];
    // 아주 오래 걸려 통째 조각이 없을 때: 분과 초를 따로 붙인다
    const d = /^(\S+ 분) (\S+ 초)$/.exec(dur);
    if (d) out.push([clip(`총 ${d[1]}`, GAP.phrase), clip(d[2], GAP.comma), clip(`${count} 걸렸어요.`)]);
    return out;
  },
  // 기록 비교: "흰색 십자가 평균보다 삼 초 빨랐어요."
  (s) => {
    const m = /^(.+ 평균보다) (\S+ 초 (?:빨랐어요|느렸어요)\.)$/.exec(s);
    return m ? [[clip(m[1], GAP.phrase), clip(m[2])]] : [];
  },
  // 초만 있는 짧은 문장과 공식 연습 정리 (숫자만 바뀐다)
  (s) => (/^\S+ 초\.$/.test(s) || /^알아보는 데 \S+ 초 걸렸어요\.$/.test(s) ? [[clip(s)]] : []),
];

/** 어떤 방법으로 풀렸는지 (보고·시험용): 통째 / 문장 잇기 / 숫자 본 / 쉼표 나누기 / 못 품 */
export type PlanKind = "whole" | "sentences" | "template" | "comma" | "none";

/** 문장 하나를 클립으로: 통째 → 숫자 본 → 쉼표 나누기 */
function planOneK(text: string, has: (t: string) => boolean): { clips: Clip[]; kind: PlanKind } | null {
  if (has(text)) return { clips: [clip(text)], kind: "whole" };
  for (const tmpl of TEMPLATES) {
    for (const parts of tmpl(text)) {
      if (parts.every((p) => has(p.text))) return { clips: parts.map((p) => ({ ...p })), kind: "template" };
    }
  }
  // 쉼표로 늘어놓은 동작 목록 ("오른쪽 위로, 윗면 왼쪽으로, 오른쪽 아래로.")
  const bits = text.split(", ");
  if (bits.length > 1 && bits.every((b) => has(b))) {
    return { clips: bits.map((b, i) => clip(b, i < bits.length - 1 ? GAP.comma : 0)), kind: "comma" };
  }
  return null;
}

function planOne(text: string, has: (t: string) => boolean): Clip[] | null {
  return planOneK(text, has)?.clips ?? null;
}

/** 문장을 재생할 클립 차례로 바꾼다. 못 풀면 null */
export function planClips(text: string, has: (t: string) => boolean): Clip[] | null {
  return planWithKind(text, has).clips;
}

/** planClips 와 같고, 어떤 방법으로 풀었는지도 알려 준다 */
export function planWithKind(text: string, has: (t: string) => boolean): { clips: Clip[] | null; kind: PlanKind } {
  if (has(text)) return { clips: [clip(text)], kind: "whole" };
  const ss = sentences(text);
  if (ss.length <= 1) {
    const one = planOneK(text, has);
    return one ?? { clips: null, kind: "none" };
  }
  const out: Clip[] = [];
  const kinds = new Set<PlanKind>();
  for (const s of ss) {
    const part = planOneK(s, has);
    if (!part) return { clips: null, kind: "none" };
    kinds.add(part.kind);
    part.clips[part.clips.length - 1].gap = endGap(s);
    out.push(...part.clips);
  }
  out[out.length - 1].gap = 0;
  return { clips: out, kind: kinds.has("comma") ? "comma" : kinds.has("template") ? "template" : "sentences" };
}

/**
 * 합성 목록을 만들 때 쓰는 나누기 (클립이 있는지 모르는 상태에서).
 * 기준은 재생할 때와 같다: 문장 단위로 자르고, 숫자가 든 문장은 본대로 조각낸다.
 * 문장 단위로 잘라 두면 코치가 문장을 새로 엮어도 그대로 읽을 수 있다.
 */
export function buildPieces(text: string): string[] {
  return sentences(text).flatMap((s) => {
    const alt = TEMPLATES.map((t) => t(s)).find((a) => a.length);
    return alt ? alt[0].map((c) => c.text) : [s];
  });
}

// --- 합성할 때 모델에 넣을 글 ------------------------------------------------------

const LETTER_KO: Record<string, string> = {
  a: "에이", b: "비", c: "씨", d: "디", e: "이", f: "에프", g: "지", h: "에이치", i: "아이", j: "제이", k: "케이", l: "엘", m: "엠",
  n: "엔", o: "오", p: "피", q: "큐", r: "알", s: "에스", t: "티", u: "유", v: "브이", w: "더블유", x: "엑스", y: "와이", z: "제트",
};
const COUNT_KO = ["영", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉", "열"];

/**
 * 화면 글자를 읽는 소리대로 바꾼다. 합성기는 영어 약자와 "2개"를 엉뚱하게 읽는다
 * (F2L → 뭉개짐, PLL Jb → 끝이 잘림, 2개 → "이 개"). 앱이 찾는 열쇠 글은 그대로 두고 소리만 바꾼다.
 */
export function readAloud(text: string): string {
  return text
    .replace(/F2L/g, "에프투엘")
    .replace(/(\d+) ?개/g, (m, d: string) => (Number(d) <= 10 ? `${COUNT_KO[Number(d)]} 개` : m))
    .replace(/[A-Za-z]+/g, (w) => [...w.toLowerCase()].map((c) => LETTER_KO[c] ?? c).join(""));
}

/** 조각 하나를 합성할 때 모델에 넣을 글. 이어지는 자리면 끝을 내려 읽지 않게 쉼표를 붙인다. */
export function ttsTextFor(piece: string): string {
  const say = readAloud(piece);
  return /[.!?,]$/.test(say) ? say : `${say},`;
}

// --- 클립 재생 -------------------------------------------------------------------

/** 이어 재생할 파일 하나와 그 뒤의 쉼(ms) */
export interface SeqItem {
  url: string;
  gap: number;
}

/** 파일을 재생하는 일. 시험에서는 가짜로 갈아 끼운다. */
export interface ClipPlayer {
  play(url: string): Promise<void>;
  stop(): void;
  /** 뒤에 이어 붙일 파일을 미리 받아 둔다 (조각 사이가 벌어지지 않게) */
  preload?(urls: string[]): void;
  /**
   * 여러 파일을 정한 쉼대로 빈틈없이 이어 재생한다. 못 하면 false (그러면 하나씩 재생한다).
   * 끊겨서 끝난 것도 true.
   */
  playSeq?(items: SeqItem[]): Promise<boolean>;
}

/** 디코딩한 소리에서 가장자리의 거의 무음(봉우리보다 60dB 작은 소리)을 뺀 구간 */
interface Span {
  buf: AudioBuffer;
  start: number;
  dur: number;
}

function edgeSpan(buf: AudioBuffer): Span {
  const x = buf.getChannelData(0);
  let peak = 0;
  for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
  const thr = peak * 1e-3;
  let a = 0;
  let b = x.length - 1;
  while (a < b && Math.abs(x[a]) <= thr) a++;
  while (b > a && Math.abs(x[b]) <= thr) b--;
  return { buf, start: a / buf.sampleRate, dur: (b + 1 - a) / buf.sampleRate };
}

/**
 * Web Audio로 조각을 시계에 맞춰 이어 붙인다. <audio>를 하나씩 틀면 파일마다 시작이 늦어
 * 조각 사이가 들쭉날쭉 벌어지는데, 이렇게 하면 쉼이 정한 그대로 들어간다.
 * AAC 파일 앞에 붙는 인코더 여백도 여기서 깎아 내므로 브라우저마다 달라지지 않는다.
 */
class WebAudioSeq {
  private ctx: AudioContext | null = null;
  private bufs = new Map<string, Promise<Span | null>>();
  private live: AudioBufferSourceNode[] = [];
  private finish: (() => void) | null = null;
  private gen = 0;

  constructor() {
    // 브라우저는 사용자가 누르기 전에는 소리를 막는다: 첫 클릭·키 입력 때 풀어 둔다
    const unlock = () => void this.context()?.resume().catch(() => {});
    for (const ev of ["pointerdown", "keydown", "touchend"]) globalThis.addEventListener?.(ev, unlock, { capture: true, passive: true });
  }

  static supported(): boolean {
    const g = globalThis as { AudioContext?: unknown; webkitAudioContext?: unknown };
    return typeof g.AudioContext === "function" || typeof g.webkitAudioContext === "function";
  }

  private context(): AudioContext | null {
    if (this.ctx) return this.ctx;
    const g = globalThis as unknown as { AudioContext?: typeof AudioContext; webkitAudioContext?: typeof AudioContext };
    const C = g.AudioContext ?? g.webkitAudioContext;
    if (!C) return null;
    try {
      this.ctx = new C();
    } catch {
      return null;
    }
    return this.ctx;
  }

  load(url: string): Promise<Span | null> {
    let p = this.bufs.get(url);
    if (p) {
      this.bufs.delete(url); // 최근에 쓴 것을 뒤로 (오래된 것부터 버린다)
      this.bufs.set(url, p);
      return p;
    }
    const ctx = this.context();
    if (!ctx) return Promise.resolve(null);
    p = fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.arrayBuffer();
      })
      .then((b) => ctx.decodeAudioData(b))
      .then(edgeSpan)
      .catch(() => null);
    this.bufs.set(url, p);
    if (this.bufs.size > 64) this.bufs.delete(this.bufs.keys().next().value!);
    return p;
  }

  async play(items: SeqItem[]): Promise<boolean> {
    const mine = this.gen;
    const ctx = this.context();
    if (!ctx) return false;
    const running = () => ctx.state === "running";
    if (!running()) await Promise.race([ctx.resume().catch(() => {}), sleep(300)]);
    if (!running()) return false;
    const spans = await Promise.all(items.map((i) => this.load(i.url)));
    if (mine !== this.gen) return true; // 받는 사이에 끊겼다
    if (spans.some((s) => !s)) return false;
    let t = ctx.currentTime + 0.02;
    const srcs: AudioBufferSourceNode[] = [];
    try {
      spans.forEach((s, i) => {
        const src = ctx.createBufferSource();
        src.buffer = s!.buf;
        src.connect(ctx.destination);
        src.start(t, s!.start, s!.dur);
        srcs.push(src);
        t += s!.dur + Math.max(items[i].gap, -s!.dur * 500) / 1000;
      });
    } catch {
      this.live = srcs;
      this.stop(); // 시계에 못 올렸다: 올린 것까지 거두고 하나씩 재생으로 넘긴다
      return false;
    }
    this.live = srcs;
    return new Promise((resolve) => {
      const done = () => {
        if (this.live === srcs) {
          this.live = [];
          this.finish = null;
        }
        resolve(true);
      };
      this.finish = done;
      srcs[srcs.length - 1].onended = done;
    });
  }

  stop(): void {
    this.gen++;
    for (const s of this.live) {
      try {
        s.stop();
        s.disconnect();
      } catch {
        /* 이미 끝난 것 */
      }
    }
    this.live = [];
    this.finish?.();
    this.finish = null;
  }
}

// 목소리를 바꿀 때마다 엔진을 새로 만들어도 AudioContext 는 하나만 쓴다 (브라우저마다 개수 제한이 있다)
let shared: WebAudioSeq | null | undefined;
function sharedSeq(): WebAudioSeq | null {
  if (shared === undefined) shared = WebAudioSeq.supported() ? new WebAudioSeq() : null;
  return shared;
}

/** 브라우저 재생: 한 파일은 <audio>로, 여러 조각은 Web Audio로 이어 붙인다 */
export class AudioClipPlayer implements ClipPlayer {
  private audio: HTMLAudioElement | null = null;
  private done: (() => void) | null = null;
  private cache = new Map<string, HTMLAudioElement>();
  private seq: WebAudioSeq | null = sharedSeq();

  preload(urls: string[]): void {
    if (this.seq && urls.length > 1) {
      for (const url of urls) void this.seq.load(url);
      return;
    }
    for (const url of urls) {
      if (this.cache.has(url)) continue;
      const a = new Audio(url);
      a.preload = "auto";
      this.cache.set(url, a);
      if (this.cache.size > 24) this.cache.delete(this.cache.keys().next().value!);
    }
  }

  play(url: string): Promise<void> {
    const a = this.cache.get(url) ?? new Audio(url);
    this.cache.delete(url); // 재생 중인 것은 따로 들고 있는다
    return new Promise((resolve) => {
      this.audio = a;
      this.done = resolve;
      a.currentTime = 0;
      a.onended = () => resolve();
      a.onerror = () => resolve();
      a.play().catch(() => resolve());
    });
  }

  playSeq(items: SeqItem[]): Promise<boolean> {
    return this.seq ? this.seq.play(items) : Promise.resolve(false);
  }

  stop(): void {
    this.seq?.stop();
    this.audio?.pause();
    this.audio = null;
    this.done?.();
    this.done = null;
  }
}

/**
 * 미리 합성한 음성 파일 재생. manifest는 { 문장(또는 조각): 파일 이름 } 꼴.
 *
 * 목록에 없는 문장은 조각을 이어 붙여 읽는다. 그래도 못 푸는 문장은 읽을 수 있는 문장만 읽고
 * 나머지는 콘솔에 알린다 — 여기서 브라우저 음성을 부르면 대사마다 목소리가 바뀌기 때문이다.
 * (화면에는 글로 나오므로 내용은 놓치지 않는다. 클립 목록을 아예 못 불러온 경우는 부르는 쪽이
 *  엔진 자체를 WebSpeechEngine 으로 바꾼다 — main.ts 의 loadClips. 목록이 비어 있으면 여기서 그렇게 한다.)
 */
export class ClipEngine implements SpeechEngine {
  private manifest: Record<string, string>;
  private base: string;
  private fallback: SpeechEngine;
  private player: ClipPlayer;
  private empty: boolean;
  private epoch = 0;
  private lastEnd = 0;
  private wake: (() => void) | null = null;
  private warned = new Set<string>();
  hits = 0;
  misses = 0;
  /** 클립으로 못 푼 문장 (개발 중 확인용) */
  readonly unresolved: string[] = [];
  /** 조각 사이 쉼 배율 (시험에서 0으로 둔다) */
  gapScale = 1;

  constructor(manifest: Record<string, string>, base: string, fallback: SpeechEngine, player?: ClipPlayer) {
    this.manifest = manifest;
    this.base = base;
    this.fallback = fallback;
    this.player = player ?? new AudioClipPlayer();
    this.empty = Object.keys(manifest).length === 0;
  }

  private hasClip = (t: string): boolean => Object.hasOwn(this.manifest, t);

  /** 통째 클립이 있는가 */
  has(text: string): boolean {
    return this.hasClip(text);
  }

  /** 통째로든 조각으로든 읽을 수 있는가 */
  canSpeak(text: string): boolean {
    return planClips(text, this.hasClip) !== null;
  }

  prepare(text: string): void {
    const plan = planClips(text, this.hasClip);
    if (plan) this.player.preload?.(plan.map((c) => this.base + this.manifest[c.text]));
  }

  async speak(text: string): Promise<void> {
    if (this.empty) return this.fallback.speak(text);
    let plan = planClips(text, this.hasClip);
    if (plan) this.hits++;
    else plan = this.partial(text);
    if (!plan.length) return;
    const mine = this.epoch;
    // 앞 대사와 붙지 않게 한 박자 쉰다 (끊고 들어온 바로잡기는 바로 말한다)
    const rest = this.lastEnd + GAP.utter * this.gapScale - Date.now();
    if (rest > 0) await this.pause(rest);
    if (mine !== this.epoch) return;
    await this.run(plan);
    if (mine === this.epoch) this.lastEnd = Date.now();
  }

  /** 통째로는 못 풀었다: 문장별로 되는 것만 읽고 빠진 것을 알린다 */
  private partial(text: string): Clip[] {
    const got: Clip[] = [];
    const lost: string[] = [];
    for (const s of sentences(text)) {
      const p = planOne(s, this.hasClip);
      if (p) {
        p[p.length - 1].gap = endGap(s);
        got.push(...p);
      } else lost.push(s);
    }
    this.misses++;
    if (!this.warned.has(text)) {
      this.warned.add(text);
      this.unresolved.push(text);
      console.warn(`[음성] 클립이 없는 대사: ${JSON.stringify(lost.join(" "))} — web/test/inventory.ts 와 scripts/tts_build_clips.py 를 다시 돌려 주세요.`);
    }
    if (got.length) got[got.length - 1].gap = 0;
    return got;
  }

  private async run(plan: Clip[]): Promise<void> {
    const mine = this.epoch;
    const urls = plan.map((c) => this.base + this.manifest[c.text]);
    if (plan.length > 1 && this.player.playSeq) {
      const ok = await this.player.playSeq(plan.map((c, i) => ({ url: urls[i], gap: c.gap * this.gapScale }))).catch(() => false);
      if (ok || mine !== this.epoch) return;
    }
    // 하나씩 재생 (Web Audio를 못 쓰는 브라우저). 겹쳐 붙이기는 못 하므로 쉼 0으로 둔다.
    this.player.preload?.(urls);
    for (let i = 0; i < plan.length; i++) {
      if (mine !== this.epoch) return;
      await this.player.play(urls[i]);
      const g = Math.max(0, plan[i].gap) * this.gapScale;
      if (g && mine === this.epoch) await this.pause(g);
    }
  }

  /** 끊기면 바로 깨는 쉼 */
  private pause(ms: number): Promise<void> {
    return new Promise((resolve) => {
      const t = setTimeout(() => {
        this.wake = null;
        resolve();
      }, ms);
      this.wake = () => {
        clearTimeout(t);
        this.wake = null;
        resolve();
      };
    });
  }

  cancel(): void {
    this.epoch++;
    this.lastEnd = 0;
    this.wake?.();
    this.fallback.cancel();
    this.player.stop();
  }
}

const sleep = (ms: number) => (ms > 0 ? new Promise<void>((r) => setTimeout(r, ms)) : Promise.resolve());

/** 소리 없이 기록만 하는 엔진 (테스트용): 읽는 데 글자 수에 비례한 시간이 걸린다고 가정 */
export class FakeEngine implements SpeechEngine {
  spoken: string[] = [];
  cancelled = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private done: (() => void) | null = null;
  msPerChar = 1;

  speak(text: string): Promise<void> {
    this.spoken.push(text);
    return new Promise((resolve) => {
      this.done = resolve;
      this.timer = setTimeout(resolve, text.length * this.msPerChar);
    });
  }

  cancel(): void {
    this.cancelled++;
    if (this.timer) clearTimeout(this.timer);
    this.done?.();
    this.done = null;
  }
}
