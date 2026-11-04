/*
 * Realtime voice client for any frontend (Cloudflare Pages, Netlify, a static
 * file). Drop-in module: no bundler, no dependencies.
 *
 *   import { createVoiceClient } from "./s2s.js";
 *
 *   const voice = createVoiceClient({
 *     url:   "wss://YOUR-BACKEND-HOST/v1/realtime",
 *     token: TOKEN,                       // see README: caller token
 *     onStatus: (text) => (label.textContent = text),
 *     onTranscript: (who, text) => append(who, text),
 *   });
 *
 *   await voice.connect();                  // must be inside a click handler
 *   await voice.setVoice("M4");             // F1-F5 female, M1-M5 male
 *   voice.close();
 *
 * The protocol is OpenAI Realtime over WebSocket: 16 kHz PCM16 up, 16 kHz
 * PCM16 down, JSON events both ways.
 */

const SAMPLE_RATE = 16000;

export function createVoiceClient({ url, token, onStatus = () => {}, onTranscript = () => {}, echoCancellation = true }) {
  let ws = null;
  let micCtx = null, outCtx = null, node = null, stream = null;
  let nextAt = 0;
  let activeResponse = false;
  let pendingVoice = null;
  let ready = false;

  const status = (s) => onStatus(s);

  // A browser cannot set headers on a WebSocket, so the token travels in the
  // subprotocol list. The server reads "token.<T>" and still negotiates
  // "realtime" as the protocol.
  function protocols() {
    if (token && /^[A-Za-z0-9._~-]+$/.test(token)) return ["realtime", `token.${token}`];
    return ["realtime"];
  }

  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  function b64(bytes) {
    let s = "";
    const a = new Uint8Array(bytes);
    // Chunked: spreading a whole audio block blows the argument limit.
    for (let i = 0; i < a.length; i += 0x8000) s += String.fromCharCode(...a.subarray(i, i + 0x8000));
    return btoa(s);
  }

  function unb64(str) {
    const bin = atob(str);
    const out = new Int16Array(bin.length / 2);
    for (let i = 0; i < out.length; i++) out[i] = bin.charCodeAt(i * 2) | (bin.charCodeAt(i * 2 + 1) << 8);
    return out;
  }

  function play(int16) {
    if (!outCtx) outCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
    const buf = outCtx.createBuffer(1, int16.length, SAMPLE_RATE);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < int16.length; i++) ch[i] = int16[i] / 32768;
    const src = outCtx.createBufferSource();
    src.buffer = buf;
    src.connect(outCtx.destination);
    const at = Math.max(outCtx.currentTime, nextAt);
    src.start(at);
    nextAt = at + buf.duration;   // gapless queue
  }

  function handle(ev) {
    switch (ev.type) {
      case "session.created":
        ready = true;
        status("siap.");
        break;
      case "input_audio_buffer.speech_started":
        status("mendengar...");
        break;
      case "response.created":
        activeResponse = true;
        status("berpikir...");
        break;
      case "response.output_audio.delta":
        play(unb64(ev.delta));
        status("berbicara...");
        break;
      case "response.output_audio_transcript.delta":
      case "response.audio_transcript.delta":
        onTranscript("assistant", ev.delta || "");
        break;
      case "conversation.item.input_audio_transcription.completed":
        onTranscript("user", ev.transcript || "");
        break;
      case "response.done":
        activeResponse = false;
        status("siap.");
        // A voice picked mid-reply takes effect on the next reply; switching
        // halfway would splice two voices into one answer.
        if (pendingVoice) { const v = pendingVoice; pendingVoice = null; setVoice(v); }
        break;
      case "error":
        // The server rejects a bad token this way, then closes.
        status(`ditolak: ${ev.error?.message || "unknown error"}`);
        break;
    }
  }

  async function connect() {
    if (ws && ws.readyState === WebSocket.OPEN) return;
    status("menghubungkan...");
    ws = new WebSocket(url, protocols());

    await new Promise((ok, fail) => {
      let settled = false;
      const done = (err) => { if (!settled) { settled = true; err ? fail(err) : ok(); } };
      const underlying = (e) => handle(JSON.parse(e.data));

      ws.onmessage = (e) => {
        const ev = JSON.parse(e.data);
        // Resolve only on session.created: the server accepts the socket even
        // when the token is wrong, then sends an error and closes.
        if (ev.type === "session.created") done();
        else if (ev.type === "error") done(new Error(ev.error?.message || "ditolak"));
        underlying(e);
      };
      ws.onclose = (e) => {
        ready = false;
        status(e.code === 1008 ? "ditolak: token salah." : `terputus (${e.code}).`);
        done(new Error(e.code === 1008 ? "token salah" : `terputus (${e.code})`));
      };
      ws.onerror = () => done(new Error("socket gagal dibuka"));
    });

    // Mic after the socket is confirmed: no pointless permission prompt when
    // the token is rejected.
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation, noiseSuppression: true },
    });
    micCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
    await micCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([`
      class Tap extends AudioWorkletProcessor {
        process(inputs) {
          const ch = inputs[0][0];
          if (ch) this.port.postMessage(new Float32Array(ch));
          return true;
        }
      }
      registerProcessor("tap", Tap);
    `], { type: "text/javascript" })));

    node = new AudioWorkletNode(micCtx, "tap");
    node.port.onmessage = ({ data }) => {
      if (!ready || !ws || ws.readyState !== WebSocket.OPEN) return;
      const pcm = new Int16Array(data.length);
      for (let i = 0; i < data.length; i++) pcm[i] = Math.max(-1, Math.min(1, data[i])) * 32767;
      send({ type: "input_audio_buffer.append", audio: b64(pcm.buffer) });
    };
    micCtx.createMediaStreamSource(stream).connect(node);
    status("bicara sekarang - saya dengar.");
  }

  function setVoice(voice) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (activeResponse) { pendingVoice = voice; status(`suara ${voice} aktif di jawaban berikutnya.`); return; }
    send({
      type: "session.update",
      event_id: `voice_${Date.now()}`,
      session: { type: "realtime", audio: { output: { voice } } },
    });
    status(`suara: ${voice}.`);
  }

  function close() {
    ready = false;
    if (node) node.port.onmessage = null;
    if (micCtx) micCtx.close().catch(() => {});
    if (outCtx) outCtx.close().catch(() => {});
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (ws) ws.close();
    ws = micCtx = outCtx = node = stream = null;
  }

  return {
    connect, setVoice, close,
    get connected() { return ready; },
    // Escape hatch for tests and custom transports; not part of the happy path.
    get socket() { return ws; },
  };
}

/** Language and voice options; open route, no token needed. */
export async function capabilities(baseUrl) {
  const r = await fetch(`${baseUrl}/v1/realtime/capabilities`);
  if (!r.ok) throw new Error(`capabilities ${r.status}`);
  return r.json();
}
