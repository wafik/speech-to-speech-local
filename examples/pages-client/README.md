# Menyambungkan frontend (Cloudflare Pages) ke backend suara

Frontend tidak perlu tahu apa pun soal STT/TTS. Ia hanya perlu tiga hal:
**URL backend**, **token**, dan `s2s.js`.

```
browser ──login──> pages.dev (Pages Function + cookie sesi)
        └─/app───> wss://<backend>/v1/realtime?  (token di subprotocol)
                        └─> laptop: VAD → ASR → LLM → TTS → audio balik
```

## 1. Nyalakan backend

Di laptop:

```powershell
make router-low      # cetak token, contoh: Token: xxxxx
make tunnel          # publish ke hostname (butuh _ws_token dari langkah 1)
```

Catat dua nilai: **hostname** (`wss://...`) dan **token**. Token tersimpan di
`_ws_token`; mau pin sendiri: `make router-low WS_TOKEN=pilihanmu`.

Cek cepat dari laptop bahwa backend hidup:

```powershell
curl http://127.0.0.1:8765/v1/realtime/capabilities
```

## 2. Simpan token sebagai environment variable di Pages

Cloudflare Dashboard → **Workers & Pages** → project frontend Anda →
**Settings → Environment variables** → tambah:

| Nama | Nilai | Tipe |
|---|---|---|
| `S2S_URL` | `wss://<hostname-bakend>` | Plaintext |
| `S2S_TOKEN` | token dari langkah 1 | **Secret** |

Keduanya harus di-set untuk **Production** (dan Preview kalau dipakai).

Token jangan ditulis di file HTML/JS repo. Kalau terpaksa (karena malas bikin
fungsi), sadari konsekuensinya: `S2S_TOKEN` di JS statis bisa dibaca siapa pun
lewat devtools atau "view source", tapi halaman `/app` sendiri juga sudah di
balik login — jadi risikonya sebatas orang yang sudah login. Opsi yang lebih
rapi ada di langkah 3b.

## 3a. Cara cepat: suntik lewat route yang sudah dilindungi login

Kalau `/app` disajikan oleh Pages Function yang memeriksa cookie sesi, kirim
konfigurasi di response HTML-nya. Di fungsi itu, sebelum `return`:

```js
const html = await env.ASSETS.fetch(request);      // atau template milikmu
const body = (await html.text()).replace(
  "</head>",
  `<script>window.S2S = ${JSON.stringify({
      url: env.S2S_URL,
      token: env.S2S_TOKEN,
    })};</script></head>`
);
return new Response(body, { headers: { "content-type": "text/html; charset=utf-8" } });
```

Dengan begitu token hanya dikirim ke pengguna yang **sudah login**, bukan ke
siapa pun yang membuka halaman publik.

## 3b. Cara paling aman: token tidak pernah menyentuh browser

Bikin fungsi `/api/voice-ticket` yang memeriksa cookie sesi lalu mengembalikan
token. Panggil saat tombol mikrofon ditekan:

```js
const r = await fetch("/api/voice-ticket");         // 401 kalau belum login
const { url, token } = await r.json();
```

Isi fungsinya (Pages Function):

```js
export async function onRequestGet({ request, env }) {
  const session = await validateSession(request, env);   // fungsi login yang sudah ada
  if (!session) return new Response('{"error":"unauthorized"}', { status: 401 });
  return Response.json({ url: env.S2S_URL, token: env.S2S_TOKEN });
}
```

Bedanya dengan 3a: browser tetap menerima token, tapi hanya setelah login dan
hanya saat diminta — bukan tertanam di HTML yang mungkin di-cache atau
dibagikan.

## 4. Pasang `s2s.js` di `/app`

Salin `examples/pages-client/s2s.js` ke project Pages (mis. `public/s2s.js`),
lalu di halaman chat:

```html
<button id="go">Mulai bicara</button>
<select id="voice">
  <option value="F1">F1 - wanita</option>
  <option value="F2">F2 - wanita</option>
  <option value="F3">F3 - wanita</option>
  <option value="F4">F4 - wanita</option>
  <option value="F5">F5 - wanita</option>
  <option value="M1">M1 - pria</option>
  <option value="M2">M2 - pria</option>
  <option value="M3">M3 - pria</option>
  <option value="M4">M4 - pria</option>
  <option value="M5">M5 - pria</option>
</select>
<div id="status">Siap.</div>
<div id="log"></div>

<script type="module">
import { createVoiceClient } from "./s2s.js";

const status = (s) => (document.querySelector("#status").textContent = s);
const log = (who, text) => {
  const row = document.createElement("div");
  row.textContent = `${who}: ${text}`;
  document.querySelector("#log").prepend(row);
};

let voice = null;

document.querySelector("#go").onclick = async () => {
  if (!voice) {
    voice = createVoiceClient({
      url: window.S2S.url,          // dari langkah 3a, atau hasil /api/voice-ticket
      token: window.S2S.token,
      onStatus: status,
      onTranscript: log,
    });
  }
  try {
    await voice.connect();           // harus di dalam handler klik: butuh gesture
    await voice.setVoice(document.querySelector("#voice").value);
  } catch (e) {
    status(`gagal: ${e.message}`);
  }
};

document.querySelector("#voice").onchange = (e) => voice?.setVoice(e.target.value);
</script>
```

Catatan penting soal browser:

- `connect()` **harus** dipanggil dari handler klik (atau gesture lain).
  AudioContext dan izin mikrofon ditolak tanpa gesture pengguna.
- Halaman harus **HTTPS** (`pages.dev` sudah). `wss://` wajib dari HTTPS;
  `ws://` akan diblokir browser.
- Microphone butuh izin; di iframe juga butuh `allow="microphone"`.

## 4b. Cek cepat tanpa menyentuh Pages

Buka `http://127.0.0.1:8080/test.html?url=wss://<backend>/v1/realtime&token=<token>`
setelah `python -m http.server 8080 --directory examples/pages-client` — halaman
itu memuat `s2s.js` dari origin yang **berbeda** dari backend, jadi ia menguji
persis jalur yang dipakai Pages: subprotocol token, `wss://`, dan CORS untuk
`fetch` capabilities.

## 5. Uji

1. Buka URL Pages Anda, login.
2. Masuk `/app`, tekan **Mulai bicara**, izinkan mikrofon.
3. Status harus berubah: `menghubungkan...` → `siap.` → `mendengar...` →
   `berpikir...` → `berbicara...`.
4. Bicara satu kalimat pendek dalam bahasa Indonesia; tunggu jeda ~400 ms.

Kalau berhasil, audio balasan keluar dari speaker dan transkrip muncul di log.

## Kalau gagal

| Gejala | Sebab | Perbaikan |
|---|---|---|
| Status `ditolak: Unauthorized...` | token salah/kosong | samakan `S2S_TOKEN` dengan `_ws_token`, lihat log startup backend |
| `terputus (1008)` | token ditolak **atau** slot penuh | pesan `All 2 session slots are in use` = ada sesi lain aktif |
| `terputus (1006)` | tunnel mati / hostname 502 | `make tunnel` di laptop; cek `cloudflared` hidup |
| `gagal: socket gagal dibuka` | URL salah atau `ws://` dari HTTPS | pakai `wss://` dan hostname yang benar |
| Tidak ada suara tapi status `berbicara...` | autoplay diblokir | pastikan `connect()` dari klik, bukan saat load |
| `capabilities 404` | URL backend salah | `S2S_URL` harus `wss://host` tanpa `/v1/realtime` |

`S2S_URL` berisi **origin saja** (`wss://host`). `s2s.js` menambahkan
`/v1/realtime` sendiri.

## Yang perlu diingat

- **Token = kunci bersama.** Siapa pun yang memegangnya bisa bicara, tanpa
  audit per pengguna. Mencabut akses = ganti token + deploy ulang Pages.
  Login di Pages hanya menjaga halaman, bukan socket.
- **Satu-dua sesi saja.** Backend jalan dengan 2 slot; pengguna ketiga saat dua
  lain aktif akan ditolak `1008`.
- **Audio lewat Cloudflare** sebelum sampai laptop; untuk obrolan biasa wajar,
  jangan untuk hal sensitif.
