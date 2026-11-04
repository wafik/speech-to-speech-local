# Desain: sesi suara online dengan ganti suara per sesi

Tanggal: 2026-11-04
Status: menunggu persetujuan

## Kenapa

Pipeline sudah jalan lokal di laptop ini (MX150 2 GB, 9router untuk LLM,
Supertonic q4 di CPU). Yang belum bisa:

1. **Suara tidak bisa diganti per sesi.** `session.update` untuk
   `audio.output.voice` diterima dan dibalas `session.updated` sukses, tapi
   Supertonic mengabaikannya. Handler lain di upstream (Kokoro, Qwen3-TTS,
   Kitten) sudah membaca nilai ini per giliran; Supertonic yang ketinggalan.
2. **Belum bisa diakses dari luar** dengan aman.

Pemakainya 1-2 orang yang **tidak pernah bersamaan**, jadi satu slot pipeline
cukup. Kliennya berupa **web app di browser** milik pengguna sendiri, yang
menyambung ke server ini lewat URL/domain.

## Tujuan

- Web app di browser bisa bicara dengan pipeline lewat `wss://` di domain sendiri.
- Tiap sesi memilih suaranya sendiri (wanita F1-F5, pria M1-M5) tanpa restart.
- Hanya ada satu sesi pada satu waktu; orang kedua mendapat pesan yang jelas.
- Akses dijaga Cloudflare Access: hanya email yang didaftarkan bisa masuk.

## Non-tujuan

- Multi-sesi bersamaan. Menambah slot di laptop ini berarti proses pipeline dan
  instance TTS tambahan, dan CPU-nya sudah jadi batas.
- Proxy teks `/v1/chat/completions` untuk publik. Tanpa `--enable_llm_proxy`
  endpoint ini menjawab `501 not_implemented`; itu perilaku yang diinginkan,
  supaya kredit 9router tidak bisa dipakai orang lain untuk teks bebas.
- Aplikasi klien. Yang dibangun di sini sisi server; klien milik pengguna.
- Suara baru di luar 10 style yang ada.

## Batasan

- Windows, PowerShell 5.1, ASCII-only untuk file `.ps1`.
- Supertonic memuat graph saat setup; mengganti voice hanya membaca JSON style
  (`get_voice_style()`), jadi murah dan boleh dilakukan per giliran.
- Browser hanya mengizinkan mikrofon di halaman HTTPS, dan halaman HTTPS
  dilarang menyambung ke `ws://`. Karena itu di balik tunnel harus `wss://`.
- Cloudflare Access tidak bisa dipasang pada quick tunnel `*.trycloudflare.com`.
  Butuh named tunnel + domain di akun Cloudflare. Ini langkah manual pengguna.
- Audio pipeline 16 kHz. Klien harus mengirim PCM 16 kHz mono.

## Bagian yang sudah diputuskan pengguna

| Keputusan | Nilai |
|---|---|
| Klien | Web app di browser, boleh di origin lain (Cloudflare Pages) |
| Auth | Token bersama di backend; login/gate milik frontend masing-masing |
| Pemakai | Siapa pun yang bisa membuka frontend; 2 slot di backend |
| Suara | Supertonic, bisa dipilih per sesi |
| Domain | dipilih pengguna; hostname aslinya sengaja tidak ditulis di repo |

## Rancangan

### 1. Supertonic membaca voice per sesi

`speech-to-speech/src/speech_to_speech/TTS/supertonic_tts_handler.py`

Mengikuti pola `kokoro_handler.py`:

- Tambah cache `self._style_cache: dict[str, Style]` berisi voice awal.
- Di `process()`, ambil voice dari `tts_input.response.audio.output.voice`
  (prioritas pertama), lalu `tts_input.runtime_config.session.audio.output.voice`.
  Kalau kosong, pakai `self.voice` seperti sekarang.
- Nama dinormalisasi uppercase. Kalau tidak ada di `AVAILABLE_VOICES`
  (`F1..F5`, `M1..M5`), catat warning dan pakai voice sebelumnya - **jangan**
  melempar exception, karena kesalahan nama tidak boleh mematikan satu giliran.
- Style diambil dari cache atau `get_voice_style()`, lalu dipakai pada
  `synthesize()`.

Karena kode upstream berada di `speech-to-speech/` yang di-`.gitignore`, ini
**patch lokal**, bukan commit ke upstream. Perlu dicatat di README repo kita
dan dicek ulang kalau upstream di-`git pull`.

### 2. UI lokal: pilih suara

`ui/index.html`

- Deteksi skema: halaman `https:` -> `wss://${location.host}/v1/realtime`;
  selain itu tetap `ws://${location.hostname}:8765/v1/realtime`.
- Tambah dropdown 10 voice. Perubahan mengirim
  `{"type":"session.update","session":{"audio":{"output":{"voice":"M2"}}}}`.
- Ganti voice **ditunda** kalau ada respons berjalan, dan dikirim setelah
  `response.done` - pola yang sama dengan `voice_controls.py:121`.

### 3. Jalankan server untuk publik

Tanpa perubahan kode, `run.ps1` cukup dipanggil:

```powershell
run.ps1 -Mode serve -Model <9router model> -ApiBase http://127.0.0.1:20128/v1 -LowLatency
```

tanpa `--enable_llm_proxy`. (`run.ps1` saat ini **selalu** menambahkan flag itu
di mode serve; itu harus diubah menjadi opt-in, misalnya `-EnableLlmProxy`.)

### 4. Named tunnel

```powershell
cloudflared tunnel login
cloudflared tunnel create speech
cloudflared tunnel route dns speech voice.<domain>
cloudflared tunnel run --url http://localhost:8765 speech
```

### 4b. Auth: gate di frontend, token di backend

Keputusan yang berubah setelah versi pertama: pengguna ingin **frontend
terpisah** (mis. Cloudflare Pages) dan **bebas siapa saja yang bisa membuka
frontend itu**. Itu membatalkan Cloudflare Access di hostname backend:

- Access memasang cookie first-party untuk hostname backend. Dari origin lain
  cookie itu third-party dan diblokir browser modern.
- Browser juga tidak bisa mengirim header auth pada WebSocket.

Jadi gate dipindah ke tempat pengguna login (frontend masing-masing), dan
backend memeriksa **token bersama** yang tidak bergantung pada origin:

| Pemanggil | Cara mengirim token |
|---|---|
| Browser | subprotocol: `new WebSocket(url, ["realtime", "token.<T>"])` |
| curl / Python / Go | `Authorization: Bearer <T>` atau `?token=<T>` |

- Token dibuat `run.ps1` sekali dan disimpan di `_ws_token` (gitignored),
  atau dipatok lewat `-WsToken` / `$env:SPEECH_WS_TOKEN`.
- Token diperiksa **sebelum** slot diklaim. Token salah -> `error` event lalu
  `1008`, tanpa memakan slot; socket **di-accept dulu** supaya browser bisa
  menampilkan pesan, bukan sekadar 403.
- `/v1/realtime/capabilities` tetap terbuka (daftar bahasa/suara, bukan data
  sesi). `/v1/pool`, `/v1/usage`, dan socket dijaga.
- Tanpa token terkonfigurasi, server tidak memeriksa apa pun - benar untuk
  `127.0.0.1`, salah untuk di belakang tunnel.
- Ini **shared secret, bukan identitas per orang**: siapa pun yang memegangnya
  masuk. Mencabut akses = ganti token + deploy ulang frontend.

Konsekuensi yang harus disadari: tidak ada audit per pengguna, dan frontend
yang membocorkan token membocorkan akses. Untuk 1-2 orang yang tidak pernah
bersamaan itu proporsional; kalau nanti perlu per-orang, ganti ke Access di
hostname backend dan selesaikan masalah cookie-nya (frontend satu origin, atau
proxy Pages Function).

### 5. Batas yang perlu disebut ke pengguna

- Cloudflare memutus TLS di sisinya, jadi isi percakapan lewat jalur publik
  berakhir di Cloudflare sebelum masuk laptop. Untuk obrolan biasa wajar.
- Server dijalankan dengan dua slot (`-Pipelines 2`), bukan satu: slot bekas
  sesi yang baru ditutup masih tertahan ~1,4 detik, dan handler yang macet
  dikarantina sampai 180 detik. Dengan satu slot, dua keadaan itu mengunci
  satu-satunya pengguna keluar. Dua orang yang benar-benar bersamaan tetap
  ditolak `1008` "All 2 session slots are in use" - itu disengaja.
- Voice dibekukan per respons: `session.update` yang datang di tengah jawaban
  berlaku untuk jawaban berikutnya, supaya satu jawaban tidak berpindah suara
  di antara kalimat.

## Yang diserahkan ke pembangun

- Nama voice dikirim huruf besar/kecil (`f1` vs `F1`) - normalisasi di server,
  bukan di klien.
- Apakah dropdown perlu menampilkan contoh ("F1 - wanita"), itu urusan UI.
- Format pesan error ke pengguna.
- Struktur file kalau patch Supertonic perlu dites terpisah.

## Verifikasi

- Unit/integrasi: kirim `session.update` voice `M1` lalu `F3` dalam satu sesi,
  pastikan `synthesize()` dipanggil dengan style yang berbeda (bisa dengan
  monkeypatch atau log).
- End-to-end lokal: `bench/ws_smoke.py` dengan tambahan event `session.update`
  sebelum kirim audio, cek suara berubah.
- Dari luar: buka `https://voice.<domain>` di browser, login, bicara, ganti
  voice, dan cek `nvidia-smi` serta log pipeline bahwa request datang lewat
  tunnel.
