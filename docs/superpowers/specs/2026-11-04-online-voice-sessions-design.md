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
| Klien | Web app di browser |
| Auth | Cloudflare Access (login Google/OTP per email) |
| Pemakai | 1-2 orang, tidak bersamaan, 1 slot cukup |
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

Lalu di dashboard Cloudflare Zero Trust: buat Access application untuk
`voice.<domain>`, policy `Allow` dengan daftar email pengguna.

### 5. Batas yang perlu disebut ke pengguna

- Cloudflare memutus TLS di sisinya, jadi isi percakapan lewat jalur publik
  berakhir di Cloudflare sebelum masuk laptop. Untuk obrolan biasa wajar.
- Sesi kedua sementara sesi pertama aktif mendapat `1008` dengan pesan
  "All 1 session slots are in use". Web app sebaiknya menampilkan itu apa
  adanya, bukan mencoba reconnect terus-menerus.

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
