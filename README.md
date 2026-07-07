# Silent-Face-Anti-Spoofing (TFLite, webcam demo)

*[English](#english) · [Bahasa Indonesia](#bahasa-indonesia)*

**Source**: [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/master)
(MIT-licensed, see `silent_face_anti_spoofing/LICENSE`).

---

## English

Trimmed down from that upstream repo to just what's needed to run the two
MiniFASNet anti-spoofing models, already converted to TFLite, against a live
webcam feed combined with randomized blink/smile challenges.


### Simple explanation

`challenge_test.py` opens your webcam and checks two things at once to decide
if a real, live person is in front of the camera (not a photo or video
replay):

1. **Face detection** — MediaPipe finds your face and tracks things like
   eye-blink and smile signals every frame.
2. **Anti-spoof score** — two small AI models (`.tflite` files) look at the
   cropped face image and score it "real" or "fake" based on texture — a
   printed photo or screen replay looks different up close than real skin.
3. **Challenges** — you're asked to blink and smile, in a random order each
   time (so a pre-recorded video can't be scripted to match).
4. **Final verdict** — if 5 frames in a row score "fake," it stops early and
   says "FAKE." Otherwise, once you've completed both challenges, at least
   70% of all scored frames must have read "real" for it to say
   `REAL & LIVE`.

Run it with `python challenge_test.py`, press `q` to quit or `r` to retry.

### Example output

- [`mobile.mov`](mobile.mov) — recorded from the Flutter app's on-device liveness flow (the ported version of this logic)
- [`webcam.mov`](webcam.mov) — recorded from this repo's `challenge_test.py` running against a laptop webcam

### What's here

| Path | Purpose |
|---|---|
| `silent_face_anti_spoofing/challenge_test.py` | **Entry point.** Live webcam demo: MediaPipe blink+smile challenges (randomized order) combined with the TFLite anti-spoof models' real/fake score |
| `silent_face_anti_spoofing/resources/anti_spoof_models_tflite/*.tflite` | The two MiniFASNet model variants, ~1.8M each — their combined score decides real vs. fake |
| `silent_face_anti_spoofing/resources/anti_spoof_models/*.pth` | Original PyTorch weights the `.tflite` files were converted from (via `litert-torch`) — kept for provenance; nothing in this folder runs them directly anymore |
| `silent_face_anti_spoofing/src/generate_patches.py` | `CropImage` — expands the face bbox by each model's scale factor, crops, resizes to that model's input size |
| `silent_face_anti_spoofing/src/utility.py` | `parse_model_name`/`get_kernel` — derives input size + scale from a model filename like `2.7_80x80_MiniFASNetV2.tflite` |

### How it works

1. **Face + landmarks**: MediaPipe `FaceLandmarker` detects the face and its
   blendshapes (`eyeBlinkLeft/Right`, `mouthSmileLeft/Right`) every frame.
2. **Anti-spoof score**: the face bbox is cropped per-model (each model has
   its own zoom-out `scale`), resized to 80x80, and fed to both `.tflite`
   models. Their softmax outputs are summed, argmax'd, halved — same combine
   logic as upstream's `test.py`. **No `/255` normalization** — these models
   were trained on raw `[0, 255]` float pixels (upstream's own `ToTensor` has
   that division commented out); this is the one thing worth getting wrong
   only once.
3. **Challenges**: a `ChallengeRunner` shuffles `[blink, smile]` order per
   session (defeats a pre-scripted replay timed to a fixed sequence). Blink
   requires an eyes-closed frame followed by an eyes-open frame; smile
   requires both `mouthSmileLeft/Right` blendshapes above threshold.
4. **Verdict**: anti-spoof scoring runs continuously, independent of
   challenge progress. 5 consecutive "fake" frames aborts early. Once all
   challenges complete, at least 70% of scored frames must have read "real"
   for a final `REAL & LIVE` — a replay that fakes the challenge motion but
   scores fake on texture still fails.

### Running it

```bash
cd silent_face_anti_spoofing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python challenge_test.py                   # 'q' to quit, 'r' to reset/retry
python challenge_test.py --max_seconds 15  # auto-quit, for scripted smoke tests
```

---

## Bahasa Indonesia

Dipangkas dari repo upstream tersebut menjadi hanya yang diperlukan untuk
menjalankan dua model anti-spoofing MiniFASNet, yang sudah dikonversi ke
TFLite, terhadap feed webcam langsung dengan challenge kedip/senyum yang
diacak.

### Penjelasan sederhana

`challenge_test.py` membuka webcam kamu dan mengecek dua hal sekaligus untuk
memastikan yang ada di depan kamera adalah orang asli yang hidup (bukan foto
atau rekaman video):

1. **Deteksi wajah** — MediaPipe mendeteksi wajah kamu dan memantau sinyal
   seperti kedipan mata dan senyum di setiap frame.
2. **Skor anti-spoofing** — dua model AI kecil (file `.tflite`) melihat
   gambar wajah yang sudah dipotong, lalu menilai apakah teksturnya "asli"
   atau "palsu" — foto cetak atau rekaman dari layar HP terlihat berbeda dari
   kulit asli kalau dilihat dari dekat.
3. **Challenge (tantangan)** — kamu diminta berkedip dan senyum, dengan
   urutan acak setiap kali dijalankan (supaya video yang direkam sebelumnya
   tidak bisa disetel mengikuti urutan tetap).
4. **Keputusan akhir** — kalau 5 frame berturut-turut terbaca "palsu," sesi
   langsung berhenti dan hasilnya "FAKE." Kalau tidak, setelah kedua
   challenge selesai, minimal 70% dari semua frame yang dinilai harus
   terbaca "asli" agar hasilnya `REAL & LIVE`.

Jalankan dengan `python challenge_test.py`, tekan `q` untuk keluar atau `r`
untuk mencoba ulang.

### Contoh hasil

- [`mobile.mov`](mobile.mov) — rekaman dari alur liveness aplikasi Flutter di perangkat (versi hasil porting logika ini)
- [`webcam.mov`](webcam.mov) — rekaman dari `challenge_test.py` di repo ini, berjalan dengan webcam laptop

### Isi folder ini

| Path | Fungsi |
|---|---|
| `silent_face_anti_spoofing/challenge_test.py` | **Entry point.** Demo webcam langsung: challenge kedip+senyum MediaPipe (urutan acak) digabung dengan skor asli/palsu dari model anti-spoof TFLite |
| `silent_face_anti_spoofing/resources/anti_spoof_models_tflite/*.tflite` | Dua varian model MiniFASNet, ~1.8M masing-masing — skor gabungannya menentukan asli vs. palsu |
| `silent_face_anti_spoofing/resources/anti_spoof_models/*.pth` | Bobot PyTorch asli yang menjadi sumber konversi ke `.tflite` (via `litert-torch`) — disimpan untuk keperluan provenance; tidak ada kode di folder ini yang menjalankannya langsung lagi |
| `silent_face_anti_spoofing/src/generate_patches.py` | `CropImage` — memperluas bounding box wajah sesuai faktor skala tiap model, memotong (crop), lalu me-resize ke ukuran input model tersebut |
| `silent_face_anti_spoofing/src/utility.py` | `parse_model_name`/`get_kernel` — menentukan ukuran input + skala dari nama file model seperti `2.7_80x80_MiniFASNetV2.tflite` |

### Cara kerjanya

1. **Wajah + landmark**: MediaPipe `FaceLandmarker` mendeteksi wajah dan
   blendshape-nya (`eyeBlinkLeft/Right`, `mouthSmileLeft/Right`) di setiap
   frame.
2. **Skor anti-spoof**: bounding box wajah dipotong per-model (setiap model
   punya faktor zoom-out `scale` sendiri), di-resize ke 80x80, lalu diproses
   oleh kedua model `.tflite`. Output softmax keduanya dijumlahkan, diambil
   argmax-nya, dibagi dua — logika penggabungan yang sama seperti `test.py`
   milik upstream. **Tidak ada normalisasi `/255`** — model-model ini
   dilatih dengan piksel float mentah `[0, 255]` (`ToTensor` milik upstream
   sendiri sengaja mengomentari baris pembagian itu); ini adalah satu hal
   yang cukup sekali saja salah, jangan diulang.
3. **Challenge**: `ChallengeRunner` mengacak urutan `[blink, smile]` setiap
   sesi (supaya rekaman video yang sudah disiapkan sebelumnya tidak bisa
   ditata ulang mengikuti urutan tetap). Blink memerlukan frame mata-tertutup
   diikuti frame mata-terbuka; smile memerlukan kedua blendshape
   `mouthSmileLeft/Right` di atas ambang batas.
4. **Keputusan**: penilaian anti-spoof berjalan terus-menerus, terlepas dari
   progres challenge. 5 frame "palsu" berturut-turut langsung menghentikan
   sesi. Setelah semua challenge selesai, minimal 70% dari frame yang dinilai
   harus terbaca "asli" untuk hasil akhir `REAL & LIVE` — rekaman yang bisa
   memalsukan gerakan challenge tapi tetap terbaca palsu dari teksturnya
   akan tetap gagal.

### Menjalankannya

```bash
cd silent_face_anti_spoofing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python challenge_test.py                   # 'q' untuk keluar, 'r' untuk reset/coba lagi
python challenge_test.py --max_seconds 15  # auto-berhenti, untuk smoke test terskrip
```
