# 🎙️ chedice

Applicazione web per la **trascrizione di video** con **riconoscimento dei parlanti**
(speaker diarization), accelerata su **GPU NVIDIA** e pronta per Docker.

- **Speech-to-text**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) con
  modello `large-v3` e rilevamento automatico della lingua.
- **Riconoscimento parlanti senza token**: embedding vocali
  [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (SpeechBrain,
  modello non-gated) + clustering per coseno. Nessun account HuggingFace richiesto.
- **Sorgenti**: file caricato (video/audio) **oppure link YouTube** (download via
  [yt-dlp](https://github.com/yt-dlp/yt-dlp)).
- **Output scaricabili**: `SRT`, `VTT`, `TXT`, `JSON`, con possibilità di **rinominare
  i parlanti** prima del download.
- **UI web**: interfaccia leggera senza build step (HTML/CSS/JS vanilla), drag & drop,
  progresso in tempo reale e anteprima colorata per parlante.

---

## Requisiti

- GPU NVIDIA con driver recenti (testato su RTX 4070 Ti, 12 GB).
- Docker + Docker Compose + **NVIDIA Container Toolkit**.

Verifica rapida:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Avvio

```bash
# (opzionale) personalizza la configurazione
cp .env.example .env

# build + run
docker compose up --build
```

Apri **http://localhost:8000**.

> ⚠️ **Primo avvio**: al primo job vengono scaricati i modelli (Whisper `large-v3`
> ~3 GB + ECAPA). Restano in cache nel volume `model-cache`, quindi succede una sola volta.

## Come funziona la pipeline

0. **Download** (solo job YouTube) — `yt-dlp` scarica la traccia audio migliore.
1. **Estrazione audio** — `ffmpeg` estrae audio mono a 16 kHz dal video.
2. **Trascrizione** — faster-whisper produce segmenti con timestamp (VAD attivo;
   se il VAD rimuove tutto l'audio, es. musica cantata, si ritenta senza VAD).
3. **Diarization** — finestre scorrevoli sulle regioni parlate → embedding ECAPA →
   clustering per coseno → ogni segmento è attribuito al parlante prevalente.
4. **Esportazione** — TXT/SRT/VTT/JSON generati al volo al download, applicando gli
   eventuali nomi personalizzati dei parlanti.

Un singolo worker in background elabora un job alla volta, così l'uso della VRAM
resta limitato anche con più upload in coda.

## Configurazione

Tutte le variabili (prefisso `APP_`) sono documentate in [`.env.example`](.env.example).
Le più utili:

| Variabile | Default | Descrizione |
|---|---|---|
| `APP_WHISPER_MODEL` | `large-v3` | Modello Whisper (`tiny`…`large-v3`). |
| `APP_DEVICE` | `cuda` | `cuda` o `cpu`. |
| `APP_COMPUTE_TYPE` | `float16` | `float16` / `int8_float16` / `int8`. |
| `APP_LANGUAGE` | *(auto)* | Forza una lingua (es. `it`). |
| `APP_SPEAKER_SEPARATION` | `0.65` | Separazione coseno minima per accettare >1 parlante (auto). Alza il valore se una voce sola viene divisa in più parlanti. |
| `APP_MAX_SPEAKERS` | `10` | Limite superiore di parlanti in modalità automatica. |

Dall'interfaccia puoi anche forzare la **lingua** e il **numero di parlanti** per singolo job
(consigliato se conosci quanti sono: migliora la precisione della diarization).

## Sviluppo locale (senza Docker)

Serve Python 3.10+, `ffmpeg`, e PyTorch con CUDA installati manualmente.

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Struttura

```
app/
  main.py            # FastAPI: upload, YouTube, stato, rinomina, download
  config.py          # impostazioni (env / .env)
  models.py          # Job e Segment
  jobs.py            # store persistente + worker in background
  utils.py           # helper condivisi (nomi file sicuri)
  pipeline/
    download.py      # download audio da URL (yt-dlp)
    audio.py         # estrazione audio (ffmpeg)
    transcribe.py    # faster-whisper (con fallback anti-VAD)
    diarize.py       # embedding ECAPA + clustering
    formats.py       # export SRT/VTT/TXT/JSON (nomi parlanti applicati al volo)
    pipeline.py      # orchestrazione end-to-end
  static/            # UI (index.html, style.css, app.js)
```

## Note

- I dati (upload, trascrizioni, metadati) vivono in `./data/` (montato nel container).
- La diarization è **best-effort**: se fallisce, la trascrizione viene comunque salvata.
