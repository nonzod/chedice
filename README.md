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
- **Campioni vocali**: per ogni parlante rilevato puoi **ascoltare un breve estratto**
  (pulsante ▶) e verificare che la diarization sia corretta prima di rinominarlo.
- **Categorie**: assegna una **categoria** a ogni trascrizione per raggrupparle e
  ritrovarle facilmente (con autocompletamento delle categorie già usate).
- **Cronologia e archivio**: pannello "Recenti" con stato live dei job, riapertura e
  cancellazione; le trascrizioni completate restano archiviate nel database.
- **Assistente AI (opzionale)**: collega un modello LLM via API **OpenAI-compatibile**
  (OpenRouter, Ollama remoto, ecc.) dalle impostazioni ⚙️ e fai domande sul testo o
  chiedi un riassunto direttamente dalla schermata della trascrizione.
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

## Uso dell'interfaccia

- **Carica** un video/audio (drag & drop o selezione) oppure incolla un **link YouTube**.
  Prima di avviare puoi impostare lingua, numero di parlanti e categoria.
- **Progresso live**: la barra mostra lo stage corrente (download → estrazione →
  trascrizione → diarization) aggiornato in tempo reale.
- **Verifica delle voci**: a trascrizione completata, accanto a ogni parlante c'è un
  pulsante ▶ per ascoltare un breve campione audio di quella voce (estratto al volo con
  `ffmpeg` e messo in cache). Utile per capire "chi è chi" prima di rinominare.
- **Rinomina parlanti**: assegna un nome a `SPEAKER 1`, `SPEAKER 2`, … I nomi vengono
  applicati al volo alla vista e a tutti i download (TXT/SRT/VTT/JSON).
- **Categorie**: assegna una categoria alla trascrizione (in fase di creazione o dopo,
  dal box 🏷️). Il campo suggerisce le categorie già in uso.
- **Cronologia**: il pannello "Recenti" elenca i job con il loro stato; clicca per
  riaprirli o usa ✕ per eliminarli (rimuove anche file sorgente e cache correlati).

### Assistente AI (LLM)

La connessione al modello si configura **dall'interfaccia** (icona ⚙️ in alto a destra),
non da env, e viene salvata nel database (tabella `app_settings`):

- **Base URL** — endpoint OpenAI-compatibile, es. `https://openrouter.ai/api/v1` oppure
  `http://mio-ollama:11434/v1`.
- **API key** — token del provider (lascia vuoto per un Ollama locale senza auth).
- **Modello** — es. `openai/gpt-4o-mini` (OpenRouter) o `llama3.1` (Ollama).
- **Temperatura** — creatività della risposta (default `0.3`).

Una volta abilitato, sotto ogni trascrizione completata compare il box **🤖 Chiedi all'AI**:
scrivi una richiesta libera (es. «*Fammi un riassunto*» o «*Cosa afferma Marco nel video?*»)
o usa i preset. La trascrizione (coi nomi dei parlanti) viene passata come contesto e lo
scambio domanda/risposta resta salvato nel job.

La risposta è in **streaming end-to-end**: i token arrivano dal modello e compaiono
nell'interfaccia man mano che vengono generati (endpoint `text/plain` in streaming).
Questo evita anche i timeout, perché i dati continuano a fluire. Con modelli grossi o su
CPU il primo avvio "a freddo" può essere lungo: regola il timeout tra i chunk con
`APP_LLM_REQUEST_TIMEOUT` (default `300` secondi) e assicurati che Ollama sia
raggiungibile in rete (`OLLAMA_HOST=0.0.0.0:11434`). A fine generazione lo scambio
domanda/risposta viene salvato nel job.

## Sviluppo locale (senza Docker)

Serve Python 3.10+, `ffmpeg`, e PyTorch con CUDA installati manualmente.

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API HTTP

| Metodo & path | Descrizione |
|---|---|
| `POST /api/jobs` | Crea un job da un file caricato (multipart: `file`, `language`, `num_speakers`, `category`). |
| `POST /api/jobs/youtube` | Crea un job da un link YouTube (JSON: `url`, `language`, `num_speakers`, `category`). |
| `GET /api/jobs` | Elenca i job (senza segmenti) per la cronologia. |
| `GET /api/jobs/{id}` | Dettaglio di un job, inclusi trascrizione e Q&A AI. |
| `DELETE /api/jobs/{id}` | Elimina un job e i file/cache correlati. |
| `GET /api/jobs/{id}/download/{fmt}` | Scarica la trascrizione (`srt`/`vtt`/`txt`/`json`). |
| `PATCH /api/jobs/{id}/speakers` | Imposta i nomi personalizzati dei parlanti. |
| `GET /api/jobs/{id}/sample/{speaker}` | Campione audio (`mp3`) di un parlante rilevato. |
| `PATCH /api/jobs/{id}/category` | Imposta o rimuove la categoria di un job. |
| `POST /api/jobs/{id}/ask` | Domanda/richiesta AI sul testo (JSON: `prompt`); risposta in streaming `text/plain`, salvata nel job a fine generazione. |
| `GET /api/categories` | Categorie distinte già in uso (per l'autocompletamento). |
| `GET /api/archive` | Trascrizioni completate (solo metadati) con i link di download. |
| `GET /api/config` | Configurazione LLM corrente. |
| `PUT /api/config` | Aggiorna la configurazione LLM (`enabled`, `base_url`, `api_key`, `model`, `temperature`). |

## Struttura

```
app/
  main.py            # FastAPI: upload, YouTube, stato, rinomina, campioni, categorie, AI, download, config
  config.py          # impostazioni (env / .env)
  models.py          # Job e Segment
  jobs.py            # store persistente (SQLite) + worker in background + config app
  llm.py             # client LLM OpenAI-compatibile (riassunti / Q&A sul testo)
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

- I dati vivono in `./data/` (montato nel container): gli upload in `./data/uploads/`,
  mentre trascrizioni e metadati (nomefile/URL, segmenti, nomi parlanti, categoria e
  cronologia Q&A dell'AI) sono salvati in un database SQLite `./data/chedice.db`. La
  configurazione LLM risiede nella tabella `app_settings` dello stesso database. I formati
  SRT/VTT/TXT/JSON sono generati al volo al download. Eventuali vecchi job in
  `./data/jobs/*.json` vengono migrati nel DB al primo avvio; le colonne aggiunte in
  seguito (`category`, `ai_messages`) sono migrate automaticamente all'avvio.
- `GET /api/archive` elenca le trascrizioni completate leggendo direttamente dal DB (solo
  metadati: nomefile/URL, data, lingua, durata, parlanti) con i link di download per ogni formato.
- La diarization è **best-effort**: se fallisce, la trascrizione viene comunque salvata.
