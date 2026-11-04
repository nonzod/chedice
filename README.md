# 🎙️ chedice

Applicazione web per la **trascrizione di video** con **riconoscimento dei parlanti**
(speaker diarization), accelerata su **GPU NVIDIA** e pronta per Docker.

- **Speech-to-text**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) con
  modello `large-v3` (o un altro scelto dall'interfaccia), timestamp per parola e
  rilevamento automatico della lingua.
- **Riconoscimento parlanti senza token**: embedding vocali
  [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (SpeechBrain,
  modello non-gated) + clustering per coseno. Nessun account HuggingFace richiesto.
- **Sorgenti**: file caricato (video/audio) **oppure link YouTube** (download via
  [yt-dlp](https://github.com/yt-dlp/yt-dlp)).
- **Output scaricabili**: `SRT`, `VTT`, `TXT`, `JSON`, con possibilità di **rinominare
  i parlanti** prima del download.
- **Campioni vocali**: per ogni parlante rilevato puoi **ascoltare un breve estratto**
  (pulsante ▶) e verificare che la diarization sia corretta prima di rinominarlo.
- **Scene chiave**: analizza fotogrammi campionati e ottieni descrizioni con timestamp,
  ambientazione, soggetti e azioni con un modello vision locale.
- **Categorie**: assegna una **categoria** a ogni trascrizione per raggrupparle e
  ritrovarle facilmente (con autocompletamento delle categorie già usate).
- **Cronologia e archivio**: pannello "Recenti" con stato live dei job, annullamento,
  ripresa, riapertura e cancellazione; pagina **🗂️ Archivio** con le registrazioni
  raggruppate in card per categoria e una **scheda dettagli** per ogni registrazione.
- **Correzioni**: modifica inline del testo trascritto, riassegnazione del parlante frase
  per frase, unione di due voci della stessa persona e ricalcolo dei parlanti.
- **Pulizia automatica**: retention configurabile dei file originali e dei job vecchi.
- **Assistente AI (opzionale)**: configura dalle impostazioni ⚙️ uno o più provider —
  **Ollama** o qualsiasi **API OpenAI-compatibile** (OpenAI, OpenRouter, llama.cpp, vLLM,
  LM Studio…) — in ordine di priorità, con **fallback automatico** al successivo se uno non
  risponde. Serve per domande sul testo, riassunti e descrizione delle scene.
- **UI web**: interfaccia leggera senza build step (HTML/CSS/JS vanilla), drag & drop,
  progresso in tempo reale e anteprima colorata per parlante.

---

## Requisiti

- GPU NVIDIA con driver recenti (testato su RTX 4070 Ti, 12 GB).
- Docker + Docker Compose + **NVIDIA Container Toolkit**. Il runtime llama.cpp viene
  eseguito come servizio Docker privato sulla stessa macchina.

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

Al primo avvio llama.cpp scarica il modello vision quantizzato Qwen3-VL 2B nel volume locale.

> ⚠️ **Primo avvio**: al primo job vengono scaricati i modelli (Whisper `large-v3`
> ~3 GB + ECAPA). Restano in cache nel volume `model-cache`, quindi succede una sola volta.

## Come funziona la pipeline

0. **Download** (solo job YouTube) — `yt-dlp` scarica la traccia audio migliore.
1. **Estrazione audio** — `ffmpeg` estrae audio mono a 16 kHz dal video.
2. **Trascrizione** — faster-whisper produce segmenti con timestamp (VAD attivo;
   se il VAD rimuove tutto l'audio, es. musica cantata, si ritenta senza VAD).
3. **Diarization** — finestre scorrevoli sulle regioni parlate → embedding ECAPA →
   clustering per coseno → smoothing a maggioranza delle etichette delle finestre →
   attribuzione **per parola** (grazie ai timestamp per parola di Whisper): i turni più
   brevi di `APP_MIN_TURN_SECONDS` vengono assorbiti dai vicini e i segmenti sono spezzati
   nei punti in cui cambia il parlante, anche a metà frase.
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
| `APP_WHISPER_MODELS` | `tiny,…,large-v3` | Modelli selezionabili dall'interfaccia per singolo job. |
| `APP_WINDOW_SECONDS` / `APP_HOP_SECONDS` | `2.0` / `1.0` | Finestre degli embedding: più corte = cambi di parlante più precisi ma embedding più rumorosi. |
| `APP_SMOOTHING_RADIUS` | `1` | Smoothing a maggioranza sulle finestre vicine (0 = disattivo). |
| `APP_MIN_TURN_SECONDS` | `0.8` | Durata minima di un turno dentro una frase (evita sfarfallii di una parola). |
| `APP_MEDIA_RETENTION_DAYS` | `0` | Elimina il file originale dei job conclusi dopo N giorni (0 = mai); modificabile da ⚙️. |
| `APP_JOB_RETENTION_DAYS` | `0` | Elimina interamente i job conclusi dopo N giorni (0 = mai); modificabile da ⚙️. |

Dall'interfaccia puoi anche scegliere il **modello Whisper**, forzare la **lingua** e il **numero di parlanti** per singolo job
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
  riaprirli, ℹ️ per i dettagli o ✕ per eliminarli (rimuove anche file sorgente e cache).
- **Annulla / riprendi**: un job in coda o in elaborazione si annulla con **✕ Annulla**;
  un job annullato o fallito si riprende con **↻ Riprendi**, anche con un altro modello.
- **Archivio**: la pagina 🗂️ mostra una card per categoria (numero di registrazioni, durata
  totale, ultime voci); cliccando una categoria si vedono le sue registrazioni, con ricerca
  per titolo, download e la **scheda dettagli** (data, durata, file, dimensione, lingua,
  modello, parlanti con tempo di parola, stato delle analisi).
- **Modifica testo**: ✏️ *Modifica testo* rende ogni frase modificabile e permette di
  cambiarne il parlante; un testo vuoto elimina la frase.
- **Voci riconosciute male**: nel box *🩺 Voci riconosciute male?* puoi **unire** due voci
  che appartengono alla stessa persona, oppure **ricalcolare** i parlanti con un numero
  forzato quando due persone sono finite sulla stessa voce (il testo resta invariato).
- **Pulizia automatica**: in ⚙️ imposta dopo quanti giorni eliminare i file originali
  (la trascrizione resta in archivio) e/o i job interi; la pulizia gira ogni ora o con
  *Pulisci ora*.

### Modelli AI

Le impostazioni ⚙️ tengono separati i due ruoli, ognuno con la propria **catena di provider**:

- **Assistente testuale** — domande sulla trascrizione e prompt. Non può usare il servizio
  vision `http://llama:8080`.
- **Analisi visiva** — descrizione dei fotogrammi; di default usa il runtime llama.cpp
  dedicato `http://llama:8080` con `Qwen3-VL-2B-Instruct-GGUF:Q4_K_M`.

Per ciascun ruolo puoi aggiungere più provider, ognuno di tipo:

| Tipo | Endpoint usato | Esempio di URL |
|---|---|---|
| **Ollama** | API nativa `/api/chat` (streaming NDJSON, immagini supportate) | `http://host.docker.internal:11434` |
| **API OpenAI-compatibile** | `/chat/completions` (streaming SSE) con API key opzionale (`Authorization: Bearer`) | `https://api.openai.com/v1`, `https://openrouter.ai/api/v1`, `http://llama:8080` |

L'ordine delle card (frecce ↑ ↓, badge `#1`, `#2`…) è la **priorità**. Con l'opzione
*«Se un provider non risponde, usa il successivo in ordine di priorità»* attiva, una richiesta
che fallisce (server irraggiungibile, errore HTTP, risposta vuota o — per le scene — incompleta)
passa automaticamente al provider successivo; disattivandola viene usato solo il primo provider
attivo. Ogni card ha il pulsante **🔌 Prova connessione**, che verifica raggiungibilità, API key e
presenza del modello (e suggerisce i modelli disponibili). Le API key restano nel database e non
vengono mai rimandate al browser: lasciando il campo vuoto la chiave salvata viene mantenuta.
Nella cronologia dell'AI ogni risposta riporta il provider che l'ha generata.

> ⚠️ Con un provider remoto (OpenAI, OpenRouter…) la trascrizione — o i fotogrammi, per
> l'analisi visiva — vengono inviati a quel servizio. Per restare completamente in locale usa
> solo Ollama/llama.cpp sulla tua macchina o rete.

Le configurazioni precedenti (un solo endpoint per ruolo) vengono convertite automaticamente
in un provider OpenAI-compatibile.

Il download iniziale del modello richiede una connessione internet; elaborazione e richieste
successive avvengono sul PC. Per le scene viene inviato al modello un massimo di otto
fotogrammi ridimensionati. L'app mette la richiesta in coda con le trascrizioni e rilascia prima i
modelli vocali dalla GPU.

Nel pannello **Scene chiave** puoi scegliere il livello **Base**, **Medio** o **Avanzato**:
regola la profondità della descrizione e il budget di generazione (768, 2048 o 4096 token).
Il runtime vision usa un contesto da 16K e disabilita il token-budget speso in ragionamenti
nascosti, così i token disponibili sono destinati alla descrizione visibile.
Ogni livello chiede di completare tutte le scene fino all'ultimo fotogramma.
Il runtime deve segnalare una chiusura normale: se raggiunge il limite di token o lo stream si
interrompe, il job viene marcato come incompleto, mostra un avviso e conserva l'eventuale testo
parziale per permettere di riprovare.

Una volta abilitato, sotto ogni trascrizione completata compare il box **🤖 Chiedi all'AI**:
scrivi una richiesta libera (es. «*Fammi un riassunto*» o «*Cosa afferma Marco nel video?*»)
o usa i preset. La trascrizione (coi nomi dei parlanti) viene passata come contesto e lo
scambio domanda/risposta resta salvato nel job.

La risposta è in **streaming end-to-end**: i token arrivano dal modello e compaiono
nell'interfaccia man mano che vengono generati (endpoint `text/plain` in streaming).
Questo evita anche i timeout, perché i dati continuano a fluire. Con modelli grossi o su
CPU il primo avvio "a freddo" può essere lungo: regola il timeout tra i chunk con
`APP_LLM_REQUEST_TIMEOUT` (default `300` secondi) e assicurati che il servizio `llama`
sia avviato. A fine generazione lo scambio
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
| `POST /api/jobs` | Crea un job da un file caricato (multipart: `file`, `language`, `num_speakers`, `category`, `whisper_model`). |
| `POST /api/jobs/youtube` | Crea un job da un link YouTube (JSON: `url`, `language`, `num_speakers`, `category`, `whisper_model`). |
| `GET /api/models` | Modelli Whisper selezionabili e modello predefinito. |
| `GET /api/jobs/{id}/details` | Scheda dettagli di un job (file, dimensione, durata, modello, statistiche parlanti…). |
| `POST /api/jobs/{id}/cancel` | Annulla un job in coda o in elaborazione. |
| `POST /api/jobs/{id}/retry` | Riprende un job annullato o fallito (JSON opzionale: `whisper_model`). |
| `PATCH /api/jobs/{id}/segments` | Modifica inline (JSON: `edits: [{index, text?, speaker?}]`; testo vuoto elimina la frase). |
| `POST /api/jobs/{id}/speakers/merge` | Unisce due voci (JSON: `source`, `target`). |
| `POST /api/jobs/{id}/rediarize` | Ricalcola i parlanti sul testo esistente (JSON: `num_speakers`, opzionale). |
| `GET/PUT /api/retention` | Legge/aggiorna la retention (`media_days`, `job_days`) e lo spazio occupato. |
| `POST /api/retention/cleanup` | Applica subito la pulizia. |
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
| `GET /api/config` | Configurazione dell'assistente testuale (`enabled`, `fallback`, `temperature`, `providers`). |
| `PUT /api/config` | Aggiorna l'assistente testuale; l'ordine di `providers` è la priorità. |
| `GET/PUT /api/vision-config` | Legge e aggiorna la catena di provider del modello vision. |
| `POST /api/llm/test` | Prova un provider (JSON: `role`, `provider`): raggiungibilità, API key, modello. |
| `POST /api/jobs/{id}/describe-scenes` | Analizza le scene (`detail`: `base`, `medium` o `advanced`; predefinito `medium`). |

## Struttura

```
app/
  main.py            # FastAPI: upload, YouTube, stato, rinomina, campioni, categorie, AI, download, config
  config.py          # impostazioni (env / .env)
  models.py          # Job e Segment
  jobs.py            # store persistente (SQLite) + worker in background + config app
  llm.py             # client LLM multi-provider (Ollama / OpenAI-compatibile) con fallback
  vision.py          # analisi di fotogrammi con llama.cpp locale
  utils.py           # helper condivisi (nomi file sicuri)
  pipeline/
    download.py      # download media video da URL (yt-dlp)
    audio.py         # estrazione audio e fotogrammi (ffmpeg)
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
