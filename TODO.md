# TODO — miglioramenti chedice

Elenco dei possibili miglioramenti futuri. In cima quelli più utili/richiesti.

## ✅ Fatto
- [x] **Assistente AI (LLM)** — maschera di configurazione ⚙️ per collegare un modello
      via API OpenAI-compatibile (OpenRouter / Ollama remoto), salvata in SQLite
      (`app_settings`). Box "Chiedi all'AI" sotto la trascrizione: prompt libero o
      preset (riassunto, punti chiave, to-do); la trascrizione è passata come contesto
      e lo scambio Q&A è persistito nel job. Endpoint `GET/PUT /api/config`,
      `POST /api/jobs/{id}/ask`.
- [x] Rinomina dei parlanti prima del download (mappa `SPEAKER N → nome`, applicata
      al volo su TXT/SRT/VTT/JSON).
- [x] **Trascrizione da URL YouTube** — endpoint `POST /api/jobs/youtube`, download
      audio con `yt-dlp` (`bestaudio`), titolo del video come nome dei file, campo URL
      nell'UI con stato "Scaricamento video…", gestione errori (privato/età/geo/live).
- [x] Fallback anti-VAD: se il filtro VAD rimuove tutto l'audio (es. musica cantata),
      la trascrizione viene rieseguita senza VAD invece di restituire vuoto.
- [x] Rilevamento numero parlanti più robusto: gate "singolo vs multi" basato sulla
      separazione coseno tra i due gruppi vocali (invece di una soglia fissa che
      frammentava una voce sola in più parlanti) + filtro dei silenzi + scelta di k
      via silhouette con preferenza per meno parlanti.

## ⚡ Progresso live (SSE)
Sostituire il polling ogni 1,5 s con Server-Sent Events per aggiornamenti istantanei.
- [ ] Endpoint `GET /api/jobs/{id}/events` (SSE) che emette progresso/stage.
- [ ] Il worker pubblica gli update su una coda per-job; l'UI si collega con
      `EventSource` e ricade sul polling se la connessione cade.

## 🗣️ Diarization più fine
Migliorare la precisione quando un parlante cambia a metà frase.
- [ ] Abilitare `word_timestamps=True` in faster-whisper.
- [ ] Assegnare il parlante a livello di parola e spezzare i segmenti Whisper nei
      punti di cambio parlante (invece dell'assegnazione per segmento).
- [ ] Valutare embedding su finestre più corte + smoothing per ridurre gli sfarfallii.

## 🔒 Sicurezza / deploy in rete
Necessario se l'app viene esposta oltre il localhost.
- [ ] Autenticazione (token/basic auth o reverse proxy con login).
- [ ] Limite di upload lato server già presente (`APP_MAX_UPLOAD_MB`); aggiungere
      rate limiting e validazione MIME/estensione.
- [ ] HTTPS via reverse proxy (Caddy/Traefik/Nginx).

## 🧹 Gestione dati / qualità
- [ ] Pulizia automatica dei job vecchi (retention configurabile) per non riempire il disco.
- [ ] Ripresa/annullamento di un job in coda dall'UI.
- [ ] Scelta del modello Whisper dall'interfaccia (oltre che via env).
- [ ] Editing inline del testo trascritto prima del download.
- [ ] Test automatici: unit test per `formats`/diarization e uno smoke test dell'API.
- [ ] CI (lint con `ruff` + i test) e healthcheck nel `docker-compose.yml`.
