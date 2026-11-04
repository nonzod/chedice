# TODO — miglioramenti chedice

Elenco dei possibili miglioramenti futuri. In cima quelli più utili/richiesti.

## UI 
- [x] Migliorare la visualizzazione delle registrazioni in DB
	  Visualizzare le registrazioni in una pagina "archivio" ad hoc, visualizzarle come card "Categorie" cliccando su categoria si vedono tutte le registrazioni di quella categoria
- [x] Al click su un video vedere i dettagli scritti a db, come data, lunghezza, file etc.. in una scheda ordinata

## 🗣️ Diarization più fine
Migliorare la precisione quando un parlante cambia a metà frase.
- [x] Abilitare `word_timestamps=True` in faster-whisper.
- [x] Assegnare il parlante a livello di parola e spezzare i segmenti Whisper nei
      punti di cambio parlante (invece dell'assegnazione per segmento).
- [x] Valutare embedding su finestre più corte + smoothing per ridurre gli sfarfallii.

## 🧹 Gestione dati / qualità
- [x] Pulizia automatica dei job vecchi (retention configurabile) per non riempire il disco.
- [x] Ripresa/annullamento di un job in coda dall'UI.
- [x] Scelta del modello Whisper dall'interfaccia (oltre che via env).
- [x] Editing inline del testo trascritto prima del download.
- [x] Dare la possibilità di indicare le voci riconosciute male, es due persone diverse per la stessa voce o stessa persona per due voci etc..
