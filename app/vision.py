"""Local llama.cpp vision analysis for sampled video frames."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from app import llm
from app.pipeline import audio


def describe_video(
    source: Path,
    duration: float | None,
    config: llm.LLMConfig,
    timeout: int,
    detail: str = "medium",
) -> str:
    """Extract representative frames and describe them with the local llama.cpp server."""
    if not config.is_ready:
        raise llm.LLMError("Abilita il modello locale e configuralo nelle impostazioni ⚙️.")
    if not llm.is_vision_base_url(config.base_url):
        raise llm.LLMError("Configura l'endpoint locale dedicato al modello vision.")

    with tempfile.TemporaryDirectory(prefix="chedice-frames-") as temp_dir:
        frames = audio.extract_video_frames(source, Path(temp_dir), duration)
        labels = "\n".join(
            f"Immagine {index + 1}: timestamp {timestamp:.1f} secondi"
            for index, (timestamp, _) in enumerate(frames)
        )
        profiles = {
            "base": (
                "Profilo BASE: sii molto conciso. Riporta solo intervalli temporali e "
                "per ogni scena una frase con ambiente, soggetti e azione principale. "
                "Massimo 2 frasi per scena."
            ),
            "medium": (
                "Profilo MEDIO: descrivi ambiente, soggetti e caratteristiche visibili, "
                "azioni e cambi di scena. Usa punti elenco brevi e accorpa fotogrammi "
                "consecutivi della stessa scena."
            ),
            "advanced": (
                "Profilo AVANZATO: fornisci una descrizione ricca ma fedele per ogni "
                "scena: ambiente e sfondo, soggetti e loro posizione, abbigliamento o "
                "tratti distintivi osservabili, oggetti, azioni, gesti e transizioni. "
                "Evidenzia anche la continuità tra scene quando utile. Non ripetere "
                "dettagli invariati inutilmente."
            ),
        }
        budgets = {"base": 512, "medium": 1536, "advanced": 3072}
        detail = detail if detail in profiles else "medium"
        prompt = (
            "Analizza tutti i fotogrammi estratti in ordine temporale da un video e "
            "completa l'analisi di TUTTE le scene fino all'ultima immagine. In italiano, "
            "usa per ogni scena un intervallo [inizio - fine] e non lasciare elenchi "
            "aperti o frasi incomplete. Non interrompere la risposta prima dell'ultima "
            "scena. Distingui ciò che è chiaramente visibile dalle ipotesi; non "
            "identificare persone reali e non inventare dettagli non osservabili.\n\n"
            + profiles[detail] + "\n\n"
            + labels
        )
        content: list[dict] = [{"type": "text", "text": prompt}]
        for _, path in frames:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            })
        return llm.chat(
            config,
            [{"role": "user", "content": content}],
            max_tokens=budgets[detail],
            read_timeout=timeout,
            require_complete=True,
        )
