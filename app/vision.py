"""Vision analysis of sampled video frames (provider chain with fallback)."""

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
    """Extract representative frames and describe them with the vision provider chain."""
    config.providers = [p for p in config.providers if llm.is_vision_base_url(p.base_url)]
    if not config.is_ready:
        raise llm.LLMError("Abilita il modello vision e configuralo nelle impostazioni ⚙️.")

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
            "Massimo una frase per scena."
            ),
            "medium": (
            "Profilo MEDIO: descrivi ambiente, soggetti e caratteristiche visibili, "
            "azioni e cambi di scena. Usa al massimo 2 frasi brevi per scena e accorpa "
            "fotogrammi consecutivi della stessa scena."
            ),
            "advanced": (
                "Profilo AVANZATO: fornisci una descrizione ricca ma fedele per ogni "
                "scena: ambiente e sfondo, soggetti e loro posizione, abbigliamento o "
                "tratti distintivi osservabili, oggetti, azioni, gesti e transizioni. "
                "Evidenzia anche la continuità tra scene quando utile. Non ripetere "
                "dettagli invariati inutilmente. Massimo 4 frasi per scena."
            ),
        }
        # Eight sampled frames need far fewer tokens than the previous loose
        # prompt encouraged; the server context also reserves room for images.
        budgets = {"base": 768, "medium": 2048, "advanced": 4096}
        detail = detail if detail in profiles else "medium"
        prompt = (
            "Analizza i fotogrammi forniti in ordine temporale e descrivi tutte le scene "
            "fino all'ultima immagine. In italiano, crea una voce numerata per ogni scena, "
            "con timestamp approssimativo e testo conciso. Unisci immagini consecutive "
            "solo se mostrano chiaramente la stessa scena. Non ripetere formule come "
            "'l'immagine è chiaramente visibile', non aggiungere introduzioni o conclusioni "
            "e non ripetere dettagli invariati. Usa solo elementi osservabili, senza "
            "identificare persone reali o inventare dettagli. Chiudi l'ultima voce con una "
            "frase completa.\n\n"
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
