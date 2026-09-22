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
        prompt = (
            "Analizza questi fotogrammi estratti in ordine temporale da un video. "
            "Descrivi in italiano, con tono concreto, le scene chiave: timestamp, "
            "ambientazione, soggetti visibili, abbigliamento o caratteristiche "
            "distintive, azioni e cambi di scena. Distingui ciò che è chiaramente "
            "visibile dalle ipotesi; non identificare persone reali e non inventare "
            "dettagli non osservabili. Raggruppa fotogrammi consecutivi quando la "
            "scena non cambia e usa elenchi leggibili.\n\n" + labels
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
            read_timeout=timeout,
        )
