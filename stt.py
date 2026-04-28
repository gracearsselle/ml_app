import io
import logging
from typing import Optional

from groq import Groq


logger = logging.getLogger(__name__)


def transcribe(
    audio_bytes: bytes,
    filename: str = "audio.webm",
    language: Optional[str] = None,
) -> str:
    client = Groq()

    file_obj = io.BytesIO(audio_bytes)

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else "webm"
    mime_type_map = {
        "webm": "audio/webm",
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "opus": "audio/opus",
    }
    mime_type = mime_type_map.get(extension, "audio/webm")

    kwargs = dict(
        file=(filename, file_obj, mime_type),
        model="whisper-large-v3-turbo",
        response_format="text",
    )
    if language:
        kwargs["language"] = language

    try:
        result = client.audio.transcriptions.create(**kwargs)
    except Exception as e:
        logger.error("Groq Whisper transcription error: %s", e)
        raise

    return result if isinstance(result, str) else result.text
