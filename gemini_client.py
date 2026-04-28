import io
import logging
import wave
from typing import Optional

from google import genai
from google.genai import types


logger = logging.getLogger(__name__)

MODEL = "gemini-3.1-flash-live-preview"

SYSTEM_INSTRUCTION = (
    "You are a helpful, friendly, and concise AI assistant. "
    "Respond in the same language the user writes in. "
    "If the user writes in French, respond in French. "
    "If the user writes in English, respond in English."
)


class GeminiLiveClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(
            api_key=api_key, http_options={"api_version": "v1alpha"}
        )

    async def chat_text(self, prompt: str) -> tuple[bytes, Optional[str]]:
        logger.info("Opening Live session for prompt: %r", prompt[:80])
        audio_chunks: list[bytes] = []
        transcription_parts: list[str] = []

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=types.Content(
                parts=[types.Part(text=SYSTEM_INSTRUCTION)]
            ),
        )

        try:
            async with self.client.aio.live.connect(
                model=MODEL, config=config
            ) as session:
                logger.info("Session opened, sending prompt via realtime_input...")

                await session.send_realtime_input(text=prompt)
                logger.info("Text sent, signaling end of input...")

                await session.send_realtime_input(audio_stream_end=True)
                logger.info("End-of-input sent, waiting for response...")

                msg_count = 0
                async for response in session.receive():
                    msg_count += 1

                    if response.setup_complete is not None:
                        logger.debug("[%d] Setup complete", msg_count)
                        continue

                    if response.server_content is not None:
                        sc = response.server_content

                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if part.text:
                                    logger.debug(
                                        "[%d] Text part: %r",
                                        msg_count,
                                        part.text[:100],
                                    )
                                if part.inline_data is not None:
                                    audio_chunks.append(part.inline_data.data)
                                    logger.debug(
                                        "[%d] Audio chunk: %d bytes, mime=%s",
                                        msg_count,
                                        len(part.inline_data.data),
                                        part.inline_data.mime_type,
                                    )

                        if sc.output_transcription and sc.output_transcription.text:
                            transcription_parts.append(sc.output_transcription.text)
                            logger.debug(
                                "[%d] Transcription chunk: %r",
                                msg_count,
                                sc.output_transcription.text[:100],
                            )

                        if sc.generation_complete:
                            logger.info("[%d] Generation complete", msg_count)

                        if sc.turn_complete:
                            logger.info(
                                "[%d] Turn complete (reason: %s)",
                                msg_count,
                                sc.turn_complete_reason,
                            )
                            break

                    if response.go_away is not None:
                        logger.warning(
                            "[%d] GoAway received, time_left=%s",
                            msg_count,
                            response.go_away.time_left,
                        )
                        break

                logger.info(
                    "Session ended after %d messages, %d audio chunks",
                    msg_count,
                    len(audio_chunks),
                )

        except Exception as e:
            logger.error("Gemini Live API error: %s", e, exc_info=True)
            raise

        if not audio_chunks:
            logger.warning("No audio chunks received")
            return b"", "".join(transcription_parts) or None

        transcription = "".join(transcription_parts)
        wav_bytes = self._pcm_to_wav(b"".join(audio_chunks))
        logger.info(
            "WAV output: %d bytes, transcription: %r", len(wav_bytes), transcription
        )
        return wav_bytes, transcription or None

    @staticmethod
    def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        return buffer.getvalue()
