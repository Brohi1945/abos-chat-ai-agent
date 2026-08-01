"""
edge_tts_plugin.py — a small custom LiveKit Agents TTS plugin around
the free `edge-tts` library (unofficial wrapper around Microsoft
Edge's "Read Aloud" voices — same Asad/Uzma Urdu neural voices Azure
Speech uses, but zero signup/key needed).

Why this file exists instead of an official plugin: LiveKit does not
ship an edge-tts plugin (it's community/extensible by design — see
docs.livekit.io/agents/integrations/plugins/). This class is written
directly against LiveKit's documented TTS plugin interface
(`tts.TTS` / `tts.ChunkedStream` / `tts.AudioEmitter`), the same
interface the official ElevenLabs/Azure/OpenAI plugins implement —
verified against the *current* source of livekit-plugins-elevenlabs
(github.com/livekit/agents) on 2026-08-01, not guessed from memory.
It only implements the simple non-streaming "chunked" path (send full
text, get back one audio blob), which is all edge-tts supports anyway
— no partial-sentence streaming, unlike ElevenLabs' websocket plugin.

⚠️ Same caveat as agent.py: if `tts.TTS`/`tts.ChunkedStream`'s exact
method signatures have moved by the time you deploy, diff this file
against a current official plugin (e.g. livekit-plugins-openai's
tts.py, which uses the same simple chunked pattern) and adjust.

⚠️ Reliability note (why this is meant to be temporary): edge-tts is
unofficial — it works by reverse-engineering Microsoft Edge's
"Read Aloud" feature, not a documented/supported API. It can break
without notice if Microsoft changes that endpoint. Fine for building
and testing now; switch to `azure.TTS` (official, same Asad/Uzma
voices, small setup) once the Azure account verification issue is
sorted — see the commented-out block in agent.py.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import edge_tts
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
)
from livekit.agents import tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

# edge-tts's default output format for every voice, including
# ur-PK-AsadNeural — 24kHz mono mp3. If you ever pass a custom
# `output_format` to edge_tts.Communicate, update this to match.
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class _EdgeTTSOptions:
    voice: str
    rate: str = "+0%"
    volume: str = "+0%"


class EdgeTTS(tts.TTS):
    def __init__(self, *, voice: str = "ur-PK-AsadNeural", rate: str = "+0%", volume: str = "+0%") -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._opts = _EdgeTTSOptions(voice=voice, rate=rate, volume=volume)

    @property
    def provider(self) -> str:
        return "edge-tts"

    def update_options(self, *, voice: str | None = None) -> None:
        if voice:
            self._opts.voice = voice

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "ChunkedStream":
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        return None


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: EdgeTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: EdgeTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        opts = self._tts._opts
        try:
            communicate = edge_tts.Communicate(
                self._input_text, voice=opts.voice, rate=opts.rate, volume=opts.volume
            )

            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                mime_type="audio/mp3",
            )

            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and chunk.get("data"):
                    output_emitter.push(chunk["data"])

            output_emitter.flush()
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            raise APIConnectionError() from e
