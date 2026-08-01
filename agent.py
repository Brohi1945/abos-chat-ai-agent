"""
abos-chat-ai-agent — Phase 4.6a foundation skeleton.

This is a SEPARATE small Python service, deployed to LiveKit Cloud
(NOT to Vercel — Vercel functions are stateless/short-lived and can't
hold a live WebRTC/agent connection for the duration of a phone call).
It's the thing that actually joins the LiveKit room, listens to the
customer over the mic, and talks back.

⚠️ IMPORTANT — read before deploying:
LiveKit's Agents SDK moves fast (new class-based `AgentSession`/`Agent`
API replaced the older `VoicePipelineAgent` pattern during 2025-2026).
The plugin imports and pipeline wiring below are written to the
*current documented pattern* as of this write-up (2026-08-01,
docs.livekit.io/agents + docs.livekit.io/agents/integrations/groq),
but before you deploy for real:
  1. Run the official LiveKit quickstart once (`lk app create` or
     https://docs.livekit.io/agents/start/voice-ai/) to scaffold a
     fresh agent against whatever SDK version is current *the day you
     deploy* — LiveKit's own scaffolding is always correct for the
     installed version, safer than trusting any one snapshot of code.
  2. Diff that scaffold's imports/class names against this file. If
     `AgentSession`/`Agent` have been renamed again, swap them here —
     the ABOS-specific logic below (metadata parsing, Urdu/English
     voice selection, system prompt, greeting) stays the same either way.

--- TTS PROVIDER: currently edge-tts (2026-08-01) ---
Using the free `edge-tts` library (see edge_tts_plugin.py) with the
Asad male voice (ur-PK-AsadNeural), per instruction — Azure Speech
account verification is stuck for now. edge-tts needs zero signup/key
and uses the exact same Microsoft neural voices Azure does, so the
voice itself sounds identical; the only difference is edge-tts is an
unofficial/reverse-engineered wrapper (see edge_tts_plugin.py's
docstring for the reliability caveat).

Switching back to Azure later is a 2-line change — the commented-out
block below shows exactly what to swap. No other file needs to change.

What this skeleton does (Phase 4.6a — foundation only):
  - Joins the LiveKit room the Vercel app dispatched it into
  - Reads call metadata (customer name, preferred language, greeting
    text) — all set by /api/ai-call-connect.js on the Vercel side
  - Picks Urdu or English STT/TTS based on the customer's
    preferred_language (falls back to Urdu — most ABOS customers are
    Urdu-first)
  - Speaks the configured greeting, then holds a plain conversational
    loop (Groq Whisper → Groq Llama → edge-tts voice)

What it does NOT do yet (Phase 4.6b, next):
  - No order-taking tools (add_to_order / confirm_order / etc.) — the
    text-chat AI in api/_lib/aiAgentTools.js already has these; wiring
    the same tools into this voice agent is the very next phase.
  - No persistent customer memory lookup (Phase 4.1's
    abos_chat_customer_memory) — also a Phase 4.6b add.
  - No call transcript saved back to abos_chat_messages.
"""

import json
import logging

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import groq, silero

from edge_tts_plugin import EdgeTTS

load_dotenv()  # reads .env in this folder — LIVEKIT_URL/API_KEY/API_SECRET, GROQ_API_KEY.
# Without this call, python-dotenv being in requirements.txt does nothing by
# itself; os.environ would stay empty unless you export the vars in your
# shell some other way. `lk agent deploy` (LiveKit Cloud hosting) sets env
# vars its own way at deploy time — see abos-chat-ai-agent/README.md — this
# load_dotenv() call is what makes `python agent.py dev` (local testing)
# work off a plain .env file.

# --- Azure Speech version (switch back once account verification is
# sorted — same Asad/Uzma voices, official/more production-stable) ---
# from livekit.plugins import azure
# import os
# tts = azure.TTS(
#     voice=voice,
#     speech_key=os.environ["AZURE_SPEECH_KEY"],
#     speech_region=os.environ["AZURE_SPEECH_REGION"],
# )
# ...then use `tts=tts` in the AgentSession(...) call below instead of
# the EdgeTTS(...) line.

logger = logging.getLogger("abos-chat-ai-agent")

# Per instruction: Asad (male) is the default Urdu voice for now.
DEFAULT_VOICE_UR = "ur-PK-AsadNeural"
DEFAULT_VOICE_EN = "en-US-JennyNeural"

DEFAULT_GREETING_UR = (
    "Assalam-o-Alaikum! Main store ka AI assistant hoon, owner abhi available "
    "nahi hain. Main aapki kaise madad kar sakta hoon?"
)
DEFAULT_GREETING_EN = "Hi! This is the store's AI assistant, the owner is unavailable right now. How can I help you?"

SYSTEM_PROMPT_UR = (
    "Aap ABOS store ke AI phone assistant hain. Customer se Roman Urdu mein baat "
    "karein — chota, seedha, dost-anay lehja rakhein, jaisay ek achi sales-person "
    "phone pe baat karti hai. Agar customer order dena chahay, tafseel poochein "
    "(item, quantity) aur unhein batayein ke ek insaan staff member jald follow-up "
    "karega — abhi is phase mein AI khud order finalize nahi karta."
)
SYSTEM_PROMPT_EN = (
    "You are ABOS store's AI phone assistant. Speak naturally and briefly, like a "
    "helpful salesperson on the phone. If the customer wants to place an order, "
    "note the details clearly (item, quantity) and tell them a human staff member "
    "will follow up shortly — order placement isn't wired into voice calls yet."
)


def prewarm(proc: JobProcess):
    # Voice-activity detection model, loaded once per worker process
    # and reused across every call it handles — not per-call.
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    metadata = {}
    try:
        metadata = json.loads(ctx.job.metadata or "{}")
    except json.JSONDecodeError:
        logger.warning("Could not parse job metadata, using defaults: %r", ctx.job.metadata)

    is_english = (metadata.get("preferred_language") or "").lower() == "english"

    voice = metadata.get("voice_en") if is_english else metadata.get("voice_ur")
    voice = voice or (DEFAULT_VOICE_EN if is_english else DEFAULT_VOICE_UR)

    greeting = metadata.get("greeting_en") if is_english else metadata.get("greeting_ur")
    greeting = greeting or (DEFAULT_GREETING_EN if is_english else DEFAULT_GREETING_UR)

    system_prompt = SYSTEM_PROMPT_EN if is_english else SYSTEM_PROMPT_UR
    customer_name = metadata.get("customer_name", "Customer")

    logger.info(
        "AI call starting — call_id=%s customer=%s lang=%s voice=%s (edge-tts)",
        metadata.get("call_id"), customer_name, "en" if is_english else "ur", voice,
    )

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=groq.STT(
            model="whisper-large-v3-turbo",
            language="en" if is_english else "ur",
        ),
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=EdgeTTS(voice=voice),
    )

    await session.start(
        agent=Agent(instructions=system_prompt),
        room=ctx.room,
    )

    await session.generate_reply(instructions=f"Say exactly this greeting, word for word: {greeting}")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # MUST match the agentName used in api/_lib/livekitServer.js's
            # dispatchAgent() call on the Vercel side, or the job never
            # reaches this worker.
            agent_name="abos-chat-ai-caller",
        )
    )
