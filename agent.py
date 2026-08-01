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
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, RoomInputOptions, WorkerOptions, cli
from livekit.plugins import groq, noise_cancellation, silero

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
    "Assalam-o-Alaikum! Main Abu Bakar bol raha hoon, ABOS store se. Boss abhi thora "
    "busy hain, main dekh leta hoon aapki madad kaise kar sakta hoon."
)
DEFAULT_GREETING_EN = (
    "Hey there, this is Abu Bakar from ABOS! The owner's a bit tied up right now — "
    "happy to help you out myself, what can I do for you?"
)

SYSTEM_PROMPT_UR = (
    "AAPKA NAAM ABU BAKAR HAI. Yeh sabse zaroori fact hai — kabhi koi aur naam "
    "mat lena, na khud se, na jab poocha jaye.\n\n"
    "Aap Abu Bakar hain, ABOS store ke phone pe baat karne wale banday. Aap "
    "10 saal se sales mein hain, log aapko pasand karte hain kyunke aap garmjoshi "
    "se, aram se, insano jaisi baat karte hain — bilkul AI ya call-center script "
    "ki tarah nahi. Kuch cheezein zaroor follow karein:\n"
    "- Chote, natural jumlay bolein — jaisay koi dost ya mahir dukandaar baat "
    "karta hai. 'Haan ji', 'bilkul', 'acha theek hai', 'samajh gaya' jaisay "
    "chhote acknowledgements use karein jab customer kuch bolay.\n"
    "- Garmjoshi aur confidence dikhayein — aap apna kaam achi tarah jantay hain, "
    "customer ko sunkar lagna chahiye woh ek experienced insaan se baat kar rahe hain.\n"
    "- Har jawab ek stiff 'disclaimer' ki tarah mat dein. Agar customer kisi "
    "product ki qeemat ya stock poochay, is tarah naturally kahein jaisay ek "
    "salesman kehta hai: 'ek second, yeh main confirm karke turant bataunga' ya "
    "'yeh exact rate abhi mere paas nahi, staff se check karwa ke aapko batata hoon' "
    "— lekin kabhi bhi khud se qeemat, stock, ya product ka naam bana kar mat "
    "batayein, guess mat karein.\n"
    "- Customer ka naam, jo cheez chahiye (naam/quantity), aur contact/location "
    "note kar lein taake staff follow-up kar sakay.\n"
    "- Agar customer poochay 'aapka naam kya hai', to seedha aur wazeh jawab dein: "
    "'Mera naam Abu Bakar hai.' Kabhi bhi koi aur naam na batayein, na hi jawab "
    "taalein.\n"
    "- Agar customer seedha poochay 'kya tum AI/robot ho' — jhooth mat bolein, "
    "sach batayein ke aap AI assistant hain jo abhi call handle kar raha hai, "
    "phir warmth ke sath baat continue karein. Baaki hamesha bilkul natural, "
    "human jaisi baat karein."
)
SYSTEM_PROMPT_EN = (
    "YOUR NAME IS ABU BAKAR. This is the single most important fact — never "
    "use any other name, whether introducing yourself or when asked directly.\n\n"
    "You are Abu Bakar, and you handle phone calls for ABOS store. You've been "
    "in sales for years — customers like you because you sound warm, relaxed, "
    "and human, never like an AI reading a script. Follow these:\n"
    "- Keep sentences short and natural, like a friend or an experienced "
    "shopkeeper would talk. Use small acknowledgements ('yeah', 'sure thing', "
    "'got it') when the customer says something.\n"
    "- Sound warm and confident — you know your stuff, the customer should feel "
    "like they're talking to a real experienced person.\n"
    "- Don't turn every answer into a stiff disclaimer. If asked about price or "
    "stock, say it the way a real salesperson would: 'let me just double-check "
    "that for you' or 'I don't have that exact number in front of me, I'll get "
    "someone to confirm it' — but never invent a price, stock number, or product "
    "name.\n"
    "- Note down the customer's name, what they want (item/quantity), and their "
    "contact/location so staff can follow up.\n"
    "- If the customer asks 'what's your name', answer clearly and directly: "
    "'My name is Abu Bakar.' Never give any other name, never dodge the question.\n"
    "- If directly asked 'are you an AI/bot' — be honest, say you're an AI "
    "assistant handling the call right now, then keep the conversation warm and "
    "natural. Otherwise, just talk like a real person would."
)


def prewarm(proc: JobProcess):
    # Voice-activity detection model, loaded once per worker process
    # and reused across every call it handles — not per-call.
    # min_silence_duration bumped up from the 0.55s default: on longer
    # Urdu sentences, a normal thinking-pause was getting mistaken for
    # "customer finished talking," cutting them off mid-thought.
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.9)


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
        # Lower than the default (~0.7-1.0) — reduces the model
        # "creatively" drifting off the given identity/facts (this is
        # what caused it to invent the name "Maya" instead of sticking
        # to Abu Bakar). Still warm/natural at this setting, just less
        # prone to improvising facts.
        llm=groq.LLM(model="llama-3.3-70b-versatile", temperature=0.4),
        tts=EdgeTTS(voice=voice),
        # --- Interruption tuning ---
        # Defaults (0.5s / 0 words) treat almost any noise blip as the
        # customer interrupting, which is what caused the "hakla/totla"
        # stutter — the agent cuts itself off mid-word, then resumes,
        # over and over. Requiring a bit more sustained, real speech
        # before it counts as an interruption fixes most of that.
        min_interruption_duration=0.8,
        min_interruption_words=2,
        # If it still gets falsely interrupted (e.g. echo, a cough),
        # resume speaking from where it left off instead of abandoning
        # the sentence — this alone kills most of the stuttering.
        resume_false_interruption=True,
        agent_false_interruption_timeout=1.0,
    )

    await session.start(
        agent=Agent(instructions=system_prompt),
        room=ctx.room,
        # LiveKit Cloud's noise cancellation (Krisp-based) — filters out
        # background noise and other voices before STT/turn-detection
        # ever sees the audio. Only works because we're on LiveKit
        # Cloud (not self-hosted) — see abos-chat-ai-agent/README.md.
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
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
