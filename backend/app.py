# =============================================================================
# KindEyes — backend/app.py (v4 — Full Video AD Script Generation)
#
# NEW PIPELINE:
#   1. Download the YouTube video
#   2. Extract audio from video
#   3. Send BOTH video frames + audio to Gemini to generate a full
#      professional AD script with [sound cues] in brackets
#   4. Parse the script — narrator reads descriptions, skips [brackets]
#   5. Generate TTS audio for each narration segment
#   6. Mix narrator audio with original video audio using ffmpeg
#   7. Return combined MP3 — ready to play alongside the video
#
# KEY INSIGHT: Instead of detecting "gaps", Gemini watches the full video
# and writes the script the way a human AD writer would — describing visual
# action and noting [sound cues] where the narrator should stay silent.
# =============================================================================

import os, json, base64, subprocess, tempfile, time, re, io
import requests
import yt_dlp
from pathlib import Path
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# ── Auto-install ffmpeg on cloud servers (Render, Railway) ────────────────────
# On local Windows/Mac, ffmpeg is already installed manually.
# On Linux cloud servers, we install it automatically on first start.
import platform
if platform.system() == "Linux":
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
        if result.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, OSError):
        print("[KindEyes] Installing ffmpeg on cloud server…")
        subprocess.run(
            ["apt-get", "install", "-y", "ffmpeg"],
            capture_output=True
        )
        print("[KindEyes] ffmpeg installed.")

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_VISION_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)
GEMINI_PRO_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-pro:generateContent"
)
ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"
ELEVENLABS_TTS_URL    = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

MAX_VIDEO_MB  = 180   # Gemini inline video limit
SAMPLE_EVERY  = 3     # extract a frame every N seconds for script generation

# ── AD Script Generation Prompt ───────────────────────────────────────────────

AD_SCRIPT_PROMPT = """You are a professional Audio Description (AD) writer for the blind and visually impaired.

I am giving you a video. Watch it carefully and write a complete Audio Description script.

CRITICAL RULES:
- NEVER describe during dialogue — use [dialogue] to mark those moments
- Describe BEFORE dialogue starts (set the scene) and AFTER dialogue ends (describe what changed)
- Only describe what is visually happening that cannot be heard
- Each description must be 10 words or fewer
- Be objective — "He lowers his head" not "He looks sad"
- Focus on: character actions, scene changes, important objects, on-screen text

SCRIPT FORMAT:
- Each line: [TIMESTAMP] description text
- Timestamp format: M:SS (e.g. 0:05, 1:23)
- Use [dialogue] for spoken dialogue moments — narrator stays silent
- Use [music] for music-only moments — narrator stays silent  
- Use [sound effects] for action sounds — narrator stays silent

EXAMPLE OUTPUT:
0:02 A boy sits alone at a wooden table.
0:05 [dialogue]
0:09 He picks up a fork and stares at vegetables.
0:12 [dialogue]
0:16 A woman enters from the left doorway.
0:19 She places a hand on his shoulder gently.
0:22 [dialogue]
0:26 The boy looks up slowly at her face.
0:28 [music]

IMPORTANT: The goal is to describe the VISUAL story during non-dialogue moments only.
If a scene starts with dialogue immediately, use [dialogue] first then describe what you see between lines.

Now write the complete AD script for this video. Return ONLY the script lines, nothing else."""

# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"ok": True, "message": "KindEyes server is running"})

# ── Voices ────────────────────────────────────────────────────────────────────

@app.route("/api/voices")
def list_voices():
    key = request.args.get("key", "")
    if not key:
        return jsonify({"voices": []})
    try:
        r = requests.get(
            ELEVENLABS_VOICES_URL,
            headers={"xi-api-key": key},
            timeout=10
        )
        if r.status_code == 200:
            voices = [
                {"id": v["voice_id"], "name": v["name"]}
                for v in r.json().get("voices", [])
            ]
            return jsonify({"voices": voices})
    except Exception as e:
        print(f"[KindEyes] Voices error: {e}")
    return jsonify({"voices": []})

# ── Main pipeline ─────────────────────────────────────────────────────────────

@app.route("/api/process", methods=["POST"])
def process():
    data           = request.get_json(force=True)
    youtube_url    = data.get("url", "").strip()
    gemini_key     = data.get("geminiKey", "").strip()
    elevenlabs_key = data.get("elevenLabsKey", "").strip()
    voice_id       = data.get("voiceId", "").strip()

    if not youtube_url:
        return jsonify({"error": "YouTube URL is required"}), 400
    if not gemini_key:
        return jsonify({"error": "Gemini API key is required"}), 400

    def generate():
        def event(kind, payload):
            return f"data: {json.dumps({'type': kind, **payload})}\n\n"

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            try:
                # ── Step 1: Video info ─────────────────────────────────────
                yield event("progress", {"step": "Fetching video info…", "pct": 5})
                info     = get_video_info(youtube_url)
                title    = info.get("title", "Unknown")
                thumb    = info.get("thumbnail", "")
                duration = float(info.get("duration") or 0)
                yield event("info", {
                    "title": title, "thumbnail": thumb, "duration": duration
                })

                # ── Step 2: Download video ─────────────────────────────────
                yield event("progress", {"step": "Downloading video…", "pct": 10})
                video_path = download_video(youtube_url, tmpdir)

                # ── Step 3: Extract audio from video ───────────────────────
                yield event("progress", {"step": "Extracting audio track…", "pct": 20})
                audio_path = extract_audio_from_video(video_path, tmpdir)

                # ── Step 4: Generate AD script via Gemini ──────────────────
                yield event("progress", {
                    "step": "Gemini is watching the video and writing the AD script…",
                    "pct": 30
                })
                script_lines = generate_ad_script(
                    video_path, audio_path, gemini_key, duration, tmpdir
                )
                yield event("progress", {
                    "step": f"Script ready — {len(script_lines)} lines generated",
                    "pct": 55
                })
                yield event("script", {"lines": script_lines})

                # ── Step 5: Generate TTS for narrator lines ────────────────
                yield event("progress", {
                    "step": "Generating narrator audio for each description…",
                    "pct": 60
                })

                # Only voice lines that are NOT sound cues
                # Also skip lines that are dialogue cues — let original audio speak
                DIALOGUE_KEYWORDS = ["dialogue", "dialog", "speaking", "says", "talking", "conversation"]
                narrator_lines = [
                    l for l in script_lines
                    if not l["is_sound_cue"]
                    and not any(kw in l["text"].lower() for kw in DIALOGUE_KEYWORDS)
                ]
                descriptions   = []
                total          = len(narrator_lines)

                for i, line in enumerate(narrator_lines):
                    pct = 60 + int((i / max(total, 1)) * 25)
                    yield event("progress", {
                        "step": f"Voicing line {i+1} of {total}: \"{line['text'][:40]}\"",
                        "pct":  pct
                    })

                    audio_b64 = make_tts(line["text"], elevenlabs_key, voice_id)
                    desc = {
                        "timestamp":   line["timestamp"],
                        "timestamp_s": line["timestamp_s"],
                        "text":        line["text"],
                        "audioBase64": audio_b64,
                    }
                    descriptions.append(desc)
                    yield event("description", {"index": i, "desc": desc})
                    time.sleep(0.2)

                # ── Step 6: Mix combined audio ─────────────────────────────
                yield event("progress", {
                    "step": "Mixing narrator audio with original video audio…",
                    "pct": 88
                })

                combined_b64 = None
                if descriptions:
                    try:
                        combined_b64 = mix_audio_ffmpeg(
                            audio_path, descriptions, tmpdir
                        )
                        yield event("progress", {
                            "step": "Combined audio track ready!",
                            "pct": 97
                        })
                    except Exception as e:
                        print(f"[KindEyes] Mix error: {e}")

                yield event("progress", {"step": "Complete!", "pct": 100})
                yield event("done", {
                    "descriptions":        descriptions,
                    "scriptLines":         script_lines,
                    "combinedAudioBase64": combined_b64,
                })

            except Exception as e:
                import traceback; traceback.print_exc()
                yield event("error", {"message": str(e)})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# ── Step 1: Video info ────────────────────────────────────────────────────────

def get_video_info(url):
    ydl_opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info

# ── Step 2: Download video ────────────────────────────────────────────────────

def download_video(url, tmpdir):
    out_tmpl = str(tmpdir / "video.%(ext)s")
    ydl_opts = {
        "format":      "bestvideo[height<=480][ext=mp4]+bestaudio/best[height<=480]/best",
        "outtmpl":     out_tmpl,
        "noplaylist":  True,
        "quiet":       True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    matches = list(tmpdir.glob("video.*"))
    if not matches:
        raise Exception("Video download failed")
    return matches[0]

# ── Step 3: Extract audio ─────────────────────────────────────────────────────

def extract_audio_from_video(video_path, tmpdir):
    audio_path = tmpdir / "audio.mp3"
    result = subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-vn", "-ar", "44100", "-ac", "2", "-ab", "128k",
        "-y", str(audio_path)
    ], capture_output=True, timeout=120)
    if result.returncode != 0:
        raise Exception(f"Audio extraction failed: {result.stderr.decode()[:200]}")
    return audio_path

# ── Step 4: Generate AD Script via Gemini ────────────────────────────────────

def generate_ad_script(video_path, audio_path, api_key, duration, tmpdir):
    """
    Send sampled video frames + audio to Gemini to generate a full AD script.

    We extract one frame every SAMPLE_EVERY seconds and send them all
    together with the audio. Gemini watches the full visual timeline
    and writes a professional AD script with [sound cues].
    """
    print(f"[KindEyes] Extracting frames every {SAMPLE_EVERY}s for AD script…")

    # Extract frames every N seconds
    frames = []
    t = 1.0
    while t < duration:
        frame_b64 = extract_frame(video_path, t)
        if frame_b64:
            frames.append({"time": t, "data": frame_b64})
        t += SAMPLE_EVERY

    print(f"[KindEyes] Extracted {len(frames)} frames, sending to Gemini…")

    # Build Gemini request with all frames
    parts = []

    # Add context about timing
    parts.append({
        "text": f"This video is {fmt_time(duration)} long. "
                f"I am sending you one frame every {SAMPLE_EVERY} seconds. "
                f"Use the frame sequence to understand the visual timeline.\n\n"
                f"{AD_SCRIPT_PROMPT}"
    })

    # Add frames with timestamp labels
    for frame in frames:
        parts.append({"text": f"[Frame at {fmt_time(frame['time'])}]"})
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data":      frame["data"]
            }
        })

    # Also send audio for Gemini to understand dialogue/sound cues
    audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if audio_size_mb <= 15:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        parts.append({"text": "Here is the audio track:"})
        parts.append({
            "inline_data": {
                "mime_type": "audio/mpeg",
                "data":      audio_b64
            }
        })

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "maxOutputTokens": 8192,
            "temperature":     0.2,
        }
    }

    r = requests.post(
        f"{GEMINI_PRO_URL}?key={api_key}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=180
    )

    if r.status_code != 200:
        raise Exception(f"Gemini script error {r.status_code}: {r.text[:300]}")

    raw_script = (r.json()
                   .get("candidates", [{}])[0]
                   .get("content", {})
                   .get("parts", [{}])[0]
                   .get("text", ""))

    print(f"[KindEyes] Raw script received:\n{raw_script[:500]}…")

    return parse_ad_script(raw_script)


def parse_ad_script(raw_script):
    """
    Parse the raw script text into structured lines.

    Expected format per line:
        0:05 A boy sits alone at a wooden table.
        0:08 [dialogue]
        0:11 He picks up a fork and stares at vegetables.

    Returns list of:
        { timestamp, timestamp_s, text, is_sound_cue }
    """
    lines = []
    # Match lines starting with a timestamp like 0:05 or 1:23
    pattern = re.compile(r"^\s*(\d{1,2}):(\d{2})\s+(.+)$", re.MULTILINE)

    for match in pattern.finditer(raw_script):
        minutes    = int(match.group(1))
        seconds    = int(match.group(2))
        text       = match.group(3).strip()
        timestamp_s = minutes * 60 + seconds

        # Detect sound cues — text wrapped in [brackets]
        is_sound_cue = bool(re.match(r"^\[.+\]$", text))

        lines.append({
            "timestamp":   f"{minutes}:{seconds:02d}",
            "timestamp_s": timestamp_s,
            "text":        text,
            "is_sound_cue": is_sound_cue,
        })

    lines.sort(key=lambda l: l["timestamp_s"])

    print(f"[KindEyes] Parsed {len(lines)} script lines "
          f"({sum(1 for l in lines if not l['is_sound_cue'])} narrator, "
          f"{sum(1 for l in lines if l['is_sound_cue'])} sound cues)")

    return lines

# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frame(video_path, timestamp):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        frame_path = Path(f.name)
    try:
        result = subprocess.run([
            "ffmpeg", "-ss", str(timestamp), "-i", str(video_path),
            "-vframes", "1", "-q:v", "4",
            "-vf", "scale=640:-1",  # smaller for batch frame sending
            "-y", str(frame_path)
        ], capture_output=True, timeout=15)
        if result.returncode != 0 or not frame_path.exists():
            return None
        if frame_path.stat().st_size == 0:
            return None
        with open(frame_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[KindEyes] Frame error at {timestamp}s: {e}")
        return None
    finally:
        if frame_path.exists():
            frame_path.unlink()

# ── TTS ───────────────────────────────────────────────────────────────────────

def make_tts(text, elevenlabs_key, voice_id):
    if elevenlabs_key and voice_id:
        result = tts_elevenlabs(text, elevenlabs_key, voice_id)
        if result:
            return result
    return tts_gtts(text)


def tts_elevenlabs(text, api_key, voice_id):
    r = requests.post(
        ELEVENLABS_TTS_URL.format(voice_id=voice_id),
        headers={
            "xi-api-key":   api_key,
            "Content-Type": "application/json",
            "Accept":       "audio/mpeg"
        },
        json={
            "text":     text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {
                "stability":         0.55,
                "similarity_boost":  0.80,
                "style":             0.0,
                "use_speaker_boost": True
            }
        },
        timeout=30
    )
    if r.status_code != 200:
        print(f"[KindEyes] ElevenLabs error {r.status_code}: {r.text[:150]}")
        return None
    return base64.b64encode(r.content).decode("utf-8")


def tts_gtts(text):
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en", slow=False).write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"[KindEyes] gTTS error: {e}")
        return None

# ── Mix audio with ffmpeg ─────────────────────────────────────────────────────

def mix_audio_ffmpeg(audio_path, descriptions, tmpdir):
    """
    Mix original video audio with narrator clips using ffmpeg.

    HOW IT WORKS:
    - Original audio plays at full volume throughout
    - At each narrator timestamp, the original audio ducks to 25% volume
    - Narrator clip plays at 2x volume over the ducked background
    - After narrator finishes, original audio restores to full volume
    - Result: a seamless audio track — dialogue, effects, music all intact
      with narrator voice naturally inserted during visual moments
    """
    print(f"[KindEyes] Mixing {len(descriptions)} narrator clips into audio…")

    # Save each narrator clip
    clip_files = []
    valid_descs = []
    for i, desc in enumerate(descriptions):
        if not desc.get("audioBase64"):
            continue
        clip_path = tmpdir / f"narrator_{i}.mp3"
        with open(clip_path, "wb") as f:
            f.write(base64.b64decode(desc["audioBase64"]))

        # Get duration of clip using ffprobe
        probe = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", str(clip_path)
        ], capture_output=True, timeout=10)
        clip_dur = 3.0  # default fallback
        try:
            probe_data = json.loads(probe.stdout)
            clip_dur = float(probe_data["streams"][0].get("duration", 3.0))
        except:
            pass

        desc["_clip_dur"] = clip_dur
        clip_files.append(clip_path)
        valid_descs.append(desc)

    if not clip_files:
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Build ffmpeg filter_complex
    # Strategy:
    #   1. Original audio: apply volume envelope — duck during each narration window
    #   2. Each narrator clip: delay to its timestamp, boost volume
    #   3. amix all together

    inputs = ["-i", str(audio_path)]
    for clip_path in clip_files:
        inputs += ["-i", str(clip_path)]

    # Build volume envelope for original audio (duck during narration)
    # Format: volume=enable='between(t,start,end)':volume=0.25
    duck_filters = []
    for desc in valid_descs:
        start = desc["timestamp_s"]
        end   = start + desc.get("_clip_dur", 3.0) + 0.3  # tiny tail
        duck_filters.append(f"between(t,{start:.2f},{end:.2f})")

    if duck_filters:
        duck_expr = "+".join(duck_filters)
        # Duck to 50% during narration (was 20%), restore to 100% otherwise
        orig_filter = f"[0:a]volume=enable='{duck_expr}':volume=0.5,volume=1.0[orig]"
    else:
        orig_filter = "[0:a]volume=1.0[orig]"

    filter_parts = [orig_filter]
    mix_labels   = ["[orig]"]

    # Each narrator clip: delay + comfortable volume (1.5x — not too loud)
    for i, (desc, clip_path) in enumerate(zip(valid_descs, clip_files)):
        delay_ms = int(desc["timestamp_s"] * 1000)
        label    = f"[n{i}]"
        filter_parts.append(
            f"[{i+1}:a]adelay={delay_ms}|{delay_ms},volume=1.5{label}"
        )
        mix_labels.append(label)

    # Mix all together
    n = len(mix_labels)
    filter_parts.append(
        f"{''.join(mix_labels)}amix=inputs={n}:duration=first:normalize=0[out]"
    )

    filter_complex = ";".join(filter_parts)
    output_path    = tmpdir / "combined.mp3"

    cmd = [
        "ffmpeg",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ar", "44100",
        "-ab", "128k",
        "-y", str(output_path)
    ]

    print(f"[KindEyes] Running ffmpeg mix ({len(clip_files)} clips)…")
    result = subprocess.run(cmd, capture_output=True, timeout=180)

    if result.returncode != 0 or not output_path.exists():
        err = result.stderr.decode("utf-8", errors="replace")[:500]
        print(f"[KindEyes] Mix failed: {err}")
        # Fallback: return original audio
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    size_kb = output_path.stat().st_size / 1024
    print(f"[KindEyes] Combined audio ready — {size_kb:.0f}KB")
    with open(output_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_time(s):
    m = int(s // 60); sec = int(s % 60)
    return f"{m}:{sec:02d}"

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n👁  KindEyes Backend  (v4 — Full Video AD Script)")
    print("━" * 50)
    print(f"Server running on port {port}")
    print("━" * 50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
