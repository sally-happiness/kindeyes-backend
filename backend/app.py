# =============================================================================
# KindEyes — backend/app.py (v5 — Cloud Deployment)
#
# CHANGES FROM v4:
#   - YouTube cookies support (bypasses YouTube bot detection on cloud servers)
#   - Fallback Gemini API key (users don't need to enter a key)
#   - Auto-installs ffmpeg on Linux cloud servers
#   - Reads PORT from environment variable (required by Render/Railway)
# =============================================================================

import os, json, base64, subprocess, tempfile, time, re, io
import requests
import yt_dlp
from pathlib import Path
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import platform

# ── Auto-install ffmpeg on cloud servers ──────────────────────────────────────
if platform.system() == "Linux":
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
        if result.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, OSError):
        print("[KindEyes] Installing ffmpeg…")
        subprocess.run(["apt-get", "install", "-y", "ffmpeg"], capture_output=True)
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
    "gemini-2.5-flash:generateContent"
)
ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"
ELEVENLABS_TTS_URL    = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

# ── YOUR GEMINI KEY HERE ──────────────────────────────────────────────────────
# Paste your Gemini API key below (between the quotes).
# Users who don't enter their own key will use this one automatically.
# Get a free key at: https://aistudio.google.com/app/apikey
DEFAULT_GEMINI_KEY = "AIzaSyCZjyfA4vqTEP5OXsxb5oPKM8H-LVKtnms"

# ── YouTube cookies ───────────────────────────────────────────────────────────
# Path to your exported YouTube cookies file.
# Upload cookies.txt to your GitHub repo alongside app.py.
# If the file doesn't exist, yt-dlp will try without cookies.
COOKIES_FILE = "cookies.txt"

SAMPLE_EVERY = 3   # extract a frame every N seconds for AD script

# ── AD Script Prompt ──────────────────────────────────────────────────────────

AD_SCRIPT_PROMPT = """You are a professional Audio Description (AD) writer for the blind and visually impaired.

I am giving you a video. Watch it carefully and write a complete Audio Description script.

CRITICAL RULES:
- READ ALL ON-SCREEN TEXT: speech bubbles, text overlays, captions, signs, titles — read them word for word as they appear. This is the most important rule for animated films that tell their story through text.
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

IMPORTANT: For animations that use text bubbles as storytelling, reading that text IS the audio description.

Now write the complete AD script for this video. Return ONLY the script lines, nothing else."""

# ── yt-dlp options with cookies ───────────────────────────────────────────────

def base_ydl_opts():
    """
    Base yt-dlp options.
    Reads YouTube cookies from environment variable YOUTUBE_COOKIES
    (set on Render dashboard) and writes them to a temp file.
    This avoids file path issues on cloud servers.
    """
    opts = {
        "quiet":       True,
        "no_warnings": True,
        "noplaylist":  True,
    }

    cookies_content = os.environ.get("YOUTUBE_COOKIES", "").strip()

    if cookies_content:
        # Write cookies to a temp file that yt-dlp can read
        cookies_tmp = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
        with open(cookies_tmp, "w", encoding="utf-8") as f:
            f.write(cookies_content)
        opts["cookiefile"] = cookies_tmp
        print("[KindEyes] Using cookies from environment variable ✓")
    elif os.path.exists("cookies.txt"):
        opts["cookiefile"] = "cookies.txt"
        print("[KindEyes] Using cookies from cookies.txt file ✓")
    else:
        print("[KindEyes] No cookies found — YouTube may block downloads")

    return opts

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
    elevenlabs_key = data.get("elevenLabsKey", "").strip()
    voice_id       = data.get("voiceId", "").strip()

    # Use user's key if provided, otherwise fall back to default key
    gemini_key = data.get("geminiKey", "").strip() or DEFAULT_GEMINI_KEY

    if not youtube_url:
        return jsonify({"error": "YouTube URL is required"}), 400
    if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY_HERE":
        return jsonify({"error": "Gemini API key is not configured. Please add it to app.py or enter one in Settings."}), 400

    def generate():
        def event(kind, payload):
            return f"data: {json.dumps({'type': kind, **payload})}\n\n"

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            try:
                # Step 1: Video info
                yield event("progress", {"step": "Fetching video info…", "pct": 5})
                info     = get_video_info(youtube_url)
                title    = info.get("title", "Unknown")
                thumb    = info.get("thumbnail", "")
                duration = float(info.get("duration") or 0)
                yield event("info", {"title": title, "thumbnail": thumb, "duration": duration})

                # Step 2: Download video
                yield event("progress", {"step": "Downloading video…", "pct": 10})
                video_path = download_video(youtube_url, tmpdir)

                # Step 3: Extract audio
                yield event("progress", {"step": "Extracting audio track…", "pct": 20})
                audio_path = extract_audio_from_video(video_path, tmpdir)

                # Step 4: Generate AD script
                yield event("progress", {"step": "Gemini is watching the video and writing the AD script…", "pct": 30})
                script_lines = generate_ad_script(video_path, audio_path, gemini_key, duration, tmpdir)
                yield event("progress", {"step": f"Script ready — {len(script_lines)} lines generated", "pct": 55})
                yield event("script", {"lines": script_lines})

                # Step 5: Generate TTS
                yield event("progress", {"step": "Generating narrator audio…", "pct": 60})

                DIALOGUE_KEYWORDS = ["dialogue", "dialog", "speaking", "says", "talking", "conversation"]
                narrator_lines = [
                    l for l in script_lines
                    if not l["is_sound_cue"]
                    and not any(kw in l["text"].lower() for kw in DIALOGUE_KEYWORDS)
                ]
                descriptions = []
                total = len(narrator_lines)

                for i, line in enumerate(narrator_lines):
                    pct = 60 + int((i / max(total, 1)) * 25)
                    yield event("progress", {
                        "step": f"Voicing line {i+1} of {total}: \"{line['text'][:40]}\"",
                        "pct": pct
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

                # Step 6: Mix audio
                yield event("progress", {"step": "Mixing narrator audio with original video audio…", "pct": 88})
                combined_b64 = None
                if descriptions:
                    try:
                        combined_b64 = mix_audio_ffmpeg(audio_path, descriptions, tmpdir)
                        yield event("progress", {"step": "Combined audio track ready!", "pct": 97})
                    except Exception as e:
                        print(f"[KindEyes] Mix error: {e}")

                yield event("progress", {"step": "Complete!", "pct": 100})
                yield event("done", {"descriptions": descriptions, "scriptLines": script_lines, "combinedAudioBase64": combined_b64})

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
    opts = base_ydl_opts()
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info

# ── Step 2: Download video ────────────────────────────────────────────────────

def download_video(url, tmpdir):
    out_tmpl = str(tmpdir / "video.%(ext)s")
    opts = base_ydl_opts()
    opts.update({
        "format":  "bestvideo[height<=480][ext=mp4]+bestaudio/best[height<=480]/best",
        "outtmpl": out_tmpl,
    })
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    matches = list(tmpdir.glob("video.*"))
    if not matches:
        raise Exception("Video download failed — no file found")
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

# ── Step 4: Generate AD Script ────────────────────────────────────────────────

def generate_ad_script(video_path, audio_path, api_key, duration, tmpdir):
    print(f"[KindEyes] Extracting frames every {SAMPLE_EVERY}s…")
    frames = []
    t = 1.0
    while t < duration:
        frame_b64 = extract_frame(video_path, t)
        if frame_b64:
            frames.append({"time": t, "data": frame_b64})
        t += SAMPLE_EVERY

    print(f"[KindEyes] Sending {len(frames)} frames to Gemini…")

    parts = []
    parts.append({
        "text": f"This video is {fmt_time(duration)} long. "
                f"I am sending you one frame every {SAMPLE_EVERY} seconds. "
                f"Use the frame sequence to understand the visual timeline.\n\n"
                f"{AD_SCRIPT_PROMPT}"
    })

    for frame in frames:
        parts.append({"text": f"[Frame at {fmt_time(frame['time'])}]"})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": frame["data"]}})

    audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if audio_size_mb <= 15:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        parts.append({"text": "Here is the audio track:"})
        parts.append({"inline_data": {"mime_type": "audio/mpeg", "data": audio_b64}})

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"maxOutputTokens": 8192, "temperature": 0.2}
    }

    r = requests.post(
        f"{GEMINI_PRO_URL}?key={api_key}",
        headers={"Content-Type": "application/json"},
        json=payload, timeout=180
    )

    if r.status_code != 200:
        raise Exception(f"Gemini script error {r.status_code}: {r.text[:300]}")

    raw_script = (r.json()
                   .get("candidates", [{}])[0]
                   .get("content", {})
                   .get("parts", [{}])[0]
                   .get("text", ""))

    print(f"[KindEyes] Script received ({len(raw_script)} chars)")
    return parse_ad_script(raw_script)


def parse_ad_script(raw_script):
    lines = []
    pattern = re.compile(r"^\s*(\d{1,2}):(\d{2})\s+(.+)$", re.MULTILINE)
    for match in pattern.finditer(raw_script):
        minutes     = int(match.group(1))
        seconds     = int(match.group(2))
        text        = match.group(3).strip()
        timestamp_s = minutes * 60 + seconds
        is_sound_cue = bool(re.match(r"^\[.+\]$", text))
        lines.append({
            "timestamp":    f"{minutes}:{seconds:02d}",
            "timestamp_s":  timestamp_s,
            "text":         text,
            "is_sound_cue": is_sound_cue,
        })
    lines.sort(key=lambda l: l["timestamp_s"])
    print(f"[KindEyes] Parsed {len(lines)} script lines")
    return lines

# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frame(video_path, timestamp):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        frame_path = Path(f.name)
    try:
        result = subprocess.run([
            "ffmpeg", "-ss", str(timestamp), "-i", str(video_path),
            "-vframes", "1", "-q:v", "4", "-vf", "scale=640:-1",
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
        headers={"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
        json={"text": text, "model_id": "eleven_turbo_v2_5",
              "voice_settings": {"stability": 0.55, "similarity_boost": 0.80, "style": 0.0, "use_speaker_boost": True}},
        timeout=30
    )
    if r.status_code != 200:
        print(f"[KindEyes] ElevenLabs error {r.status_code}")
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

# ── Audio mixing ──────────────────────────────────────────────────────────────

def mix_audio_ffmpeg(audio_path, descriptions, tmpdir):
    print(f"[KindEyes] Mixing {len(descriptions)} narrator clips…")

    clip_files  = []
    valid_descs = []

    for i, desc in enumerate(descriptions):
        if not desc.get("audioBase64"):
            continue
        clip_path = tmpdir / f"narrator_{i}.mp3"
        with open(clip_path, "wb") as f:
            f.write(base64.b64decode(desc["audioBase64"]))

        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(clip_path)],
            capture_output=True, timeout=10
        )
        clip_dur = 3.0
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

    inputs = ["-i", str(audio_path)]
    for clip_path in clip_files:
        inputs += ["-i", str(clip_path)]

    duck_filters = []
    for desc in valid_descs:
        start = desc["timestamp_s"]
        end   = start + desc.get("_clip_dur", 3.0) + 0.3
        duck_filters.append(f"between(t,{start:.2f},{end:.2f})")

    if duck_filters:
        duck_expr  = "+".join(duck_filters)
        orig_filter = f"[0:a]volume=enable='{duck_expr}':volume=0.8,volume=1.0[orig]"
    else:
        orig_filter = "[0:a]volume=1.0[orig]"

    filter_parts = [orig_filter]
    mix_labels   = ["[orig]"]

    for i, (desc, clip_path) in enumerate(zip(valid_descs, clip_files)):
        delay_ms = int(desc["timestamp_s"] * 1000)
        label    = f"[n{i}]"
        filter_parts.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms},volume=1.5{label}")
        mix_labels.append(label)

    n = len(mix_labels)
    filter_parts.append(f"{''.join(mix_labels)}amix=inputs={n}:duration=first:normalize=0[out]")

    filter_complex = ";".join(filter_parts)
    output_path    = tmpdir / "combined.mp3"

    cmd = [
        "ffmpeg", *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ar", "44100", "-ab", "128k",
        "-y", str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=180)

    if result.returncode != 0 or not output_path.exists():
        print(f"[KindEyes] Mix failed — returning original audio")
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
    print("\n👁  KindEyes Backend (v5 — Cloud Deployment)")
    print("━" * 50)
    print(f"Server running on port {port}")
    has_cookies = bool(os.environ.get("YOUTUBE_COOKIES")) or os.path.exists("cookies.txt")
print(f"Cookies: {'configured ✓' if has_cookies else 'NOT configured — YouTube may block downloads'}")
    print(f"Default Gemini key: {'configured ✓' if DEFAULT_GEMINI_KEY != 'YOUR_GEMINI_API_KEY_HERE' else 'NOT SET — add your key!'}")
    print("━" * 50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
