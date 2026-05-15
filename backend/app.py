# =============================================================================
# KindEyes — backend/app.py (v6 — Final Cloud Version)
# Reads YouTube cookies from Render environment variable YOUTUBE_COOKIES
# =============================================================================

import os, json, base64, subprocess, tempfile, time, re, io, platform
import requests
import yt_dlp
from pathlib import Path
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

if platform.system() == "Linux":
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True)
        if r.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, OSError):
        print("[KindEyes] Installing ffmpeg...")
        subprocess.run(["apt-get", "install", "-y", "ffmpeg"], capture_output=True)
        print("[KindEyes] ffmpeg installed.")

app = Flask(__name__)
CORS(app)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"
ELEVENLABS_TTS_URL    = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

# ── PASTE YOUR GEMINI KEY BELOW ───────────────────────────────────────────────
DEFAULT_GEMINI_KEY = "YOUR_GEMINI_KEY_HERE"

SAMPLE_EVERY = 3

AD_SCRIPT_PROMPT = """You are a professional Audio Description (AD) writer for the blind and visually impaired.

I am giving you a video. Watch it carefully and write a complete Audio Description script.

CRITICAL RULES:
- READ ALL ON-SCREEN TEXT: speech bubbles, text overlays, captions, signs — read them word for word
- NEVER describe during dialogue — use [dialogue] to mark those moments
- Describe BEFORE dialogue starts and AFTER dialogue ends
- Only describe what is visually happening that cannot be heard
- Each description must be 10 words or fewer
- Be objective — "He lowers his head" not "He looks sad"

SCRIPT FORMAT:
- Each line: [TIMESTAMP] description text
- Timestamp format: M:SS (e.g. 0:05, 1:23)
- Use [dialogue] for spoken dialogue — narrator stays silent
- Use [music] for music only — narrator stays silent
- Use [sound effects] for action sounds — narrator stays silent

EXAMPLE:
0:02 A boy sits alone at a wooden table.
0:05 [dialogue]
0:09 He picks up a fork and stares at vegetables.
0:16 A woman enters from the left doorway.
0:22 [dialogue]
0:26 The boy looks up slowly at her face.
0:28 [music]

Now write the complete AD script. Return ONLY the script lines, nothing else."""


def base_ydl_opts():
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
    cookies_content = os.environ.get("YOUTUBE_COOKIES", "").strip()
    if cookies_content:
        cookies_tmp = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
        with open(cookies_tmp, "w", encoding="utf-8") as f:
            f.write(cookies_content)
        opts["cookiefile"] = cookies_tmp
        print("[KindEyes] Cookies loaded from environment variable")
    elif os.path.exists("cookies.txt"):
        opts["cookiefile"] = "cookies.txt"
        print("[KindEyes] Cookies loaded from cookies.txt")
    else:
        print("[KindEyes] No cookies found")
    return opts


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "message": "KindEyes server is running"})
@app.route("/api/debug")
def debug():
    cookies_env = os.environ.get("YOUTUBE_COOKIES", "")
    cookies_file = os.path.exists("cookies.txt")
    return jsonify({
        "cookies_env_length": len(cookies_env),
        "cookies_env_preview": cookies_env[:100] if cookies_env else "EMPTY",
        "cookies_file_exists": cookies_file,
    })


@app.route("/api/voices")
def list_voices():
    key = request.args.get("key", "")
    if not key:
        return jsonify({"voices": []})
    try:
        r = requests.get(ELEVENLABS_VOICES_URL, headers={"xi-api-key": key}, timeout=10)
        if r.status_code == 200:
            voices = [{"id": v["voice_id"], "name": v["name"]} for v in r.json().get("voices", [])]
            return jsonify({"voices": voices})
    except Exception as e:
        print(f"[KindEyes] Voices error: {e}")
    return jsonify({"voices": []})


@app.route("/api/process", methods=["POST"])
def process():
    data           = request.get_json(force=True)
    youtube_url    = data.get("url", "").strip()
    elevenlabs_key = data.get("elevenLabsKey", "").strip()
    voice_id       = data.get("voiceId", "").strip()
    gemini_key     = data.get("geminiKey", "").strip() or DEFAULT_GEMINI_KEY

    if not youtube_url:
        return jsonify({"error": "YouTube URL is required"}), 400
    if not gemini_key or gemini_key == "YOUR_GEMINI_KEY_HERE":
        return jsonify({"error": "Gemini API key not configured."}), 400

    def generate():
        def event(kind, payload):
            return f"data: {json.dumps({'type': kind, **payload})}\n\n"

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            try:
                yield event("progress", {"step": "Fetching video info...", "pct": 5})
                info     = get_video_info(youtube_url)
                title    = info.get("title", "Unknown")
                thumb    = info.get("thumbnail", "")
                duration = float(info.get("duration") or 0)
                yield event("info", {"title": title, "thumbnail": thumb, "duration": duration})

                yield event("progress", {"step": "Downloading video...", "pct": 10})
                video_path = download_video(youtube_url, tmpdir)

                yield event("progress", {"step": "Extracting audio track...", "pct": 20})
                audio_path = extract_audio(video_path, tmpdir)

                yield event("progress", {"step": "Gemini is watching the video and writing the AD script...", "pct": 30})
                script_lines = generate_ad_script(video_path, audio_path, gemini_key, duration)
                yield event("progress", {"step": f"Script ready — {len(script_lines)} lines", "pct": 55})
                yield event("script", {"lines": script_lines})

                yield event("progress", {"step": "Generating narrator audio...", "pct": 60})
                SKIP_KEYWORDS = ["dialogue", "dialog", "speaking", "talking", "conversation"]
                narrator_lines = [
                    l for l in script_lines
                    if not l["is_sound_cue"]
                    and not any(kw in l["text"].lower() for kw in SKIP_KEYWORDS)
                ]
                descriptions = []
                total = len(narrator_lines)

                for i, line in enumerate(narrator_lines):
                    pct = 60 + int((i / max(total, 1)) * 25)
                    yield event("progress", {"step": f"Voicing line {i+1} of {total}...", "pct": pct})
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

                yield event("progress", {"step": "Mixing audio track...", "pct": 88})
                combined_b64 = None
                if descriptions:
                    try:
                        combined_b64 = mix_audio(audio_path, descriptions, tmpdir)
                        yield event("progress", {"step": "Combined audio ready!", "pct": 97})
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


def get_video_info(url):
    with yt_dlp.YoutubeDL(base_ydl_opts()) as ydl:
        return ydl.extract_info(url, download=False)


def download_video(url, tmpdir):
    opts = base_ydl_opts()
    opts.update({
        "format":  "bestvideo[height<=480][ext=mp4]+bestaudio/best[height<=480]/best",
        "outtmpl": str(tmpdir / "video.%(ext)s"),
    })
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    matches = list(tmpdir.glob("video.*"))
    if not matches:
        raise Exception("Video download failed")
    return matches[0]


def extract_audio(video_path, tmpdir):
    audio_path = tmpdir / "audio.mp3"
    r = subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-vn", "-ar", "44100", "-ac", "2", "-ab", "128k",
        "-y", str(audio_path)
    ], capture_output=True, timeout=120)
    if r.returncode != 0:
        raise Exception(f"Audio extraction failed: {r.stderr.decode()[:200]}")
    return audio_path


def generate_ad_script(video_path, audio_path, api_key, duration):
    frames = []
    t = 1.0
    while t < duration:
        fb64 = extract_frame(video_path, t)
        if fb64:
            frames.append({"time": t, "data": fb64})
        t += SAMPLE_EVERY

    parts = [{"text": f"Video is {fmt_time(duration)} long. One frame every {SAMPLE_EVERY}s.\n\n{AD_SCRIPT_PROMPT}"}]
    for frame in frames:
        parts.append({"text": f"[Frame at {fmt_time(frame['time'])}]"})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": frame["data"]}})

    if audio_path.stat().st_size / (1024*1024) <= 15:
        with open(audio_path, "rb") as f:
            ab64 = base64.b64encode(f.read()).decode("utf-8")
        parts.append({"text": "Audio track:"})
        parts.append({"inline_data": {"mime_type": "audio/mpeg", "data": ab64}})

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"maxOutputTokens": 8192, "temperature": 0.2}
    }
    r = requests.post(
        f"{GEMINI_URL}?key={api_key}",
        headers={"Content-Type": "application/json"},
        json=payload, timeout=180
    )
    if r.status_code != 200:
        raise Exception(f"Gemini error {r.status_code}: {r.text[:300]}")

    raw = (r.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", ""))
    return parse_script(raw)


def parse_script(raw):
    lines = []
    for match in re.finditer(r"^\s*(\d{1,2}):(\d{2})\s+(.+)$", raw, re.MULTILINE):
        m, s, text = int(match.group(1)), int(match.group(2)), match.group(3).strip()
        lines.append({
            "timestamp":    f"{m}:{s:02d}",
            "timestamp_s":  m * 60 + s,
            "text":         text,
            "is_sound_cue": bool(re.match(r"^\[.+\]$", text)),
        })
    lines.sort(key=lambda l: l["timestamp_s"])
    return lines


def extract_frame(video_path, timestamp):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        fp = Path(f.name)
    try:
        r = subprocess.run([
            "ffmpeg", "-ss", str(timestamp), "-i", str(video_path),
            "-vframes", "1", "-q:v", "4", "-vf", "scale=640:-1",
            "-y", str(fp)
        ], capture_output=True, timeout=15)
        if r.returncode != 0 or not fp.exists() or fp.stat().st_size == 0:
            return None
        with open(fp, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return None
    finally:
        if fp.exists():
            fp.unlink()


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


def mix_audio(audio_path, descriptions, tmpdir):
    clip_files, valid_descs = [], []
    for i, desc in enumerate(descriptions):
        if not desc.get("audioBase64"):
            continue
        cp = tmpdir / f"narrator_{i}.mp3"
        with open(cp, "wb") as f:
            f.write(base64.b64decode(desc["audioBase64"]))
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(cp)],
            capture_output=True, timeout=10
        )
        clip_dur = 3.0
        try:
            clip_dur = float(json.loads(probe.stdout)["streams"][0].get("duration", 3.0))
        except:
            pass
        desc["_clip_dur"] = clip_dur
        clip_files.append(cp)
        valid_descs.append(desc)

    if not clip_files:
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    inputs = ["-i", str(audio_path)]
    for cp in clip_files:
        inputs += ["-i", str(cp)]

    duck_filters = [
        f"between(t,{d['timestamp_s']:.2f},{d['timestamp_s'] + d.get('_clip_dur', 3.0) + 0.3:.2f})"
        for d in valid_descs
    ]
    orig_filter = (
        f"[0:a]volume=enable='{'+'.join(duck_filters)}':volume=0.8,volume=1.0[orig]"
        if duck_filters else "[0:a]volume=1.0[orig]"
    )

    filter_parts = [orig_filter]
    mix_labels   = ["[orig]"]
    for i, (desc, cp) in enumerate(zip(valid_descs, clip_files)):
        delay_ms = int(desc["timestamp_s"] * 1000)
        label    = f"[n{i}]"
        filter_parts.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms},volume=1.5{label}")
        mix_labels.append(label)

    n = len(mix_labels)
    filter_parts.append(f"{''.join(mix_labels)}amix=inputs={n}:duration=first:normalize=0[out]")

    output_path = tmpdir / "combined.mp3"
    cmd = ["ffmpeg", *inputs, "-filter_complex", ";".join(filter_parts),
           "-map", "[out]", "-ar", "44100", "-ab", "128k", "-y", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, timeout=180)

    if result.returncode != 0 or not output_path.exists():
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    with open(output_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def fmt_time(s):
    return f"{int(s//60)}:{int(s%60):02d}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    has_cookies = bool(os.environ.get("YOUTUBE_COOKIES")) or os.path.exists("cookies.txt")
    key_ok = DEFAULT_GEMINI_KEY != "AIzaSyCZjyfA4vqTEP5OXsxb5oPKM8H-LVKtnms"
    print("\n👁  KindEyes Backend (v6)")
    print("━" * 40)
    print(f"Port:       {port}")
    print(f"Cookies:    {'configured' if has_cookies else 'NOT configured'}")
    print(f"Gemini key: {'configured' if key_ok else 'NOT SET'}")
    print("━" * 40 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
