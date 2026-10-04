"""Render the three phase-two LIBERO modes on one shared wall clock."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

try:
    import imageio_ffmpeg
except ImportError:  # The phase2 extra installs this; a system ffmpeg remains a useful fallback.
    imageio_ffmpeg = None


PANEL_WIDTH, HEIGHT = 480, 820
BG, PANEL, TEXT, MUTED = "#0d1b24", "#172a36", "#edf3ef", "#9fb2bd"
COLORS = {"pi05": "#8ecae6", "vlm-jev-triggered": "#b6d58f", "vlm-jev-dense": "#c7a8e8",
          "vlm-chunk-no-jev": "#f4a261"}
LABELS = {"pi05": "pi0.5 direct", "vlm-jev-triggered": "Triggered VLM + Jev",
          "vlm-jev-dense": "Dense VLM + Jev", "vlm-chunk-no-jev": "Triggered VLM, no Jev"}


def font(size):
    for path in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


FONTS = {size: font(size) for size in (13, 15, 18, 22, 28)}


def wrap(draw, value, x, y, width, *, size=15, color=TEXT, lines=2):
    current, row = "", 0
    tokens = re.findall(r"\S+\s*", str(value)) if " " in str(value) else list(str(value))
    for token in tokens:
        if current and draw.textlength(current + token, font=FONTS[size]) > width:
            draw.text((x, y + row * (size + 5)), current, font=FONTS[size], fill=color)
            row += 1
            current = ""
            if row >= lines:
                return
        current += token
    if current and row < lines:
        draw.text((x, y + row * (size + 5)), current, font=FONTS[size], fill=color)


class Episode:
    def __init__(self, root):
        self.root = root
        self.row = json.loads((root / "episode.json").read_text())
        self.frames = self.row["frames"]
        if not self.frames:
            raise ValueError(f"Episode has no recorded frames: {root}")
        self.times = [frame["wall_seconds"] for frame in self.frames]
        self.duration = self.row.get("rollout_seconds", self.times[-1])
        self.cache = {}

    def image(self, relative):
        if relative not in self.cache:
            self.cache[relative] = Image.open(self.root / relative).convert("RGB").rotate(180)
        return self.cache[relative]

    def state(self, elapsed):
        frame_index = max(0, bisect.bisect_right(self.times, elapsed) - 1)
        completed = [item for item in self.row["decisions"] if item["inference_end_seconds"] <= elapsed]
        decision = completed[-1] if completed else None
        active = [call for call in self.row["api_calls"]
                  if call["start_seconds"] <= elapsed < call["end_seconds"]]
        calls = [call for call in self.row["api_calls"] if call["end_seconds"] <= elapsed]
        return self.frames[frame_index], decision, active[0] if active else None, calls


def render(episode_paths, output, speed=8., fps=12):
    episodes = [Episode(path) for path in episode_paths]
    expected = {"pi05", "vlm-jev-triggered", "vlm-jev-dense"}
    modes = {episode.row["mode"] for episode in episodes}
    if modes not in (expected, set(LABELS)) or len(episodes) != len(modes):
        raise ValueError("Expected the three main modes, optionally plus the no-Jev ablation")
    episodes.sort(key=lambda episode: tuple(LABELS).index(episode.row["mode"]))
    width = PANEL_WIDTH * len(episodes)
    fingerprints = {episode.row.get("initial_fingerprint") for episode in episodes}
    cases = {json.dumps(episode.row["case"], sort_keys=True) for episode in episodes}
    if len(fingerprints) != 1 or len(cases) != 1:
        raise ValueError("Episodes are not a paired reset of the same case")
    output.mkdir(parents=True, exist_ok=False)
    end = max(episode.duration for episode in episodes)
    count = math.ceil(end / speed * fps) + fps * 2
    video = output / "comparison.mp4"
    gif = output / "comparison.gif"
    sample_indices = {round(index * (count - 1) / 59) for index in range(60)}
    gif_frames = []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe() if imageio_ffmpeg else shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install the phase2 extra or make ffmpeg available on PATH")
    descriptor, temporary_name = tempfile.mkstemp(prefix="phase2-comparison-", suffix=".mp4")
    os.close(descriptor)
    temporary_video = Path(temporary_name)
    encoder = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{HEIGHT}", "-r", str(fps), "-i", "-",
        "-an", "-vcodec", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary_video)], stdin=subprocess.PIPE)
    try:
        for frame_id in range(count):
            elapsed = min(end, frame_id / fps * speed)
            canvas = Image.new("RGB", (width, HEIGHT), BG)
            draw = ImageDraw.Draw(canvas)
            draw.text((22, 14), "jev-embodied · phase-two paired evaluation", font=FONTS[28], fill=TEXT)
            draw.text((22, 55), f"shared wall clock · {speed:g}x · {elapsed:.1f}/{end:.1f}s · model waits included",
                      font=FONTS[15], fill=MUTED)
            for column, episode in enumerate(episodes):
                x = column * PANEL_WIDTH + 10
                mode = episode.row["mode"]
                color = COLORS[mode]
                frame, decision, active, calls = episode.state(elapsed)
                draw.rounded_rectangle((x, 88, x + 460, 790), radius=9, fill=PANEL)
                draw.text((x + 14, 102), LABELS[mode], font=FONTS[22], fill=color)
                completed = elapsed >= episode.duration
                if completed:
                    status = "success" if episode.row["success"] else episode.row["status"]
                elif active:
                    status = "waiting for local VLM" if active["provider"] == "chat" else "waiting for Jev"
                else:
                    status = "executing action"
                vlm_calls = sum(call["provider"] == "chat" for call in calls)
                jev_calls = sum(call["provider"] == "jev" for call in calls)
                draw.text((x + 14, 137), f"step {frame['step']} · VLM {vlm_calls} · Jev {jev_calls}",
                          font=FONTS[15], fill=MUTED)
                draw.text((x + 14, 164), status, font=FONTS[18], fill="#f2ce82" if active else color)
                external = episode.image(frame["images"]["external"]["path"]).resize((432, 432))
                canvas.paste(external, (x + 14, 198))
                wrist = episode.image(frame["images"]["wrist"]["path"]).resize((122, 122))
                canvas.paste(wrist, (x + 324, 213))
                draw.text((x + 328, 342), "wrist camera", font=FONTS[13], fill=MUTED)
                if decision:
                    if mode == "pi05":
                        headline = f"action chunk {decision['executed_chunk_length']}/{decision['chunk_length']}"
                        detail = f"inference {decision['inference_latency_ms']:.0f} ms"
                    else:
                        selected = decision["selection"]
                        family = decision.get("candidate_chunks", {}).get(selected, {}).get("family", selected)
                        headline = (("deterministic chunk: " if mode == "vlm-chunk-no-jev" else "Jev chunk: ")
                                    + str(family))
                        confidence = decision.get("confidence") or {}
                        detail = ("nominal candidate; no Jev request" if mode == "vlm-chunk-no-jev" else
                                  f"max={confidence.get('max_probability', 0):.1%} · "
                                  f"margin={confidence.get('top_two_margin', 0):.1%}")
                    draw.text((x + 14, 646), headline, font=FONTS[18], fill=color)
                    draw.text((x + 14, 674), detail, font=FONTS[13], fill=MUTED)
                    if mode != "pi05":
                        plan = decision.get("vlm_keyframe", {})
                        trigger = ", ".join(decision.get("vlm_trigger_reasons", [])) or "cached plan reused"
                        wrap(draw, f"phase: {plan.get('phase', '-')} · {plan.get('summary', '-')}",
                             x + 14, 700, 430, size=15, lines=2)
                        wrap(draw, "trigger: " + trigger, x + 14, 746, 430, size=13, color=MUTED, lines=1)
                else:
                    draw.text((x + 14, 646), "waiting for first decision", font=FONTS[18], fill=MUTED)
            draw.text((22, 798), "same task · paired reset · official success predicate · raw dual-camera observations",
                      font=FONTS[13], fill=MUTED)
            if frame_id == 0:
                canvas.save(output / "poster.png")
            if frame_id == count - 1:
                canvas.save(output / "final.png")
            if frame_id in sample_indices:
                gif_frames.append(canvas.resize((width // 2, HEIGHT // 2), Image.Resampling.LANCZOS))
            encoder.stdin.write(canvas.tobytes())
    finally:
        encoder.stdin.close()
    if encoder.wait():
        temporary_video.unlink(missing_ok=True)
        raise RuntimeError("Video encoder failed")
    try:
        shutil.copy2(temporary_video, video)
    finally:
        temporary_video.unlink(missing_ok=True)
    gif_frames[0].save(gif, save_all=True, append_images=gif_frames[1:], duration=130,
                       loop=0, optimize=True, disposal=2)
    metadata = {"speed": speed, "fps": fps, "wall_seconds": end,
                "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "gif_sha256": hashlib.sha256(gif.read_bytes()).hexdigest(),
                "gif_frames": len(gif_frames),
                "episodes": [{"mode": episode.row["mode"], "path": str(episode.root / "episode.json"),
                              "success": episode.row["success"]} for episode in episodes]}
    (output / "media.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"video": str(video), "wall_seconds": end}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=8.)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.speed <= 64 or not 5 <= args.fps <= 30:
        parser.error("speed must be 1..64 and fps must be 5..30")
    render(args.episodes, args.output, args.speed, args.fps)
