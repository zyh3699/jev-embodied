"""Render the three phase-two LIBERO modes on one shared wall clock."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

try:
    import imageio_ffmpeg
except ImportError:  # The phase2 extra installs this; a system ffmpeg remains a useful fallback.
    imageio_ffmpeg = None


PANEL_WIDTH, WIDTH, HEIGHT = 480, 1440, 820
BG, PANEL, TEXT, MUTED = "#0d1b24", "#172a36", "#edf3ef", "#9fb2bd"
COLORS = {"pi05": "#8ecae6", "vlm-jev-triggered": "#b6d58f", "vlm-jev-dense": "#c7a8e8"}
LABELS = {"pi05": "π0.5 直接动作", "vlm-jev-triggered": "按需 VLM + Jev", "vlm-jev-dense": "逐轮 VLM + Jev"}


def font(size):
    for path in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.truetype("DejaVuSans.ttf", size)


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
    if len(episodes) != 3 or {episode.row["mode"] for episode in episodes} != set(LABELS):
        raise ValueError("Expected one pi05, one triggered, and one dense episode")
    episodes.sort(key=lambda episode: tuple(LABELS).index(episode.row["mode"]))
    fingerprints = {episode.row.get("initial_fingerprint") for episode in episodes}
    cases = {json.dumps(episode.row["case"], sort_keys=True) for episode in episodes}
    if len(fingerprints) != 1 or len(cases) != 1:
        raise ValueError("Episodes are not a paired reset of the same case")
    output.mkdir(parents=True, exist_ok=False)
    end = max(episode.duration for episode in episodes)
    count = math.ceil(end / speed * fps) + fps * 2
    video = output / "comparison.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe() if imageio_ffmpeg else shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install the phase2 extra or make ffmpeg available on PATH")
    encoder = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "-",
        "-an", "-vcodec", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE)
    try:
        for frame_id in range(count):
            elapsed = min(end, frame_id / fps * speed)
            canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
            draw = ImageDraw.Draw(canvas)
            draw.text((22, 14), "jev-embodied · 二阶段三路线配对实验", font=FONTS[28], fill=TEXT)
            draw.text((22, 55), f"共同墙钟 · {speed:g}× · {elapsed:.1f}/{end:.1f}s · 包含模型等待",
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
                    status = "成功" if episode.row["success"] else episode.row["status"]
                elif active:
                    status = "等待本地 VLM" if active["provider"] == "chat" else "等待 Jev"
                else:
                    status = "执行动作"
                vlm_calls = sum(call["provider"] == "chat" for call in calls)
                jev_calls = sum(call["provider"] == "jev" for call in calls)
                draw.text((x + 14, 137), f"step {frame['step']} · VLM {vlm_calls} · Jev {jev_calls}",
                          font=FONTS[15], fill=MUTED)
                draw.text((x + 14, 164), status, font=FONTS[18], fill="#f2ce82" if active else color)
                external = episode.image(frame["images"]["external"]["path"]).resize((432, 432))
                canvas.paste(external, (x + 14, 198))
                wrist = episode.image(frame["images"]["wrist"]["path"]).resize((122, 122))
                canvas.paste(wrist, (x + 324, 213))
                draw.text((x + 328, 342), "腕部相机", font=FONTS[13], fill=MUTED)
                if decision:
                    if mode == "pi05":
                        headline = f"动作块 {decision['executed_chunk_length']}/{decision['chunk_length']}"
                        detail = f"推理 {decision['inference_latency_ms']:.0f} ms"
                    else:
                        headline = f"Jev: {decision['selection']}"
                        confidence = decision.get("confidence", {})
                        detail = (f"max={confidence.get('max_probability', 0):.1%} · "
                                  f"margin={confidence.get('top_two_margin', 0):.1%}")
                    draw.text((x + 14, 646), headline, font=FONTS[18], fill=color)
                    draw.text((x + 14, 674), detail, font=FONTS[13], fill=MUTED)
                    if mode != "pi05":
                        plan = decision.get("vlm_plan", {})
                        trigger = ", ".join(decision.get("vlm_trigger_reasons", [])) or "复用缓存计划"
                        wrap(draw, f"阶段：{plan.get('phase', '—')} · {plan.get('summary', '—')}",
                             x + 14, 700, 430, size=15, lines=2)
                        wrap(draw, "触发：" + trigger, x + 14, 746, 430, size=13, color=MUTED, lines=1)
                else:
                    draw.text((x + 14, 646), "等待首次决策", font=FONTS[18], fill=MUTED)
            draw.text((22, 798), "同任务 · 同初始状态 · 官方成功判定 · 原始双相机观测", font=FONTS[13], fill=MUTED)
            if frame_id == 0:
                canvas.save(output / "poster.png")
            if frame_id == count - 1:
                canvas.save(output / "final.png")
            encoder.stdin.write(canvas.tobytes())
    finally:
        encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("Video encoder failed")
    metadata = {"speed": speed, "fps": fps, "wall_seconds": end,
                "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "episodes": [{"mode": episode.row["mode"], "path": str(episode.root / "episode.json"),
                              "success": episode.row["success"]} for episode in episodes]}
    (output / "media.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"video": str(video), "wall_seconds": end}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=8.)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.speed <= 64 or not 5 <= args.fps <= 30:
        parser.error("speed must be 1..64 and fps must be 5..30")
    render(args.episodes, args.output, args.speed, args.fps)
