"""Render two recorded Panda episodes on one shared wall-clock timeline.

Model waits are reconstructed from the recorded per-call latency. The small
remainder between total wall time and recorded model latency is assigned to the
recorded qpos frames, so both lanes finish at their measured wall time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from embodied_jev.physics import ASSETS, build_scene


WIDTH, HEIGHT, FPS = 1280, 720, 10
BG, INK, MUTED, LINE = "#f3f6f3", "#173b32", "#65756e", "#d8e2dc"
GREEN, BLUE, AMBER = "#23765d", "#4268a8", "#b57918"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size):
    for path in ("/System/Library/Fonts/STHeiti Medium.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.truetype("DejaVuSans.ttf", size)


FONTS = {n: font(n) for n in (15, 18, 22, 28)}


class Lane:
    def __init__(self, path, label, color):
        self.path, self.label, self.color = path, label, color
        self.episode = json.loads(path.read_text())
        episode = self.episode
        if episode.get("format") != "embodied-jev-episode-v1" or not episode.get("frames"):
            raise ValueError(f"Not a recorded Panda episode: {path}")
        xml, initial_target, _ = build_scene(episode["task"], episode["seed"], episode["scene_config"])
        scene_hash = hashlib.sha256(xml.replace(str(ASSETS), "ASSETS").encode()).hexdigest()
        if scene_hash != episode["scene_hash"]:
            raise ValueError(f"Scene source differs from recording: {path}")
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=560, height=420)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [.43, 0, .035]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 1., 180., -65.
        self.initial_target = np.asarray(initial_target)
        self.target_ids = [i for i in range(self.model.ngeom)
                           if self.model.geom(i).name == "support" or
                           (self.model.geom(i).name or "").startswith("tray_wall_")]
        self.target_positions = self.model.geom_pos[self.target_ids].copy()
        self.frames = episode["frames"]
        self.history = {int(row["cycle"]): row for row in episode["history"]}
        self.wall = float(episode["wall_seconds"])
        self.timeline = self._timeline()

    def _timeline(self):
        waits = {}
        for cycle, row in self.history.items():
            waits[cycle] = sum(float(item.get("latency_ms", 0)) / 1000
                               for item in (row.get("intent", {}), row.get("decision", {}))
                               if item.get("model_call"))
        model_time = sum(waits.values())
        motion_time = self.wall - model_time
        if motion_time < 0:
            raise ValueError("Recorded call latency exceeds episode wall time")
        movable = [frame for frame in self.frames if int(frame.get("cycle", 0)) > 0]
        per_frame = motion_time / max(1, len(movable))
        timeline, elapsed, index = [], 0., 0
        for cycle in sorted(self.history):
            cycle_frames = [i for i, f in enumerate(self.frames) if int(f.get("cycle", 0)) == cycle]
            timeline.append((elapsed, elapsed + waits[cycle], index, "等待模型", cycle))
            elapsed += waits[cycle]
            for index in cycle_frames:
                timeline.append((elapsed, elapsed + per_frame, index, "执行动作", cycle))
                elapsed += per_frame
        # Floating-point normalization makes the last event end exactly at wall_seconds.
        if timeline:
            a, _, i, state, cycle = timeline[-1]
            timeline[-1] = (a, self.wall, i, state, cycle)
        return timeline

    def state(self, elapsed):
        if elapsed >= self.wall:
            return len(self.frames) - 1, "成功", max(self.history)
        for start, end, index, state, cycle in self.timeline:
            if start <= elapsed < end:
                return index, state, cycle
        return 0, "准备", 0

    def scene(self, index):
        frame = self.frames[index]
        self.data.qpos[:] = frame["qpos"]
        self.data.time = frame["time"]
        target = np.asarray(frame.get("evaluation_target", self.initial_target))
        self.model.geom_pos[self.target_ids] = self.target_positions + target - self.initial_target
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.camera)
        return Image.fromarray(self.renderer.render().copy())

    def close(self):
        self.renderer.close()


def render(jev_path, gpt_path, output, speed, overwrite):
    outputs = {ext: output.with_suffix(ext) for ext in (".mp4", ".gif", ".png", ".json")}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Output exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    lanes = [Lane(jev_path, "Jev", GREEN), Lane(gpt_path, "GPT-6 Astra", BLUE)]
    maximum = max(lane.wall for lane in lanes)
    duration = maximum / speed
    frame_count = round(duration * FPS) + FPS * 2

    def frame(output_time):
        wall_time = min(output_time * speed, maximum)
        image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, WIDTH, 76), fill=INK)
        draw.text((24, 17), "Panda · 同一墙钟时间轴", font=FONTS[28], fill="white")
        draw.text((860, 23), f"实验时间 {wall_time:05.1f} s  ·  {speed:g}× 播放", font=FONTS[18], fill="white")
        for col, lane in enumerate(lanes):
            x = 24 + col * 628
            idx, state, cycle = lane.state(wall_time)
            draw.rounded_rectangle((x, 94, x + 604, 676), radius=14, fill="white", outline=LINE)
            draw.text((x + 18, 112), lane.label, font=FONTS[28], fill=lane.color)
            draw.text((x + 185, 118), f"实测完成时间 {lane.wall:.2f} s", font=FONTS[18], fill=INK)
            state_color = AMBER if state == "等待模型" else lane.color
            draw.rounded_rectangle((x + 455, 109, x + 582, 143), radius=7, fill=state_color)
            tw = FONTS[18].getlength(state)
            draw.text((x + 455 + (127 - tw) / 2, 115), state, font=FONTS[18], fill="white")
            image.paste(lane.scene(idx), (x + 22, 158))
            row = lane.history.get(cycle, {})
            label = row.get("label", "等待首个决策" if cycle == 0 else "任务完成")
            draw.text((x + 22, 590), f"阶段 {cycle}/8 · {label}", font=FONTS[18], fill=INK)
            progress = min(wall_time / lane.wall, 1.)
            draw.rounded_rectangle((x + 22, 630, x + 582, 650), radius=8, fill="#e8eeea")
            draw.rounded_rectangle((x + 22, 630, x + 22 + 560 * progress, 650), radius=8, fill=lane.color)
            shown = min(wall_time, lane.wall)
            draw.text((x + 22, 654), f"{shown:.1f} / {lane.wall:.1f} s", font=FONTS[15], fill=MUTED)
        draw.text((24, 694), "等待区间来自每次 API 的实测延迟；动作来自保存的真实 qpos；两侧使用同一时间轴。",
                  font=FONTS[15], fill=MUTED)
        return image

    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
               "-an", "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(outputs[".mp4"])]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    poster = None
    try:
        for n in range(frame_count):
            encoder.stdin.write(frame(n / FPS).tobytes())
        poster = frame(duration)
    finally:
        encoder.stdin.close()
        for lane in lanes:
            lane.close()
    if encoder.wait():
        raise RuntimeError("Video encoder failed")
    poster.save(outputs[".png"])
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(outputs[".mp4"]),
                    "-vf", "fps=10,scale=960:-1:flags=lanczos", "-loop", "0", str(outputs[".gif"])], check=True)
    metadata = {
        "timing": "Shared recorded wall-clock timeline; output uniformly accelerated.",
        "playback_speed": speed, "fps": FPS, "duration_seconds": frame_count / FPS,
        "model_calls_during_render": 0, "physics_steps_during_render": 0,
        "episodes": [{"label": lane.label, "path": str(lane.path), "sha256": digest(lane.path),
                      "wall_seconds": lane.wall} for lane in lanes],
        "artifacts": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                      for ext, p in outputs.items() if ext != ".json"},
    }
    outputs[".json"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jev", type=Path, required=True)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.speed <= 32:
        parser.error("--speed must be 1..32")
    render(args.jev, args.gpt, args.output, args.speed, args.overwrite)


if __name__ == "__main__":
    main()
