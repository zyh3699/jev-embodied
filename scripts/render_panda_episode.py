"""Render a recorded built-in Panda episode to verified MP4, GIF and poster.

The renderer only replays saved qpos frames. It never calls a model or advances
physics, so the media is an auditable visualization of the recorded experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from embodied_jev.physics import ASSETS, build_scene

WIDTH, HEIGHT, FPS = 1280, 720, 15
INK, MUTED, GREEN, RED, BG = "#173b32", "#65756e", "#23765d", "#ad4a43", "#f3f6f3"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def font(size):
    for path in ("/System/Library/Fonts/STHeiti Medium.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.truetype("DejaVuSans.ttf", size)


FONTS = {n: font(n) for n in (16, 19, 23, 30)}


def render(episode_path, output, title, gif_speed):
    episode = json.loads(episode_path.read_text())
    if episode.get("format") != "embodied-jev-episode-v1" or not episode.get("frames"):
        raise ValueError("Expected a recorded jev-embodied Panda episode")
    xml, initial_target, _ = build_scene(episode["task"], episode["seed"], episode["scene_config"])
    if digest(xml.replace(str(ASSETS), "ASSETS").encode()) != episode["scene_hash"]:
        raise ValueError("Scene source differs from the recorded episode")
    model, data = mujoco.MjModel.from_xml_string(xml), None
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(800, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(600, model.vis.global_.offheight)
    target_ids = [i for i in range(model.ngeom) if model.geom(i).name == "support" or
                  (model.geom(i).name or "").startswith("tray_wall_")]
    target_positions = model.geom_pos[target_ids].copy()
    renderer = mujoco.Renderer(model, width=800, height=600)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.43, 0, .035]
    camera.distance, camera.azimuth, camera.elevation = 1., 180., -65.
    history = {row["cycle"]: row for row in episode.get("history", [])}
    source = episode["frames"]
    selected = [source[0]]
    for cycle in sorted({f["cycle"] for f in source}):
        rows = [f for f in source if f["cycle"] == cycle]
        selected.extend(rows[round(i / 11 * (len(rows) - 1))] for i in range(12))
    selected.extend([source[-1]] * (FPS * 2))

    def draw_frame(frame):
        data.qpos[:] = frame["qpos"]
        data.time = frame["time"]
        target = np.asarray(frame.get("evaluation_target", initial_target))
        model.geom_pos[target_ids] = target_positions + target - initial_target
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        scene = Image.fromarray(renderer.render().copy())
        image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        image.paste(scene, (20, 92))
        d = ImageDraw.Draw(image)
        d.rectangle((0, 0, WIDTH, 72), fill=INK)
        d.text((24, 18), "jev-embodied", font=FONTS[30], fill="white")
        d.text((330, 22), title, font=FONTS[23], fill="white")
        cycle = int(frame.get("cycle", 0))
        step = history.get(cycle)
        status_color = GREEN if episode.get("success") else RED
        d.rounded_rectangle((844, 96, 1254, 684), radius=14, fill="white", outline="#d8e2dc")
        d.text((872, 124), episode.get("model") or episode.get("provider", "unknown"), font=FONTS[23], fill=INK)
        d.text((872, 164), f"任务：{episode['task']}  ·  seed {episode['seed']}", font=FONTS[19], fill=MUTED)
        d.text((872, 202), f"控制：{episode['control_mode']}  ·  周期 {cycle}/{len(history)}", font=FONTS[19], fill=MUTED)
        if step:
            d.text((872, 266), "当前阶段", font=FONTS[16], fill=MUTED)
            d.text((872, 296), step.get("label") or step.get("phase", "—"), font=FONTS[30], fill=INK)
            action = step.get("action", {})
            d.text((872, 354), f"动作：{action.get('id', '—')}", font=FONTS[19], fill=GREEN)
            decision = step.get("decision", {})
            latency = decision.get("latency_ms")
            d.text((872, 392), f"本次模型等待：{latency / 1000:.2f} s" if isinstance(latency, (int, float)) else "本次模型等待：—",
                   font=FONTS[19], fill=MUTED)
            after = step.get("after", {})
            d.text((872, 438), "持物：" + ("是" if after.get("held") else "否"), font=FONTS[19], fill=INK)
            d.text((872, 474), f"最大抬升：{after.get('max_lift_m', 0):.3f} m", font=FONTS[19], fill=INK)
        terminal = frame is source[-1] or frame == source[-1]
        if terminal:
            label = "物理成功" if episode.get("success") else "实验未完成"
            d.rounded_rectangle((872, 564, 1218, 624), radius=9, fill=status_color)
            d.text((897, 580), label, font=FONTS[23], fill="white")
        d.text((24, 696), "由保存的真实 qpos 离线重绘 · 无模型调用 · 无补帧轨迹", font=FONTS[16], fill=MUTED)
        return image

    outputs = {ext: output.with_suffix(ext) for ext in (".mp4", ".gif", ".png", ".json")}
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(p.exists() for p in outputs.values()):
        raise FileExistsError("Output exists")
    poster = draw_frame(source[-1])
    encoder = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(outputs[".mp4"])], stdin=subprocess.PIPE)
    try:
        for frame in selected:
            encoder.stdin.write(draw_frame(frame).tobytes())
    finally:
        encoder.stdin.close()
        renderer.close()
    if encoder.wait():
        raise RuntimeError("Video encoder failed")
    poster.save(outputs[".png"])
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(outputs[".mp4"]),
        "-vf", f"setpts=PTS/{gif_speed},fps=12,scale=960:-1:flags=lanczos", "-loop", "0", str(outputs[".gif"])], check=True)
    metadata = {"episode": str(episode_path), "episode_sha256": digest(episode_path.read_bytes()),
        "model": episode.get("model"), "success": episode.get("success"), "frames": len(selected), "fps": FPS,
        "model_calls_during_render": 0, "physics_steps_during_render": 0,
        "artifacts": {p.name: {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
                      for ext, p in outputs.items() if ext != ".json"}}
    outputs[".json"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"outputs": [str(p) for p in outputs.values()]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--gif-speed", type=float, default=4)
    args = parser.parse_args()
    if not math.isfinite(args.gif_speed) or not 1 <= args.gif_speed <= 16:
        parser.error("gif-speed must be 1..16")
    render(args.episode, args.output, args.title, args.gif_speed)


if __name__ == "__main__":
    main()
