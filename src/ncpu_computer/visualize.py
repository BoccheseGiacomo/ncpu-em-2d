from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .config import ExperimentConfig
from .tape import TapeLayout, interpret_tape, quantize, validate_symbols


_NEGATIVE = (59, 76, 192)
_NEUTRAL = (245, 245, 245)
_POSITIVE = (180, 4, 38)


def evolution_phase(step: int, free_steps: int, supervision_steps: int) -> str:
    if any(type(value) is not int for value in (step, free_steps, supervision_steps)):
        raise TypeError("evolution times must be integers")
    if step < 0 or free_steps < 0 or supervision_steps < 1:
        raise ValueError("evolution times are invalid")
    if step <= free_steps:
        return "Free evolution"
    if step <= free_steps + supervision_steps:
        return "Supervision window"
    return "Beyond training window"


def _validate_rollout(rollout: torch.Tensor) -> None:
    if rollout.ndim != 4 or min(rollout.shape) < 1:
        raise ValueError("rollout must have shape (time, channels, height, width)")
    if not torch.is_floating_point(rollout) or not torch.isfinite(rollout).all():
        raise ValueError("rollout must contain finite floating-point states")


def _colorize(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp(-1.0, 1.0)
    neutral = values.new_tensor(_NEUTRAL)
    negative = values.new_tensor(_NEGATIVE)
    positive = values.new_tensor(_POSITIVE)
    endpoint = torch.where((values < 0).unsqueeze(-1), negative, positive)
    return (
        (neutral + values.abs().unsqueeze(-1) * (endpoint - neutral))
        .round()
        .to(torch.uint8)
    )


def rollout_rgb(rollout: torch.Tensor, io_channel: int) -> torch.Tensor:
    _validate_rollout(rollout)
    if not 0 <= io_channel < rollout.shape[1]:
        raise ValueError("I/O channel index does not exist in rollout")
    return _colorize(rollout[:, io_channel])


def _value_color(value: float) -> tuple[int, int, int]:
    value = max(-1.0, min(1.0, value))
    endpoint = _NEGATIVE if value < 0 else _POSITIVE
    return tuple(
        round(center + abs(value) * (extreme - center))
        for center, extreme in zip(_NEUTRAL, endpoint)
    )


def save_gif(
    rollout: torch.Tensor,
    path: str | Path,
    *,
    layout: TapeLayout,
    config: ExperimentConfig,
    program_tile: torch.Tensor,
    task_name: str,
    input_symbols: str,
    target_symbols: str,
    free_steps: int,
    supervision_steps: int,
    output_mode: str = "single",
    duration_ms: int = 80,
    scale: int = 24,
) -> Path:
    if duration_ms < 1 or scale < 1:
        raise ValueError("duration_ms and scale must be positive")
    if output_mode not in {"single", "multiple"}:
        raise ValueError("output_mode must be 'single' or 'multiple'")
    validate_symbols(input_symbols, allow_empty=True)
    validate_symbols(target_symbols, allow_empty=True)
    if not task_name.strip():
        raise ValueError("task_name cannot be empty")
    if (
        len(input_symbols) > layout.tape_slots
        or len(target_symbols) > layout.tape_slots
    ):
        raise ValueError("input or target exceeds the visualized tape")
    config.validate()
    _validate_rollout(rollout)
    if tuple(rollout.shape[1:]) != (config.model.channels, layout.height, layout.width):
        raise ValueError("rollout dimensions do not match the layout and model")
    expected_tile = (
        config.model.program_channels,
        layout.height,
        config.geometry.stride,
    )
    if tuple(program_tile.shape) != expected_tile:
        raise ValueError(f"program_tile must have shape {expected_tile}")

    states = rollout.detach().cpu()
    frames = rollout_rgb(states, config.model.io_channel)
    tape_values = layout.extract_tape(states[:, config.model.io_channel])
    raw_tapes = tuple(
        "".join({-1: "0", 0: "B", 1: "1"}[value] for value in row)
        for row in quantize(tape_values).tolist()
    )
    tile_images = [
        Image.fromarray(_colorize(channel).cpu().numpy()).resize(
            (config.geometry.stride * scale, layout.height * scale),
            Image.Resampling.NEAREST,
        )
        for channel in program_tile.detach().cpu()
    ]
    images = []
    for step, frame in enumerate(frames):
        grid = Image.fromarray(frame.numpy()).resize(
            (layout.width * scale, layout.height * scale), Image.Resampling.NEAREST
        )
        images.append(
            _annotate_frame(
                grid,
                tile_images,
                step,
                len(frames) - 1,
                raw_tapes[step],
                layout,
                config,
                task_name,
                input_symbols,
                target_symbols,
                free_steps,
                supervision_steps,
                output_mode,
                scale,
            )
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0
    )
    return path


def _annotate_frame(
    frame: Image.Image,
    tile_images: list[Image.Image],
    step: int,
    total_steps: int,
    raw_tape: str,
    layout: TapeLayout,
    config: ExperimentConfig,
    task_name: str,
    input_symbols: str,
    target_symbols: str,
    free_steps: int,
    supervision_steps: int,
    output_mode: str,
    scale: int,
) -> Image.Image:
    left, top, bottom, gap = 18, 110, 38, 22
    tile_width = config.geometry.stride * scale
    tile_stack_height = len(tile_images) * (layout.height * scale + 18)
    content_width = frame.width + gap + tile_width
    content_height = max(frame.height, tile_stack_height)
    image = Image.new(
        "RGB",
        (max(520, content_width + 2 * left), top + content_height + bottom),
        "#161a1f",
    )
    image.paste(frame, (left, top))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    foreground = "#e7ebef"
    interpreted = interpret_tape(
        torch.tensor(
            [1.0 if s == "1" else -1.0 if s == "0" else 0.0 for s in raw_tape]
        ),
        output_mode,
    )
    decoded = " | ".join(interpreted.binary_strings) if interpreted.valid else "invalid"
    phase = evolution_phase(step, free_steps, supervision_steps)
    draw.text(
        (left, 6), f"t = {step} / {total_steps}    {phase}", fill=foreground, font=font
    )
    draw.text((left, 25), f"Raw tape: {raw_tape}", fill=foreground, font=font)
    draw.text((left, 44), f"Decoded: {decoded}", fill=foreground, font=font)
    draw.text(
        (left, 63),
        f"Task: {task_name}    Input: {input_symbols or '<empty>'}    "
        f"Target: {target_symbols or '<empty>'}",
        fill=foreground,
        font=font,
    )
    draw.text((left, 84), "Shared mutable I/O channel", fill=foreground, font=font)
    for index, (row, column) in enumerate(layout.tape_coordinates):
        x, y = left + column * scale, top + row * scale
        draw.rectangle(
            (x, y, x + scale - 1, y + scale - 1),
            outline="#ffcc33",
            width=max(1, scale // 12),
        )
        if scale >= 12:
            draw.text(
                (x + scale // 2, y - 2),
                str(index),
                fill="#ffcc33",
                font=font,
                anchor="ms",
            )
    tile_x = left + frame.width + gap
    tile_y = top
    for channel, tile in enumerate(tile_images):
        draw.text(
            (tile_x, tile_y - 14), f"Program {channel}", fill=foreground, font=font
        )
        image.paste(tile, (tile_x, tile_y))
        tile_y += tile.height + 18
    _draw_scale(draw, left, top + content_height + 7, font, foreground)
    return image


def _draw_scale(draw, left: int, top: int, font, foreground) -> None:
    width, height = 140, 7
    for offset in range(width):
        value = -1.0 + 2.0 * offset / (width - 1)
        draw.line(
            (left + offset, top, left + offset, top + height), fill=_value_color(value)
        )
    draw.rectangle((left, top, left + width - 1, top + height), outline="#8d949c")
    label_y = top + height + 2
    draw.text((left, label_y), "0 (-1)", fill=foreground, font=font)
    draw.text(
        (left + width // 2, label_y), "B (0)", fill=foreground, font=font, anchor="ma"
    )
    draw.text(
        (left + width, label_y), "1 (+1)", fill=foreground, font=font, anchor="ra"
    )
