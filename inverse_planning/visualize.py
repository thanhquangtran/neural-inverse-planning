from __future__ import annotations

from html import escape

from inverse_planning.simulate import Trajectory
from inverse_planning.task import GridworldTask, Location

GOAL_COLORS = (
    "#D41159",
    "#FFC20A",
    "#1A85FF",
    "#007D68",
    "#785EF0",
    "#D55E00",
)


def _cell_origin(row: int, col: int, cell_size: int) -> tuple[int, int]:
    return col * cell_size, row * cell_size


def _circle(loc: Location, cell_size: int, color: str, radius_scale: float = 0.25, stroke: str = "none") -> str:
    x, y = _cell_origin(loc[0], loc[1], cell_size)
    cx = x + cell_size / 2
    cy = y + cell_size / 2
    r = cell_size * radius_scale
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" stroke="{stroke}" stroke-width="2" />'


def _agent_icon(loc: Location, cell_size: int, color: str = "#111") -> str:
    x, y = _cell_origin(loc[0], loc[1], cell_size)
    cx = x + cell_size / 2
    cy = y + cell_size / 2
    head_r = cell_size * 0.12
    body_top = cy - cell_size * 0.02
    body_bottom = cy + cell_size * 0.18
    arm_y = cy + cell_size * 0.02
    leg_y = cy + cell_size * 0.32
    stroke_w = max(2, cell_size // 16)
    return (
        f'<g stroke="{color}" stroke-width="{stroke_w}" stroke-linecap="round" stroke-linejoin="round" fill="none">'
        f'<circle cx="{cx}" cy="{cy - cell_size * 0.20}" r="{head_r}" fill="#fffdf7" />'
        f'<line x1="{cx}" y1="{body_top}" x2="{cx}" y2="{body_bottom}" />'
        f'<line x1="{cx - cell_size * 0.16}" y1="{arm_y}" x2="{cx + cell_size * 0.16}" y2="{arm_y}" />'
        f'<line x1="{cx}" y1="{body_bottom}" x2="{cx - cell_size * 0.14}" y2="{leg_y}" />'
        f'<line x1="{cx}" y1="{body_bottom}" x2="{cx + cell_size * 0.14}" y2="{leg_y}" />'
        "</g>"
    )


def _polyline(points: list[Location], cell_size: int, color: str) -> str:
    if not points:
        return ""
    coords = []
    for row, col in points:
        x, y = _cell_origin(row, col, cell_size)
        coords.append(f"{x + cell_size / 2},{y + cell_size / 2}")
    joined = " ".join(coords)
    return (
        f'<polyline points="{joined}" fill="none" stroke="{color}" '
        f'stroke-width="{max(2, cell_size // 8)}" stroke-linecap="round" stroke-linejoin="round" />'
    )


def render_gridworld_svg(
    task: GridworldTask,
    trajectory: Trajectory | None = None,
    upto_step: int | None = None,
    cell_size: int = 64,
    title: str | None = None,
) -> str:
    h, w = task.shape
    width = w * cell_size
    height = h * cell_size
    parts: list[str] = []

    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + (36 if title else 0)}" viewBox="0 0 {width} {height + (36 if title else 0)}">')
    if title:
        parts.append(
            f'<text x="{width / 2}" y="24" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="18" fill="#222">{escape(title)}</text>'
        )
    y_offset = 36 if title else 0
    parts.append(f'<g transform="translate(0,{y_offset})">')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f7f4ea" stroke="#111" stroke-width="2" />')

    for row in range(h):
        for col in range(w):
            x, y = _cell_origin(row, col, cell_size)
            fill = "#1f1f1f" if task.grid[row, col] else "#fffdf7"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'fill="{fill}" stroke="#c9c2b8" stroke-width="1" />'
            )

    for idx, goal in enumerate(task.goal_locs):
        color = GOAL_COLORS[idx % len(GOAL_COLORS)]
        x, y = _cell_origin(goal[0], goal[1], cell_size)
        size = cell_size * 0.28
        cx = x + cell_size / 2
        cy = y + cell_size / 2
        points = [
            f"{cx},{cy - size}",
            f"{cx + size},{cy}",
            f"{cx},{cy + size}",
            f"{cx - size},{cy}",
        ]
        parts.append(f'<polygon points="{" ".join(points)}" fill="{color}" stroke="#222" stroke-width="2" />')

    start = task.init_loc
    x, y = _cell_origin(start[0], start[1], cell_size)
    parts.append(
        f'<rect x="{x + cell_size * 0.2}" y="{y + cell_size * 0.2}" '
        f'width="{cell_size * 0.6}" height="{cell_size * 0.6}" fill="none" '
        f'stroke="#444" stroke-width="3" stroke-dasharray="6 4" />'
    )

    if trajectory is not None:
        if upto_step is None:
            upto_step = len(trajectory.actions)
        upto_step = max(0, min(upto_step, len(trajectory.actions)))
        path_points = trajectory.positions[: upto_step + 1]
        parts.append(_polyline(path_points, cell_size, "#111"))
        for pos in path_points[:-1]:
            parts.append(_circle(pos, cell_size, "#111", radius_scale=0.10))
        current = path_points[-1]
        parts.append(_circle(current, cell_size, "#f4efe4", radius_scale=0.28, stroke="#111"))
        parts.append(_agent_icon(current, cell_size, color="#111"))

    parts.append("</g></svg>")
    return "".join(parts)


def render_trajectory_frames_html(
    task: GridworldTask,
    trajectory: Trajectory,
    columns: int = 4,
    cell_size: int = 48,
) -> str:
    frames: list[str] = []
    total_steps = len(trajectory.actions)
    for step in range(total_steps + 1):
        frame = render_gridworld_svg(
            task,
            trajectory=trajectory,
            upto_step=step,
            cell_size=cell_size,
            title=f"t = {step}",
        )
        frames.append(
            '<div style="padding:8px; background:#faf7f0; border:1px solid #ddd;">'
            f"{frame}</div>"
        )
    return (
        '<div style="display:grid; gap:12px; '
        f'grid-template-columns: repeat({columns}, minmax(0, 1fr));">'
        + "".join(frames)
        + "</div>"
    )
