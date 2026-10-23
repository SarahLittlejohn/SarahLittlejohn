"""Qubit weather: a daily weather map of the IBM Quantum fleet.

Reads calibration data for up to four IBM Quantum processors (no jobs are
submitted, so no QPU time is used), keeps a rolling history per backend and
renders qubit-weather.svg as 2x2 small multiples.

    python qubit_weather.py           # live, needs IBM_QUANTUM_TOKEN
    python qubit_weather.py --demo    # fake Heron snapshots, no token, no history written
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import escape

MAX_BACKENDS = 4
STORM_THRESHOLD = 0.08
STRETCH_LENGTH = 5
HISTORY_DAYS = 60
SPARKLINE_DAYS = 14
MIN_ROW_LENGTH = 3
TWO_QUBIT_GATES = ("cz", "ecr", "cx")
FORECAST_CHANGE_THRESHOLD = 0.03

# One log-scale colour ramp shared by every panel, for both readout and 2q error.
RAMP_MIN = 0.001
RAMP_MAX = 0.2
RAMP_STOPS = (
    (0.0, "#1c5cab"),
    (0.3, "#3987e5"),
    (0.5, "#1baf7a"),
    (0.68, "#eda100"),
    (0.85, "#eb6834"),
    (1.0, "#d03b3b"),
)
RAMP_TICKS = (0.001, 0.003, 0.01, 0.03, 0.1)
MISSING_COLOUR = "#9a9993"

WIDTH = 1000
MARGIN = 32
SIDEBAR_WIDTH = 210
SIDEBAR_GAP = 36
PANEL_GAP = 28
MAX_CELL = 40
SERIF = "Georgia, 'Iowan Old Style', 'Palatino Linotype', 'Book Antiqua', 'Times New Roman', serif"
SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"


@dataclass
class Snapshot:
    backend: str
    num_qubits: int
    gate: str | None
    calibrated: datetime | None
    processor: str | None = None
    t1: dict[int, float] = field(default_factory=dict)
    t2: dict[int, float] = field(default_factory=dict)
    readout: dict[int, float] = field(default_factory=dict)
    edges: dict[tuple[int, int], float | None] = field(default_factory=dict)


@dataclass
class Layout:
    coords: dict[int, tuple[int, int]]
    rows: list[list[int]]
    bad_edges: list[tuple[int, int]]

    @property
    def size(self) -> tuple[int, int]:
        xs = [x for x, _ in self.coords.values()] or [0]
        ys = [y for _, y in self.coords.values()] or [0]
        return max(xs) + 1, max(ys) + 1


@dataclass
class Panel:
    snapshot: Snapshot
    layout: Layout
    summary: dict[str, float | None]
    stretch: tuple[list[int], float] | None
    previous: dict | None
    storms: list[int]


def main() -> None:
    args = parse_args()
    backends = load_demo_backends() if args.demo else load_live_backends()
    snapshots = [read_snapshot(b) for b in backends]
    history_path = Path(args.history)
    history = load_history(history_path)

    if not args.demo:
        for snapshot in snapshots:
            history = record_reading(history, snapshot, summarise(snapshot))
        save_history(history_path, history)

    Path(args.out).write_text(render_fleet(snapshots, history), encoding="utf-8")
    print(f"Wrote {args.out} for {', '.join(s.backend for s in snapshots)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="use fake backends; needs no token and writes no history")
    parser.add_argument("--out", default="qubit-weather.svg")
    parser.add_argument("--history", default="data/history.json")
    return parser.parse_args()


def load_demo_backends() -> list:
    from qiskit_ibm_runtime.fake_provider import FakeFez, FakeKingston, FakeMarrakesh, FakeTorino

    return sorted((FakeFez(), FakeTorino(), FakeKingston(), FakeMarrakesh()), key=lambda b: b.name)


def load_live_backends() -> list:
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        sys.exit("IBM_QUANTUM_TOKEN is not set (use --demo to run without one)")
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=os.environ.get("IBM_QUANTUM_INSTANCE") or None,
    )
    backends = sorted(service.backends(operational=True, simulator=False), key=lambda b: b.name)
    if not backends:
        sys.exit("No operational backends are available to this account")
    return backends[:MAX_BACKENDS]


def read_snapshot(backend) -> Snapshot:
    target = backend.target
    snapshot = Snapshot(
        backend=backend.name,
        num_qubits=target.num_qubits,
        gate=next((g for g in TWO_QUBIT_GATES if g in target.operation_names), None),
        calibrated=calibration_time(backend),
        processor=processor_name(backend),
    )

    for q, props in enumerate(target.qubit_properties or []):
        if props is None:
            continue
        if valid(getattr(props, "t1", None)):
            snapshot.t1[q] = props.t1
        if valid(getattr(props, "t2", None)):
            snapshot.t2[q] = props.t2

    if "measure" in target.operation_names:
        for qargs, props in target["measure"].items():
            if qargs and props is not None and valid(props.error):
                snapshot.readout[qargs[0]] = props.error

    if snapshot.gate:
        for qargs, props in target[snapshot.gate].items():
            if not qargs or len(qargs) != 2:
                continue
            key = tuple(sorted(qargs))
            error = props.error if props is not None and valid(props.error) else None
            current = snapshot.edges.get(key)
            # Directed gates (ecr) appear once per native direction; keep the better one.
            snapshot.edges[key] = error if current is None else (current if error is None else min(current, error))
    return snapshot


def calibration_time(backend) -> datetime | None:
    try:
        props = backend.properties()
        stamp = getattr(props, "last_update_date", None) if props is not None else None
    except Exception:
        return None
    if isinstance(stamp, str):
        stamp = datetime.fromisoformat(stamp)
    if not isinstance(stamp, datetime):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def processor_name(backend) -> str | None:
    try:
        kind = getattr(backend, "processor_type", None) or getattr(backend.configuration(), "processor_type", None)
    except Exception:
        return None
    if not isinstance(kind, dict) or not kind.get("family"):
        return None
    return f"{kind['family']} r{kind['revision']}" if kind.get("revision") else kind["family"]


def valid(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def summarise(snapshot: Snapshot) -> dict[str, float | None]:
    return {
        "t1_us": median_or_none([v * 1e6 for v in snapshot.t1.values()]),
        "t2_us": median_or_none([v * 1e6 for v in snapshot.t2.values()]),
        "readout": median_or_none(list(snapshot.readout.values())),
        "two_q": median_or_none([v for v in snapshot.edges.values() if v is not None]),
    }


def median_or_none(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def load_history(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def record_reading(history: dict[str, list[dict]], snapshot: Snapshot, summary: dict) -> dict[str, list[dict]]:
    calibrated = snapshot.calibrated or datetime.now(timezone.utc)
    entry = {"date": calibrated.date().isoformat(), "calibrated": calibrated.isoformat(), **summary}
    existing = history.get(snapshot.backend, [])
    if any(e["calibrated"] == entry["calibrated"] for e in existing):
        return history
    cutoff = (calibrated.date() - timedelta(days=HISTORY_DAYS)).isoformat()
    kept = [e for e in existing if e["date"] != entry["date"] and e["date"] > cutoff]
    return {**history, snapshot.backend: sorted(kept + [entry], key=lambda e: e["date"])}


def save_history(path: Path, history: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def infer_layout(snapshot: Snapshot) -> Layout:
    """Use the heavy-hex layout if every coupler fits on it, otherwise a square grid."""
    heavy_hex = heavy_hex_layout(snapshot)
    if not heavy_hex.bad_edges:
        return heavy_hex
    square = square_layout(snapshot)
    return square if len(square.bad_edges) < len(heavy_hex.bad_edges) else heavy_hex


def heavy_hex_layout(snapshot: Snapshot) -> Layout:
    """Place qubits on a grid using IBM's numbering.

    Consecutive indices joined by an edge form rows; short runs are bridge
    qubits. Each row is offset so that its bridges sit directly between
    their two neighbours.
    """
    n = snapshot.num_qubits
    adj = adjacency(snapshot)
    runs: list[list[int]] = [[0]] if n else []
    for q in range(1, n):
        if q - 1 in adj[q]:
            runs[-1].append(q)
        else:
            runs.append([q])
    rows = [r for r in runs if len(r) >= MIN_ROW_LENGTH] or runs[:1]
    row_of = {q: i for i, row in enumerate(rows) for q in row}
    pos = {q: j for row in rows for j, q in enumerate(row)}
    bridges = [q for q in range(n) if q not in row_of]

    offsets, row_y = [0], [0]
    for i in range(len(rows) - 1):
        offset, gap = link_rows(i, rows, adj, row_of, pos, bridges, offsets[i])
        offsets.append(offset)
        row_y.append(row_y[i] + gap)

    coords = {q: (offsets[row_of[q]] + pos[q], row_y[row_of[q]]) for q in row_of}
    pending = list(bridges)
    while pending:
        placed = [q for q in pending if place_bridge(q, adj, coords)]
        if not placed:
            # Isolated qubits: park them in a line below the chip.
            bottom = max((y for _, y in coords.values()), default=0) + 2
            for k, q in enumerate(pending):
                coords[q] = (k, bottom)
            break
        pending = [q for q in pending if q not in placed]

    return finish_layout(snapshot, coords, rows)


def adjacency(snapshot: Snapshot) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {q: set() for q in range(snapshot.num_qubits)}
    for a, b in snapshot.edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def link_rows(i, rows, adj, row_of, pos, bridges, offset) -> tuple[int, int]:
    for b in bridges:
        above = [m for m in adj[b] if row_of.get(m) == i]
        below = [m for m in adj[b] if row_of.get(m) == i + 1]
        if above and below:
            return offset + pos[above[0]] - pos[below[0]], 2
    for q in rows[i]:
        for m in adj[q]:
            if row_of.get(m) == i + 1:
                return offset + pos[q] - pos[m], 1
    return offset, 2


def place_bridge(q, adj, coords) -> bool:
    neighbours = sorted((coords[m] for m in adj[q] if m in coords), key=lambda c: c[1])
    if not neighbours:
        return False
    x, y = neighbours[0]
    coords[q] = (x, y + 1)
    return True


def square_layout(snapshot: Snapshot) -> Layout:
    """Row-major square grid; the row width is the most common non-adjacent index gap."""
    n = snapshot.num_qubits
    gaps = Counter(abs(a - b) for a, b in snapshot.edges if abs(a - b) > 1)
    width = gaps.most_common(1)[0][0] if gaps else max(1, math.ceil(math.sqrt(n)))
    coords = {q: (q % width, q // width) for q in range(n)}
    rows = [list(range(start, min(start + width, n))) for start in range(0, n, width)]
    return finish_layout(snapshot, coords, rows)


def finish_layout(snapshot: Snapshot, coords: dict[int, tuple[int, int]], rows: list[list[int]]) -> Layout:
    min_x = min((x for x, _ in coords.values()), default=0)
    coords = {q: (x - min_x, y) for q, (x, y) in coords.items()}
    bad_edges = [
        (a, b) for a, b in snapshot.edges if abs(coords[a][0] - coords[b][0]) + abs(coords[a][1] - coords[b][1]) != 1
    ]
    return Layout(coords=coords, rows=rows, bad_edges=bad_edges)


def best_stretch(snapshot: Snapshot, layout: Layout) -> tuple[list[int], float] | None:
    best = None
    for row in layout.rows:
        for start in range(len(row) - STRETCH_LENGTH + 1):
            chain = row[start : start + STRETCH_LENGTH]
            score = stretch_score(chain, snapshot)
            if score is not None and (best is None or score > best[1]):
                best = (chain, score)
    return best


def stretch_score(chain: list[int], snapshot: Snapshot) -> float | None:
    score = 1.0
    for q in chain:
        if q not in snapshot.readout:
            return None
        score *= 1 - snapshot.readout[q]
    for a, b in zip(chain, chain[1:]):
        error = snapshot.edges.get((a, b))
        if error is None:
            return None
        score *= 1 - error
    return score


def build_panel(snapshot: Snapshot, history: dict[str, list[dict]]) -> Panel:
    layout = infer_layout(snapshot)
    if layout.bad_edges:
        print(f"Warning: {len(layout.bad_edges)} couplers on {snapshot.backend} do not fit the grid", file=sys.stderr)
    return Panel(
        snapshot=snapshot,
        layout=layout,
        summary=summarise(snapshot),
        stretch=best_stretch(snapshot, layout),
        previous=previous_entry(history.get(snapshot.backend, []), snapshot),
        storms=sorted(
            (q for q, e in snapshot.readout.items() if e > STORM_THRESHOLD),
            key=lambda q: snapshot.readout[q],
            reverse=True,
        ),
    )


def reading_date(snapshot: Snapshot) -> date:
    return (snapshot.calibrated or datetime.now(timezone.utc)).date()


def previous_entry(entries: list[dict], snapshot: Snapshot) -> dict | None:
    current = reading_date(snapshot).isoformat()
    earlier = [e for e in entries if e["date"] < current]
    return earlier[-1] if earlier else None


def relative_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return (current - previous) / previous


def outlook(panel: Panel) -> str:
    share = len(panel.storms) / max(panel.snapshot.num_qubits, 1)
    if not panel.storms:
        return "Clear skies"
    if len(panel.storms) <= 2 or share < 0.02:
        return "Mostly fair"
    return "Changeable" if share < 0.06 else "Unsettled"


def panel_forecast(panel: Panel) -> str:
    sentences = [f"{outlook(panel)}."]
    if panel.stretch:
        chain = panel.stretch[0]
        sentences.append(f"Best {STRETCH_LENGTH}-qubit stretch: {chain[0]} to {chain[-1]}.")
    coherence = coherence_sentence(panel)
    if coherence:
        sentences.append(coherence)
    return " ".join(sentences)


def coherence_sentence(panel: Panel) -> str | None:
    if panel.previous is None:
        return None
    moves = []
    for label, key in (("T1", "t1_us"), ("T2", "t2_us")):
        change = relative_change(panel.summary[key], panel.previous.get(key))
        if change is not None and abs(change) >= FORECAST_CHANGE_THRESHOLD:
            moves.append(f"{label} {'up' if change > 0 else 'down'} {abs(change) * 100:.0f}%")
    if not moves:
        return None
    then = date.fromisoformat(panel.previous["date"])
    when = "on yesterday" if reading_date(panel.snapshot) - then == timedelta(days=1) else f"since {short_date(then)}"
    return f"{' and '.join(moves)} {when}."


def storm_report(panel: Panel) -> str:
    if not panel.storms:
        return "No storm cells"
    if len(panel.storms) == 1:
        return f"1 storm cell, over qubit {panel.storms[0]}"
    return f"{len(panel.storms)} storm cells, the worst over qubit {panel.storms[0]}"


def fleet_headline(panels: list[Panel]) -> str:
    ranked = sorted((p for p in panels if p.summary["two_q"] is not None), key=lambda p: p.summary["two_q"])
    if len(ranked) >= 2:
        return f"Stormiest today: {ranked[-1].snapshot.backend}. Clearest: {ranked[0].snapshot.backend}."
    if len(panels) == 1:
        return f"{outlook(panels[0])} over {panels[0].snapshot.backend}."
    return "Qubit weather over the IBM Quantum fleet"


def short_date(d: date) -> str:
    return f"{d.day} {d.strftime('%b')}"


def ramp_colour(value: float | None) -> str:
    if value is None:
        return MISSING_COLOUR
    return interpolate(ramp_position(value))


def ramp_position(value: float) -> float:
    clamped = min(max(value, RAMP_MIN), RAMP_MAX)
    return math.log(clamped / RAMP_MIN) / math.log(RAMP_MAX / RAMP_MIN)


def interpolate(t: float) -> str:
    for (t0, c0), (t1, c1) in zip(RAMP_STOPS, RAMP_STOPS[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            a, b = to_oklab(c0), to_oklab(c1)
            return from_oklab(tuple(x + (y - x) * f for x, y in zip(a, b)))
    return RAMP_STOPS[-1][1]


def to_oklab(hex_colour: str) -> tuple[float, float, float]:
    r, g, b = (srgb_to_linear(int(hex_colour[i : i + 2], 16) / 255) for i in (1, 3, 5))
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def from_oklab(lab: tuple[float, ...]) -> str:
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )
    return "#" + "".join(f"{round(min(max(linear_to_srgb(c), 0), 1) * 255):02x}" for c in rgb)


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * max(c, 0) ** (1 / 2.4) - 0.055


def render_fleet(snapshots: list[Snapshot], history: dict[str, list[dict]]) -> str:
    panels = [build_panel(s, history) for s in snapshots]
    gates = {(s.gate or "2q").upper() for s in snapshots}
    gate = gates.pop() if len(gates) == 1 else "2q"

    out: list[str] = []
    headline = fleet_headline(panels)
    out.append(text(MARGIN, 52, headline, "serif ink", 28))
    out.append(text(MARGIN, 80, fleet_subtitle(snapshots), "sans ink2", 14))
    top = 118

    grid_width = WIDTH - 2 * MARGIN - SIDEBAR_WIDTH - SIDEBAR_GAP
    cols = 1 if len(panels) == 1 else 2
    panel_width = (grid_width - (cols - 1) * PANEL_GAP) / cols
    max_cols = max(p.layout.size[0] for p in panels)
    max_rows = max(p.layout.size[1] for p in panels)
    cell = min(MAX_CELL, panel_width / max_cols)
    forecasts = [wrap(panel_forecast(p), panel_width, 15, 0.45) for p in panels]
    header_height = 52 + 20 * max(len(f) for f in forecasts)
    panel_height = header_height + max_rows * cell

    for i, (panel, forecast_lines) in enumerate(zip(panels, forecasts)):
        left = MARGIN + (i % cols) * (panel_width + PANEL_GAP)
        panel_top = top + (i // cols) * (panel_height + PANEL_GAP)
        render_panel(out, panel, forecast_lines, left, panel_top, panel_width, header_height, cell)

    grid_bottom = top + math.ceil(len(panels) / cols) * (panel_height + PANEL_GAP) - PANEL_GAP
    legend_bottom = legend(out, MARGIN, grid_bottom + 30, cell, gate, snapshots)
    sidebar_bottom = sidebar(out, WIDTH - MARGIN - SIDEBAR_WIDTH, top, panels, history, gate)
    height = math.ceil(max(legend_bottom, sidebar_bottom) + MARGIN)

    description = " ".join(
        [headline] + [f"{p.snapshot.backend}: {panel_forecast(p)} {storm_report(p)}." for p in panels]
    )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="qw-title qw-desc">',
            f'<title id="qw-title">Qubit weather: {escape(headline)}</title>',
            f'<desc id="qw-desc">{escape(description)}</desc>',
            style(cell),
            ramp_gradient(),
            f'<rect class="bg" width="{WIDTH}" height="{height}" rx="12"/>',
            *out,
            "</svg>",
            "",
        ]
    )


def fleet_subtitle(snapshots: list[Snapshot]) -> str:
    count = len(snapshots)
    lead = f"Qubit weather across {count} IBM Quantum processor{'s' if count != 1 else ''}"
    stamps = sorted(s.calibrated for s in snapshots if s.calibrated)
    if not stamps:
        return f"{lead}. Calibration times not reported."
    first, last = stamps[0], stamps[-1]
    if first.date() == last.date():
        span = f"{first:%H:%M}" if first == last else f"{first:%H:%M} to {last:%H:%M}"
        return f"{lead}, calibrated {first.day} {first:%B %Y}, {span} UTC."
    return f"{lead}, calibrated {short_date(first.date())} {first:%H:%M} to {short_date(last.date())} {last:%H:%M} UTC."


def style(cell: float) -> str:
    return f"""<style>
  .bg {{ fill: #ffffff; }}
  .ink {{ fill: #1f2328; }}
  .ink2 {{ fill: #59636e; }}
  .rule {{ stroke: #d1d9e0; stroke-width: 1; }}
  .serif {{ font-family: {SERIF}; }}
  .sans {{ font-family: {SANS}; }}
  .num {{ font-variant-numeric: tabular-nums; }}
  .halo {{ stroke: #ffffff; stroke-width: 4; stroke-linejoin: round; paint-order: stroke; }}
  .qubit {{ stroke: #ffffff; stroke-width: {max(1, cell * 0.05):.1f}; }}
  .coupler {{ stroke-width: {max(2, cell * 0.11):.1f}; stroke-linecap: round; }}
  .coupler-missing {{ stroke: {MISSING_COLOUR}; stroke-width: 1.5; stroke-dasharray: 2 2; }}
  .storm {{ fill: none; stroke: #1f2328; stroke-width: 1.3; stroke-dasharray: 2.5 2; }}
  .stretch {{ fill: #1f2328; fill-opacity: 0.09; stroke: #1f2328; stroke-opacity: 0.35; stroke-width: 1; }}
  .better {{ fill: #1a7f37; }}
  .worse {{ fill: #cf222e; }}
  .spark {{ fill: none; stroke: #59636e; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
  .spark-dot {{ fill: #1f2328; }}
  @media (prefers-color-scheme: dark) {{
    .bg {{ fill: #0d1117; }}
    .ink {{ fill: #f0f6fc; }}
    .ink2 {{ fill: #9198a1; }}
    .rule {{ stroke: #3d444d; }}
    .halo {{ stroke: #0d1117; }}
    .qubit {{ stroke: #0d1117; }}
    .storm {{ stroke: #f0f6fc; }}
    .stretch {{ fill: #f0f6fc; fill-opacity: 0.12; stroke: #f0f6fc; stroke-opacity: 0.4; }}
    .better {{ fill: #3fb950; }}
    .worse {{ fill: #ff7b72; }}
    .spark {{ stroke: #9198a1; }}
    .spark-dot {{ fill: #f0f6fc; }}
  }}
</style>"""


def ramp_gradient() -> str:
    stops = "".join(f'<stop offset="{i / 20:.2f}" stop-color="{interpolate(i / 20)}"/>' for i in range(21))
    return f'<defs><linearGradient id="qw-ramp" x1="0" x2="1" y1="0" y2="0">{stops}</linearGradient></defs>'


def render_panel(out, panel: Panel, forecast_lines, left, top, width, header_height, cell) -> None:
    snapshot = panel.snapshot
    size = f"{snapshot.num_qubits} qubits" + (f", {snapshot.processor}" if snapshot.processor else "")
    out.append(f'<g id="panel-{escape(snapshot.backend)}">')
    out.append(text(left, top + 18, snapshot.backend, "serif ink", 20))
    out.append(text(left + width, top + 18, size, "sans ink2", 13, anchor="end"))
    y = top + 42
    for line in forecast_lines:
        out.append(text(left, y, line, "serif ink", 15, italic=True))
        y += 20
    out.append(text(left, y, storm_report(panel), "sans ink2", 13))
    out.append(map_layer(panel, left, top + header_height, cell))
    out.append("</g>")


def map_layer(panel: Panel, left: float, top: float, cell: float) -> str:
    snapshot, coords, stretch = panel.snapshot, panel.layout.coords, panel.stretch
    gate = (snapshot.gate or "2q").upper()

    def px(q: int) -> tuple[float, float]:
        x, y = coords[q]
        return left + (x + 0.5) * cell, top + (y + 0.5) * cell

    parts = []
    if stretch:
        chain = stretch[0]
        (x0, y0), (x1, _) = px(chain[0]), px(chain[-1])
        pad = cell * 0.42
        parts.append(
            f'<rect class="stretch" x="{x0 - pad:.1f}" y="{y0 - pad:.1f}" width="{x1 - x0 + 2 * pad:.1f}" '
            f'height="{2 * pad:.1f}" rx="{pad:.1f}"><title>{snapshot.backend}: best {STRETCH_LENGTH}-qubit stretch, '
            f"{chain[0]} to {chain[-1]}\nFidelity product {stretch[1]:.3f}</title></rect>"
        )
    for (a, b), error in sorted(snapshot.edges.items()):
        (xa, ya), (xb, yb) = px(a), px(b)
        tip = f"{snapshot.backend} coupler {a}-{b}\n{gate} error {pct(error)}"
        line = f'x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}"><title>{tip}</title></line>'
        if error is None:
            parts.append(f'<line class="coupler-missing" {line}')
        else:
            parts.append(f'<line class="coupler" stroke="{ramp_colour(error)}" {line}')
    for q in range(snapshot.num_qubits):
        x, y = px(q)
        error = snapshot.readout.get(q)
        storm = error is not None and error > STORM_THRESHOLD
        tip = (
            f"{snapshot.backend} qubit {q}{' (storm cell)' if storm else ''}\nReadout error {pct(error)}\n"
            f"T1 {micro(snapshot.t1.get(q))}, T2 {micro(snapshot.t2.get(q))}"
        )
        ring = f'<circle class="storm" cx="{x:.1f}" cy="{y:.1f}" r="{cell * 0.48:.1f}"/>' if storm else ""
        parts.append(
            f'<g><title>{tip}</title>{ring}'
            f'<circle class="qubit" fill="{ramp_colour(error)}" cx="{x:.1f}" cy="{y:.1f}" r="{cell * 0.28:.1f}"/></g>'
        )
    if stretch:
        chain = stretch[0]
        (x0, y0), (x1, _) = px(chain[0]), px(chain[-1])
        label = f"{chain[0]} to {chain[-1]}"
        # Labels above a top-row stretch would collide with the panel header, so put those below.
        label_y = y0 - cell * 0.42 - 4 if y0 - top > cell else y0 + cell * 0.42 + 12
        parts.append(text((x0 + x1) / 2, label_y, label, "sans ink num halo", 11, anchor="middle", weight=600))
    return "\n".join(parts)


def legend(out, left, top, cell, gate, snapshots) -> float:
    out.append(text(left, top, f"Error rate on one log scale for every processor: qubits show readout error, couplers show {gate} error", "sans ink2", 13))
    bar_top = top + 14
    bar_left = left + 44
    bar_width = 300
    out.append(text(left, bar_top + 10, "Calm", "sans ink2", 12))
    out.append(f'<rect x="{bar_left}" y="{bar_top}" width="{bar_width}" height="10" rx="5" fill="url(#qw-ramp)"/>')
    out.append(text(bar_left + bar_width + 8, bar_top + 10, "Stormy", "sans ink2", 12))
    for tick in RAMP_TICKS:
        x = bar_left + ramp_position(tick) * bar_width
        out.append(f'<line class="rule" x1="{x:.1f}" y1="{bar_top + 12}" x2="{x:.1f}" y2="{bar_top + 16}"/>')
        out.append(text(x, bar_top + 29, f"{tick * 100:g}%", "sans ink2 num", 12, anchor="middle"))

    keys = bar_top + 56
    radius = max(cell * 0.28, 6)
    x = left + radius + 4
    out.append(f'<circle class="storm" cx="{x:.1f}" cy="{keys - 4}" r="{radius * 1.7:.1f}"/>')
    out.append(f'<circle class="qubit" fill="{ramp_colour(0.1)}" cx="{x:.1f}" cy="{keys - 4}" r="{radius:.1f}"/>')
    out.append(text(x + 22, keys, f"Storm cell: readout error above {STORM_THRESHOLD * 100:g}%", "sans ink2", 13))
    x2 = left + 290
    out.append(f'<rect class="stretch" x="{x2}" y="{keys - 12}" width="30" height="16" rx="8"/>')
    out.append(text(x2 + 38, keys, f"Best {STRETCH_LENGTH}-qubit stretch: highest product of (1 - error) along a row", "sans ink2", 13))
    bottom = keys + 6
    if any(len(s.readout) < s.num_qubits or None in s.edges.values() for s in snapshots):
        bottom += 26
        out.append(f'<circle class="qubit" fill="{MISSING_COLOUR}" cx="{x:.1f}" cy="{bottom - 4}" r="{radius:.1f}"/>')
        out.append(text(x + 22, bottom, "Grey: not reported", "sans ink2", 13))
    return bottom


def sidebar(out, left, top, panels: list[Panel], history, gate) -> float:
    out.append(text(left, top + 18, "Fleet readings", "serif ink", 20))
    out.append(text(left, top + 40, "Median of each processor's median", "sans ink2", 12))
    previous_dates = {p.previous["date"] for p in panels if p.previous}
    since = f" since {short_date(date.fromisoformat(previous_dates.pop()))}" if len(previous_dates) == 1 else " on last reading"
    stats = (
        ("T1", "t1_us", micro_value, True),
        ("T2", "t2_us", micro_value, True),
        ("Readout error", "readout", pct, False),
        (f"{gate} error", "two_q", pct, False),
    )
    y = top + 64
    for label, key, fmt, higher_is_better in stats:
        value = median_or_none([p.summary[key] for p in panels])
        compared = [p for p in panels if p.previous and p.summary[key] is not None and p.previous.get(key) is not None]
        change = relative_change(
            median_or_none([p.summary[key] for p in compared]),
            median_or_none([p.previous[key] for p in compared]),
        )
        out.append(text(left, y + 12, label, "sans ink2", 13))
        out.append(text(left, y + 40, fmt(value), "sans ink num", 25, weight=600))
        out.append(change_text(left, y + 60, change, higher_is_better, since))
        y += 84

    out.append(f'<line class="rule" x1="{left}" y1="{y - 6}" x2="{left + SIDEBAR_WIDTH}" y2="{y - 6}"/>')
    out.append(text(left, y + 16, f"{gate} error, last {SPARKLINE_DAYS} days", "sans ink2", 13))
    return sparkline(out, left, y + 30, fleet_series(panels, history))


def fleet_series(panels: list[Panel], history) -> list[tuple[date, float]]:
    end = max(reading_date(p.snapshot) for p in panels)
    start = end - timedelta(days=SPARKLINE_DAYS - 1)
    by_date: dict[date, list[float]] = {}
    for panel in panels:
        for entry in history.get(panel.snapshot.backend, []):
            day = date.fromisoformat(entry["date"])
            if start <= day <= end and entry.get("two_q") is not None:
                by_date.setdefault(day, []).append(entry["two_q"])
    return [(day, statistics.median(values)) for day, values in sorted(by_date.items())]


def change_text(x, y, change, higher_is_better, since) -> str:
    if change is None:
        return text(x, y, "No earlier reading", "sans ink2", 13)
    if abs(change) < 0.005:
        return text(x, y, f"No change{since}", "sans ink2", 13)
    better = (change > 0) == higher_is_better
    arrow = "▲" if change > 0 else "▼"
    word = "better" if better else "worse"
    return text(x, y, f"{arrow} {abs(change) * 100:.1f}% {word}{since}", f"sans num {word}", 13)


def sparkline(out, left, top, series: list[tuple[date, float]]) -> float:
    height, width = 54, SIDEBAR_WIDTH
    if len(series) < 2:
        out.append(text(left, top + 18, "Builds up as daily readings arrive", "sans ink2", 13))
        return top + 24

    start = series[-1][0] - timedelta(days=SPARKLINE_DAYS - 1)
    values = [v for _, v in series]
    low, high = min(values), max(values)
    span = (high - low) or high or 1
    coords = [
        (
            left + 4 + (day - start).days / (SPARKLINE_DAYS - 1) * (width - 8),
            top + 4 + (1 - (value - low) / span) * (height - 8),
        )
        for day, value in series
    ]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords))
    out.append(f'<path class="spark" d="{path}"/>')
    for (day, value), (x, y) in zip(series, coords):
        out.append(f'<circle fill-opacity="0" cx="{x:.1f}" cy="{y:.1f}" r="7"><title>{short_date(day)}: {pct(value)}</title></circle>')
    lx, ly = coords[-1]
    out.append(f'<circle class="spark-dot" cx="{lx:.1f}" cy="{ly:.1f}" r="4"/>')
    out.append(text(left, top + height + 18, f"Low {pct(low)}", "sans ink2 num", 12))
    out.append(text(left + width, top + height + 18, f"High {pct(high)}", "sans ink2 num", 12, anchor="end"))
    return top + height + 24


def text(x, y, content, cls, size, anchor="start", weight=None, italic=False) -> str:
    attrs = f' font-weight="{weight}"' if weight else ""
    attrs += ' font-style="italic"' if italic else ""
    attrs += f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" font-size="{size}"{attrs}>{escape(content)}</text>'


def wrap(content: str, width: float, size: float, char_em: float) -> list[str]:
    limit = int(width / (size * char_em))
    lines, line = [], ""
    for word in content.split():
        candidate = f"{line} {word}".strip()
        if len(candidate) > limit and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + ([line] if line else [])


def pct(value: float | None) -> str:
    if value is None:
        return "not reported"
    return f"{value * 100:.2f}%" if value < 0.1 else f"{value * 100:.1f}%"


def micro(value: float | None) -> str:
    return "not reported" if value is None else f"{value * 1e6:.0f} µs"


def micro_value(value_us: float | None) -> str:
    return "Not reported" if value_us is None else f"{value_us:.0f} µs"


if __name__ == "__main__":
    main()
