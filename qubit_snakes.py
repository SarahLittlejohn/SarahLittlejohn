"""Qubit cats and ladders: one quantum dice roll a day on an IBM Quantum chip.

The board is the heavy-hex layout of IBM's Fez processor (the fake_fez
calibration snapshot). Row qubits are the squares, bridge qubits are the
ladders, and noisy qubits are Schrodinger's cats that send you back. Each day the dice is rolled by
measuring three qubits in superposition on a real IBM Quantum processor.

    python qubit_snakes.py           # roll on a real QPU, needs IBM_QUANTUM_TOKEN
    python qubit_snakes.py --demo    # roll on a local simulator, state not saved
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

CAT_READOUT = 0.02
KITTEN_READOUT = 0.012
GREEN_RUNG = 0.01
PENALTY = {"cat": 3, "kitten": 2}
DICE_QUBITS = 3
DICE_SHOTS = 16
JOB_WAIT_SECONDS = 40 * 60
HISTORY_SHOWN = 20
MIN_ROW_LENGTH = 3

WIDTH = 1000
MARGIN = 32
CELL = 38
PADDING = 30
QUBIT_RADIUS = 13
BRIDGE_RADIUS = 10
SIDEBAR_GAP = 30
SERIF = "Georgia, 'Iowan Old Style', 'Palatino Linotype', 'Book Antiqua', 'Times New Roman', serif"
SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

# The chip is a physical object, so it keeps the same colours in light and dark mode.
PACKAGE = "#1b2130"
SUBSTRATE = "#232b3d"
GOLD = "#c8a24a"
COUPLER = "#5d6a88"
QUBIT_FILL = {"calm": "#dfe7fb", "kitten": "#f6c86b", "cat": "#ef7f72"}
QUBIT_INK = {"calm": "#1b2130", "kitten": "#5c3a00", "cat": "#5f140c"}
PENALTY_INK = {"kitten": "#f6c86b", "cat": "#ff8f82"}
LADDER_OPEN = "#3fb950"
LADDER_BLOCKED = "#ff6b6b"
PAWN = "#a371f7"
PIP = "#1f2328"


@dataclass
class Ladder:
    foot: int
    middle: int
    top: int
    is_open: bool
    reason: str | None
    couplers: dict[int, float]


@dataclass
class Board:
    name: str
    path: list[int]
    coords: dict[int, tuple[int, int]]
    readout: dict[int, float]
    ladders: dict[int, Ladder]
    gate: str

    def risk(self, qubit: int) -> str:
        error = self.readout.get(qubit)
        if error is None or error > CAT_READOUT:
            return "cat"
        return "kitten" if error > KITTEN_READOUT else "calm"

    @property
    def last(self) -> int:
        return len(self.path) - 1


@dataclass
class Roll:
    value: int
    bits: str
    discarded: int
    backend: str
    job_id: str | None
    rolled_at: str


def main() -> None:
    args = parse_args()
    board = build_board(load_board_backend())
    state_path = Path(args.state)
    state = load_state(state_path)

    if args.demo:
        roll = simulated_roll()
    else:
        roll, pending = quantum_roll(state.get("pending_job"))
        if roll is None:
            save_state(state_path, {**state, "pending_job": pending})
            print(f"No roll yet; job {pending} is still queued" if pending else "No usable roll; trying again tomorrow")
            return

    state = advance(state, board, roll)
    if not args.demo:
        save_state(state_path, state)
    Path(args.out).write_text(render(board, state), encoding="utf-8")
    print(f"Rolled a {roll.value} on {roll.backend}: {event_text(board, state['event'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="roll on a local simulator; needs no token, saves no state")
    parser.add_argument("--out", default="qubit-snakes.svg")
    parser.add_argument("--state", default="data/game.json")
    return parser.parse_args()


def load_board_backend():
    from qiskit_ibm_runtime.fake_provider import FakeFez

    return FakeFez()


def build_board(backend) -> Board:
    target = backend.target
    gate = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    readout = {
        qargs[0]: props.error
        for qargs, props in target["measure"].items()
        if props is not None and props.error is not None
    }
    edges: dict[tuple[int, int], float] = {}
    for qargs, props in target[gate].items():
        error = props.error if props is not None and props.error is not None else 1.0
        key = tuple(sorted(qargs))
        edges[key] = min(edges.get(key, 1.0), error)

    coords, rows = heavy_hex_layout(target.num_qubits, edges)
    # Snake the path from the bottom row upwards, alternating direction.
    ordered_rows = sorted(rows, key=lambda row: coords[row[0]][1], reverse=True)
    path = [
        q
        for k, row in enumerate(ordered_rows)
        for q in sorted(row, key=lambda q: coords[q][0], reverse=k % 2 == 1)
    ]
    on_path = set(path)
    ladders = {}
    for middle in (q for q in range(target.num_qubits) if q not in on_path):
        ends = sorted((m for (a, b) in edges for m in (a, b) if middle in (a, b) and m != middle), key=lambda m: coords[m][1])
        if len(ends) != 2 or not all(e in on_path for e in ends):
            continue
        top, foot = ends
        couplers = {e: edges[tuple(sorted((middle, e)))] for e in ends}
        reason = None
        if readout.get(middle, 1.0) > CAT_READOUT:
            reason = "cat"
        elif max(couplers.values()) >= GREEN_RUNG:
            reason = "coupler"
        ladders[foot] = Ladder(foot=foot, middle=middle, top=top, is_open=reason is None, reason=reason, couplers=couplers)
    return Board(name=backend.name, path=path, coords=coords, readout=readout, ladders=ladders, gate=gate.upper())


def heavy_hex_layout(n: int, edges: dict) -> tuple[dict[int, tuple[int, int]], list[list[int]]]:
    """Consecutive indices joined by an edge form rows; bridge qubits sit between them."""
    adj: dict[int, set[int]] = {q: set() for q in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    runs: list[list[int]] = [[0]]
    for q in range(1, n):
        if q - 1 in adj[q]:
            runs[-1].append(q)
        else:
            runs.append([q])
    rows = [r for r in runs if len(r) >= MIN_ROW_LENGTH]
    row_of = {q: i for i, row in enumerate(rows) for q in row}
    pos = {q: j for row in rows for j, q in enumerate(row)}
    bridges = [q for q in range(n) if q not in row_of]

    offsets = [0]
    for i in range(len(rows) - 1):
        offset = offsets[i]
        for b in bridges:
            above = [m for m in adj[b] if row_of.get(m) == i]
            below = [m for m in adj[b] if row_of.get(m) == i + 1]
            if above and below:
                offset = offsets[i] + pos[above[0]] - pos[below[0]]
                break
        offsets.append(offset)

    coords = {q: (offsets[row_of[q]] + pos[q], 2 * row_of[q]) for q in row_of}
    for b in bridges:
        neighbours = sorted((coords[m] for m in adj[b] if m in coords), key=lambda c: c[1])
        if neighbours:
            coords[b] = (neighbours[0][0], neighbours[0][1] + 1)
    min_x = min(x for x, _ in coords.values())
    return {q: (x - min_x, y) for q, (x, y) in coords.items()}, rows


def simulated_roll() -> Roll:
    from qiskit.primitives import StatevectorSampler

    result = StatevectorSampler().run([dice_circuit()], shots=DICE_SHOTS).result()
    return roll_from_bitstrings(result[0].data.meas.get_bitstrings(), "local simulator", None)


def quantum_roll(pending_job: str | None) -> tuple[Roll | None, str | None]:
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.exceptions import RuntimeJobTimeoutError

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        sys.exit("IBM_QUANTUM_TOKEN is not set (use --demo to run without one)")
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=os.environ.get("IBM_QUANTUM_INSTANCE") or None,
    )

    job = service.job(pending_job) if pending_job else None
    if job is not None and str(job.status()) in ("ERROR", "CANCELLED"):
        job = None
    if job is None:
        backend = service.least_busy(operational=True, simulator=False)
        circuit = generate_preset_pass_manager(optimization_level=1, backend=backend).run(dice_circuit())
        job = SamplerV2(mode=backend).run([circuit], shots=DICE_SHOTS)

    try:
        job.wait_for_final_state(timeout=JOB_WAIT_SECONDS)
    except RuntimeJobTimeoutError:
        return None, job.job_id()
    if str(job.status()) != "DONE":
        return None, None
    bitstrings = job.result()[0].data.meas.get_bitstrings()
    try:
        return roll_from_bitstrings(bitstrings, job.backend().name, job.job_id()), None
    except ValueError:
        return None, None


def dice_circuit():
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(DICE_QUBITS)
    circuit.h(range(DICE_QUBITS))
    circuit.measure_all()
    return circuit


def roll_from_bitstrings(bitstrings: list[str], backend: str, job_id: str | None) -> Roll:
    """Three qubits give 0 to 7; outcomes 6 and 7 are discarded so the dice stays fair."""
    for discarded, bits in enumerate(bitstrings):
        if int(bits, 2) < 6:
            return Roll(
                value=int(bits, 2) + 1,
                bits=bits,
                discarded=discarded,
                backend=backend,
                job_id=job_id,
                rolled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
    raise ValueError("every shot measured 6 or 7")


def take_turn(board: Board, position: int, roll: int) -> tuple[int, dict]:
    landed = min(position + roll, board.last)
    event = {"kind": "move", "from": position, "landed": landed, "to": landed, "qubit": board.path[landed]}
    if landed == board.last:
        return landed, {**event, "kind": "finish"}
    qubit = board.path[landed]
    risk = board.risk(qubit)
    if risk != "calm":
        to = max(landed - PENALTY[risk], 0)
        return to, {**event, "kind": risk, "to": to}
    ladder = board.ladders.get(qubit)
    if ladder is None:
        return landed, event
    if not ladder.is_open:
        return landed, {**event, "kind": "blocked", "middle": ladder.middle, "reason": ladder.reason}
    to = board.path.index(ladder.top)
    return to, {**event, "kind": "ladder", "to": to, "middle": ladder.middle}


def advance(state: dict, board: Board, roll: Roll) -> dict:
    new_game = not state or state.get("finished", False)
    position = 0 if new_game else state["position"]
    to, event = take_turn(board, position, roll.value)
    return {
        "game": state.get("game", 0) + 1 if new_game else state["game"],
        "turn": 1 if new_game else state["turn"] + 1,
        "position": to,
        "previous": position,
        "finished": event["kind"] == "finish",
        "games_finished": state.get("games_finished", 0) + (event["kind"] == "finish"),
        "roll": asdict(roll),
        "event": event,
        "history": roll_history(state) + [history_entry(asdict(roll))],
        "pending_job": None,
    }


def roll_history(state: dict) -> list[dict]:
    """Every roll so far; older state files only kept the latest roll, so start from that."""
    if "history" in state:
        return state["history"]
    return [history_entry(state["roll"])] if state.get("roll") else []


def history_entry(roll: dict) -> dict:
    return {key: roll[key] for key in ("value", "bits", "backend", "job_id", "rolled_at")}


def load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def event_text(board: Board, event: dict) -> str:
    landed, to = event["landed"] + 1, event["to"] + 1
    qubit = event["qubit"]
    start = f"Moved from square {event['from'] + 1} to {landed}"
    kind = event["kind"]
    if kind == "finish":
        return f"{start} and reached the finish! A new game starts tomorrow."
    if kind in PENALTY:
        who = "Schrödinger's cat" if kind == "cat" else "a kitten"
        return (
            f"{start}, where {who} on qubit {qubit} ({pct(board.readout.get(qubit))} readout error) "
            f"knocked you back {PENALTY[kind]} to square {to}."
        )
    if kind == "ladder":
        return f"{start} and climbed the ladder through bridge qubit {event['middle']} to square {to}."
    if kind == "blocked":
        why = "a cat sitting on the bridge qubit" if event["reason"] == "cat" else "a noisy coupler"
        return f"{start}, but the ladder through bridge qubit {event['middle']} is blocked by {why}."
    return f"{start}."


def render(board: Board, state: dict) -> str:
    out: list[str] = []
    out.append(text(MARGIN, 52, "Qubit cats and ladders", "serif ink", 30))
    out.append(
        text(
            MARGIN,
            80,
            f"One quantum dice roll a day, played on the qubits of IBM's Fez processor ({board.name} calibration data).",
            "sans ink2",
            14,
        )
    )
    top = 108
    cols = max(x for x, _ in board.coords.values()) + 1
    rows = max(y for _, y in board.coords.values()) + 1
    chip_width = cols * CELL + 2 * PADDING
    chip_height = rows * CELL + 2 * PADDING
    origin_x, origin_y = MARGIN + PADDING, top + PADDING

    def centre(q: int) -> tuple[float, float]:
        x, y = board.coords[q]
        return origin_x + (x + 0.5) * CELL, origin_y + (y + 0.5) * CELL

    out.append(chip_package(MARGIN, top, chip_width, chip_height))
    out.extend(path_turns(board, centre))
    out.extend(row_couplers(board, centre))
    for ladder in board.ladders.values():
        out.append(ladder_svg(board, ladder, centre))
    for index, qubit in enumerate(board.path):
        out.append(qubit_svg(board, index, qubit, centre(qubit)))
    if state:
        if state["previous"] != state["position"]:
            out.append(pawn(*centre(board.path[state["previous"]]), ghost=True))
        out.append(pawn(*centre(board.path[state["position"]])))
    else:
        out.append(pawn(*centre(board.path[0])))

    legend_bottom = legend(out, MARGIN, top + chip_height + 34, board.gate)
    sidebar_left = MARGIN + chip_width + SIDEBAR_GAP
    sidebar_bottom = sidebar(out, sidebar_left, top, WIDTH - MARGIN - sidebar_left, board, state)
    height = max(legend_bottom, sidebar_bottom) + MARGIN

    summary = event_text(board, state["event"]) if state else "Waiting for the first roll."
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="qs-title qs-desc">',
            '<title id="qs-title">Qubit cats and ladders</title>',
            f'<desc id="qs-desc">{escape(summary)}</desc>',
            STYLE,
            f'<rect class="bg" width="{WIDTH}" height="{height}" rx="12"/>',
            *out,
            "</svg>",
            "",
        ]
    )


STYLE = f"""<style>
  .bg {{ fill: #ffffff; }}
  .ink {{ fill: #1f2328; }}
  .ink2 {{ fill: #59636e; }}
  .serif {{ font-family: {SERIF}; }}
  .sans {{ font-family: {SANS}; }}
  @media (prefers-color-scheme: dark) {{
    .bg {{ fill: #0d1117; }}
    .ink {{ fill: #f0f6fc; }}
    .ink2 {{ fill: #9198a1; }}
  }}
</style>"""


def chip_package(x: float, y: float, width: float, height: float) -> str:
    pins = []
    for px in frange_up(x + 24, x + width - 20, 22):
        pins.append(f'<rect x="{px:.1f}" y="{y - 5}" width="6" height="7" rx="1" fill="{GOLD}"/>')
        pins.append(f'<rect x="{px:.1f}" y="{y + height - 2}" width="6" height="7" rx="1" fill="{GOLD}"/>')
    for py in frange_up(y + 24, y + height - 20, 22):
        pins.append(f'<rect x="{x - 5}" y="{py:.1f}" width="7" height="6" rx="1" fill="{GOLD}"/>')
        pins.append(f'<rect x="{x + width - 2}" y="{py:.1f}" width="7" height="6" rx="1" fill="{GOLD}"/>')
    return "".join(
        pins
        + [
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="14" fill="{PACKAGE}" stroke="{GOLD}" stroke-width="2"/>',
            f'<rect x="{x + 8}" y="{y + 8}" width="{width - 16}" height="{height - 16}" rx="8" fill="{SUBSTRATE}"/>',
        ]
    )


def frange_up(start: float, stop: float, step: float) -> list[float]:
    values, value = [], start
    while value <= stop:
        values.append(value)
        value += step
    return values


def path_turns(board: Board, centre) -> list[str]:
    """Dotted arrows round the end of each row, showing which way play continues."""
    parts = []
    cols = max(x for x, _ in board.coords.values())
    for a, b in zip(board.path, board.path[1:]):
        if board.coords[a][1] == board.coords[b][1]:
            continue
        (ax, ay), (bx, by) = centre(a), centre(b)
        side = 1 if board.coords[a][0] == cols else -1
        edge = ax + side * QUBIT_RADIUS
        bulge = edge + side * 20
        parts.append(
            f'<path d="M{edge:.1f},{ay:.1f} C{bulge:.1f},{ay:.1f} {bulge:.1f},{by:.1f} {edge + side * 3:.1f},{by:.1f}" '
            f'fill="none" stroke="{GOLD}" stroke-width="1.5" stroke-dasharray="2 3" opacity="0.8"/>'
        )
        tip = edge + side * 1
        parts.append(
            f'<path d="M{tip:.1f},{by:.1f} l{side * 6:.1f},-4 l0,8 Z" fill="{GOLD}" opacity="0.9"/>'
        )
    return parts


def row_couplers(board: Board, centre) -> list[str]:
    parts = []
    for a, b in zip(board.path, board.path[1:]):
        if board.coords[a][1] != board.coords[b][1]:
            continue
        (ax, ay), (bx, by) = centre(a), centre(b)
        parts.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{COUPLER}" stroke-width="3"/>')
    return parts


def ladder_svg(board: Board, ladder: Ladder, centre) -> str:
    (fx, fy), (mx, my), (tx, ty) = centre(ladder.foot), centre(ladder.middle), centre(ladder.top)
    colour = LADDER_OPEN if ladder.is_open else LADDER_BLOCKED
    dash = "" if ladder.is_open else ' stroke-dasharray="4 3"'
    segments = ((fy - QUBIT_RADIUS, my + BRIDGE_RADIUS), (my - BRIDGE_RADIUS, ty + QUBIT_RADIUS))
    parts = []
    for y0, y1 in segments:
        for x in (fx - 4, fx + 4):
            parts.append(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" stroke="{colour}" stroke-width="2"{dash}/>')
        for y in frange(y0 - 3, y1 + 2, 5):
            parts.append(f'<line x1="{fx - 4:.1f}" y1="{y:.1f}" x2="{fx + 4:.1f}" y2="{y:.1f}" stroke="{colour}" stroke-width="1.5"/>')

    risk = board.risk(ladder.middle)
    if ladder.is_open:
        mark = f'<path d="M{mx:.1f},{my - 5:.1f} l4.5,5 h-3 v4.5 h-3 v-4.5 h-3 Z" fill="{LADDER_OPEN}"/>'
        status = "Open: climb!"
    else:
        mark = (
            f'<path d="M{mx - 3.5:.1f},{my - 3.5:.1f} l7,7 M{mx + 3.5:.1f},{my - 3.5:.1f} l-7,7" '
            f'stroke="{LADDER_BLOCKED if risk == "calm" else QUBIT_INK[risk]}" stroke-width="2" stroke-linecap="round"/>'
        )
        status = "Blocked by a cat on the bridge qubit" if ladder.reason == "cat" else "Blocked by a noisy coupler"
    couplers = " and ".join(pct(ladder.couplers[end]) for end in (ladder.foot, ladder.top))
    tip = (
        f"Bridge qubit {ladder.middle}: a ladder from square {board.path.index(ladder.foot) + 1} "
        f"to {board.path.index(ladder.top) + 1}\nReadout error {pct(board.readout.get(ladder.middle))}\n"
        f"{board.gate} errors {couplers}\n{status}"
    )
    ears = cat_ears(mx, my, BRIDGE_RADIUS, risk) if risk != "calm" else ""
    return (
        f"<g><title>{tip}</title>{''.join(parts)}{ears}"
        f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="{BRIDGE_RADIUS}" fill="{QUBIT_FILL[risk]}" stroke="{colour}" stroke-width="2"/>'
        f"{mark}</g>"
    )


def frange(start: float, stop: float, step: float) -> list[float]:
    values, value = [], start
    while value >= stop:
        values.append(value)
        value -= step
    return values


def qubit_svg(board: Board, index: int, qubit: int, centre: tuple[float, float]) -> str:
    cx, cy = centre
    risk = board.risk(qubit)
    tip = f"Square {index + 1}: qubit {qubit}\nReadout error {pct(board.readout.get(qubit))}"
    if risk != "calm":
        who = "Schrödinger's cat" if risk == "cat" else "Kitten"
        tip += f"\n{who}: back {PENALTY[risk]}"
    parts = [f"<g><title>{tip}</title>"]
    if risk != "calm":
        parts.append(cat_ears(cx, cy, QUBIT_RADIUS, risk))
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{QUBIT_RADIUS}" fill="{QUBIT_FILL[risk]}"/>')
    ink = QUBIT_INK[risk]
    if risk == "calm":
        parts.append(number(cx, cy + 3.5, index + 1, ink))
    else:
        parts.append(f'<circle cx="{cx - 4:.1f}" cy="{cy - 4:.1f}" r="1.4" fill="{ink}"/><circle cx="{cx + 4:.1f}" cy="{cy - 4:.1f}" r="1.4" fill="{ink}"/>')
        parts.append(number(cx, cy + 7, index + 1, ink, size=8.5))
        parts.append(
            f'<text x="{cx + QUBIT_RADIUS + 5:.1f}" y="{cy + QUBIT_RADIUS + 4:.1f}" font-size="9.5" font-weight="700" '
            f'text-anchor="end" fill="{PENALTY_INK[risk]}" font-family="{SANS}">−{PENALTY[risk]}</text>'
        )
    if index == 0:
        parts.append(f'<text x="{cx:.1f}" y="{cy + QUBIT_RADIUS + 14:.1f}" font-size="10" text-anchor="middle" fill="{GOLD}" font-family="{SANS}" font-weight="600">Start</text>')
    if index == board.last:
        parts.append(f'<text x="{cx:.1f}" y="{cy - QUBIT_RADIUS - 6:.1f}" font-size="10" text-anchor="middle" fill="{GOLD}" font-family="{SANS}" font-weight="600">Finish ★</text>')
    parts.append("</g>")
    return "".join(parts)


def cat_ears(cx: float, cy: float, radius: float, risk: str) -> str:
    size = 1.0 if risk == "cat" else 0.75
    left = f"M{cx - radius * 0.85:.1f},{cy - radius * 0.35:.1f} L{cx - radius * 0.75:.1f},{cy - radius - 7 * size:.1f} L{cx - radius * 0.2:.1f},{cy - radius * 0.85:.1f} Z"
    right = f"M{cx + radius * 0.85:.1f},{cy - radius * 0.35:.1f} L{cx + radius * 0.75:.1f},{cy - radius - 7 * size:.1f} L{cx + radius * 0.2:.1f},{cy - radius * 0.85:.1f} Z"
    return f'<path d="{left} {right}" fill="{QUBIT_FILL[risk]}" stroke="{QUBIT_INK[risk]}" stroke-width="0.8" stroke-linejoin="round"/>'


def number(x: float, y: float, value: int, ink: str, size: float = 10) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="700" text-anchor="middle" fill="{ink}" font-family="{SANS}">{value}</text>'


def pawn(cx: float, cy: float, ghost: bool = False) -> str:
    """The pawn stands on top of its qubit, with a ring round it, so the square number stays readable."""
    ring = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{QUBIT_RADIUS + 3}" fill="none" stroke="{PAWN}" stroke-width="2.5"{" stroke-dasharray=\"3 2\"" if ghost else ""}/>'
    cx, cy = cx + 9, cy - 19
    style = (
        f'fill="none" stroke="{PAWN}" stroke-width="1.5" stroke-dasharray="3 2"'
        if ghost
        else f'fill="{PAWN}" stroke="#ffffff" stroke-width="1.5"'
    )
    label = "Previous square" if ghost else "You are here"
    return (
        f"<g><title>{label}</title>{ring}"
        f'<path d="M{cx - 11:.1f},{cy + 14:.1f} Q{cx:.1f},{cy + 8:.1f} {cx + 11:.1f},{cy + 14:.1f} L{cx + 5.5:.1f},{cy:.1f} Q{cx:.1f},{cy - 3:.1f} {cx - 5.5:.1f},{cy:.1f} Z" {style}/>'
        f'<circle cx="{cx:.1f}" cy="{cy - 7:.1f}" r="7" {style}/>'
        "</g>"
    )


def legend(out: list[str], left: float, top: float, gate: str) -> float:
    y = top
    out.append(f'<circle cx="{left + 10}" cy="{y - 4}" r="9" fill="{QUBIT_FILL["calm"]}" stroke="{COUPLER}" stroke-width="1.5"/>')
    out.append(text(left + 26, y, "Every circle is a qubit on the chip; the number is its square in the game", "sans ink2", 13))
    y += 28
    out.append(cat_ears(left + 10, y - 4, 9, "cat") + f'<circle cx="{left + 10}" cy="{y - 4}" r="9" fill="{QUBIT_FILL["cat"]}"/>')
    out.append(text(left + 26, y, f"Schrödinger's cat: readout error above {pct(CAT_READOUT)}, back {PENALTY['cat']}", "sans ink2", 13))
    out.append(cat_ears(left + 390, y - 4, 9, "kitten") + f'<circle cx="{left + 390}" cy="{y - 4}" r="9" fill="{QUBIT_FILL["kitten"]}"/>')
    out.append(text(left + 406, y, f"Kitten: {pct(KITTEN_READOUT)} to {pct(CAT_READOUT)}, back {PENALTY['kitten']}", "sans ink2", 13))
    y += 28
    out.append(mini_ladder(left + 10, y - 4, LADDER_OPEN, False))
    out.append(text(left + 26, y, f"Bridge qubit with every {gate} coupler below {pct(GREEN_RUNG)} error: a ladder you can climb", "sans ink2", 13))
    y += 28
    out.append(mini_ladder(left + 10, y - 4, LADDER_BLOCKED, True))
    out.append(text(left + 26, y, "Blocked ladder: a cat on the bridge qubit, or a noisy coupler", "sans ink2", 13))
    return y + 8


def mini_ladder(cx: float, cy: float, colour: str, dashed: bool) -> str:
    dash = ' stroke-dasharray="3 2"' if dashed else ""
    parts = [f'<line x1="{x}" y1="{cy - 11}" x2="{x}" y2="{cy + 11}" stroke="{colour}" stroke-width="2"{dash}/>' for x in (cx - 4, cx + 4)]
    parts += [f'<line x1="{cx - 4}" y1="{y}" x2="{cx + 4}" y2="{y}" stroke="{colour}" stroke-width="1.5"/>' for y in (cy - 8, cy - 4, cy + 4, cy + 8)]
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{QUBIT_FILL["calm"]}" stroke="{colour}" stroke-width="1.5"/>')
    return "".join(parts)


def sidebar(out: list[str], left: float, top: float, width: float, board: Board, state: dict) -> float:
    out.append(text(left, top + 18, "Today's quantum roll", "serif ink", 20))
    die_top = top + 36
    roll = state.get("roll") if state else None
    out.append(die(left, die_top, 84, roll["value"] if roll else None))
    if roll is None:
        out.append(text(left + 100, die_top + 48, "Waiting for the", "sans ink2", 14))
        out.append(text(left + 100, die_top + 66, "first roll", "sans ink2", 14))
        return die_top + 100
    out.append(text(left + 100, die_top + 34, f"Rolled a {roll['value']}", "sans ink", 20, weight=600))
    out.append(text(left + 100, die_top + 56, f"Measured |{roll['bits']}⟩", "sans ink2", 13))
    out.append(text(left + 100, die_top + 74, f"on {roll['backend']}", "sans ink2", 13))

    y = die_top + 112
    discarded = f" ({roll['discarded']} today)" if roll["discarded"] else ""
    how = (
        "Three qubits were put into superposition with Hadamard gates and measured, giving 0 to 7. "
        f"Outcomes 6 and 7 are discarded to keep the dice fair{discarded}, then 1 is added."
    )
    for line in wrap(how, width, 12.5, 0.52):
        out.append(text(left, y, line, "sans ink2", 12.5))
        y += 17
    y += 16
    out.append(text(left, y, "What happened", "serif ink", 17))
    y += 22
    for line in wrap(event_text(board, state["event"]), width, 15, 0.45):
        out.append(text(left, y, line, "serif ink", 15, italic=True))
        y += 20
    y += 14
    facts = [
        ("Square", f"{state['position'] + 1} of {len(board.path)}"),
        ("Game", f"{state['game']}, turn {state['turn']}"),
        ("Games finished", str(state["games_finished"])),
    ]
    for label, value in facts:
        out.append(text(left, y, label, "sans ink2", 13))
        out.append(text(left + width, y, value, "sans ink", 13, anchor="end", weight=600))
        y += 22
    y += 10
    history = roll_history(state)
    count = len(history)
    out.append(text(left, y, f"Roll history ({count} roll{'s' if count != 1 else ''})", "sans ink2", 13))
    y += 10
    size, gap, per_row = 20, 3.5, 10
    shown = history[-HISTORY_SHOWN:]
    for k, entry in enumerate(shown):
        x = left + (k % per_row) * (size + gap)
        row_y = y + (k // per_row) * (size + gap)
        day = datetime.fromisoformat(entry["rolled_at"])
        tip = f"Roll {count - len(shown) + k + 1}: {entry['value']} on {entry['backend']}, {day.day} {day:%b %Y}\nMeasured |{entry['bits']}⟩"
        out.append(f"<g><title>{escape(tip)}</title>{die(x, row_y, size, entry['value'])}</g>")
    return y + ((len(shown) - 1) // per_row + 1) * (size + gap)


def die(x: float, y: float, size: float, value: int | None) -> str:
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}" rx="{size * 0.18:.1f}" fill="#ffffff" stroke="{PIP}" stroke-width="{max(1, size / 40):.1f}"/>'
    ]
    if value is None:
        parts.append(f'<text x="{x + size / 2:.1f}" y="{y + size * 0.68:.1f}" font-size="{size * 0.5:.0f}" text-anchor="middle" fill="{PIP}" font-family="{SANS}">?</text>')
    else:
        spots = {
            1: [(1, 1)],
            2: [(0, 0), (2, 2)],
            3: [(0, 0), (1, 1), (2, 2)],
            4: [(0, 0), (2, 0), (0, 2), (2, 2)],
            5: [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)],
            6: [(0, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)],
        }[value]
        for col, row in spots:
            cx = x + size * (0.25 + 0.25 * col)
            cy = y + size * (0.25 + 0.25 * row)
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{size * 0.085:.1f}" fill="{PIP}"/>')
    return "".join(parts)


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
    return "not reported" if value is None else f"{value * 100:.1f}%"


if __name__ == "__main__":
    main()
