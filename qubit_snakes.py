"""Qubit snakes and ladders: one quantum dice roll a day on an IBM Quantum board.

The board is the heavy-hex layout of IBM's Fez processor (the fake_fez
calibration snapshot). Row qubits are the squares, bridge qubits are the
ladders, and noisy qubits send you back. Each day the dice is rolled by
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

STORM_READOUT = 0.02
CHOPPY_READOUT = 0.012
GREEN_RUNG = 0.01
STORM_PENALTY = 3
CHOPPY_PENALTY = 2
DICE_QUBITS = 3
DICE_SHOTS = 16
JOB_WAIT_SECONDS = 40 * 60
RECENT_ROLLS = 10
MIN_ROW_LENGTH = 3

WIDTH = 1000
MARGIN = 32
CELL = 40
FRAME = 14
SIDEBAR_GAP = 34
SERIF = "Georgia, 'Iowan Old Style', 'Palatino Linotype', 'Book Antiqua', 'Times New Roman', serif"
SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

# The board is a physical object, so it keeps the same colours in light and dark mode.
WOOD = "#8a5a2e"
BOARD = "#e9dbb8"
TILE_LIGHT = "#fbf5e6"
TILE_DARK = "#f1e4c6"
TILE_LINE = "#cdb98f"
TILE_INK = "#6b5a3a"
CHOPPY_TILE = "#f6c86b"
CHOPPY_INK = "#7a4b00"
STORM_TILE = "#ef8f80"
STORM_INK = "#7d1d12"
LADDER_OPEN = "#2f7d32"
LADDER_BLOCKED = "#b3261e"
PAWN = "#6929c4"
PIP = "#1f2328"


@dataclass
class Ladder:
    foot: int
    middle: int
    top: int
    is_open: bool
    reason: str | None


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
        if error is None or error > STORM_READOUT:
            return "storm"
        return "choppy" if error > CHOPPY_READOUT else "calm"

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
        rungs = [edges[tuple(sorted((middle, e)))] for e in ends]
        reason = None
        if readout.get(middle, 1.0) > STORM_READOUT:
            reason = "storm"
        elif max(rungs) >= GREEN_RUNG:
            reason = "rung"
        ladders[foot] = Ladder(foot=foot, middle=middle, top=top, is_open=reason is None, reason=reason)
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
        to = max(landed - (STORM_PENALTY if risk == "storm" else CHOPPY_PENALTY), 0)
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
        "recent_rolls": (state.get("recent_rolls", []) + [roll.value])[-RECENT_ROLLS:],
        "pending_job": None,
    }


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
    if kind in ("storm", "choppy"):
        penalty = STORM_PENALTY if kind == "storm" else CHOPPY_PENALTY
        weather = "a storm" if kind == "storm" else "choppy water"
        return f"{start}: {weather} on qubit {qubit} ({pct(board.readout.get(qubit))} readout error). Back {penalty} to square {to}."
    if kind == "ladder":
        return f"{start} and climbed the green ladder through qubit {event['middle']} to square {to}."
    if kind == "blocked":
        why = "a storm on its middle qubit" if event["reason"] == "storm" else "a shaky rung"
        return f"{start}, but the ladder through qubit {event['middle']} is blocked by {why}."
    return f"{start}."


def render(board: Board, state: dict) -> str:
    out: list[str] = []
    out.append(text(MARGIN, 52, "Qubit snakes and ladders", "serif ink", 30))
    out.append(
        text(
            MARGIN,
            80,
            f"One quantum dice roll a day, on a board made from the layout of IBM's Fez processor ({board.name} calibration data).",
            "sans ink2",
            14,
        )
    )
    top = 108
    cols = max(x for x, _ in board.coords.values()) + 1
    rows = max(y for _, y in board.coords.values()) + 1
    board_left = MARGIN + FRAME
    board_top = top + FRAME
    out.append(
        f'<rect x="{MARGIN}" y="{top}" width="{cols * CELL + 2 * FRAME}" height="{rows * CELL + 2 * FRAME}" rx="14" fill="{WOOD}"/>'
    )
    out.append(f'<rect x="{board_left}" y="{board_top}" width="{cols * CELL}" height="{rows * CELL}" rx="4" fill="{BOARD}"/>')

    def centre(q: int) -> tuple[float, float]:
        x, y = board.coords[q]
        return board_left + (x + 0.5) * CELL, board_top + (y + 0.5) * CELL

    for index, qubit in enumerate(board.path):
        out.append(tile(board, index, qubit, centre(qubit)))
    for ladder in board.ladders.values():
        out.append(ladder_svg(board, ladder, centre))
    if state:
        if state["previous"] != state["position"]:
            out.append(pawn(*centre(board.path[state["previous"]]), ghost=True))
        out.append(pawn(*centre(board.path[state["position"]])))
    else:
        out.append(pawn(*centre(board.path[0])))

    board_bottom = top + rows * CELL + 2 * FRAME
    legend_bottom = legend(out, MARGIN, board_bottom + 30)
    sidebar_left = MARGIN + cols * CELL + 2 * FRAME + SIDEBAR_GAP
    sidebar_bottom = sidebar(out, sidebar_left, top, WIDTH - MARGIN - sidebar_left, board, state)
    height = max(legend_bottom, sidebar_bottom) + MARGIN

    summary = event_text(board, state["event"]) if state else "Waiting for the first roll."
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="qs-title qs-desc">',
            '<title id="qs-title">Qubit snakes and ladders</title>',
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


def tile(board: Board, index: int, qubit: int, centre: tuple[float, float]) -> str:
    cx, cy = centre
    x, y = cx - CELL / 2, cy - CELL / 2
    risk = board.risk(qubit)
    fill = {"storm": STORM_TILE, "choppy": CHOPPY_TILE}.get(risk, TILE_DARK if index % 2 else TILE_LIGHT)
    tip = f"Square {index + 1}: qubit {qubit}\nReadout error {pct(board.readout.get(qubit))}"
    if risk != "calm":
        tip += f"\n{'Storm' if risk == 'storm' else 'Choppy'}: back {STORM_PENALTY if risk == 'storm' else CHOPPY_PENALTY}"
    label = "Start" if index == 0 else "Finish" if index == board.last else str(index + 1)
    parts = [
        f'<g><title>{tip}</title>',
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{CELL}" height="{CELL}" fill="{fill}" stroke="{TILE_LINE}" stroke-width="1"/>',
        f'<text x="{x + 3:.1f}" y="{y + 11:.1f}" font-size="{8 if index in (0, board.last) else 9.5}" fill="{TILE_INK}" font-family="{SANS}">{label}</text>',
    ]
    if risk != "calm":
        ink = STORM_INK if risk == "storm" else CHOPPY_INK
        penalty = STORM_PENALTY if risk == "storm" else CHOPPY_PENALTY
        parts.append(snake(cx - 1, cy + 3, ink, 1.0 if risk == "storm" else 0.8))
        parts.append(
            f'<text x="{x + CELL - 3:.1f}" y="{y + CELL - 4:.1f}" font-size="10" font-weight="700" text-anchor="end" fill="{ink}" font-family="{SANS}">−{penalty}</text>'
        )
    if index == board.last:
        parts.append(star(cx, cy + 4, 9))
    parts.append("</g>")
    return "".join(parts)


def snake(x: float, y: float, colour: str, scale: float) -> str:
    return (
        f'<g transform="translate({x:.1f} {y:.1f}) scale({scale})">'
        f'<path d="M-11,5 C-8,-4 -3,-4 -1,1 S6,7 9,-3" fill="none" stroke="{colour}" stroke-width="3.2" stroke-linecap="round"/>'
        f'<circle cx="9.5" cy="-4" r="3" fill="{colour}"/>'
        f'<circle cx="10.3" cy="-4.8" r="0.9" fill="#ffffff"/>'
        "</g>"
    )


def star(cx: float, cy: float, r: float) -> str:
    points = []
    for k in range(10):
        radius = r if k % 2 == 0 else r * 0.45
        angle = -math.pi / 2 + k * math.pi / 5
        points.append(f"{cx + radius * math.cos(angle):.1f},{cy + radius * math.sin(angle):.1f}")
    return f'<polygon points="{" ".join(points)}" fill="#e3a008" stroke="{TILE_INK}" stroke-width="0.8"/>'


def ladder_svg(board: Board, ladder: Ladder, centre) -> str:
    (fx, fy), (tx, ty) = centre(ladder.foot), centre(ladder.top)
    colour = LADDER_OPEN if ladder.is_open else LADDER_BLOCKED
    dash = "" if ladder.is_open else ' stroke-dasharray="5 3"'
    rail = 7
    # Stop at the tile edges so the ladder never hides a square's number or penalty.
    y0, y1 = fy - CELL / 2 + 2, ty + CELL / 2 - 2
    rungs = "".join(
        f'<line x1="{fx - rail:.1f}" y1="{y:.1f}" x2="{fx + rail:.1f}" y2="{y:.1f}" stroke="{colour}" stroke-width="2.5"{dash}/>'
        for y in frange(y0 - 4, y1 + 3, 8)
    )
    rails = "".join(
        f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" stroke="{colour}" stroke-width="3" stroke-linecap="round"{dash}/>'
        for x in (fx - rail, fx + rail)
    )
    middle = ladder.middle
    mx, my = centre(middle)
    risk = board.risk(middle)
    dot = {"storm": STORM_TILE, "choppy": CHOPPY_TILE}.get(risk, TILE_LIGHT)
    status = "Open: climb!" if ladder.is_open else "Blocked by a storm on the middle qubit" if ladder.reason == "storm" else "Blocked by a shaky rung"
    tip = (
        f"Ladder through qubit {middle}: square {board.path.index(ladder.foot) + 1} to {board.path.index(ladder.top) + 1}\n"
        f"Middle qubit readout error {pct(board.readout.get(middle))}\n{status}"
    )
    badge = (
        f'<path d="M{mx - 3:.1f},{my - 3:.1f} L{mx + 3:.1f},{my + 3:.1f} M{mx + 3:.1f},{my - 3:.1f} L{mx - 3:.1f},{my + 3:.1f}" stroke="{LADDER_BLOCKED}" stroke-width="2" stroke-linecap="round"/>'
        if not ladder.is_open
        else ""
    )
    return (
        f"<g><title>{tip}</title>{rungs}{rails}"
        f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="7" fill="{dot}" stroke="{colour}" stroke-width="2"/>{badge}</g>'
    )


def frange(start: float, stop: float, step: float) -> list[float]:
    values, value = [], start
    while value >= stop:
        values.append(value)
        value -= step
    return values


def pawn(cx: float, cy: float, ghost: bool = False) -> str:
    cx, cy = cx + 4, cy + 5
    style = (
        f'fill="none" stroke="{PAWN}" stroke-width="1.5" stroke-dasharray="3 2" opacity="0.7"'
        if ghost
        else f'fill="{PAWN}" stroke="#ffffff" stroke-width="1.5"'
    )
    label = "Previous square" if ghost else "You are here"
    return (
        f"<g><title>{label}</title>"
        f'<path d="M{cx - 12:.1f},{cy + 15:.1f} Q{cx:.1f},{cy + 8:.1f} {cx + 12:.1f},{cy + 15:.1f} L{cx + 6:.1f},{cy:.1f} Q{cx:.1f},{cy - 3:.1f} {cx - 6:.1f},{cy:.1f} Z" {style}/>'
        f'<circle cx="{cx:.1f}" cy="{cy - 8:.1f}" r="7.5" {style}/>'
        "</g>"
    )


def legend(out: list[str], left: float, top: float) -> float:
    y = top
    items = [
        (f'<rect x="{left}" y="{y - 13}" width="18" height="18" fill="{TILE_LIGHT}" stroke="{TILE_LINE}"/>', "Calm qubit"),
        (f'<rect x="{left}" y="{y - 13}" width="18" height="18" fill="{CHOPPY_TILE}" stroke="{TILE_LINE}"/>', f"Choppy: readout error {pct(CHOPPY_READOUT)} to {pct(STORM_READOUT)}, back {CHOPPY_PENALTY}"),
        (f'<rect x="{left}" y="{y - 13}" width="18" height="18" fill="{STORM_TILE}" stroke="{TILE_LINE}"/>', f"Storm: readout error above {pct(STORM_READOUT)}, back {STORM_PENALTY}"),
    ]
    x = left
    widths = (130, 320, 260)
    for (swatch, label), width in zip(items, widths):
        out.append(swatch.replace(f'x="{left}"', f'x="{x}"'))
        out.append(text(x + 26, y, label, "sans ink2", 13))
        x += width
    y += 30
    out.append(mini_ladder(left + 9, y - 4, LADDER_OPEN, False))
    out.append(text(left + 26, y, f"Green ladder: every rung below {pct(GREEN_RUNG)} CZ error, so climb it", "sans ink2", 13))
    out.append(mini_ladder(left + 439, y - 4, LADDER_BLOCKED, True))
    out.append(text(left + 456, y, "Blocked: a storm on the middle qubit or a shaky rung", "sans ink2", 13))
    return y + 8


def mini_ladder(cx: float, cy: float, colour: str, dashed: bool) -> str:
    dash = ' stroke-dasharray="3 2"' if dashed else ""
    parts = [f'<line x1="{x}" y1="{cy - 10}" x2="{x}" y2="{cy + 10}" stroke="{colour}" stroke-width="2.5"{dash}/>' for x in (cx - 5, cx + 5)]
    parts += [f'<line x1="{cx - 5}" y1="{y}" x2="{cx + 5}" y2="{y}" stroke="{colour}" stroke-width="2"/>' for y in (cy - 6, cy, cy + 6)]
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
    out.append(text(left, y, "Recent rolls", "sans ink2", 13))
    y += 10
    size, gap = 20, 3.5
    for k, value in enumerate(state["recent_rolls"]):
        out.append(die(left + k * (size + gap), y, size, value))
    return y + size


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
