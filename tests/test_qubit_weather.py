from datetime import datetime, timezone

import pytest
from qiskit_ibm_runtime.fake_provider import (
    FakeBrisbane,
    FakeFez,
    FakeKingston,
    FakeMarrakesh,
    FakeTorino,
)

import qubit_weather as qw


def square_lattice(rows: int, cols: int) -> qw.Snapshot:
    """Nighthawk-style square lattice, numbered row by row."""
    edges = {}
    for q in range(rows * cols):
        if q % cols < cols - 1:
            edges[(q, q + 1)] = 0.002
        if q + cols < rows * cols:
            edges[(q, q + cols)] = 0.003
    return qw.Snapshot(
        backend="square",
        num_qubits=rows * cols,
        gate="cz",
        calibrated=datetime(2026, 10, 23, 6, tzinfo=timezone.utc),
        readout={q: 0.01 for q in range(rows * cols)},
        edges=edges,
    )


@pytest.mark.parametrize("fake", [FakeFez, FakeTorino, FakeBrisbane, FakeKingston, FakeMarrakesh])
def test_heavy_hex_layout_puts_every_coupler_at_length_one(fake):
    snapshot = qw.read_snapshot(fake())

    layout = qw.infer_layout(snapshot)

    assert layout.bad_edges == []
    assert len(layout.coords) == snapshot.num_qubits
    assert len(set(layout.coords.values())) == snapshot.num_qubits


def test_square_lattice_fits_on_a_grid():
    snapshot = square_lattice(10, 12)

    layout = qw.infer_layout(snapshot)

    assert layout.bad_edges == []
    assert layout.size == (12, 10)


def test_square_layout_is_used_when_heavy_hex_does_not_fit():
    snapshot = square_lattice(4, 4)
    # A numbering where consecutive indices run down columns breaks the row inference.
    snapshot.edges = {tuple(sorted((r * 4 + c, (r + 1) * 4 + c))): 0.002 for r in range(3) for c in range(4)}
    snapshot.edges |= {(q, q + 1): 0.002 for q in range(16) if q % 4 < 3}

    layout = qw.infer_layout(snapshot)

    assert layout.bad_edges == []


def test_demo_forecast_for_fake_fez():
    snapshot = qw.read_snapshot(FakeFez())

    panel = qw.build_panel(snapshot, {})

    assert qw.panel_forecast(panel) == "Mostly fair. Best 5-qubit stretch: 122 to 126."
    assert qw.storm_report(panel) == "1 storm cell, over qubit 113"


def test_render_survives_missing_values():
    snapshot = qw.read_snapshot(FakeFez())
    for q in range(0, snapshot.num_qubits, 5):
        snapshot.t1.pop(q, None)
        snapshot.t2.pop(q, None)
        snapshot.readout.pop(q, None)
    for key in list(snapshot.edges)[::7]:
        snapshot.edges[key] = None
    empty = qw.Snapshot(backend="empty", num_qubits=3, gate=None, calibrated=None)

    svg = qw.render_fleet([snapshot, empty], {})

    assert "Readout error not reported" in svg
    assert "Grey: not reported" in svg


@pytest.mark.parametrize("count", [1, 2, 3])
def test_fleet_renders_with_fewer_than_four_backends(count):
    snapshots = [qw.read_snapshot(f()) for f in (FakeFez, FakeTorino, FakeKingston)][:count]

    svg = qw.render_fleet(snapshots, {})

    assert svg.count('<g id="panel-') == count


def test_headline_ranks_by_median_two_qubit_error():
    snapshots = [qw.read_snapshot(f()) for f in (FakeFez, FakeTorino, FakeKingston)]

    panels = [qw.build_panel(s, {}) for s in snapshots]

    assert qw.fleet_headline(panels) == "Stormiest today: fake_torino. Clearest: fake_kingston."


def test_record_reading_replaces_same_day_and_skips_repeat_calibrations():
    snapshot = square_lattice(2, 3)
    summary = qw.summarise(snapshot)

    history = qw.record_reading({}, snapshot, summary)
    repeat = qw.record_reading(history, snapshot, summary)
    snapshot.calibrated = snapshot.calibrated.replace(hour=9)
    same_day = qw.record_reading(history, snapshot, summary)

    assert repeat is history
    assert len(same_day["square"]) == 1
    assert same_day["square"][0]["calibrated"].startswith("2026-10-23T09:00")


def test_record_reading_keeps_sixty_days():
    snapshot = square_lattice(2, 3)
    summary = qw.summarise(snapshot)
    old = {"square": [{"date": f"2026-0{m}-01", "calibrated": f"2026-0{m}-01T06:00:00+00:00"} for m in (6, 9)]}

    history = qw.record_reading(old, snapshot, summary)

    assert [e["date"] for e in history["square"]] == ["2026-09-01", "2026-10-23"]
