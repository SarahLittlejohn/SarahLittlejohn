import pytest

import qubit_snakes as qs


@pytest.fixture(scope="module")
def board():
    return qs.build_board(qs.load_board_backend())


def a_roll(value=3):
    return qs.Roll(value=value, bits="010", discarded=0, backend="test", job_id=None, rolled_at="2026-10-23T07:30:00+00:00")


def square_of(board, qubit):
    return board.path.index(qubit)


def first_calm_square(board, start=1):
    return next(i for i in range(start, board.last) if board.risk(board.path[i]) == "calm" and board.path[i] not in board.ladders)


def test_board_snakes_up_from_the_bottom_left(board):
    assert len(board.path) == 128
    assert board.path[:2] == [140, 141]
    assert board.path[16] == 135
    assert board.path[-1] == 0


def test_bridge_qubits_become_ladders(board):
    blocked = {ladder.middle: ladder.reason for ladder in board.ladders.values() if not ladder.is_open}

    assert len(board.ladders) == 28
    assert blocked == {17: "coupler", 78: "cat", 116: "coupler", 117: "coupler", 119: "cat", 139: "coupler"}
    assert all(square_of(board, l.top) > square_of(board, l.foot) for l in board.ladders.values())


def test_plain_move(board):
    target = first_calm_square(board, 2)

    to, event = qs.take_turn(board, target - 2, 2)

    assert (to, event["kind"]) == (target, "move")


@pytest.mark.parametrize(("risk", "penalty"), [("cat", 3), ("kitten", 2)])
def test_noisy_qubits_send_you_back(board, risk, penalty):
    landed = next(i for i in range(6, board.last) if board.risk(board.path[i]) == risk)

    to, event = qs.take_turn(board, landed - 1, 1)

    assert event["kind"] == risk
    assert to == landed - penalty


def test_penalty_never_goes_below_start(board):
    board = qs.Board(board.name, board.path, board.coords, {**board.readout, board.path[1]: 0.5}, board.ladders, board.gate)

    to, event = qs.take_turn(board, 0, 1)

    assert (to, event["kind"]) == (0, "cat")


def test_open_ladder_is_climbed(board):
    ladder = board.ladders[143]
    foot = square_of(board, 143)

    to, event = qs.take_turn(board, foot - 3, 3)

    assert ladder.is_open
    assert (to, event["kind"], event["middle"]) == (square_of(board, ladder.top), "ladder", 136)


def test_blocked_ladder_is_not_climbed(board):
    ladder = next(l for l in board.ladders.values() if not l.is_open and board.risk(l.foot) == "calm")
    square = square_of(board, ladder.foot)

    to, event = qs.take_turn(board, square - 1, 1)

    assert (to, event["kind"], event["reason"]) == (square, "blocked", ladder.reason)


def test_reaching_the_end_finishes_and_next_roll_starts_a_new_game(board):
    finished = qs.advance({"game": 2, "turn": 40, "position": board.last - 2, "games_finished": 1, "recent_rolls": []}, board, a_roll(6))
    restarted = qs.advance(finished, board, a_roll(1))

    assert finished["finished"] and finished["position"] == board.last
    assert finished["games_finished"] == 2
    assert (restarted["game"], restarted["turn"], restarted["previous"]) == (3, 1, 0)


def test_every_roll_is_kept_in_the_history(board):
    state = {}
    for value in (1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5):
        state = qs.advance(state, board, a_roll(value))

    assert len(state["history"]) == 23
    assert [e["value"] for e in state["history"][:3]] == [1, 2, 3]


def test_history_starts_from_the_last_roll_of_an_older_state_file(board):
    older = qs.advance({}, board, a_roll(6))
    del older["history"]
    older["recent_rolls"] = [6]

    state = qs.advance(older, board, a_roll(2))

    assert [e["value"] for e in state["history"]] == [6, 2]
    assert state["position"] == older["position"] + 2 or state["event"]["kind"] != "move"


@pytest.mark.parametrize(("bitstrings", "value", "discarded"), [(["000"], 1, 0), (["101"], 6, 0), (["111", "110", "011"], 4, 2)])
def test_dice_discards_six_and_seven(bitstrings, value, discarded):
    roll = qs.roll_from_bitstrings(bitstrings, "test", None)

    assert (roll.value, roll.discarded) == (value, discarded)


def test_dice_with_no_usable_shots_raises():
    with pytest.raises(ValueError):
        qs.roll_from_bitstrings(["110", "111"], "test", None)


def test_simulated_roll_is_a_dice_value():
    assert 1 <= qs.simulated_roll().value <= 6


@pytest.mark.parametrize("kind", ["move", "cat", "kitten", "ladder", "blocked", "finish"])
def test_every_event_renders(board, kind):
    event = {"kind": kind, "from": 0, "landed": 5, "to": 3, "qubit": 145, "middle": 136, "reason": "coupler"}
    state = qs.advance({}, board, a_roll()) | {"event": event}

    svg = qs.render(board, state)

    assert qs.event_text(board, event).split(",")[0] in svg


def test_render_before_first_roll(board):
    assert "Waiting for the" in qs.render(board, {})
