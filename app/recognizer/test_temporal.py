from app.recognizer.temporal import TemporalRecognizer


def make_result(name, similarity, recognized=True):
    return {
        "recognized": recognized,
        "name": name,
        "similarity": similarity,
    }


def test_recognizes_after_minimum_votes():
    temporal = TemporalRecognizer(
        window_size=5,
        min_votes=3,
    )

    temporal.update(
        1,
        make_result("Dhruv", 0.70),
    )

    temporal.update(
        1,
        make_result("Dhruv", 0.72),
    )

    result = temporal.update(
        1,
        make_result("Dhruv", 0.71),
    )

    assert result["recognized"] is True
    assert result["name"] == "Dhruv"


def test_unknown_does_not_get_recognized():
    temporal = TemporalRecognizer(
        window_size=5,
        min_votes=3,
    )

    result = temporal.update(
        1,
        make_result(
            None,
            0.30,
            recognized=False,
        ),
    )

    assert result["recognized"] is False
    assert result["name"] is None


def test_multiple_tracks_are_independent():
    temporal = TemporalRecognizer(
        window_size=5,
        min_votes=2,
    )

    temporal.update(
        1,
        make_result("Dhruv", 0.70),
    )

    result = temporal.update(
        2,
        make_result("Alice", 0.80),
    )

    assert result["recognized"] is False

    result = temporal.update(
        1,
        make_result("Dhruv", 0.72),
    )

    assert result["recognized"] is True
    assert result["name"] == "Dhruv"

    result = temporal.update(
        2,
        make_result("Alice", 0.82),
    )

    assert result["recognized"] is True
    assert result["name"] == "Alice"


def test_majority_vote_wins():
    temporal = TemporalRecognizer(
        window_size=5,
        min_votes=3,
    )

    temporal.update(1, make_result("Dhruv", 0.70))
    temporal.update(1, make_result("Alice", 0.80))
    temporal.update(1, make_result("Dhruv", 0.71))

    result = temporal.update(
        1,
        make_result("Dhruv", 0.73),
    )

    assert result["recognized"] is True
    assert result["name"] == "Dhruv"


def test_history_is_limited_to_window():
    temporal = TemporalRecognizer(
        window_size=3,
        min_votes=3,
    )

    temporal.update(1, make_result("Alice", 0.80))
    temporal.update(1, make_result("Alice", 0.81))
    temporal.update(1, make_result("Alice", 0.82))

    result = temporal.update(
        1,
        make_result("Dhruv", 0.90),
    )

    assert result["recognized"] is False