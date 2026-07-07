import pytest

from person_detect.cli import FrameCadence


def test_frame_cadence_waits_until_next_processing_slot() -> None:
    cadence = FrameCadence(interval_seconds=0.5)
    times = iter([1.0, 1.2, 1.5])
    sleeps: list[float] = []

    assert cadence.wait(now_fn=lambda: next(times), sleep_fn=sleeps.append) == 1.0
    assert sleeps == []

    assert cadence.wait(now_fn=lambda: next(times), sleep_fn=sleeps.append) == 1.5
    assert sleeps == [pytest.approx(0.3)]
