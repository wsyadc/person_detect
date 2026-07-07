from person_detect.tracking import PersonCandidate, TargetTracker


def test_tracker_emits_anchor_and_uses_explicit_absent_marker() -> None:
    tracker = TargetTracker(face_threshold=0.38, iou_threshold=0.25, lost_seconds=5.0)

    anchored = tracker.update(
        [PersonCandidate(box=(10, 10, 100, 200), face_score=0.82)],
        now=0.0,
    )
    assert anchored.event == "锚定成功"
    assert anchored.target_box == (10, 10, 100, 200)
    assert anchored.matched_by == "face"

    briefly_lost = tracker.update([], now=2.0)
    assert briefly_lost.event is None
    assert briefly_lost.target_box is None
    assert tracker.is_absent is False

    absent = tracker.update([], now=5.1)
    assert absent.event is None
    assert absent.target_box is None
    assert tracker.is_absent is False

    tracker.mark_absent()
    assert tracker.is_absent is True

    ignored_iou_only_return = tracker.update(
        [PersonCandidate(box=(11, 12, 101, 202), face_score=None)],
        now=6.0,
    )
    assert ignored_iou_only_return.event is None
    assert ignored_iou_only_return.target_box is None
    assert tracker.is_absent is True

    returned = tracker.update(
        [PersonCandidate(box=(15, 12, 105, 202), face_score=0.76)],
        now=7.0,
    )
    assert returned.event is None
    assert returned.target_box == (15, 12, 105, 202)
    assert returned.matched_by == "face"
    assert tracker.is_absent is False


def test_tracker_uses_iou_while_target_is_temporarily_face_occluded() -> None:
    tracker = TargetTracker(face_threshold=0.38, iou_threshold=0.25, lost_seconds=5.0)
    tracker.update([PersonCandidate(box=(10, 10, 100, 200), face_score=0.9)], now=0.0)

    tracked = tracker.update(
        [PersonCandidate(box=(14, 12, 104, 202), face_score=None)],
        now=0.5,
    )

    assert tracked.event is None
    assert tracked.target_box == (14, 12, 104, 202)
    assert tracked.matched_by == "iou"
    assert tracked.score is not None
    assert 0.0 < tracked.score < 1.0
    assert tracker.is_absent is False


def test_tracker_rejects_edge_clipped_iou_match_without_face_confirmation() -> None:
    tracker = TargetTracker(face_threshold=0.38, iou_threshold=0.25, lost_seconds=5.0)
    tracker.update(
        [PersonCandidate(box=(1706, 3, 1920, 1071), face_score=0.9)],
        now=0.0,
        image_size=(1920, 1080),
    )

    lost = tracker.update(
        [PersonCandidate(box=(1700, 420, 1919, 845), face_score=None)],
        now=0.5,
        image_size=(1920, 1080),
    )

    assert lost.event is None
    assert lost.target_box is None
    assert tracker.is_absent is False
