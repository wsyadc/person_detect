import base64
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from person_detect.behavior import (
    BEHAVIOR_PROMPT,
    BehaviorAnalyzer,
    BehaviorWindow,
    build_messages,
    crop_frame,
    encode_frame_to_data_url,
)


@dataclass
class StubFuture:
    done_value: bool

    def done(self) -> bool:
        return self.done_value


class StubExecutor:
    def __init__(self) -> None:
        self.submissions: list[tuple] = []
        self.next_future = StubFuture(True)

    def submit(self, fn, *args):
        self.submissions.append((fn, args))
        return self.next_future


class StubAnalyzer:
    def classify_window(self, frame_urls):
        return ",".join(frame_urls)


def test_crop_frame_supports_base_and_expanded_target_boxes() -> None:
    frame = np.arange(100 * 120 * 3, dtype=np.uint8).reshape((100, 120, 3))

    base = crop_frame(frame, (40, 30, 80, 70), scale=1.0)
    expanded = crop_frame(frame, (40, 30, 80, 70), scale=1.5)

    assert base.shape == (40, 40, 3)
    assert np.array_equal(base, frame[30:70, 40:80])
    assert expanded.shape == (60, 60, 3)
    assert np.array_equal(expanded, frame[20:80, 30:90])


def test_encode_frame_to_data_url_returns_jpeg_data_url() -> None:
    frame = np.zeros((12, 16, 3), dtype=np.uint8)

    data_url = encode_frame_to_data_url(
        frame,
        jpeg_quality=80,
        width=32,
        height=24,
    )

    prefix = "data:image/jpeg;base64,"
    assert data_url.startswith(prefix)
    decoded = base64.b64decode(data_url.removeprefix(prefix))
    assert decoded.startswith(b"\xff\xd8")
    assert decoded.endswith(b"\xff\xd9")


def test_build_messages_contains_prompt_and_each_image_url() -> None:
    messages = build_messages(["data:image/jpeg;base64,one", "data:image/jpeg;base64,two"])

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": BEHAVIOR_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,one"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,two"},
                },
            ],
        }
    ]


def test_behavior_prompt_is_overridden_instead_of_editing_original_prompt() -> None:
    source = Path("src/person_detect/behavior.py").read_text()

    assert source.count("BEHAVIOR_PROMPT =") >= 2
    assert "Override the original prompt" in source


def test_behavior_prompt_focuses_on_behavior_only_and_requires_json() -> None:
    assert "目标人物定位已由上游人体检测、人脸匹配和跟踪模块完成" in BEHAVIOR_PROMPT
    assert "不要重新进行目标人物定位" in BEHAVIOR_PROMPT
    assert "不要输出「完全离席」" in BEHAVIOR_PROMPT
    assert '"behavior_name": "完全离席"' not in BEHAVIOR_PROMPT
    assert "仅输出一个 JSON 对象" in BEHAVIOR_PROMPT
    assert '"is_abnormal"' in BEHAVIOR_PROMPT
    assert '"behavior_type"' in BEHAVIOR_PROMPT
    assert '"behavior_name"' in BEHAVIOR_PROMPT
    assert '"evidence"' in BEHAVIOR_PROMPT


def test_behavior_window_submits_when_full_and_then_clears() -> None:
    analyzer = StubAnalyzer()
    executor = StubExecutor()
    window = BehaviorWindow(window_size=3, analyzer=analyzer, executor=executor)

    assert window.add("one") is None
    assert window.add("two") is None
    future = window.add("three")

    assert future is executor.next_future
    assert len(executor.submissions) == 1
    submitted_fn, submitted_args = executor.submissions[0]
    assert submitted_fn == analyzer.classify_window
    assert submitted_args == (["one", "two", "three"],)
    assert window.pending_count == 0


def test_behavior_window_queues_full_windows_when_previous_request_is_pending() -> None:
    analyzer = StubAnalyzer()
    executor = StubExecutor()
    executor.next_future = StubFuture(False)
    window = BehaviorWindow(window_size=2, analyzer=analyzer, executor=executor)

    first_future = window.add("one")
    assert first_future is None
    first_future = window.add("two")
    assert first_future is executor.next_future

    skipped = window.add("three")
    assert skipped is None
    skipped = window.add("four")

    assert skipped is executor.next_future
    assert len(executor.submissions) == 2
    assert window.pending_count == 0


def test_behavior_analyzer_uses_openai_compatible_chat_completion() -> None:
    class StubMessage:
        content = ' {"is_abnormal": false, "behavior_type": "", "behavior_name": "", "evidence": ""} '

    class StubChoice:
        message = StubMessage()

    class StubCompletions:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return type("Response", (), {"choices": [StubChoice()]})()

    class StubChat:
        def __init__(self) -> None:
            self.completions = StubCompletions()

    class StubClient:
        def __init__(self) -> None:
            self.chat = StubChat()

    client = StubClient()
    analyzer = BehaviorAnalyzer(client=client, model="Qwen")

    result = analyzer.classify_window(["data:image/jpeg;base64,one"])

    assert result == '{"is_abnormal": false, "behavior_type": "", "behavior_name": "", "evidence": ""}'
    call = client.chat.completions.calls[0]
    assert call["model"] == "Qwen"
    assert call["messages"] == build_messages(["data:image/jpeg;base64,one"])
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
