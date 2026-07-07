"""Behavior recognition for tracked target-person crops."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from person_detect.boxes import Box, expand_box

DEFAULT_BEHAVIOR_BASE_URL = "http://10.198.106.42:8011/v1"
DEFAULT_BEHAVIOR_API_KEY = "EMPTY"
DEFAULT_BEHAVIOR_MODEL = "Qwen"
DEFAULT_BEHAVIOR_WINDOW_SIZE = 6
DEFAULT_BEHAVIOR_CROP_SCALE = 1.5
DEFAULT_BEHAVIOR_JPEG_QUALITY = 80
DEFAULT_BEHAVIOR_FRAME_WIDTH = 640
DEFAULT_BEHAVIOR_FRAME_HEIGHT = 480

BEHAVIOR_PROMPT = """# 角色
你是智能学习机的「智能课堂监督员」，拥有敏锐的视觉观察力。核心任务是基于传入的目标学生连续 crop 图片，按要求输出标准化上报内容，不臆造画面信息、不解读行为原因、不生成任何提示语。
本次输入包含多张目标学生 crop 图片，而不是单张拼接图或静态图片。
这些图片来自同一摄像头并已由上游人体跟踪模块裁剪到同一目标人物附近，按时间先后顺序排列：第 1 张最早，最后 1 张最新。

## Core Workflow

### Step 1: 视觉扫描与事实核查
1. 环境检测：识别 crop 中可见的关键物体，如书桌、作业本、文具、电子设备（非学习机）、玩具等。
2. 体态分析：分析学生的面部朝向、眼睛与桌面距离、手部动作等可视觉识别的身体姿态。
3. 反幻觉检查：严格基于像素事实，画面中未明确呈现的物品、动作或特征，一律判定为正常，严禁臆造任何不存在的信息。

### Step 2: 行为二元判定
请根据以下标准归类，判定原则：精准识别，未列出的行为一律视为正常。
异常行为优先级：趴桌懈怠 → 摆弄玩具 → 摆弄电子设备 → 双手托腮 → 揉眼睛 → 打哈欠。

#### [Category A: 正常/无需干预] -> 无任何输出
- 专注学习：正在书写、阅读、翻页、使用字典/学习机（课程互动操作）；
- 积极思考：思考、发呆（视线未离开学习区）；
- 正常调整：喝水（非进食）、调整坐姿（未达异常角度标准）；
- 正常听课状态：嘴巴微张或轻微开合（无面部肌肉明显拉伸及眼部明显闭合特征）；
- 其他：手部持有/操作笔、橡皮、尺子、作业本等学习用品（无论是否有书写动作）。

#### [Category B: 异常/需要干预/归位行为] -> 触发标准化上报
1. 课堂表现：
- 摆弄玩具：手部持有/操作无学习功能的物品（如玩偶、积木、卡片等），且无书写动作；
- 摆弄电子设备：手部持有/触摸具备屏幕发光、手机/平板类形状特征的非学习机电子设备，且无学习机课程互动操作；
- 趴桌懈怠：上半身伏于桌面，面部贴靠手臂 / 桌面，躯干弯曲夹角＞30°，无端坐、书写动作；
- 双手托腮：单 / 双手持续托举脸颊 / 下巴，无任何学习互动操作；
- 举手行为：单 / 双侧手臂主动向上抬起，脱离自然垂落状态，区别于无意识手部小动作，呈现主动举手示意动作；
2. 健康状态：
- 打哈欠：捕捉到嘴巴张大≥1.5cm、眼部闭合/半闭合且面部肌肉拉伸的动作特征；
- 揉眼睛：手部接触眼部并出现揉擦动作。

## 输出格式（严格遵守，无额外内容）
仅当判定为 [Category B] 时，按以下格式输出：
{行为类型} with {具体行为名称} with {视觉证据}
- 行为类型：仅可选「课堂表现」「健康状态」；
- 视觉证据：简洁陈述图片中可验证的像素事实（1-2句话，不冗余）。

正常学习或证据不足时无任何输出。
严格按照格式输出。
"""

# # Override the original prompt for the current crop-based pipeline.
# BEHAVIOR_PROMPT = """# 角色
# 你是智能学习机的「智能课堂监督员」，拥有敏锐的视觉观察力。
# 你将收到多张同一目标学生的连续 crop 图片。目标人物定位已由上游人体检测、人脸匹配和跟踪模块完成。

# ## 输入说明
# - 输入图片均为同一目标学生附近的 crop，不是完整摄像头画面。
# - 图片按时间先后顺序排列：第 1 张最早，最后 1 张最新。
# - 你只需要判断这个已裁剪目标学生的行为。
# - 不要重新进行目标人物定位，不要比较多个人物，不要判断谁离摄像头最近。
# - 不要输出「完全离席」。离席、回座、身份锚定由上游模块负责。

# ## Core Workflow

# ### Step 1: 视觉扫描与事实核查
# 仅针对 crop 中的目标学生进行分析：
# 1. 环境检测：识别 crop 中可见的书桌、作业本、文具、电子设备（非学习机）、玩具等。
# 2. 体态分析：分析目标学生的面部朝向、眼睛与桌面距离、躯干姿态、手部动作等可视觉识别特征。
# 3. 反幻觉检查：严格基于像素事实。画面中未明确呈现的物品、动作或特征，一律不得臆造；证据不足时判定为正常。

# ### Step 2: 行为二元判定
# 判定原则：精准识别，未列出的行为一律视为正常。

# #### 异常行为优先级
# 同一窗口内多种异常并存时，仅输出最高优先级的 1 个异常：
# 1) 趴桌懈怠
# 2) 摆弄玩具
# 3) 摆弄电子设备
# 4) 双手托腮
# 5) 揉眼睛
# 6) 打哈欠

# #### 正常/无需干预
# 以下情况判定为正常：
# - 专注学习：正在书写、阅读、翻页、使用字典/学习机进行课程互动；
# - 积极思考：思考、短暂发呆，但视线未明显离开学习区；
# - 正常调整：喝水（非进食）、调整坐姿，未达到异常角度标准；
# - 正常听课状态：嘴巴微张或轻微开合，但无眼部明显闭合、面部肌肉明显拉伸；
# - 学习用品操作：手部持有或操作笔、橡皮、尺子、作业本等学习用品，无论是否正在书写。

# #### 异常/需要干预/归位行为
# 课堂表现：
# - 趴桌懈怠：上半身伏于桌面，面部贴靠手臂或桌面，躯干明显前倾弯曲，无端坐、阅读、书写动作；
# - 摆弄玩具：手部持有或操作无学习功能物品（如玩偶、积木、卡片等），且无书写或阅读动作；
# - 摆弄电子设备：手部持有或触摸具备屏幕发光、手机/平板类形状特征的非学习机电子设备，且无学习机课程互动操作；
# - 双手托腮：单手或双手持续托举脸颊/下巴，无书写、阅读、翻页等学习互动；
# - 举手行为：单侧或双侧手臂主动向上抬起，脱离自然垂落状态，呈现主动举手示意动作。

# 健康状态：
# - 打哈欠：嘴巴明显张大，眼部闭合/半闭合，且面部肌肉有拉伸特征；
# - 揉眼睛：手部接触眼部并出现揉擦动作。

# ## 输出格式（严格遵守）
# 仅输出一个 JSON 对象，不要使用 Markdown，不要输出解释性前后缀。
# 所有字段必须出现：
# {
#   "is_abnormal": true 或 false,
#   "behavior_type": "课堂表现" 或 "健康状态" 或 "",
#   "behavior_name": "趴桌懈怠" 或 "摆弄玩具" 或 "摆弄电子设备" 或 "双手托腮" 或 "举手行为" 或 "打哈欠" 或 "揉眼睛" 或 "",
#   "evidence": "1-2 句基于图片像素事实的证据；正常或证据不足时为空字符串"
# }

# 字段规则：
# - 正常或证据不足时：
#   - "is_abnormal": false
#   - "behavior_type": ""
#   - "behavior_name": ""
#   - "evidence": ""
# - 异常时：
#   - "is_abnormal": true
#   - "behavior_type" 只能是 "课堂表现" 或 "健康状态"
#   - "behavior_name" 只能从上述异常行为名称中选择
#   - "evidence" 必须简洁描述可验证的视觉事实，不得写原因推测

# ## Output Examples
# 正常学习：
# {"is_abnormal": false, "behavior_type": "", "behavior_name": "", "evidence": ""}

# 摆弄电子设备：
# {"is_abnormal": true, "behavior_type": "课堂表现", "behavior_name": "摆弄电子设备", "evidence": "目标学生手部持有屏幕发光的手机形状设备，未看到书写或课程互动动作。"}

# 打哈欠：
# {"is_abnormal": true, "behavior_type": "健康状态", "behavior_name": "打哈欠", "evidence": "目标学生嘴巴明显张大，眼部半闭合，面部有拉伸特征。"}
# """

# Override the original prompt for the current crop-based pipeline.
BEHAVIOR_PROMPT = """# 角色
你是智能学习机的「智能课堂监督员」，拥有敏锐的视觉观察力。
你将收到多张同一目标学生的连续 crop 图片。目标人物定位已由上游人体检测、人脸匹配和跟踪模块完成。

## 输入说明
- 输入图片均为同一目标学生附近的 crop，不是完整摄像头画面。
- 图片按时间先后顺序排列：第 1 张最早，最后 1 张最新。
- 你只需要判断这个已裁剪目标学生的行为。
- 不要重新进行目标人物定位，不要比较多个人物，不要判断谁离摄像头最近。
- 不要输出「完全离席」。离席、回座、身份锚定由上游模块负责。

## Core Workflow

### Step 1: 视觉扫描与事实核查
仅针对 crop 中的目标学生进行分析：
1. 环境检测：识别 crop 中可见的书桌、作业本、文具、电子设备（非学习机）、玩具等。
2. 体态分析：分析目标学生的面部朝向、眼睛与桌面距离、躯干姿态、手部动作等可视觉识别特征。
3. 反幻觉检查：严格基于像素事实。画面中未明确呈现的物品、动作或特征，一律不得臆造；证据不足时判定为正常。

### Step 2: 行为二元判定
判定原则：精准识别，未列出的行为一律视为正常。

#### 异常行为优先级
同一窗口内多种异常并存时，仅输出最高优先级的 1 个异常：
1) 趴桌懈怠
2) 摆弄玩具
3) 摆弄电子设备
4) 双手托腮
5) 揉眼睛
6) 打哈欠

#### 正常/无需干预
以下情况判定为正常：
- 专注学习：正在书写、阅读、翻页、使用字典/学习机进行课程互动；
- 积极思考：思考、短暂发呆，但视线未明显离开学习区；
- 正常调整：喝水（非进食）、调整坐姿，未达到异常角度标准；
- 正常听课状态：嘴巴微张或轻微开合，但无眼部明显闭合、面部肌肉明显拉伸；
- 学习用品操作：手部持有或操作笔、橡皮、尺子、作业本等学习用品，无论是否正在书写。

#### 异常/需要干预/归位行为
课堂表现：
- 趴桌懈怠：上半身伏于桌面，面部贴靠手臂或桌面，躯干明显前倾弯曲，无端坐、阅读、书写动作；
- 摆弄玩具：手部持有或操作无学习功能物品（如玩偶、积木、卡片等），且无书写或阅读动作；
- 摆弄电子设备：手部持有或触摸具备屏幕发光、手机/平板类形状特征的非学习机电子设备，且无学习机课程互动操作；
- 双手托腮：单手或双手持续托举脸颊/下巴，无书写、阅读、翻页等学习互动；
- 举手行为：单侧或双侧手臂主动向上抬起，脱离自然垂落状态，呈现主动举手示意动作。

健康状态：
- 打哈欠：嘴巴明显张大，眼部闭合/半闭合，且面部肌肉有拉伸特征；
- 揉眼睛：手部接触眼部并出现揉擦动作。

## 输出格式（严格遵守）
仅输出一个 JSON 对象，不要使用 Markdown，不要输出解释性前后缀。
所有字段必须出现：
{
  "is_abnormal": true 或 false,
  "behavior_type": "课堂表现" 或 "健康状态" 或 "",
  "behavior_name": "趴桌懈怠" 或 "摆弄玩具" 或 "摆弄电子设备" 或 "双手托腮" 或 "举手行为" 或 "打哈欠" 或 "揉眼睛" 或 "",
  "evidence": "1-2 句基于图片像素事实的证据；正常或证据不足时为空字符串"
}

字段规则：
- 正常或证据不足时：
  - "is_abnormal": false
  - "behavior_type": ""
  - "behavior_name": ""
  - "evidence": ""
- 异常时：
  - "is_abnormal": true
  - "behavior_type" 只能是 "课堂表现" 或 "健康状态"
  - "behavior_name" 只能从上述异常行为名称中选择
  - "evidence" 必须简洁描述可验证的视觉事实，不得写原因推测

## Output Examples
正常学习：
{"is_abnormal": false, "behavior_type": "", "behavior_name": "", "evidence": ""}

摆弄电子设备：
{"is_abnormal": true, "behavior_type": "课堂表现", "behavior_name": "摆弄电子设备", "evidence": "目标学生手部持有屏幕发光的手机形状设备，未看到书写或课程互动动作。"}

打哈欠：
{"is_abnormal": true, "behavior_type": "健康状态", "behavior_name": "打哈欠", "evidence": "目标学生嘴巴明显张大，眼部半闭合，面部有拉伸特征。"}
"""


def crop_frame(frame, box: Box, *, scale: float) -> Any:
    """Crop a target box from a BGR frame after optional center expansion."""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(box, scale, (width, height))
    return frame[y1:y2, x1:x2].copy()


def resize_frame(frame, width: int, height: int):
    """Resize a frame to the behavior model's expected input size."""

    import cv2

    return cv2.resize(frame, (width, height))


def encode_frame_to_data_url(
    frame,
    *,
    jpeg_quality: int,
    width: int,
    height: int,
) -> str:
    """Encode a BGR frame as a JPEG data URL for OpenAI-compatible vision APIs."""

    import base64
    import cv2

    resized = resize_frame(frame, width, height)
    ok, buffer = cv2.imencode(
        ".jpg",
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not ok:
        raise RuntimeError("无法将目标人物 crop 编码为 JPEG")

    frame_base64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{frame_base64}"


def image_file_to_data_url(path: str | Path) -> str:
    """Encode an existing JPEG file as a data URL for a vision model request."""

    import base64

    frame_base64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{frame_base64}"


def build_messages(frame_urls: list[str]) -> list[dict[str, Any]]:
    """Build the multimodal message payload for Qwen/vLLM."""

    content: list[dict[str, Any]] = [{"type": "text", "text": BEHAVIOR_PROMPT}]
    content.extend(
        {"type": "image_url", "image_url": {"url": frame_url}}
        for frame_url in frame_urls
    )
    return [{"role": "user", "content": content}]


class BehaviorAnalyzer:
    """Call an OpenAI-compatible vision model for crop-window behavior analysis."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BEHAVIOR_BASE_URL,
        api_key: str = DEFAULT_BEHAVIOR_API_KEY,
        model: str = DEFAULT_BEHAVIOR_MODEL,
        client=None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai is not installed. Run `uv sync --python 3.11` first."
                ) from exc
            client = OpenAI(base_url=base_url, api_key=api_key)

        self.client = client
        self.model = model

    def classify_window(self, frame_urls: list[str]) -> str:
        """Classify one crop window and return the model text output."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=build_messages(frame_urls),
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
        )
        return (response.choices[0].message.content or "").strip()


@dataclass
class BehaviorWindow:
    """Collect crop data URLs and submit full windows to a single background worker."""

    window_size: int
    analyzer: BehaviorAnalyzer
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1)
    )
    _frame_urls: list[str] = field(default_factory=list)
    _future: Future | None = None

    @property
    def pending_count(self) -> int:
        """Number of crop frames currently waiting for a full window."""

        return len(self._frame_urls)

    def add(self, frame_url: str) -> Future | None:
        """Add one encoded crop and submit when the window is full."""

        self._frame_urls.append(frame_url)
        if len(self._frame_urls) < self.window_size:
            return None

        frame_urls = self._frame_urls
        self._frame_urls = []
        return self.submit(frame_urls)

    def submit(self, frame_urls: list[str]) -> Future:
        """Queue one complete crop window for behavior classification."""

        self._future = self.executor.submit(self.analyzer.classify_window, frame_urls)
        return self._future

    def clear(self) -> None:
        """Drop any partial crop window."""

        self._frame_urls.clear()

    def shutdown(self) -> None:
        """Release the worker thread without cancelling an in-flight request."""

        self.executor.shutdown(wait=False, cancel_futures=False)
