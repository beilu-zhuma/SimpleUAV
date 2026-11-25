"""单个 OpenAI-compatible 多模态调用；不依赖模型 SDK 或训练框架。"""

import json
import urllib.request
from pathlib import Path

from lightning_vln.core.context import MEMORY_CHARS, REASON_CHARS, Context, image_content
from lightning_vln.core.geometry import vector
from lightning_vln.core.types import Decision, Observation, VIEWS
from lightning_vln.core.evidence import LOCALIZATION_SCHEMA, NAVIGATION_PROMPT, PROMPT_VERSION as EVIDENCE_PROMPT_VERSION, bind_target, localization_messages, navigation_schema, parse_localization
from lightning_vln.core.motion_options import NAVIGATION_PROMPT as MOTION_PROMPT, PROMPT_VERSION as MOTION_OPTIONS_PROMPT_VERSION, bind_motion


PROMPT_VERSION = "visual_search_v4_directional_explore"

from lightning_vln.core.prompts import SYSTEM_PROMPT

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["action", "view", "bbox", "reason", "memory", "vertical_direction"],
    "properties": {
        "action": {"type": "string", "enum": ["horizontal_explore", "vertical_explore", "approach", "stop"]},
        "vertical_direction": {"type": ["string", "null"], "enum": ["up", "down", None]},
        "view": {"type": "string", "enum": list(VIEWS)},
        "bbox": {"type": "array", "minItems": 4, "maxItems": 4,
                 "items": {"type": "number", "minimum": 0, "maximum": 1000}},
        "reason": {"type": "string", "minLength": 1, "maxLength": REASON_CHARS},
        "memory": {"type": "string", "maxLength": MEMORY_CHARS},
    },
}


def parse_decision(content: str, catalog=None, stop_distance=10) -> Decision:
    payload = json.loads(content)
    required = set(SCHEMA["required"]) | ({"target_id"} if catalog is not None else set())
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("模型输出字段与 schema 不一致")
    if (not isinstance(payload["action"], str) or not isinstance(payload["view"], str)
            or payload["action"] not in ("horizontal_explore", "vertical_explore", "approach", "stop") or payload["view"] not in VIEWS):
        raise ValueError("无效的 action/view")
    direction = payload["vertical_direction"]
    if ((payload["action"] == "vertical_explore" and direction not in ("up", "down"))
            or (payload["action"] != "vertical_explore" and direction is not None)):
        raise ValueError("vertical_direction 只用于垂直探索，且必须为 up/down")
    for key, limit in (("reason", REASON_CHARS), ("memory", MEMORY_CHARS)):
        if not isinstance(payload[key], str) or len(payload[key]) > limit:
            raise ValueError(f"无效或过长的 {key}")
    if not payload["reason"].strip():
        raise ValueError("reason 不能为空")
    bbox = payload["bbox"]
    if not isinstance(bbox, list) or any(type(x) not in (int, float) for x in bbox):
        raise ValueError("bbox 必须为数值数组")
    x0, y0, x1, y1 = vector(bbox, 4)
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise ValueError("bbox 必须为 [0,1000] 内的非空 xyxy")
    decision = Decision(payload["action"], payload["view"], tuple(bbox),
                        payload["reason"], payload["memory"], direction, payload.get("target_id"))
    return bind_target(decision, catalog, stop_distance) if catalog is not None else decision


class VLM:
    def __init__(self, base_url: str, model: str, log_path: Path, api_key: str = "",
                 timeout: float = 90, max_tokens: int = 1024, thinking: bool = False,
                 current_target_evidence: bool = False, motion_options: bool = False, goal_image: bytes | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.log_path, self.api_key = model, log_path, api_key
        self.timeout, self.max_tokens, self.thinking = timeout, max_tokens, thinking
        self.current_target_evidence = current_target_evidence
        if motion_options and not current_target_evidence:
            raise ValueError("移动选项需要启用当前目标视觉证据")
        self.motion_options = motion_options
        self.goal_image = goal_image
        self.isolated_model_recovery = False
        self.http_attempts = 0

    def _http_json(self, request):
        from lightning_vln.core.model_transport import request_json
        def count():self.http_attempts += 1
        return request_json(request,self.timeout,recovery=self.isolated_model_recovery,on_attempt=count)

    def localize(self, instruction, observation, iteration):
        return self._query(localization_messages(instruction, observation), LOCALIZATION_SCHEMA,
                           parse_localization, iteration, "target_localization", EVIDENCE_PROMPT_VERSION)

    def decide(self, context: Context, observation: Observation, iteration: int) -> Decision:
        if self.current_target_evidence:
            if context.current_targets is None:
                raise ValueError("当前目标证据尚未生成")
            stop_distance = context.feedback["motion_policy"]["stop_distance_m"]
            messages = context.messages(SYSTEM_PROMPT + NAVIGATION_PROMPT + (MOTION_PROMPT if context.current_motion_options is not None else ""), observation, iteration)
            schema = navigation_schema(SCHEMA, context.current_targets, stop_distance, context.current_motion_options)
            parser = lambda content: bind_motion(parse_decision(content, context.current_targets, stop_distance), context.current_motion_options)
            version = MOTION_OPTIONS_PROMPT_VERSION if self.motion_options else EVIDENCE_PROMPT_VERSION
            return self._query(messages, schema, parser, iteration, "navigation", version)
        messages = context.messages(SYSTEM_PROMPT, observation, iteration)
        return self._query(messages, SCHEMA, parse_decision, iteration, "navigation", PROMPT_VERSION)

    def _query(self, messages, schema, parser, iteration, phase, prompt_version):
        if self.goal_image is not None:
            messages = [messages[0], {"role": "user", "content": [
                {"type": "text", "text": "以下是任务给定的目标参考图，不是无人机当前视图。寻找其中的同一物体实例。"},
                image_content(self.goal_image)]}] + messages[1:]
        body = {"model": self.model, "messages": messages, "temperature": 0,
                "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": self.thinking},
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "navigation_decision" if phase == "navigation" else phase,
                    "strict": True, "schema": schema}}}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        for attempt in range(2):  # 只修复一次输出格式；不重试网络故障或运动。
            request = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
            result = self._http_json(request)
            # 先落盘原始响应，再解析；非法尝试保留，不能被修复结果覆盖。
            if self.log_path is not None:
                with self.log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"iteration": iteration, "attempt": attempt,
                                            "phase": phase, "prompt_version": prompt_version, "response": result},
                                            ensure_ascii=False) + "\n")
            choice = result["choices"][0]
            try:
                if choice.get("finish_reason") != "stop":
                    raise ValueError("模型输出没有正常结束：" + str(choice.get("finish_reason")))
                return parser(choice["message"]["content"])
            except ValueError as error:
                if attempt:
                    raise
                repair = ("不要用零框或空框表示未找到目标。选择可见探索区域，框必须非空，view与对应图片一致。"
                          if prompt_version == PROMPT_VERSION else
                          "目标核查没有候选可返回空列表；导航没有目标须选探索区域。框必须非空，view与对应图片一致。")
                prefix = ("根据同一组当前四图重新输出完整JSON；" if prompt_version == PROMPT_VERSION else
                          "根据同一组当前四图和schema重新输出完整JSON；")
                body["messages"] = messages + [
                    {"role": "assistant", "content": choice["message"]["content"]},
                    {"role": "user", "content": f"输出校验失败：{error}。" + prefix + repair}]
