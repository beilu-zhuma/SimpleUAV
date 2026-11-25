"""只消融导航输入中的文字摘要，输出格式、历史与当前证据保持原样。"""

import copy
import json

from lightning_vln.adapters.uavon.minimal.vlm import MinimalVLM


def without_text_memory(messages):
    result = copy.deepcopy(messages)
    found = 0
    # 当前观察是最后一条user消息；不能误改历史反馈、图像、候选证据或上升状态。
    content = result[-1]["content"]
    for part in content:
        if part.get("type") != "text":
            continue
        try:
            payload = json.loads(part["text"])
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and "search_memory_unverified" in payload:
            payload["search_memory_unverified"] = ""
            part["text"] = json.dumps(payload, ensure_ascii=False)
            found += 1
    if found != 1:
        raise ValueError("文字记忆消融未找到唯一的当前摘要输入")
    return result


class TuningVLM(MinimalVLM):
    text_memory_mode = "on"

    def _query(self, messages, schema, parser, iteration, phase, prompt_version):
        if self.text_memory_mode not in ("on", "off"):
            raise ValueError("未知文字记忆模式")
        if phase == "navigation" and self.text_memory_mode == "off":
            messages = without_text_memory(messages)
        return super()._query(messages, schema, parser, iteration, phase, prompt_version)
