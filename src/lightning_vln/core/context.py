"""有限关键帧、真实执行反馈和一份模型维护的短搜索记忆。"""

import base64
import json
from collections import deque

from lightning_vln.core.types import Decision, Observation


MEMORY_CHARS = 1200
REASON_CHARS = 300


def image_content(rgb: bytes) -> dict:
    return {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + base64.b64encode(rgb).decode("ascii")}}


class Context:
    def __init__(self, instruction: str, history_turns: int = 4):
        if history_turns < 0:
            raise ValueError("history_turns 必须非负")
        self.instruction = instruction
        self.turns = deque(maxlen=history_turns)
        self.memory = ""
        self.feedback = {"status": "initial"}
        self.current_targets = None
        self.current_motion_options = None

    def messages(self, system_prompt: str, observation: Observation, iteration: int) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": "搜索任务：" + self.instruction}]
        for user, assistant in self.turns:
            messages.extend([{"role": "user", "content": user},
                             {"role": "assistant", "content": assistant}])
        current = {"iteration": iteration, "position_ned_m": observation.position.tolist(),
                   "search_memory_unverified": self.memory, "last_execution": self.feedback,
                   "image_order": list(observation.views)}
        if self.current_targets is not None:
            current["current_target_evidence"] = self.current_targets
        if self.current_motion_options is not None:
            current["current_motion_options"] = self.current_motion_options
        content = [{"type": "text", "text": json.dumps(current, ensure_ascii=False)}]
        for name, view in observation.views.items():
            content.extend([{"type": "text", "text": "当前视图：" + name}, image_content(view.rgb)])
        messages.append({"role": "user", "content": content})
        return messages

    def record(self, observation: Observation, decision: Decision | None, feedback: dict):
        self.feedback = feedback
        name = decision.view if decision else "front"
        description = {"position_ned_m": observation.position.tolist(),
                       "keyframe_view": name}
        user = [{"type": "text", "text": json.dumps(description, ensure_ascii=False)},
                image_content(observation.views[name].rgb)]
        # 历史只保留所选视图、短决策与实际结果，不回填 thinking，也不复制整份 memory。
        assistant = {"decision": None, "execution": feedback}
        if decision:
            self.memory = decision.memory
            assistant["decision"] = {"action": decision.action, "view": name,
                                     "bbox": decision.bbox, "reason": decision.reason,
                                     "vertical_direction": decision.vertical_direction}
            if self.current_targets is not None:
                assistant["decision"]["target_id"] = decision.target_id
        self.turns.append((user, json.dumps(assistant, ensure_ascii=False)))
