"""本轮图像目标证据；候选身份只在当前观察中有效，不读取任务真值。"""

import copy
import json

import numpy as np

from lightning_vln.core.context import image_content
from lightning_vln.core.geometry import ground_bbox, vector
from lightning_vln.core.types import VIEWS


PROMPT_VERSION = "visual_search_v6_current_target_evidence"
from lightning_vln.core.prompts import LOCALIZATION_PROMPT
from lightning_vln.core.prompts import EVIDENCE_NAVIGATION_PROMPT as NAVIGATION_PROMPT
BBOX_SCHEMA = {"type": "array", "minItems": 4, "maxItems": 4,
               "items": {"type": "number", "minimum": 0, "maximum": 1000}}
LOCALIZATION_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["candidates"],
    "properties": {"candidates": {"type": "array", "maxItems": 4, "items": {
        "type": "object", "additionalProperties": False,
        "required": ["view", "bbox", "match", "evidence"], "properties": {
            "view": {"type": "string", "enum": list(VIEWS)},
            "bbox": BBOX_SCHEMA,
            "match": {"type": "string", "enum": ["confirmed", "possible"]},
            "evidence": {"type": "string", "minLength": 1, "maxLength": 300}}}}}}


def localization_messages(instruction, observation):
    # 不带历史、搜索记忆、无人机位置、深度、task pose/object_name或离线评分。
    content = [{"type": "text", "text": "任务：" + instruction}]
    for name, view in observation.views.items():
        content.extend([{"type": "text", "text": "当前视图：" + name}, image_content(view.rgb)])
    return [{"role": "system", "content": LOCALIZATION_PROMPT},
            {"role": "user", "content": content}]


def parse_localization(content):
    payload = json.loads(content)
    if not isinstance(payload, dict) or set(payload) != {"candidates"}:
        raise ValueError("视觉核查必须只包含 candidates")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or len(candidates) > 4:
        raise ValueError("视觉候选必须是最多四项的列表")
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"view", "bbox", "match", "evidence"}:
            raise ValueError("视觉候选字段错误")
        if (not isinstance(candidate["view"], str) or candidate["view"] not in VIEWS
                or not isinstance(candidate["match"], str) or candidate["match"] not in ("confirmed", "possible")):
            raise ValueError("视觉候选 view/match 错误")
        if (not isinstance(candidate["evidence"], str) or not candidate["evidence"].strip()
                or len(candidate["evidence"]) > 300):
            raise ValueError("视觉候选必须提供简短可见证据")
        bbox = candidate["bbox"]
        if not isinstance(bbox, list) or any(type(x) not in (int, float) for x in bbox):
            raise ValueError("视觉候选 bbox 必须是数值数组")
        x0, y0, x1, y1 = vector(bbox, 4)
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError("视觉候选 bbox 必须为非空 xyxy")
    return candidates


def ground_candidates(candidates, observation, iteration):
    catalog = []
    for index, candidate in enumerate(candidates):
        item = dict(candidate, target_id=f"frame_{iteration}_target_{index}",
                    groundable=False, grounded_distance_m=None)
        try:
            point = ground_bbox(observation.views[item["view"]], item["bbox"])
            item.update(groundable=True, grounded_distance_m=float(np.linalg.norm(point-observation.position)))
        except ValueError as error:
            item["grounding_error"] = str(error)
        catalog.append(item)
    return catalog


def can_stop(candidate, stop_distance):
    return (candidate["groundable"] and candidate["match"] == "confirmed"
            and candidate["grounded_distance_m"] <= stop_distance)


def navigation_schema(base_schema, catalog, stop_distance, motion_options=None):
    branches = []
    active = motion_options is not None and motion_options["active"]
    moves = motion_options["moves"] if active else [dict(action=a) for a in ("horizontal_explore", "vertical_explore")]
    for move in moves:
        action = move["action"]
        branch = copy.deepcopy(base_schema)
        branch["required"].append("target_id")
        branch["properties"]["target_id"] = {"type": "null"}
        branch["properties"]["action"]["enum"] = [action]
        branch["properties"]["vertical_direction"] = ({"type": "string", "enum": ["up", "down"]}
                                                       if action == "vertical_explore" else {"type": "null"})
        if active:
            branch["properties"]["view"]["enum"] = [move["view"]]
            branch["properties"]["bbox"]["enum"] = [move["bbox"]]
            branch["properties"]["vertical_direction"] = ({"type": "string", "enum": [move["vertical_direction"]]}
                                                           if action == "vertical_explore" else {"type": "null"})
        branches.append(branch)
    for candidate in catalog:
        if not candidate["groundable"]:
            continue
        actions = ["approach", "stop"] if can_stop(candidate, stop_distance) else ["approach"]
        if active:
            permitted = next((t["actions"] for t in motion_options["targets"] if t["target_id"] == candidate["target_id"]), [])
            actions = [a for a in actions if a in permitted]
            if not actions:
                continue
        branch = copy.deepcopy(base_schema)
        branch["required"].append("target_id")
        branch["properties"]["target_id"] = {"type": "string", "enum": [candidate["target_id"]]}
        branch["properties"]["action"]["enum"] = actions
        branch["properties"]["view"]["enum"] = [candidate["view"]]
        branch["properties"]["bbox"]["enum"] = [candidate["bbox"]]
        branch["properties"]["vertical_direction"] = {"type": "null"}
        branches.append(branch)
    return {"anyOf": branches}


def bind_target(decision, catalog, stop_distance):
    if decision.action not in ("approach", "stop"):
        if decision.target_id is not None:
            raise ValueError("探索不能绑定目标")
        return decision
    candidate = next((c for c in catalog if c["target_id"] == decision.target_id), None)
    if candidate is None or not candidate["groundable"]:
        raise ValueError("目标身份不属于当前可定位候选")
    if decision.view != candidate["view"] or tuple(decision.bbox) != tuple(candidate["bbox"]):
        raise ValueError("目标动作的视图/框与当前视觉证据不一致")
    if decision.action == "stop" and not can_stop(candidate, stop_distance):
        raise ValueError("停止需要本轮确认目标且满足原停止距离")
    # 执行框直接取核查器输出，避免后续动作处理改写框。
    decision.view, decision.bbox = candidate["view"], tuple(candidate["bbox"])
    return decision
