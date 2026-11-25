"""仅保留逐决策的计时标量，不持久化模型原文、图像、位姿或轨迹。"""
from contextlib import contextmanager
import json


def execution_parts(execution):
    yield execution
    fallback = execution.get('approach_fallback')
    if fallback:
        yield from execution_parts(fallback['initial_execution'])


def timing_row(record):
    timing = record['timing']
    execution = record['execution']
    parts = list(execution_parts(execution))
    def total(name):
        values = [part[name] for part in parts if name in part]
        return sum(values) if values else None
    return dict(schema='decision_timing_v1', step=record['iteration']+1,
        iteration=record['iteration'], action=(record.get('decision') or {}).get('action'),
        status=execution['status'], iteration_s=timing['iteration_s'],
        hold_s=timing.get('hold_s'), observe_s=timing.get('observe_s'),
        save_observation_s=timing.get('save_observation_s'),
        model_and_grounding_s=record.get('model_and_grounding_s'),
        execution_s=total('elapsed_s'), yaw_s=total('yaw_s'),
        braking_s=sum(part.get('stop', {}).get('elapsed_s', 0) for part in parts) if any('stop' in part for part in parts) else None,
        control_commands=sum(part.get('control_commands', 0) for part in parts),
        motion_segments=sum('control_commands' in part for part in parts))


class TimingStream:
    def __init__(self, primary, timing):
        self.primary, self.timing = primary, timing

    def write(self, text):
        row = timing_row(json.loads(text))
        result = self.primary.write(text)
        self.timing.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        return result

    def flush(self):
        self.primary.flush()
        self.timing.flush()


@contextmanager
def decision_timings(stream, path=None):
    if path is None:
        yield stream
    else:
        with path.open('x', encoding='utf-8') as timing:
            yield TimingStream(stream, timing)
