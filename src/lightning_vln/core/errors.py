"""仅明确的仿真基础设施故障可恢复；模型、配置和程序错误不在此列。"""


class SimulationUnavailable(RuntimeError):
    pass


class SimulationTimeout(SimulationUnavailable, TimeoutError):
    pass


class SimulationDisconnected(SimulationUnavailable, ConnectionError):
    pass


class SimulationControlFailure(SimulationUnavailable):
    """制动/定高未达标：当前轨迹作废，可由监督重建环境，不能放宽门槛。"""


class SimulationInitializationFailure(SimulationUnavailable):
    """出生位置未同步或定高后偏离：保留门禁，由监督有限重建环境。"""
