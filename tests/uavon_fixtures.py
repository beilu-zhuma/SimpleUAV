import math
import numpy as np
from lightning_vln.types import Decision
from test_simple_agent import FakeWorld

def up():
    return Decision("vertical_explore", "front", (400, 400, 600, 600), "升高观察", "", "up")

class YawWorld(FakeWorld):
    def __init__(self, yaw=0, turned=True):
        super().__init__()
        self.yaw = yaw
        self.turned = turned
        self.events = []

    def face(self, goal):
        self.events.append("face")
        if self.turned:
            delta = goal-self.location
            self.yaw = math.atan2(delta[1], delta[0])
        return self.turned

    def advance(self, velocity, seconds):
        self.events.append("advance")
        super().advance(velocity, seconds)

    def observe(self, with_rgb=True):
        result = super().observe(with_rgb)
        # 本测试只需水平三个相机的真实朝向；下视图保留朝下。
        for name, offset in (("front", 0), ("left", -math.pi/2), ("right", math.pi/2)):
            angle = self.yaw+offset
            result.views[name].quaternion = np.array([0, 0, math.sin(angle/2), math.cos(angle/2)])
        return result
