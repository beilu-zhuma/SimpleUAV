# UAV-ON稳定入口

```bash
python scripts/experiment.py check --config configs/experiments/uavon.local.json
python scripts/experiment.py prepare --config configs/experiments/uavon.local.json --output /results/uavon-main
python scripts/experiment.py run --output /results/uavon-main
```

- 配置由`uavon.example.json`复制，填写解释器、valset、TEST_ENVS和已授权GPU。
- 1000条排除四个问题场景后固定733条，不抽样、不补齐。
- 150决策、两阶段、历史4轮/文字记忆、上升保护/深度安全；初始90°旋转默认关闭、STOP10m、单段6m、出生+2m。
- JSON配置`"method": {"initial_yaw90": true}`开启开局旋转；`false`或省略关闭。开启占1次总预算，关闭不转且不注入旋转前图片。低层CLI仅在显式传`--initial-yaw90`时开启。
- `method.json`与`machine-config.json`冻结实际开关；切换设置需准备新输出，不修改旧实验续跑参数。
- 专用碰撞触发立即失败；出生碰撞正常结算并单独标记，不退出整批。
- SR/OSR/SPL/NE全部保留；官方20m与原geodesic distance，注明adjusted protocol。
- 配置示例采用单卡4路并发，需按实际模型与仿真总负载调整。多卡使用设备隔离，不能由显存或max-num-seqs直接推算并发。
- `stop`等待自有资源退出，`resume`继承登记结算并写新attempt，不覆盖旧记录。

完整口径见[评测协议](../benchmarks.md)。
