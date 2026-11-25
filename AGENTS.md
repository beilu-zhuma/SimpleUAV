# 贡献与维护约定

## 代码范围

- 本分支维护 UAV-ON 主方法与运行所需组件。公共入口为 `scripts/experiment.py`，配置示例位于 `configs/experiments/`。
- 通用推理与控制逻辑放在 `src/lightning_vln/core/`，UAV-ON 任务、场景及评分适配放在 `src/lightning_vln/adapters/uavon/`。
- `third_party/UAV-ON/` 保留上游参考代码和署名。策略修改应放在本项目代码中，避免改写官方评分参考。

## 运行与结果

- 安装和运行步骤见 [英文 README](README.md) 与[中文 README](README.zh-CN.md)。先执行 `check`、`prepare`；只有 `run`、`resume` 会启动模型与仿真。
- 使用当前有权使用的 GPU、端口和输出目录。数据、模型与场景资产只读复用，不覆盖已有结果。
- 进程清理必须核对登记的 UID、PID 与启动时间，只管理本次运行创建的进程，不按名称或端口批量查杀。
- 不修改运行中的冻结源码或配置。更改参数时准备新的输出目录；接续时保留已结算任务与恢复预算。
- 修改源码后，在仓库根目录执行 `python scripts/experiment.py refresh-source`，再检查并准备新的运行目录。禁止在已准备的输出目录及其子目录刷新源码清单。
- 评测协议见 [docs/benchmarks.md](docs/benchmarks.md)。保留专用碰撞检测、出生碰撞失败标记与完整 SR/OSR/SPL/NE；更改协议须同步更新文档，不能与原设置混报。

## 文档与验证

- 英文 `README.md` 与中文 `README.zh-CN.md` 的使用步骤保持一致。其他文档默认用中文，UAV-ON 使用说明放在 `docs/`，`docs/must_read_docs/` 保持简洁、以命令为主。
- 公开文档说明安装、配置、运行、结果读取及必要的协议限制；不加入内部排班、验收记录或机器运行账本。
- 不提交凭据、本机配置、模型权重、数据集或运行产物。
- 修改代码后运行相关检查。完整 CPU 检查可在不启动模型或 UE 的情况下执行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts:tests \
  python -B -m unittest discover -s tests -q
```
