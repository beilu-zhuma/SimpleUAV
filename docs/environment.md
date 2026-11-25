# UAV-ON环境与外置数据

## Python环境

控制环境使用Python3.10–3.12，依赖`requirements/uavon-control.txt`。原生AirSim消息通过项目自带RPC边界读取，不需要安装AirSim Python SDK。模型环境单独安装，完整步骤见[中文使用说明](../README.zh-CN.md)。`requirements/model-observed.txt`提供版本参考，其中PyTorch CUDA版本须与驱动及vLLM兼容，不能直接作为通用安装命令。

```bash
/path/to/control/python -m pip install -r requirements/uavon-control.txt
```

vLLM模型环境须与本机驱动、CUDA及所选模型兼容，并能找到CUDA工具链和ninja。模型启动会把`model_python`所在目录加入PATH；不要用控制环境的PATH覆盖模型工具链。不需要深度预测服务或另外的初始化SDK。

## 系统与资源

需要Linux、NVIDIA驱动、Vulkan以及允许普通用户使用的bubblewrap（bwrap）。缺少系统依赖时请联系管理员配置，入口不自动安装系统包。UE使用设备隔离；单卡同时承载模型和仿真，并发需根据两者的总负载设置，不能只由显存容量或vLLM的max-num-seqs推断。

配置中的`gpus`必须是当前获授权且空闲的物理编号。首次启动拒绝占用已有计算任务的卡；运行中出现其他任务时只保留本轮归属并继续，不清理外部进程。

## 外置输入

配置复制自`configs/experiments/uavon.example.json`：

- `paths.python`：控制环境解释器。
- `paths.model_python`：vLLM模型环境解释器。
- `models[0].path`：本地模型目录，须含config.json与所需权重。
- `paths.annotations`：官方1.0 `DATASET/UAV-ON-data/valset`目录，保留原JSON。
- `paths.scenes`：配套`TEST_ENVS`目录，场景脚本映射由项目校验。
- `paths.vulkan_icd`：有需要时填写本机NVIDIA Vulkan ICD文件，否则留空。
- 可选`paths.runtime_root`：短路径临时目录，例如`/tmp`，供vLLM Unix IPC使用；不要把IPC放在很深的实验输出路径。

相对路径以配置文件所在目录为基准。数据、模型、UE资产只读复用，日志、设置和缓存写入本轮输出与自有临时目录。新输出目录必须尚不存在，历史结果不会覆盖。

## 本地验证

`check`核验源码摘要、任务数据、场景脚本、解释器和依赖；`prepare`冻结源码、实际参数、完整身份清单与来源SHA。两者都不查询或分配GPU，不与模型/仿真服务建立导航会话。只有显式`run/resume`会分配资源。
