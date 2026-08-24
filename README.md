# micro-dl

micro-dl 是一个从零实现的小型深度学习框架，使用 NumPy/CuPy 双 Backend，目标是在
保持源码适合学习和阅读的同时，支持真实的 MNIST 与 CIFAR-100 训练。

它不依赖 PyTorch、TensorFlow 或 JAX。Tensor、自动微分、神经网络层、优化器、数据
管线、训练生命周期和 checkpoint 都由项目自身实现。

## 当前能力

- NumPy CPU Backend，以及可选的 CuPy CUDA Backend
- 动态计算图、反向传播、梯度累积和显式 Backend 转换
- Linear、Conv2d、BatchNorm2d、Dropout 和 ResidualBlock
- CrossEntropyLoss、SGD、Adam、AdamW 与学习率调度器
- MNIST 和 CIFAR-100 数据管线及标准 Experiment
- 可恢复训练、结构化指标、安全 checkpoint 和训练 profiling
- 无图形界面环境可用的训练曲线输出

当前 Backend 只表示 NumPy 或 CuPy Runtime，不包含多 GPU Device 模型。不同 Backend
的 Tensor 可以存在于同一进程，但不能隐式混算；迁移必须显式调用 `to_backend()`。

## 环境要求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- CPU 运行不需要 CUDA
- GPU 运行需要兼容的 NVIDIA 驱动、CUDA 环境和 CuPy CUDA 12 包

这个项目不绑定某一款 GPU。桌面显卡、工作站、云主机或远程训练服务器均可使用，
只要 CuPy 能正确识别其 CUDA Runtime。

## 安装

CPU / NumPy：

```bash
git clone https://github.com/SaintFore/micro-dl.git
cd micro-dl
uv sync
uv run micro-dl doctor --backend numpy
```

NVIDIA GPU / CuPy：

```bash
uv sync --extra cuda
uv run micro-dl doctor --backend cupy
```

如果 `doctor` 无法识别 GPU，先检查 NVIDIA 驱动、CUDA 与 CuPy 的兼容性。不要仅根据
显卡名称选择包；应以机器实际支持的 CUDA Runtime 为准。

## 快速开始

准备数据：

```bash
uv run micro-dl prepare-data --dataset mnist --data-dir ~/data/micro-dl
uv run micro-dl prepare-data --dataset cifar100 --data-dir ~/data/micro-dl
```

运行聚焦 smoke：

```bash
uv run micro-dl smoke --experiment mnist --backend numpy
uv run micro-dl smoke --experiment cifar100 --backend numpy
```

有 CUDA 时，把 `numpy` 替换为 `cupy`。

训练 MNIST：

```bash
uv run micro-dl train \
  --experiment mnist \
  --backend numpy \
  --epochs 10 \
  --batch-size 32 \
  --data-dir ~/data/micro-dl \
  --run-root ~/runs/micro-dl
```

训练 CIFAR-100：

```bash
uv run micro-dl train \
  --experiment cifar100 \
  --backend cupy \
  --epochs 30 \
  --batch-size 64 \
  --data-dir ~/data/micro-dl \
  --run-root ~/runs/micro-dl
```

正式训练要求 Git 工作区干净。每个 Run 会记录精确 commit，避免训练结果无法对应到
实际源码。

## Run 与 checkpoint

每次新训练创建独立目录：

```text
<run-root>/<experiment>/<UTC timestamp>-<git short SHA>/
├── config.toml
├── run.json
├── metrics.jsonl
├── train.log
├── gpu.csv
└── checkpoints/
    ├── best.zip
    └── last.zip
```

分类 Experiment 的 `best.zip` 按最高 validation accuracy 选择，`last.zip` 保存最后一个
epoch。checkpoint 不使用 pickle，并分别保存模型、优化器、调度器和随机状态。

继续同一个 Run：

```bash
uv run micro-dl train --resume <run-dir>/checkpoints/last.zip --epochs 5
```

从 checkpoint 创建配置可变的新 Run：

```bash
uv run micro-dl train --fork-from <run-dir>/checkpoints/best.zip --epochs 5
```

检查 Run 或 checkpoint：

```bash
uv run micro-dl inspect <run-dir>
uv run micro-dl checkpoint-inspect <run-dir>/checkpoints/best.zip
```

## 本地与远程运行

短任务可以直接在终端运行。远程训练建议使用 tmux、screen、systemd-run 或集群调度器，
避免 SSH 断开时终止进程。例如：

```bash
tmux new -s micro-dl
uv run micro-dl train --experiment mnist --backend cupy --epochs 10
```

重新连接：

```bash
tmux attach -t micro-dl
```

查看结构化 epoch 指标：

```bash
RUN_DIR=$(ls -td ~/runs/micro-dl/mnist/* | head -n 1)
watch -n 5 "grep '\"type\":\"epoch\"' '$RUN_DIR/metrics.jsonl' | tail -n 5"
```

CuPy Run 还会生成 `gpu.csv`；也可以使用 `watch -n 1 nvidia-smi` 查看当前 GPU 状态。

## Profiling

先用框架自带的同步 phase profiler 判断瓶颈：

```bash
uv run micro-dl profile \
  --experiment cifar100 \
  --backend cupy \
  --steps 100 \
  --batch-size 32 \
  --data-dir ~/data/micro-dl
```

报告包含 `data`、`forward`、`backward` 和 `optimizer` 的耗时与占比。CuPy profiling
会在 phase 边界同步，因此适合定位方向，不代表无插桩训练的最终吞吐。

## 测试

```bash
uv sync --group dev
uv run pytest -q
```

未安装 CuPy 或 CUDA 不可用时，GPU 合约会跳过，NumPy 核心测试仍可运行。

## 项目结构

```text
micro_dl/
├── tensor.py          # Tensor public interface
├── autograd/          # 动态图与 backward engine
├── backends/          # NumPy/CuPy capability implementations
├── ops.py             # 可微 Operators
├── nn/                # Module、状态与网络层
├── optim/             # Optimizer 与 scheduler
├── data/              # Dataset、DataLoader 与 transforms
├── training/          # Trainer 与结构化结果
├── experiments/       # MNIST/CIFAR-100 教材级装配
├── run/               # 配置、数据准备与 checkpoint
├── visualization.py
└── cli.py
```

当前架构术语与模块边界见 [`CONTEXT.md`](CONTEXT.md)。

## 已验证结果

- MNIST：98.00% official-test top-1（可读 MLP，CuPy）
- CIFAR-100：68.71% official-test top-1（ResNet-like Experiment，CuPy）

这些结果用于验证框架能够完成真实训练，不是硬件无关的性能承诺。训练速度、显存占用
和最终精度会受设备、Backend、随机种子与配置影响。

## 后续方向

Tensor、Autograd、Operators、Module、CNN 与训练基础设施已完成。
后续计划扩展 RNN 及相关序列建模能力。
