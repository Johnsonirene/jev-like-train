# 两套模型的训练流程

下面的路径都是相对路径。公共环境命令从仓库根目录执行；autojev 小节会进入 `training/autojev/`，AgentJev 小节从仓库根目录执行。不要在中途切换工作目录，否则生成图像的相对路径无法正确读取。

## 环境

Linux / WSL2、Python 3.12、CUDA GPU。按 [DEPLOYMENT.md](DEPLOYMENT.md) 安装与显卡匹配的 torch / torchvision 和编译器，再安装训练依赖：

```bash
source .venv/bin/activate
pip install -r requirements-training.txt
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
# 如需镜像，可在首次下载前启用：
# export HF_ENDPOINT=https://hf-mirror.com
```

训练内存需求高于部署。原 autojev 完整训练实际峰值进程内存约 19.24 GB、全卡 GPU 采样峰值约 10.18 GB；AgentJev 16GB 配置峰值 CUDA allocated 约 12GB。启动前检查 WSL 可用内存和 Windows commit 余量，不要只看显卡空闲。

```bash
free -g
nvidia-smi
```

Windows 宿主可用 PowerShell 查看 `Get-CimInstance Win32_OperatingSystem | Select-Object FreeVirtualMemory`；该字段以 KB 计。原完整训练要求启动时可用 WSL 内存至少 20GB、Windows commit 余量至少 30GB；可用内存或 commit 低于 8GiB 时不要启动。autojev 的资源 wrapper 保留 14GB 全卡显存和启动可用内存 80% 的预算，超限会中断训练；它不能替代宿主 commit 检查。

## autojev：公开多来源完整训练

核心代码在 `src/autojev/`：`data.py` 构建分区，`train.py` 训练和选优，`loss.py` 定义损失，`optim.py` 实现 CPUOffloadAdamW，`evaluate.py` 评测及温度拟合。`sft_pipeline.py` 及其依赖是训练入口导入所需的原模块；本次公开来源流程不调用外部教师 API 或合成文本服务。

骨干固定为 `Qwen/Qwen3.5-0.8B-Base`，revision 为 `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`；不能省略骨干参数，否则上游默认会使用 27B。

### 1. 下载固定数据源并规范化

```bash
cd training/autojev
export PYTHONPATH=../../src
export HF_HOME=cache/hf
export HF_HUB_CACHE=cache/hf/hub
export HF_DATASETS_CACHE=cache/hf/datasets
export TRITON_CACHE_DIR=cache/triton
export AUTOJEV_EVENTS=logs/data-events.jsonl
mkdir -p logs baselines runs cache
python scripts/prepare_sources.py
```

来源文件由 [public-sources.json](training/autojev/configs/public-sources.json) 固定 repository、revision、文件名和 SHA-256。脚本逐个下载并核验，流式转换 Parquet/JSON 为来源 JSONL；还下载并校验 VitaminC、MultiNLI、MASSIVE 归档，以及固定 revision 的 Nimble 转换器。来源缓存生成在 `data/full-public/raw/`。Nimble 是外部依赖，由脚本取得，不作为本仓库源码复制。

这个脚本替代了原任务中带本机绝对目录、依赖先前试点下载记录的下载/规范化脚本；转换逻辑沿用原流式写出方式，不改变 `autojev.data` 的配额、family 切分或过滤策略。首次下载受网络和外部来源可访问性影响。

### 2. 构建并审计训练、dev、temperature 分区

```bash
python scripts/run_module.py autojev.data logs/data-resources.json \
  --cache-root cache/hf \
  --source-cache data/full-public/raw \
  --nimble-source data/full-public/raw/nimble-source \
  --output data/full-public --manifest data/full-public/manifest.json \
  --train-scale 1 --max-length 4096 --seed 20260920 \
  --base-model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --image-train 1000 --image-eval 200
python scripts/audit_full_data.py
```

原完整运行产生 train=95,999、dev=1,497、temperature=983、known=3,247、test=4,994、image_test=200。family、ID、state 和图像 SHA 审计由代码检查，日志输出在 `logs/full-data-audit.json`。test/known/image_test 不参与下面的选优与校准；构建这些分区不等于模型推理评测它们。

如果是首次尝试，可以将 `--train-scale` 调小；数据分区随参数变化，不能把试点结果当作原完整训练的复现成绩。

### 3. 固定零样本参照

```bash
python scripts/run_module.py autojev.evaluate logs/baseline-resources.json \
  --local --data data/full-public/dev.jsonl \
  --output baselines/base-development.json --temperature 1 --batch-size 4 \
  --base-model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --cache-dir cache/hf/hub
```

同时生成 `baselines/base-development.predictions.jsonl`。这里的 reference 是本地零样本底座，选优结果中的 `development_target_met` 不能解释成超越商业 Jev。

### 4. 主组完整 1 epoch

```bash
python scripts/run_module.py autojev.train logs/full-noncausal-resources.json \
  --train data/full-public/train.jsonl \
  --development data/full-public/dev.jsonl \
  --temperature data/full-public/temperature.jsonl \
  --reference baselines/base-development.predictions.jsonl \
  --run runs/full-noncausal/run --output runs/full-noncausal/checkpoints \
  --base-model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --cache-dir cache/hf/hub \
  --epochs 1 --batch-size 4 --effective-batch-size 64 \
  --token-budget 4096 --max-length 4096 \
  --lr 1e-5 --warmup-ratio 0.05 --weight-decay 0.01 --brier-weight 0 \
  --eval-every 150 --public-eval-every 1000000 --resume-every 150 \
  --attention-mode noncausal_full_attention --pooling last \
  --cpu-threads 8 --seed 20260920
```

温度在独立 temperature 分区拟合。选优门槛保留：calibrated ECE/Brier 各不超过固定参照 + 0.01，eligible 中按准确率、Brier 等原规则选择。不要为了得到 selected 而放宽门槛。

需要 causal 对照时，复用同一冻结数据与 reference，将 attention-mode 改为 `causal`，同时改用 `runs/full-causal/run`、`runs/full-causal/checkpoints` 和新的资源日志文件。两组顺序运行。

首次检查训练流程时，可在主组命令上改用新的 smoke run/output，附加 `--stop-after 20 --eval-every 10 --resume-every 10`；该命令仍会运行 dev/temperature，实际耗时取决于数据规模。不要覆盖已有 run。

### 5. 保存后评测与部署

```bash
python scripts/run_module.py autojev.evaluate logs/selected-dev-resources.json \
  --local --checkpoint runs/full-noncausal/checkpoints/selected \
  --data data/full-public/dev.jsonl \
  --output evaluations/selected-dev.json --batch-size 4
```

该命令在 dev 上验证保存后的 checkpoint，不是独立 test 成绩。最终权重位于 `runs/full-noncausal/checkpoints/selected/`，`decision_config.json` 内含温度和 attention_mode；`resume.pt` 用于恢复训练，部署不需要。

本轮原完整模型尚未运行独立 test。若要最后评测未参与调参的公开 test，可将上面 data 改为 `data/full-public/test.jsonl`，另选新的 output，在模型与温度冻结后执行。不要用 test 选模型或回调温度。

## AgentJev：typed-decisions 全参训练

以下从仓库根目录执行。如果刚执行完 autojev 小节，先 `cd ../..`。

### 1. 下载数据与底座

```bash
python agentjev_api/scripts/download_data.py
python agentjev_api/scripts/download_backbone.py --target-dir models/Qwen3-0.6B
```

两个下载器都支持 `--mirror`。数据 revision 与 SHA-256 固定；底座下载器的实际 repository 是 `Qwen/Qwen3-0.6B`。目录名中写 Base 不会改变所下载的仓库，所以这里显式命名为 `Qwen3-0.6B`。

另一个已训练的 `full_qwen3_0.6b_base` 使用独立的 Qwen3-0.6B-Base 底座及分词器；若复现该组，应自行准备其匹配底座目录，并替换训练与部署的 `--backbone`，不能靠改文件夹名字切换底座。

### 2. 案例级切分

```bash
python agentjev_api/scripts/prepare_data.py
```

按固定 seed 和案例 ID 切分：960 个 train case、120 个 dev case、120 个 calibration case；官方 test 400 case 保留，展开后分别是 4,800、600、600、2,000 道题。输出 `agentjev_api/prepared/` 和 `agentjev_api/split_manifest.json`；同一案例的问题不跨分区。

### 3. 16GB 显卡配置训练

```bash
python agentjev_api/scripts/train.py \
  --backbone models/Qwen3-0.6B \
  --protocol agentjev_api/scripts/protocol_full_16g.json \
  --prepared agentjev_api/prepared \
  --run-dir agentjev_api/runs/full_qwen3_0.6b
```

配置保留原 600 步、microbatch=4、梯度累积=4、骨干 lr=1e-5、候选头 lr=1e-4、gradient checkpointing、显存比例 0.78。损失是软交叉熵加 0.1 倍候选 Brier。训练器先按 dev 未校准 soft CE 选 checkpoint，再在独立 calibration 上拟合分类型温度，最后才读取官方 test。

冒烟子集与 20 步配置也已保留：

```bash
python agentjev_api/scripts/make_smoke_subset.py
python agentjev_api/scripts/train.py \
  --backbone models/Qwen3-0.6B \
  --protocol agentjev_api/scripts/protocol_smoke.json \
  --prepared agentjev_api/prepared_smoke \
  --run-dir agentjev_api/runs/smoke
```

这套训练入口要求 CUDA；部署接口可选择 device，不能据此认为训练也支持 CPU。`--init-from` 可接兼容的已有 checkpoint；原多源 phase-4 的外部训练工程不在本仓库内，不是本次 from-scratch 两组的必要步骤。

### 4. 产物

`best.pt`、`temperatures.json`、`selection.json`、`report.json`、校准/test 预测和训练 events 保存在 run 目录。部署只需要 `best.pt`、校准文件、匹配底座及服务代码，详见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 本次补充代码的验证范围

核心训练、选优、校准和模型文件直接复用已有训练通过的源码；这次补充不重新执行两套完整训练。新增检查包含训练/数据/评测入口加载、CLI 参数、AgentJev 数据切分复现和来源转换检查；具体通过项记录在 `validation-training.json`。文件路径使用仓库相对路径，运行时输出的验证记录可以记录实际所在位置。
