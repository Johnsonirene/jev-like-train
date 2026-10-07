# 部署训练完成的模型

代码仓库和模型产物要分别准备。所有文件路径均相对于代码仓库根目录。原训练工程来源目录单独注明为相对于原工程根目录；服务器可按启动命令放置文件。

## 1. 环境与公共代码

已验证环境：Linux / WSL2、Python 3.12.3、RTX 5070 Ti 16GB、torch 2.14.1+cu130、torchvision 0.29.1+cu130、transformers 5.17.0。autojev 使用 Python 3.12 的类型语法，不能用 Python 3.10/3.11。CUDA 驱动必须支持安装的 PyTorch wheel。

autojev 的首次 CUDA 推理会触发 Triton 编译，需要 C 编译器、链接器及 Python 开发头文件。本机没有系统 gcc，因此本次验证设置了 `CC` 为训练任务自带 gcc-13；通用 Ubuntu 服务器可安装 `build-essential` 和与 Python 版本匹配的 `python3.12-dev`。如果 compiler 不在 PATH，显式设置 `CC=gcc`（确保 gcc 在 PATH 中）。本机 flash-linear-attention / fla-core 已卸载，沿用 transformers 的回退路径，部署不要求安装它们。

在仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-torch-cu130.txt
pip install -r requirements.txt
# 新服务器需要时先安装系统编译依赖：
# sudo apt-get update
# sudo apt-get install build-essential python3.12-dev
```

这些版本来自实际训练环境；其他 CUDA/PyTorch/transformers 组合没有在本次验证。两套服务可共用同一个环境，也可分别复制所需代码、创建环境。

## 2. autojev：部署整个 selected 目录

主组来源：

```text
training/autojev/runs/full-noncausal/checkpoints/selected/
```

本仓库重新训练会生成上述目录；已有训练产物在原工程的 `pplx_small/runs/full-noncausal/checkpoints/selected/`。

现有训练产物的 `selected` 是指向 `step-01050` 的软链接。应复制实际内容，不能只传软链接。

| 必需文件 | 用途 |
|---|---|
| model.safetensors | 已微调的完整骨干权重，约 1.706 GB |
| readout.safetensors | 255 维决策头，约 0.522 MB |
| decision_config.json | 温度、attention_mode、pooling、答案码和来源版本 |
| config.json | 模型结构配置 |
| tokenizer.json | 分词器 |
| tokenizer_config.json | 分词器配置 |
| processor_config.json | 文本、图像处理器配置 |
| chat_template.jinja | 与训练一致的输入模板 |

8 个文件合计 1,726,558,973 字节，约 1.73 GB。详细哈希在 [deployment-manifest.json](deployment-manifest.json)。已验证仅凭这个目录离线加载并推理，不需要另带 Qwen3.5 底座下载缓存。

代码只需 `src/autojev/{__init__,model,types,server}.py` 和运行依赖；不需要 `resume.pt`、训练集、run/events.jsonl 或所有 step 目录。温度已保存在 `decision_config.json`，无需额外 `temperatures.json`。

在训练机打包：

```bash
mkdir -p artifacts
tar -C training/autojev/runs/full-noncausal/checkpoints/selected \
  -czf artifacts/autojev-selected.tar.gz .
```

将压缩包传到服务器后，在克隆的代码仓库根目录解压并启动：

```bash
mkdir -p checkpoints/autojev
tar -xzf artifacts/autojev-selected.tar.gz -C checkpoints/autojev
source .venv/bin/activate
export PYTHONPATH=src
export AUTOJEV_CHECKPOINT=checkpoints/autojev
export AUTOJEV_API_KEY='replace-with-your-own-key'
export AUTOJEV_HOST=127.0.0.1
export PORT=8000
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m autojev.server
```

`AUTOJEV_API_KEY` 通过环境变量读取，不自动加载 `.env`。外部访问需要设置 `AUTOJEV_HOST=0.0.0.0` 或配置反向代理；每个 worker 都会加载一份模型，先使用默认单进程。

接口：`GET /health`、`GET /v1/models`、`POST /v1/systemone`，文档为 `/docs`。`/v1/models` 和决策接口启用 Bearer 鉴权；health/docs 可直接访问。

```bash
curl --noproxy '*' http://127.0.0.1:8000/health
curl --noproxy '*' http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer $AUTOJEV_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"jev-latest","state":"客户表示同一订单被扣款两次，要求退还多扣的钱。","questions":{"route":{"type":"choice","instructions":"应该交给哪个部门？","criteria":{"billing":"重复扣款及退款","tech":"软件故障"}},"refund":{"type":"noul","instructions":"客户是否要求退款？"},"urgency":{"type":"score","instructions":"紧急程度？","criteria":["低","中","高"]}}}'
```

Choice 返回 `choice/probabilities/confidence`，Noul 返回 `noul=P(true)`，Score 返回期望评分、概率和图例。`usage.output_tokens=0`。请求支持 base64 PNG/JPEG/WebP 图片。服务遇到并发忙返回 529，可按 Retry-After 重试；实际长请求显存需求随长度和题数上升，短样例验证不能证明最大负载容量。

causal 对照也可以部署：改为复制 `training/autojev/runs/full-causal/checkpoints/selected/` 的实际内容。不要混合两个 checkpoint 的文件，attention_mode/temperature 由各自配置决定。本次新增 HTTP 验证使用主组；causal 的加载已在训练最终验收中通过。

## 3. AgentJev：best.pt + 校准文件 + 对应底座

默认部署 `full_qwen3_0.6b`；本仓库训练后的文件路径为：

```text
agentjev_api/runs/full_qwen3_0.6b/best.pt
agentjev_api/runs/full_qwen3_0.6b/temperatures.json
models/Qwen3-0.6B/
```

已有 checkpoint 在原工程的 `text_jev_train/runs/full_qwen3_0.6b/`，对应底座名为 `Qwen3-0.6B`。将该底座整体复制到仓库的 `models/Qwen3-0.6B/`。

| 文件 / 目录 | 用途 |
|---|---|
| best.pt | 已训练骨干及候选决策头，2,393,802,682 字节，约 2.39 GB |
| temperatures.json | 分类型校准温度；代码允许缺省，但复现校准结果应携带 |
| 底座 config.json + model.safetensors | 现有代码先构造 Qwen3Model，再严格加载 best.pt，因此仍需底座权重 |
| 底座 tokenizer.json、tokenizer_config.json、vocab.json、merges.txt | 保留与训练匹配的完整分词资源；直接复制整个底座目录最稳妥 |
| agentjev_api/agentjev/*.py | 模型结构 |
| agentjev_api/jev_service/*.py | 协议、推理引擎及 prefix.py；prefix.py 不可遗漏 |
| agentjev_api/scripts/serve.py | 启动入口 |

本组底座约 1.52 GB，连 checkpoint 和校准文件约需 3.92 GB。底座里的 generation_config.json、README.md 不参与决策推理；转移模型时保留其 LICENSE。`best.pt` 使用 PyTorch checkpoint 格式，只加载你自己的可信训练产物。

在服务器代码根目录放置：

```text
checkpoints/agentjev/best.pt
checkpoints/agentjev/temperatures.json
models/Qwen3-0.6B/           # 从上面的对应底座复制
```

启动：

```bash
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python agentjev_api/scripts/serve.py \
  --run-dir checkpoints/agentjev \
  --backbone models/Qwen3-0.6B \
  --host 127.0.0.1 --port 8149 --device cuda:0
```

接口：`GET /health`、`GET /api/info`、`POST /api/evaluate`。本服务本身没有 API key 鉴权，对外部署应通过带鉴权的反向代理访问。

```bash
curl --noproxy '*' http://127.0.0.1:8149/api/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"id":"demo","state":"客户表示同一订单被扣款两次，要求退还多扣的钱。","questions":[{"id":"route","type":"choice","question":"应该交给哪个部门？","options":{"billing":"重复扣款及退款","tech":"软件故障"}},{"id":"refund","type":"boolean","question":"客户是否要求退款？"},{"id":"urgency","type":"score","question":"紧急程度？","levels":["低","中","高"]}]}'
```

返回 `results[].answers[]`，含 `distribution`，并按题型返回答案。`usage.generated_tokens=0`。AgentJev 和 autojev 的 JSON 结构及真假类型名称不同，调用方需使用对应协议。

若要部署 `full_qwen3_0.6b_base`，则改用这个 run 的 `best.pt/temperatures.json`，并配 `models/Qwen3-0.6B-Base/`。不要混用两组分词器和校准文件。本次新增 HTTP 验证使用 `full_qwen3_0.6b`。

## 4. 检查部署

```bash
# 在有编译器的环境中运行；脚本会启动并关闭自己的本机测试服务。
mkdir -p artifacts
python scripts/smoke_api.py autojev \
  --checkpoint checkpoints/autojev --output artifacts/validation-autojev.json
python scripts/smoke_api.py agentjev \
  --checkpoint checkpoints/agentjev --backbone models/Qwen3-0.6B \
  --output artifacts/validation-agentjev.json
```

验证覆盖真实 checkpoint 的离线启动、health、首页、三个决策题型、有限概率及归一化、无生成 token、非法请求；autojev 还覆盖 API key 和图片推理。使用新构造的样例，不重复运行训练的 test 集，也不据此评价业务准确率。本次实际运行结果见 `validation-autojev.json` 和 `validation-agentjev.json`。这次验证复用了训练环境，没有在全新服务器重新安装依赖；编译器和 CUDA 驱动由部署环境提供。
