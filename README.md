# JEV 类决策模型：两套本地 API 部署代码

这里保留已训练模型所需的推理代码、HTTP 服务、依赖及部署清单。模型权重需要单独传到服务器，GitHub 仓库本身不能代替模型文件。

| 服务 | 骨干 | 决策接口 | 题型 |
|---|---|---|---|
| Perplexity autojev | Qwen3.5-0.8B-Base | `POST /v1/systemone` | choice / noul / score |
| AgentJev | Qwen3-0.6B 或对应的 Base 版本 | `POST /api/evaluate` | choice / boolean / score |

两套都是判别式决策模型：输入状态、问题和候选描述，通过决策头得到概率，返回选择、真假概率或评分，生成 token 数为 0。它们是本地 JEV 类实现；此仓库未做商业 Jev 全协议兼容认证，也未证明达到商业 Jev 的效果。

## 当前训练结果

- autojev 的 full-noncausal 和 full-causal 均完成公开来源完整 1 epoch、1,500 步、95,999 条训练样本；均在 dev 选中 step=1050。主组温度 1.0499191408242807；dev 1,497 题准确率 85.7715%，ECE=0.017434、hard Brier=0.202987。causal 温度 1.052955333131324，dev 准确率同为 85.7715%。dev 参与选优；本轮独立 test 未运行，不能据此宣称 test 或商业 Jev 成绩。训练数据中没有 Score 题，API 支持 Score 不表示已验证其泛化质量。
- AgentJev 的 full_qwen3_0.6b 和 full_qwen3_0.6b_base 都选中 step=600。已有 typed-decisions 独立 test 2,000 题准确率分别为 77.90% 和 77.20%。它们与 autojev 使用不同训练数据和评测划分，不能直接比较 85.77% 与 77.90%。

## 仓库内容

```text
src/autojev/                 # autojev 推理、协议和 FastAPI 服务（4 个 Python 文件）
agentjev_api/agentjev/       # AgentJev 模型
agentjev_api/jev_service/    # 输入协议、推理引擎、共享前缀编码和 HTTP 服务
agentjev_api/scripts/serve.py
requirements.txt            # 两套服务共用的非 PyTorch 运行依赖
requirements-torch-cu130.txt # 本机验证使用的 CUDA 13.0 PyTorch / torchvision
DEPLOYMENT.md               # 文件清单、模型传输和启动命令
deployment-manifest.json     # 选中 checkpoint 的文件大小和 SHA-256
scripts/smoke_api.py         # 启动真实模型并验证接口，完成后关闭测试服务
validation-*.json            # 本次实际 HTTP 验证结果
LICENSE / NOTICE            # 上游来源、许可和导出副本改动说明
```

部署步骤见 [DEPLOYMENT.md](DEPLOYMENT.md)。训练数据、虚拟环境、缓存、优化器状态、中间 checkpoint 和权重均不上传普通 Git。此仓库只提供部署代码，不包含重新训练的完整流水线。

## 来源及改动

- autojev 来源：[perplexity-ai/pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)，固定源码 revision `3b45dead91dfa6d95aad6b95764a606fab2bf7a6`，Apache-2.0。沿用本机训练时的 `model.py`，包括 processor chat-template 回退。
- AgentJev 来源：[malevrigns/agent-jev](https://github.com/malevrigns/agent-jev)，Apache-2.0，经本机 `text_jev_train` 适配；baseline commit `183153817b683b941123f7c708be286c9ecae279`。
- 两套源码快照缺少服务引用的演示 HTML。导出副本的 autojev 首页改为跳转 `/docs`，AgentJev 首页在未传自定义页面时返回模型信息。模型打分逻辑未修改。

## 本次接口验证

2026-10-08（Asia/Shanghai）使用导出的代码与真实已训练权重验证，两套服务均通过：

- [autojev HTTP 验证](validation-autojev.json)：离线加载、health、首页/docs、Bearer 鉴权、三个题型、有限且归一化的概率、非法请求和图片推理。
- [AgentJev HTTP 验证](validation-agentjev.json)：离线加载、health、首页、三个题型、有限且归一化的概率和非法请求。

首次验证发现本机缺少系统 C 编译器；设置训练时已有的 gcc-13 环境后 autojev 通过。部署所需系统依赖已写入说明。验证复用了训练环境，未在全新服务器安装依赖或部署公网服务。
