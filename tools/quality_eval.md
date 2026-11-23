<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->
# 标准数据集精度对比

本工具是离线评测客户端，复用 **lm-evaluation-harness 0.4.13 的任务、答案提取与评分实现**。通过 OpenAI 兼容的 `/v1/chat/completions` 请求已有引擎，不修改 engine、scheduler 或模型模块。

## 固定的首批任务

| 任务 | 数据集版本 | 测试规模 | 提示与指标 |
| --- | --- | --- | --- |
| GSM8K CoT | `openai/gsm8k@740312add88f781978c0658806c59bc2815b9866` | test，1319 题 | 官方 8-shot 示例；strict-match / flexible-extract exact match |
| IFEval | `google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84` | train，541 题（该数据集的官方评测 split） | 0-shot；prompt / instruction level，strict / loose 四个准确率 |

GSM8K 默认输出预算 1024 tokens，IFEval 为官方 1280 tokens；默认 temperature=0、top_p=1、seed=1234、batch_size=1，用于贪心差异诊断。`--temperature`、`--top-p`、`--seed` 显式设置 API 采样参数并进入配对协议；改变采样 seed 不改变数据顺序、few-shot 示例或评分随机种子。8-shot 保留为单条 user 消息中的示例，服务端应用模型自己的 chat template。`--concurrency` 控制同时发出的 HTTP 请求数，不代表引擎 batch size。

**这些是当前协议下的引擎配对对比，不是直接复现模型排行榜。** GSM8K 默认预算是本协议的显式选择。思考模型必须记录相同的 reasoning 开关；若预算不足，使用相同 `--max-tokens` 重跑两端，不能把一端截断的结果当成退化结论。工具不静默删除 `<think>` 内容。IFEval 对思考文本和最终答案混在 content 中的输出按官方规则评分，必须先确认两端返回内容语义相同。

V4.1 的质量评测必须与上述贪心诊断分开。[模型官方说明](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash#evaluation-results) 的 instruct 评测采用 temperature=1.0、top_p=0.95、reasoning_effort=100；其部署建议也给出远高于本工具默认值的输出预算。2026-11-20 实测内部参考在 temperature=0 / effort50 / 4096 下也发生重复思考及空最终答案，不能只增加预算而忽略解码配置。采样对照可显式使用 `--temperature 1.0 --top-p 0.95 --seed 1234`，两端使用同一预算、effort 和 seed；若继续使用内部默认 effort50，必须在 metadata 中保留 50，不能称为官方榜单复现。8192 等小预算仍只是联调起点，任何截断继续阻止验收。完整采样结果应考虑多 seed 的波动，单个固定 seed 不保证两个引擎逐 token 相同或统计等价。

## 一次准备，后续离线复用

在仓库根目录，使用 Python 3.12。Mac 用于 CPU 检查、准备离线资源和分析结果。长时间设备评测的客户端应在 NPU 机器上的独立 venv 中运行，不使用 NPU、不修改推理服务环境；固定评分依赖和数据版本与 Mac 一致。

```bash
python3.12 -m venv .deps/quality-eval-venv
.deps/quality-eval-venv/bin/python -m pip install -r tools/quality_eval_requirements.txt
export HF_HOME="$PWD/.deps/quality-eval-cache"
export NLTK_DATA="$PWD/.deps/quality-nltk-data"
mkdir -p "$NLTK_DATA/tokenizers" ../artifacts/quality-eval
curl -fL https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip -o .deps/quality-punkt_tab.zip
.deps/quality-eval-venv/bin/python -c 'from pathlib import Path; import hashlib; expected="e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"; actual=hashlib.sha256(Path(".deps/quality-punkt_tab.zip").read_bytes()).hexdigest(); assert actual == expected, actual'
unzip -qo .deps/quality-punkt_tab.zip -d "$NLTK_DATA/tokenizers"
.deps/quality-eval-venv/bin/python tools/quality_eval.py prepare --output ../artifacts/quality-eval/datasets.json
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

NLTK 文件来自官方 nltk_data，下载后校验固定 SHA256；运行时再次核验英语分句资源内容。这样评分时不需要临时联网下载。数据集保存内容摘要、revision、split、条数和任务版本。每次运行额外保存 `environment.txt`，对比协议包含评分库版本和 runner 文件摘要；升级依赖或脚本后须重跑参考端。

## 身份与运行配置

分别创建 reference 和 candidate 的 metadata JSON，填写实际值（下面是字段说明，不能直接执行）：

```json
{
  "weights_sha256": "权重文件清单及内容 SHA256 清单的 SHA256（64 位小写十六进制）",
  "tokenizer_sha256": "tokenizer 配置与词表文件内容清单的 SHA256",
  "chat_template_sha256": "实际生效的 chat template 内容 SHA256",
  "precision": "w8a8/bf16，注明量化与累加方式",
  "reasoning_policy": "实际生效的 thinking 开关及相关模板参数",
  "engine_revision": "服务进程使用的源码 commit",
  "engine_tree_sha256": "实际构建源码（含未提交和未跟踪源码）的内容清单 SHA256",
  "topology": "实际 TP/DP/EP/MoE-TP 配置",
  "hardware": "设备型号、设备数、驱动、CANN、torch_npu 版本",
  "launch_command": "完整脱敏启动命令，包括模型、长度、并发及调度参数",
  "purpose": "real-checkpoint"
}
```

不能只用目录名或 safetensors 文件头当作权重内容身份，也不能只用 `git HEAD` 代表尚未提交的改动。metadata 是启动者提供的运行记录，客户端不能独立证明远端实际装载的文件或生效配置，需要与启动日志一起归档。参考端和候选端的前三个摘要、precision、reasoning_policy 必须相同；topology、engine 和硬件允许不同，并明确展示。

真实 V4.1 验收使用 `/export/home/models/DeepSeek-V4.1-Flash-w8a8`。小模型可用于客户端与服务连通性验证，但不能代替 V4.1 验收。随机六层合成模型必须标记 `pipeline-test`，不能获得验收结论。

## 运行与比较

短请求可以通过 SSH 隧道执行。长评测应直接在远端访问自己的 HTTP 服务，通过 `docker exec -d` 等方式脱离 Mac 的 SSH 会话，并在远端保存日志、原始响应、评分和进程退出码。先通过该远端客户端的 CPU 集成测试，再启动设备评测；Mac 断线后重新读取作业状态，不自动重发已完成的请求。以下端口及 model alias 需与实际服务对应；不要将其他用户的服务当作默认参考端。

```bash
# 冒烟：两端各跑每个任务前 32 题。输出目录必须是新目录。
.deps/quality-eval-venv/bin/python tools/quality_eval.py run \
  --endpoint http://127.0.0.1:18888/v1/chat/completions --model v41 \
  --metadata ../artifacts/quality-eval/reference-meta.json \
  --output ../artifacts/quality-eval/reference-smoke --limit 32
.deps/quality-eval-venv/bin/python tools/quality_eval.py run \
  --endpoint http://127.0.0.1:18889/v1/chat/completions --model v41 \
  --metadata ../artifacts/quality-eval/candidate-meta.json \
  --output ../artifacts/quality-eval/candidate-smoke --limit 32
.deps/quality-eval-venv/bin/python tools/quality_eval.py compare \
  --reference ../artifacts/quality-eval/reference-smoke/results.json \
  --candidate ../artifacts/quality-eval/candidate-smoke/results.json \
  --output ../artifacts/quality-eval/smoke-comparison.json
```

正式运行去掉 `--limit`，更换输出目录，完成全部 **1860 题/端**。根据模型需要给两端加相同 `--max-tokens 8192` 等输出预算。默认并发 1；再以相同并发值对比两端的并发路径，例如双方 `--concurrency 4`。不同并发配置不能直接混在一对比较中。

每次输出：

- `manifest.json`：运行身份、完整协议和运行状态；失败保留 running 状态，不能当完成。
- `environment.txt`：客户端实际安装版本。
- `responses.jsonl`：原始 API 响应，包括结束原因及服务端返回的 usage。
- `requests-responses.jsonl`：每条已完成响应及其完整 HTTP 请求，逐条落盘，供精确续跑使用。
- `results.json`：官方总分、标准误、每题原始/提取答案与得分、输入/目标摘要。
- compare 输出：所有指标的分数与百分点差、主指标退化/改善题号、文本变化题号、两端运行配置。GSM8K 每题有两套提取规则，题目数不会重复计算。

API 错误、缺失或非字符串 content 和异常 finish_reason 直接失败，不用占位答案继续评分，不自动重试。合法的空字符串保留给官方任务计分，并记录 `empty_answers`，不以 reasoning_content 替换。`length` 响应保留评分与原文，但阻止验收；增大相同预算后重跑两端。工具发送未截断的 messages，不在客户端裁掉输入；服务端必须对超长输入报错，不能静默截断，需结合启动配置和日志核验。

长请求的 `--timeout` 必须覆盖该并发下最大输出预算的生成时间。例如 32768 tokens 在每请求约 1.3 token/s 时超过 7 小时，4 小时 HTTP 超时并不足够。超时不等于引擎停滞，应同时观察输入/输出 token 进度。

中断后以新 `--output` 目录运行，增加 `--resume <上次输出目录>`，其余生成和评分参数保持不变。续跑严格核对评分协议、脚本摘要、引擎/模型身份和模型别名，按完整请求内容复用已完成响应，只发送缺失请求，再通过官方评分器重新生成全部样本成绩。请求并发度、采样参数、权重或引擎改变时拒绝复用；没有请求关联的旧 `responses.jsonl` 不能按响应顺序猜测恢复。`reused_responses` 记录复用数量；续跑不用于连续性能测量。

## 验收含义

默认不设通用“业界允许误差”，完整对比结果状态为 `review_required`。当前数值验收以两端完整任务成绩、百分点变化和逐题退化/改善为主；仅当任务质量退化时再追查数值内部，不把逐 token 或小模型 TV 对齐单独设为阻断项。稳定性、加载取消和性能仍分别验收。

确定本项目的分数下降预算后才传 `--max-drop-pp`，单位是**绝对百分点**，对所有输出指标生效。例如 0.5 表示最多下降 0.5 个百分点，不是“99% token 一致率”。`within_configured_budget` 只表示满足这个分数预算，**不表示统计等价、全部模型能力或实现正确性已经通过**。样本上的标准误不等于两端差值的配对置信区间；微小差值仍需检查逐题结果，必要时重复评测/做配对统计。

退出码：0=生成报告（可能仍需 review），2=退化或不具备验收条件，其他非零=运行/协议错误。`--limit`、合成测试、任何输出截断、缺少响应均不能通过验收，即使分数很高。样本、filter、逐题输入/目标或模型身份不一致时拒绝比较。

CPU 检查：

```bash
.deps/mac-cpu-venv/bin/python -m unittest discover -s tests/scripts -p test_quality_eval.py -v
# 准备数据后，离线运行真实官方任务/评分器 + 本地 HTTP 测试服务：
XLLM_QUALITY_EVAL_INTEGRATION=1 .deps/quality-eval-venv/bin/python -m unittest discover -s tests/scripts -p test_quality_eval.py -v
```

第二条用显式 `pipeline-test` 响应注入正确/错误答案，验证评分和退化识别，不能用它给真实模型报分。设备阶段依次为：真实权重加载和退出/资源验证 → 两端小样本联调 → 两端完整数据集 → 必要的并行与并发矩阵。原有算子精度、logits、稀疏选取、长上下文、取消和释放测试继续保留。

首批不涵盖中文知识、代码执行、长上下文或多模态。后续按能力阶段接入相应标准任务。经典 MMLU/HellaSwag 的 likelihood 协议不能直接套用 chat endpoint；要先验证 prompt loglikelihood 接口再接，不把生成式选择题成绩冒充原协议。

## 官方依据

- [EleutherAI harness 固定版本](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.13)
- [GSM8K 官方任务](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/tasks/gsm8k/gsm8k-cot.yaml)
- [IFEval 官方任务](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/tasks/ifeval/ifeval.yaml)
- [API 适配协议](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/docs/API_guide.md)
