# Capability Eval · 科研 Agent 开销评测器

以冻结的任务、能力状态、执行规则和预算测量固定版本的制备、加载、完整使用及离线评分。源码是 Python 包 `capability_eval`；分析核心只用标准库。首版执行器面向 Linux 单机串行，CPU 样例只验证接口和记账，不验证科研能力，也不访问正式数据。

## WSL 本地快速开始

```bash
cd /home/cyl/agentRes26
ENV="$PWD/.local/conda-envs/capability-eval-cpu"
WORK=/mnt/e/codexWorkSpace/capability-eval
mkdir -p "$WORK/tmp" "$WORK/cache" "$PWD/.local/conda-pkgs"
export TMPDIR="$WORK/tmp" XDG_CACHE_HOME="$WORK/cache" PIP_CACHE_DIR="$WORK/cache/pip"
export CONDA_REGISTER_ENVS=false CONDA_PKGS_DIRS="$PWD/.local/conda-pkgs"

# 已有该 prefix 时不重建。复装依赖需要软件包下载，不下载模型或数据。
conda create --prefix "$ENV" --file locks/linux-64-conda.explicit
"$ENV/bin/python" -m pip install -r locks/cpu-linux-py311.txt
"$ENV/bin/python" -m pip install --no-index --no-build-isolation -e .
"$ENV/bin/python" -m pip check
export PATH="$ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1

# 目录必须尚不存在。Linux 支持 unshare 时可以在断网命名空间运行。
RUN="$WORK/tmp/quickstart-01"
capability-eval validate --config examples/fixture.json
capability-eval prepare --config examples/fixture.json --run-dir "$RUN"
capability-eval run --run-dir "$RUN"
capability-eval score --run-dir "$RUN" --private examples/fixture-private.json
capability-eval analyze --run-dir "$RUN" --workload examples/workload.json
capability-eval verify --run-dir "$RUN"
```

仅需核心 CLI 时可跳过 `pip install -r`。`examples/tiny-cpu.json` 以同一固定 seed 微型 GPT-2 查验 B0、文字 T、`inputs_embeds` Z、动态 PEFT LoRA L、组合 T+L。把配置换成它并换一个空 run 路径，执行 `validate → prepare → run → analyze → verify`（不评分）。权重全部本地合成，不训练、不下载。独立执行器及 v20 读入见 [接入说明](docs/integrate-existing.md)。

```bash
# Bubblewrap 可用时，运行全部本地合同回归及离线烟测。
export CE_TEST_BWRAP="$PWD/.local/bwrap/usr/bin/bwrap"  # 或系统 bwrap 的绝对路径
DAY=$(TZ=Asia/Shanghai date +%F)
bash scripts/local_acceptance.sh \
  "/mnt/e/ResAgent2-test-records/local-diagnostics/$DAY/capability-eval-recheck-01" \
  "$WORK/tmp" \
  /mnt/e/ResAgent2-test-records/server/2026-11-15/capability-injection-bcb-symmetric-memory-v20/snapshot-20261115T123724Z/server/runs/main
```

尾部 v20 路径参数可省略；给定时只转换原文件。完整脚本要求工作树干净，记录实际源码提交，不启动 `official-contract`。Linux user namespace 不可用时脚本停止，不把未隔离运行算作通过。

## 命令和产物

| 命令 | 合同 |
|---|---|
| `validate` | 校验配置和声明的文件身份，不制备、不执行。 |
| `prepare` | 在新目录写 `plan.json` 与 `environment.json` 后才制备，最后写 `artifact_freeze.json`。保留制备失败。 |
| `run` / `wrap` | 预算和计划行先写盘；运行时失败在账。`wrap` 只接受 `serve` 和预登记的 exact argv。已预留或已启动的行默认不重派。未确认清理则停止。 |
| `score --private SPEC` | 从候选文件读入，不加载生成模型。用 `--rescore REASON` 创新版本；改评分规格还需 `--revised-spec`。原 hash 不变。 |
| `analyze --workload FILE` | 不依赖模型，读取最新的完整评分版本；重算 `resource_summary.json`、`analysis.json` 和 `report.md`。 |
| `analyze --ledger-only --output NEW_DIR` | 只需冻结文件、账本和分数，内容可重算，不校验不存在的 raw 文件；输出必须在原 run 目录外。 |
| `import-events` / `verify` | 读入原事件及校验；默认校验正常 run 的全部文件哈希。 |

`attempts.jsonl` 是状态转换日志，完整分母来自 `plan.json`；不是只统计已经有输出的行。`scores.jsonl` 是第一版副本，`scores.vNNNN.jsonl` 和对应完成回执为各版原账。`raw/` 保存候选、进程输出、日志和清理回执。`checksums.json` 覆盖报告和原始记录，是完整性核对，不是签名。大文件、正式数据和权重不要提交进仓库。

## 使用边界

- 方法只接收按字段重建的 `PublicTask`、自身状态及运行参数；私有文件不传入方法。Python SDK/plugin 和外部命令是**受信任接入**，接口分离不是对恶意同 UID 代码的沙箱；要隔离不受信任的 Agent，应另加文件系统和网络边界。官方候选由单独的 Linux Bubblewrap 评分路径执行。
- `serve` 包含能力实际使用到候选交付；诊断前向和离线评分独立记账。外层和嵌套区间不相加；峰显存取同域最大，不摊销。未测的量为 `null` 加原因，wrapper 不推造训练 token 或 allocator 峰。
- 实际设备持有仍含空闲和离线评分；投影只接受显式、同域、非共享 GPU 分账。CPU 冷启动是每项独立进程，不能用来外推暖会话。SDK 可记录 CUDA/NVML，**没有目标卡实测证据**。
- 流式更新、并发、多卡、kernel 忙碌归因、逐层 KV、超出 T+L 的内置组合不支持。官方任务及目标 4090 验收见 [服务器交接](docs/server-validation.md)；不属于本地 smoke。
