# 外部项目接入 · Inspect AI 与 v20

## Inspect AI 原执行循环

固定 `inspect-ai==0.3.139`（MIT，调用已安装的包），`transformers==4.56.2`、`torch==2.8.0+cpu`。`examples/inspect_smoke.py` 是独立任务：Inspect AI 的 `Task`、`generate(cache=False)`、评测执行循环和日志不修改。仅在自定义 ModelAPI 提供随机初始化的微型 GPT-2 CPU 后端；一条合成题、一条生成、三个新 token 上限，不进行官方评分。包装时增加可选 SDK 边界：实际调用 `.generate()` 前记 `inspect_generate`、生成后记录后端真实 token 数；对独立运行没有强依赖。

```bash
R=/home/cyl/agentRes26
P="$R/.local/conda-envs/capability-eval-cpu/bin/python"
WORK=/mnt/e/codexWorkSpace/capability-eval/tmp
mkdir -p "$WORK/inspect-original"; cd "$WORK/inspect-original"
HOME="$PWD" TMPDIR="$WORK" unshare -Urn "$P" -B "$R/examples/inspect_smoke.py"

cd "$R"
TARGET="$WORK/inspect-wrapped-01"
unshare -Urn "$P" -B -m capability_eval prepare --config examples/inspect.json --run-dir "$TARGET"
unshare -Urn "$P" -B -m capability_eval wrap --run-dir "$TARGET" --phase serve --attempt-id a000001 -- "$P" -B "$R/examples/inspect_smoke.py"
"$P" -B -m capability_eval analyze --run-dir "$TARGET" --workload examples/workload.json
```

实际原命令和包装输出一致：`{"completion": "[66, 152, 152]", "runner": "inspect_ai", "sample_id": "inspect-local-1", "status": "success"}`。包装版保存外层和进程树 CPU、日志、SDK token；内部冷加载细分、检索、训练、GPU、官方质量均没有覆盖。

其他原项目用可信的 argv 数组（不走 shell）。不修改原项目也可获得外层墙钟、退出码、进程及原输出。可细分时：

```python
from capability_eval import Recorder

recorder = Recorder.from_environment()
with recorder.span("retrieval", role="retriever"):
    recorder.reserve("retrieve")  # requests.retrieve 必须先在外层计划内预留
    hits = project_retrieve(public_prompt)
    recorder.measure("retrieved_items", len(hits), "items")
recorder.attach_artifact("result.json", label="project-output")
```

`span(phase=…, role=…)` 也可标教师／编码／工具／训练和生成接点；训练 step/token 由真实入口计数后调用 `measure`。子进程事件写独立 JSONL 并检查身份后合并，不能代替可信原项目对预算的执行。GPU 观测需声明分配边界；`metering.cuda_window` 只同步、采 allocator 与可选 NVML，不自动宣告持有。明确的排他设备持有可记录 `phase="device_hold", data={"attribution":"exclusive"}`；仅在匹配设备、时钟及归属的排他窗口内，用户才能在阶段标 `accounting="phase", data={"charge_gpu_allocation":True}`。重叠分账或离线评分重叠不进部署投影。

## 只读历史导入

```bash
SRC=/mnt/e/ResAgent2-test-records/server/2026-11-15/capability-injection-bcb-symmetric-memory-v20/snapshot-20261115T123724Z/server/runs/main
capability-eval import-events --input "$SRC" --mapping examples/v20-mapping.json --run-dir "$WORK/import-v20-01"
capability-eval analyze --run-dir "$WORK/import-v20-01"
capability-eval verify --run-dir "$WORK/import-v20-01"
```

v20 commit `2833ceccdc21728e19c3fe3f138171b07d4634e1`，使用原 `BATCH_FREEZE.json`、`ATTEMPT_LEDGER.json`、`runtime_events.jsonl`、候选及 score 文件。六组×40 题，原通过数为 15、21、20、17、15、16。保留原文件字节和映射哈希。`legacy_actor_seconds` 包含评分，不作为 `serve` 开销；Torch 峰是进程累计，不是逐尝试峰。没有原 PlanFreeze 的 JSONL 可用 `examples/flat-mapping.json`，只报观测分母，不构造配对。

## 源码边界与复用

- v20 `runtime_v20.py`、`prepare_bcb.py`、`RUNTIME_CONTRACT.md` 仅作为协议参考；未复制旧 runner、数据、模型或评分沙箱实现。v20 历史代码未假定有公开再分发许可。
- DS-1000 commit `b39aab71da6d23ef8d3cac59a7c5f834516ab334` 的 `test_ds1000.py`、`environment.yml` 用于核对评测合同；数据／仓库 CC BY-SA 4.0，不入库。
- BigCodeBench commit `09dd993f46c3fbf3a799465bb96d524edcb0b199` 的 `bigcodebench/eval/__init__.py`、`utils.py`（文件 MIT 声明）、`_special_oracle.py`、`bigcodebench/__init__.py` 仅从操作者提供的固定 checkout 调用。未 vendoring。隔离进程内用外层 Bubblewrap 替代上游 Python-only `safe_environment`，仍调用官方测试发现、执行和判定逻辑。
