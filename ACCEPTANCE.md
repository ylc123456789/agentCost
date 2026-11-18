# Capability Eval v0.1 · 2026-11-18 本地验收

执行源码：`8e467616bf26c6ce8977e733a35c127c0e0cb7e7`（本文件后续只补验收记录）。项目 `/home/cyl/agentRes26`，Ubuntu-D WSL2。专用 Conda prefix `/home/cyl/agentRes26/.local/conda-envs/capability-eval-cpu`，Python 3.11.15、Torch `2.8.0+cpu`、Transformers `4.56.2`、PEFT `0.17.1`、Inspect AI `0.3.139`。

上面是历史验收路径。当前项目根为 `/home/cyl/agentRes26/agentCost`；专用环境位于 `/home/cyl/agentRes26/agentCost/.local/conda-envs/capability-eval-cpu`。

| 验收项 | 状态 | 证据 |
|---|---|---|
| 核心 SDK / CLI / 账本 | 本地通过（`fixture_only` 合成合同） | 14/14 合同测试通过（38.997 秒）。预算早于派发；超时进程树已确认退出；强制中断和预留后建目录失败均不重发已预留项；保持计划分母。合成配对 +1/−1；6 份候选两次评分的 hash 一致。断网、`python -S`、只有账本与冻结文件的重算，三个报告逐字节一致。 |
| B0 / T / Z / LoRA / T+L | `verified_cpu`，随机初始化的微型 GPT-2 | 5/5 生成，5 次生成预留、9 次诊断前向；Z 为 `[1,3,32]` 非零输入，扰动及移除对照的 logits 最大差值约 0.00720／0.38990；LoRA 启用差值约 0.01389，禁用后相对底座差值为 0；同一次 T+L 生成有文字与非零 LoRA 两条证据。没有质量结论。 |
| 现成执行器 / 历史读入 | 本地接入通过 | Inspect AI 保留原循环，包装前后输出相同 `[66, 152, 152]`；各执行一条合成题。历史 v20 只读导入 240 条，有效通过数 15／21／20／17／15／16，原导入清单核对 1217 文件。最终代码另行复算同值。不把 v20 含评分耗时或进程累计峰当新口径。 |
| DS-1000 / BigCodeBench 官方合同 | `unverified`，未跑 | 仅合成评分隔离正／负／超时测试及官方 adapter 字段检查。正式数据执行默认拒绝。官方入口、数据哈希、3+1 dev 样本与 12 个控制已列在服务器交接。 |
| 4090 目标校准 | `unverified`，未跑 | 内置 tiny driver 仅 CPU；SDK CUDA/NVML 和设备持有分账未在目标卡验收。逐层 KV、超出 T+L 的内置组合、流式更新、并发、多卡和 kernel 忙碌归因 `unsupported`。 |

**实测和边界。** 最终本地复跑墙钟 100 秒（不含软件包安装、旧账首次转换和独立 wheel 构建）。tiny 共享制备一次约 4.531 秒、五个进程外层合计约 19.340 秒（含诊断），其中 serve 合计 1.234 秒；Inspect 包装外层约 4.331 秒。无 GPU 使用（0 GPUh）、无训练、无商业 API、无模型权重下载、未执行正式数据、未登录服务器。资源秒为本机烟测，不作正式比较或 GPU 耗时预测。

**复跑与证据。** 本地入口为 `bash scripts/local_acceptance.sh ABS_EVIDENCE_DIR ABS_TEMP_DIR [V20_RUNS_MAIN]`，见 README 的完整命令。主证据路径 `E:/ResAgent2-test-records/local-diagnostics/2026-11-18/capability-eval-v0.1-final`。入口文件 `contract-tests.log`、`proof-checks.json`、`fixture/report.md`、`tiny-cpu/report.md`、`inspect-wrapped/report.md`、`v20-recomputed/report.md`、`BUNDLE_SHA256.json`。最终 wheel 已断网构建试装，并用标准库解释器验证入口和随包 schema。旧导入源账、第一次跨版本尾差失败和修复前日志保留在同日 `capability-eval-v0.1`，不要当作最终通过记录。

**下一步需授权。** 先在用户指定的可用 Linux 服务器上执行 grader-only 合同，不需要 GPU；执行命令见 `docs/server-validation.md`。固定 12 个控制，计划候选墙钟上限 120 秒（另计 preflight/清理），零生成、零训练、零付费接口。之后另冻 4090 目标 runner 和 B0/T 的 6 份候选，只有获准后才启动，首批建议上限 1 GPUh。
