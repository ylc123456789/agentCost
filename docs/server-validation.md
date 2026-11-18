# 服务器验收交接 · 不在本地执行

本页命令只供获得服务器执行授权后使用。本地只跑合成数据和离线 v20 转换。不要在授予算力前连接、安装、传资产或发起 GPU 测试。本页不含服务器地址、SSH 密钥或凭据。

## B · 官方 grader-only（未运行）

先确认独立 Linux Conda 环境、可用的 Bubblewrap user/pid/net/mount namespace、锁定数据 hash、源码 commit 和依赖版本。所有新增服务器文件放在 `/root/autodl-tmp`。依赖使用新的 prefix，BigCodeBench 的 Parquet 读取还需 `pyarrow==19.0.1`；官方题真实依赖资格以参考通过为准，不假定本地合成测试证明了兼容。临时目录单独定位，不改全局目录；保存完整 `plan.json`、退出状态、stderr、环境和 checksum。

```bash
# 设置为已获授权的服务器路径，均应在 /root/autodl-tmp。
: "${EVAL_ROOT:?}" "${DS_GZ:?}" "${BCB_PARQUET:?}" "${BCB_EVAL_ROOT:?}" "${EVAL_PYTHON:?}" "${BWRAP:?}"
export TMPDIR="$EVAL_ROOT/tmp" CUDA_VISIBLE_DEVICES= TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR"

"$EVAL_PYTHON" -B -m capability_eval official-contract --dataset ds1000 \
  --input "$DS_GZ" --bwrap "$BWRAP" --run-dir "$EVAL_ROOT/ds1000-dev-contract-01" --allow-official

"$EVAL_PYTHON" -B -m capability_eval official-contract --dataset bcb \
  --input "$BCB_PARQUET" --evaluator-root "$BCB_EVAL_ROOT" --bwrap "$BWRAP" \
  --run-dir "$EVAL_ROOT/bcb-complete-dev-contract-01" --allow-official
```

- DS-1000：固定开发 ID 10（Pandas）、315（NumPy）、846（Sklearn）。按原 `test_execution` 的 `[insert]` 合同和存在时 `test_string`；不修补选手代码。数据 gzip SHA256 `e8c6daa9d7223976bce0296644f3933f78d7f47830669ff05cd61da62c6ba9b3`。
- BCB：`BigCodeBench/206`，已曝光，**不能用于留出**。Full/v0.1.4/Complete 已知快照 revision `b74c0d0bf70d2c0bc459be537895cca163007f1a`；Parquet SHA256 `d9a4965821c9507ebdfb551c288656b2d5fe553234f5183044333ca8a4018267`。固定官方实现及四个源码 hash；接受完整 Python module，沿用该版本 calibrated 拼接规则。
- 每题参考、失败、超时各一次；总共 12 条控制、每条墙钟上限 10 秒（另有每批 sandbox preflight 及清理），计划候选工作上限 120 秒，不是耗时预测；GPU、训练、生成、商业 API 均 0。超时是缺失，不当作 0 分。任何不合格先查原日志，不补跑正式题追分。
- `verify --run-dir …` 核验已完成的资格证据。评分沙箱隔离宿主文件、网络和命名空间；官方 Python 测试与候选同评分进程，不能把它解释成对恶意候选伪造评分的完整防御。正式使用前须核参考和控制。

## C · 单张 4090 校准（未运行）

另冻计划后才启动。待确认内容：本地已获准的 `Qwen3-4B-Instruct-2507` 权重／tokenizer／模板哈希、精度和本机驱动、三道预先合格的 dev 题、同规则 B0/T 的 6 份候选以及完整 CPU/GPU 核查。首批建议 GPU 持有上限 1 GPUh，训练和付费 API 为 0；不能自行启动实例、下载模型或延长预算。要先提供目标 runner（内置 tiny driver 不跑 GPU）；使用外部 SDK CUDA 窗口和排他持有接口，区分 allocator／整卡采样／离线评分重叠。B0/T 的结果不能给 Z/LoRA 或 T+L 的目标卡状态升级。
