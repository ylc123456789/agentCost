#!/usr/bin/env bash
# No official dataset execution, CUDA, training, model-weight download, or commercial API.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
out=${1:?usage: bash scripts/local_acceptance.sh ABS_EVIDENCE_DIR ABS_TEMP_DIR [V20_RUNS_MAIN]}
temp=${2:?separate project temporary directory required}
legacy=${3:-}
py=${CE_PYTHON:-"$repo/.local/conda-envs/capability-eval-cpu/bin/python"}
: "${CE_TEST_BWRAP:?Set CE_TEST_BWRAP to a Linux bubblewrap executable}"
[[ "$out" = /* && "$temp" = /* && "$py" = /* && "$CE_TEST_BWRAP" = /* ]]
mkdir -p "$(dirname "$out")" "$temp"
mkdir "$out"
exec > >(tee "$out/commands.log") 2>&1
started=$SECONDS
trap 'code=$?; set +x; printf "exit=%s\nwall_seconds=%s\n" "$code" "$((SECONDS-started))" > "$out/status.txt"' EXIT
export TMPDIR="$temp" XDG_CACHE_HOME="$temp/cache" XDG_CONFIG_HOME="$temp/config"
export HF_HOME="$temp/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1
export CE_TEST_RECORDS="$out/contract-snapshots"
offline() { unshare -Urn "$@"; }
ce() { offline "$py" -B -m capability_eval "$@"; }
std() { PYTHONPATH="$repo/src" offline /usr/bin/python3 -B -S -m capability_eval "$@"; }

set -x
git rev-parse HEAD > "$out/source.commit"
git status --porcelain=v1 > "$out/source.status"
git diff --check HEAD
test ! -s "$out/source.status"
cp locks/linux-64-conda.explicit locks/cpu-linux-py311.txt "$out/"
offline "$py" -m pip check
"$CE_TEST_BWRAP" --version
sha256sum "$CE_TEST_BWRAP" > "$out/bwrap.sha256"
offline "$py" -B -m unittest discover -s tests -v |& tee "$out/contract-tests.log"

ce validate --config examples/fixture.json
ce prepare --config examples/fixture.json --run-dir "$out/fixture"
ce run --run-dir "$out/fixture"
ce score --run-dir "$out/fixture" --private examples/fixture-private.json
ce score --run-dir "$out/fixture" --private examples/fixture-private.json --rescore 'same-candidate regression'
ce analyze --run-dir "$out/fixture" --workload examples/workload.json
ce verify --run-dir "$out/fixture"

ce validate --config examples/tiny-cpu.json
ce prepare --config examples/tiny-cpu.json --run-dir "$out/tiny-cpu"
ce run --run-dir "$out/tiny-cpu"
ce analyze --run-dir "$out/tiny-cpu" --workload examples/workload.json
ce verify --run-dir "$out/tiny-cpu"

cd "$out"
mkdir inspect-original
cd inspect-original
HOME="$PWD" offline "$py" -B "$repo/examples/inspect_smoke.py" > result.json
cd "$repo"
ce prepare --config examples/inspect.json --run-dir "$out/inspect-wrapped"
ce wrap --run-dir "$out/inspect-wrapped" --phase serve --attempt-id a000001 -- "$py" -B "$repo/examples/inspect_smoke.py"
ce analyze --run-dir "$out/inspect-wrapped" --workload examples/workload.json
ce verify --run-dir "$out/inspect-wrapped"
cmp "$out/inspect-original/result.json" "$out/inspect-wrapped/raw/a000001/stdout.log"

if [[ -n "$legacy" ]]; then
  ce import-events --input "$legacy" --mapping examples/v20-mapping.json --run-dir "$out/v20-import"
  std analyze --run-dir "$out/v20-import"
  std verify --run-dir "$out/v20-import"
fi

# Do not give the offline reporter raw outputs, trained assets, model packages or network.
mkdir "$out/ledger-only-input"
for name in plan.json artifact_freeze.json environment.json events.jsonl attempts.jsonl scores.v0002.jsonl scorecomplete.v0002.json; do
  cp "$out/fixture/$name" "$out/ledger-only-input/$name"
done
std analyze --run-dir "$out/ledger-only-input" --workload examples/workload.json --ledger-only --output "$out/ledger-only-rebuild"
for name in analysis.json resource_summary.json report.md; do
  cmp "$out/fixture/$name" "$out/ledger-only-rebuild/$name"
done
offline "$py" -B scripts/check_evidence.py "$out" > "$out/proof-checks.json"
