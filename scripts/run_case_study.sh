#!/usr/bin/env bash
# Case study after run_base_plan.sh: builds the FAT32 evidence image if it does not exist yet (Linux,
# dosfstools + mtools; files in data/case_study/files, see data/case_study/README.md), scans it with
# every model listed under case_study.runs in configs/paper.yaml, evaluates and rebuilds the paper assets.
# Usage: scripts/run_case_study.sh [configs/case_study/fat32_1g.yaml]
source "$(dirname "$0")/_common.sh"
config="${1:-configs/case_study/fat32_1g.yaml}"
mkdir -p results/logs
exec > >(tee -a results/logs/case_study.log) 2>&1

step "preflight"
plan=$(python - "$config" configs/paper.yaml <<'EOF'
import sys
from pathlib import Path

import yaml

case_path, paper_path = sys.argv[1:3]
case, spec = yaml.safe_load(open(case_path)), yaml.safe_load(open(paper_path))["case_study"]
if spec["image"] != case["name"]:
    raise SystemExit(f"{paper_path}: case_study.image is {spec['image']!r}, but {case_path} builds {case['name']!r}")
runs = [Path("results/runs") / run for run in spec["runs"]]
if missing := [str(run) for run in runs if not (run / "DONE").exists()]:
    raise SystemExit(f"runs not finished yet (run scripts/run_base_plan.sh first): {', '.join(missing)}")
image = Path(case["out_dir"])
if not (image / "build.json").exists() and not any(p.is_file() for p in Path(case["files_dir"]).rglob("*")):
    raise SystemExit(f"no image at {image} and no evidence files in {case['files_dir']}: see data/case_study/README.md")
print(image, *runs)
EOF
)
read -r -a plan <<< "$plan"
image="${plan[0]}"
runs=("${plan[@]:1}")
echo "image: $image | runs: ${runs[*]}"

if [[ -f "$image/build.json" ]]; then
    step "image exists ($image/build.json), not rebuilding"
else
    step "building the image (log: $image/build.log)"
    ft case-build --config "$config"
fi

step "scanning every cluster"
ft case-scan "${runs[@]}" --image "$image"

step "evaluation: transferred guarantees, signature baseline, recalibration"
ft case-eval "${runs[@]}" --image "$image"

step "paper figures and tables (paper/)"
ft report
echo
echo "case study finished: results/case_study/$(basename "$image")/"
