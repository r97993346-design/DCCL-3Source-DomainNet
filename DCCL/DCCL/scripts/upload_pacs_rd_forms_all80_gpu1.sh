#!/usr/bin/env bash
# Upload analysis-only all-80 RD formula PACS results; excludes model weights and checkpoints.
# Usage: bash scripts/upload_pacs_rd_forms_all80_gpu1.sh [run_directory]
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
branch="feature/cipt-b5b-entropy-mcc-rd"

[[ "$(git branch --show-current)" == "$branch" ]] || {
  echo "ERROR: switch to $branch before uploading." >&2; exit 1;
}
if ! git diff --cached --quiet; then
  echo "ERROR: Git already has staged files. Commit or unstage them first." >&2
  exit 1
fi

base="${CIPT_OUTPUT_ROOT:-train_output/ablation/rd_forms_all80_gpu1}"
if (( $# > 0 )); then
  src="$1"
else
  [[ -d "$base" ]] || { echo "ERROR: missing results directory $base" >&2; exit 1; }
  src="$(find "$base" -mindepth 1 -maxdepth 1 -type d -name 'run_*' | LC_ALL=C sort | tail -n 1)"
fi
[[ -n "$src" && -d "$src" ]] || { echo "ERROR: no run_* directory found." >&2; exit 1; }
src="$(cd -- "$src" && pwd -P)"
run_id="$(basename -- "$src")"
[[ "$run_id" == run_* ]] || { echo "ERROR: expected run_* directory." >&2; exit 1; }

# Avoid forced pushes or overwriting unrelated work.
git pull --ff-only origin "$branch"

dest="analysis_results/rd_forms_all80_gpu1/$run_id"
mkdir -p -- "$dest"
count=0
result_count=0
while IFS= read -r -d '' file; do
  name="$(basename -- "$file")"
  rel="$(realpath --relative-to="$src" "$file")"
  to="$dest/$rel"
  case "$name" in
    results.jsonl|log.txt|status.tsv|command.sh|git_commit.txt|launcher_snapshot.sh|config_cipt_b5b_all80_*.yaml|summary.json|summary.csv|metrics.json|metrics.csv)
      if [[ "$name" == results.jsonl ]]; then
        bytes="$(wc -c < "$file")"
        if (( bytes > 30000000 )); then
          echo "ERROR: results.jsonl larger than 30MB: $file" >&2; exit 1
        fi
        ((result_count+=1))
      fi
      mkdir -p -- "$(dirname -- "$to")"
      cp -- "$file" "$to"
      ((count+=1))
      ;;
    console.log)
      mkdir -p -- "$(dirname -- "$to")"
      tail -n 3000 "$file" > "${to%.log}_tail.log"
      ((count+=1))
      ;;
  esac
done < <(find "$src" -type f -print0)

if (( result_count < 1 )); then
  echo "ERROR: no results.jsonl found under $src. Nothing was committed." >&2
  exit 1
fi
{
  echo "PACS official B5b all-80 comparison: E1 S1 RD1"
  printf 'Source run: %s\n' "$src"
  printf 'Upload UTC: %s\n' "$(date -u +%FT%TZ)"
  printf 'Results files: %s (expected 3 complete methods)\n' "$result_count"
  printf 'Copied files: %s\n' "$count"
  echo
  if [[ -f "$src/status.tsv" ]]; then cat -- "$src/status.tsv"; fi
} > "$dest/upload_manifest.txt"

git add -f -- "$dest"
if git diff --cached --quiet; then
  echo "No new data to upload; already present in Git."
  exit 0
fi
echo "Staged files:"
git diff --cached --name-only
git commit -m "Upload PACS all80 E1 S1 RD1 metrics and logs $run_id"
git push origin "HEAD:$branch"
echo
echo "SUCCESS: uploaded $result_count results.jsonl file(s)."
printf 'https://github.com/r97993346-design/DCCL-3Source-DomainNet/tree/%s/DCCL/DCCL/%s\n' "$branch" "$dest"
