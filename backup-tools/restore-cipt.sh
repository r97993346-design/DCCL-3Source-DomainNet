#!/usr/bin/env bash
set -euo pipefail

backup=$(realpath "${1:?Usage: bash restore-cipt.sh BACKUP_DIRECTORY [verify|restore|runtime]}")
mode=${2:-verify}
disk=/root/autodl-tmp

verify() {
    (cd "$backup"; sha256sum --quiet --check SHA256SUMS)
    git bundle list-heads "$backup/cipt.bundle"
    git bundle list-heads "$backup/registry.bundle"
    echo "Backup checksums passed."
}

verify
case "$mode" in
    verify) exit 0 ;;
    restore|runtime) ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac

mountpoint -q "$disk" || { echo "Expected a mounted AutoDL data disk at $disk" >&2; exit 1; }
test "$(id -u)" -eq 0

if [[ "$mode" == runtime ]]; then
    test -f "$backup/system-runtime.tar.zst"
    command -v zstd >/dev/null
    for path in /root/miniconda3 /usr/local/cuda-11.8 /usr/local/cuda /usr/local/cuda-11 /etc/alternatives/cuda /etc/alternatives/cuda-11; do
        if [[ -e "$path" || -L "$path" ]]; then
            echo "Runtime restore requires an unused path: $path" >&2
            exit 1
        fi
    done
    zstd -dc "$backup/system-runtime.tar.zst" | tar -C / -xf -
fi

python=/root/miniconda3/envs/cipt/bin/python
test -x "$python" || { echo "Boot the preserved system image or restore the runtime first." >&2; exit 1; }
clone() {
    local bundle=$1 branch=$2 dest=$3 expected
    expected=$(git bundle list-heads "$bundle" "refs/heads/$branch" | awk '{print $1}')
    test -n "$expected"
    if [[ -e "$dest" ]]; then
        test "$(git -C "$dest" rev-parse HEAD)" = "$expected"
        test -z "$(git -C "$dest" status --porcelain)"
    else
        git clone -q --branch "$branch" "$bundle" "$dest"
    fi
}

clone "$backup/cipt.bundle" feature/pacs-fixed6-tpe-prompt-rd3-condcov3-20261127 "$disk/code"
clone "$backup/cipt.bundle" feature/vlcs-fixed6-tpe-lr1e5-prompt-rd3-condcov3-20261127 "$disk/code-vlcs"
clone "$backup/cipt.bundle" feature/terra-fixed6-tpe-lr2e5-prompt-rd3-condcov3-20261127 "$disk/code-terra"
clone "$backup/registry.bundle" registry-implementation "$disk/global-optuna-registry"

for dir in datasets weights; do
    mkdir -p "$disk/$dir"
    changes=$(rsync -nrc --existing --itemize-changes "$backup/$dir/" "$disk/$dir/")
    if [[ -n "$changes" ]]; then printf 'Conflicting %s files:\n%s\n' "$dir" "$changes" >&2; exit 1; fi
    rsync -rt --ignore-existing "$backup/$dir/" "$disk/$dir/"
    test -z "$(rsync -nrc --itemize-changes "$backup/$dir/" "$disk/$dir/")"
done

mkdir -p "$disk/outputs"
for source in "$backup"/studies/*; do
    [[ -d "$source" ]] || continue
    target="$disk/outputs/${source##*/}"
    if [[ -e "$target" ]]; then
        echo "Leaving existing study at $target"
        continue
    fi
    mkdir "$target"
    rsync -rt "$source/" "$target/"
done

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$disk/code/CLIP"
"$python" -B -m pip check
"$python" -B - <<'PY'
import clip, optuna, torch
print("PyTorch", torch.__version__, "built CUDA", torch.version.cuda, "Optuna", optuna.__version__)
assert clip.tokenize(["a photo of a"]).shape == (1, 77)
PY
echo "Restore complete. No experiment was started."
