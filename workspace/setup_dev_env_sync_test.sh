#!/bin/bash
# =============================================================================
# SECTION 1 — USER CONFIG (auto-detected)
# List the repo folder names (under ~/) you want installed as editable packages.
# This is the only section you need to edit.
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
THIS_REPO="$(basename "$(cd "$SCRIPT_DIR/.." && pwd)")"
ENV_DIR=~/envs/workspace_env_sync_test
REPOS=(
  "$THIS_REPO"   # auto-detected from this script's location
  "CQT-qibocal"      # add whatever other package-repo local clones you need
  "CQT-qibolab"
)
# When sourced by another script (e.g. run_sinq20_dev.sh), only the variables
# above are evaluated — the setup body below does not run.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
set -euo pipefail
# =============================================================================
# SECTION 2 — ENV SETUP
# =============================================================================
module load qibo
unset PYTHONPATH
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel
# =============================================================================
# SECTION 3 — RESOLVE CONSTRAINTS
# Reads compatibility_matrix.toml and determines which packages to pin
# based on the repos listed above. Packages you install as editable forks
# are skipped; everything else is pinned to the known-good version.
# =============================================================================
echo ""
echo "===== RESOLVING COMPATIBILITY CONSTRAINTS ====="
resolver_output=$(python3 "$SCRIPT_DIR/resolve_constraints.py" "${REPOS[@]}")
CONSTRAINTS_FILE=""
PINS=()
while IFS= read -r line; do
  if [[ "$line" == CONSTRAINTS_FILE* ]]; then
    CONSTRAINTS_FILE="${line#CONSTRAINTS_FILE }"
  elif [[ "$line" == PIN* ]]; then
    PINS+=("${line#PIN }")
  fi
done <<< "$resolver_output"
echo "Constraints file: $CONSTRAINTS_FILE"
if [[ ${#PINS[@]} -gt 0 ]]; then
  echo "Pins to apply: ${PINS[*]}"
fi
echo "==============================================="
echo ""
# =============================================================================
# SECTION 4 — INSTALL
# Pins are installed first, then editable repos with the constraints file.
# =============================================================================
# Install pinned dependencies first
if [[ ${#PINS[@]} -gt 0 ]]; then
  pip install "${PINS[@]}"
fi
# Install each editable repo with constraints applied
for repo in "${REPOS[@]}"; do
  echo "Installing editable: $PARENT_DIR/$repo"
  if [[ -n "$CONSTRAINTS_FILE" ]]; then
    pip install -e "$PARENT_DIR/$repo" -c "$CONSTRAINTS_FILE"
  else
    pip install -e "$PARENT_DIR/$repo"
  fi
done
# Cleanup temp constraints file
[[ -n "$CONSTRAINTS_FILE" && -f "$CONSTRAINTS_FILE" ]] && rm "$CONSTRAINTS_FILE"
# =============================================================================
# SECTION 5 — MODULE FALLBACK
# Adds the module site-packages as a fallback AFTER editable installs,
# so dev_env packages always win. Named zzz_* to sort last.
# (Needed for keysight and other packages only available in the module.)
# =============================================================================
VENV_SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')
MODULE_SITE="/mnt/scratch/envs/qibo/lib/python3.12/site-packages"
echo "$MODULE_SITE" > "$VENV_SITE/zzz_module_fallback.pth"
echo "Created $VENV_SITE/zzz_module_fallback.pth -> $MODULE_SITE"
# =============================================================================
# SECTION 6 — LOG
# =============================================================================
echo ""
echo "===== INSTALLED PACKAGES ====="
pip list | grep -iE "qibo|qibocal|qibolab|qiboml|qibo-client"
echo ""
echo "===== qibocal location ====="
pip show qibocal | grep -E "^(Name|Version|Location|Editable)"
echo "=============================="
fi # end of direct-execution guard