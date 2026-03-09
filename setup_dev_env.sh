#!/bin/bash
set -euo pipefail

# dev_env = any virtual env you'd like to use for specific development needs
ENV_DIR=~/envs/dev_env

# Load qibo module, unset python path, Create fresh dev_env
module load qibo
unset PYTHONPATH
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# Install whichever editable local packages you want to use in dev_env
pip install -e ~/qibocal-eo-fork

# Pin qibolab to 0.2.9 (compatible with qibo 0.2.x).
# qibocal pulls qibolab 0.2.12+ which requires qibo >=0.3.0, but
# qibocal itself requires qibo <0.3.0 — so we force the last compatible version.
pip install "qibolab==0.2.9"

# Lock qibo version to match qibocal 0.2.x requirements
echo "qibo<0.3.0" > /tmp/qibocal-constraints.txt
pip install -e ~/CQT-experiments-eo-fork -c /tmp/qibocal-constraints.txt
rm /tmp/qibocal-constraints.txt

# Add module site-packages as fallback (for keysight + its deps).
# Named zzz_* so it sorts AFTER any editable install .pth files
# (e.g. qibocal.pth, qibolab.pth), ensuring dev_env always wins.
VENV_SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')
MODULE_SITE="/mnt/scratch/envs/qibo/lib/python3.12/site-packages"

echo "$MODULE_SITE" > "$VENV_SITE/zzz_module_fallback.pth"
echo "Created $VENV_SITE/zzz_module_fallback.pth → $MODULE_SITE"

# 4. Log all installed versions for clarity
echo ""
echo "===== INSTALLED PACKAGES ====="
pip list | grep -iE "qibo|qibocal|qibolab|qiboml|qibo-client"
echo ""
echo "===== qibocal location ====="
pip show qibocal | grep -E "^(Name|Version|Location|Editable)"
echo "=============================="
