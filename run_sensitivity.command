#!/bin/zsh
# Sensitivity analysis: ResNet18 cross-validation with a looser merge threshold (125 panel groups).
cd "$(dirname "$0")" || exit 1

# Make conda available (the .zshrc of an interactive Terminal normally does this)
[ -f "$HOME/.zshrc" ] && source "$HOME/.zshrc" >/dev/null 2>&1
if ! command -v conda >/dev/null 2>&1; then
  for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
              "/opt/homebrew/Caskroom/miniconda/base" "/opt/homebrew/anaconda3" "/opt/anaconda3" "/usr/local/anaconda3"; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then source "$base/etc/profile.d/conda.sh"; break; fi
  done
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.zsh hook 2>/dev/null)"
  conda activate pv-health
fi
echo "Python: $(which python3)"
if ! python3 -c "import torch, matplotlib, sklearn" 2>/dev/null; then
  echo ""
  echo "FEL: hittar inte torch/matplotlib i den har Python-miljon. Oppna Terminal, skriv"
  echo "  conda activate pv-health"
  echo "och kor sedan raden som jag skickade i chatten."
  read -k 1
  exit 1
fi
caffeinate -i sh -c '
  python3 src/cnn_cv.py --model resnet18 --group-file data/processed/group_merge_t0995.csv --tag t0995
'
echo ""
echo "Klart. Du kan stanga det har fonstret."
read -k 1
