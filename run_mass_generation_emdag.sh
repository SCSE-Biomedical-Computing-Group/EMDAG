# run_mass_generation.sh
#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

# Get test set size
N=$(python - <<'PY'
from emdag.utils.misc import load_config
from emdag.datasets import get_dataset
cfg,_ = load_config('./configs/test/emdag_test.yml')
print(len(get_dataset(cfg.dataset.test)))
PY
)
echo "Test set size: $N"
for idx in $(seq 0 $((N-1))); do
  python design_testset.py $idx \
    --config ./configs/test/emdag_test.yml \
    --out_root ./results_emdag \
    --device cuda --batch_size 16 --seed 2022
done

python -m emdag.tools.relax \
  --root "./results_emdag" \

python -m emdag.tools.eval \
  --root "./results_emdag" \