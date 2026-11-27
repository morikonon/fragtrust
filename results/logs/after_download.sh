cd "$HOME/fragtrust"
for n in test train val; do
    until grep -q "^EXIT=" results/logs/download_$n.log; do sleep 30; done
done
if grep -q "^EXIT=0" results/logs/download_test.log && grep -q "^EXIT=0" results/logs/download_train.log && grep -q "^EXIT=0" results/logs/download_val.log; then
    ls -la data/raw/4k_1
    bash scripts/run_base_plan.sh
else
    echo "a download failed; see results/logs/download_*.log"
fi
