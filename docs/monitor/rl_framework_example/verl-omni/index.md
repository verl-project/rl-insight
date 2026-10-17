# Monitor a verl-omni Training Run

This example adds RL-Insight to the Qwen-Image OCR LoRA FlowGRPO V1 recipe, using vLLM-Omni rollout and TransferQueue. Instructions are maintained here; `examples/monitor/verl-omni/` links to this guide.

## Prepare verl-omni and its data

Follow the [verl-omni installation guide](https://github.com/verl-project/verl-omni/blob/main/docs/start/install.md) and [FlowGRPO quickstart](https://github.com/verl-project/verl-omni/blob/main/docs/start/flowgrpo_quickstart.md), including OCR dependencies and model access.

RL-Insight requires Python >= 3.10; verl-omni has its own higher minimum (>= 3.11 in the referenced installation guide). Use a Python version supported by the entire training environment.

This guide follows the [upstream script at revision 6195598](https://github.com/verl-project/verl-omni/blob/619559873f6f34edb6078bac6cc5b953b27fa6f3/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_v1.sh): one node with four 80 GB GPUs, policy `Qwen/Qwen-Image`, and reward model `Qwen/Qwen3-VL-8B-Instruct`. The recipe uses host-memory offload between phases; follow upstream host-memory and kernel requirements as well as GPU requirements.

Launch arguments were checked against source, not executed on training hardware. Use the referenced revision for this script, or review newer recipe changes before running.

To prepare a separate checkout at that revision, run the following, then follow its installation and FlowGRPO setup guides:

```bash
git clone https://github.com/verl-project/verl-omni.git verl-omni-monitor-example
cd verl-omni-monitor-example
git checkout --detach 619559873f6f34edb6078bac6cc5b953b27fa6f3
```

The commands below use Bash in the Linux training environment. The monitoring server can run on a separate supported host.

Install this RL-Insight checkout in both training and server environments:

```bash
python -m pip install -e /path/to/rl-insight
```

From the **verl-omni repository root**, obtain the [raw OCR dataset](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr) following the upstream quickstart and place its training and test files under `$HOME/data/ocr`, then preprocess it:

```bash
mkdir -p "$HOME/data/ocr/qwen_image"
python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
    --input_dir "$HOME/data/ocr" \
    --output_dir "$HOME/data/ocr/qwen_image"
```

This produces `train.parquet` and `test.parquet`. The command below passes those paths explicitly. Ensure policy and reward weights can be downloaded, or override `actor_rollout_ref.model.path` and `reward.reward_model.model_path` with local copies.

Before launching, verify imports and both prepared datasets in the active environment:

```bash
python3 -c "import ray, rl_insight, verl_omni, vllm_omni; print('Training and monitoring imports OK')"
test -s "$HOME/data/ocr/qwen_image/train.parquet"
test -s "$HOME/data/ocr/qwen_image/test.parquet"
```

## Start the verl-omni example

On the **monitoring host**, follow [Server Installation](../../server_installation.md), then run:

```bash
rl-insight server install
rl-insight server start --detach
```

In the prepared **training environment**, from the **verl-omni root**, replace the server address and launch:

```bash
export RL_INSIGHT_SERVER_URL="http://<server-ip>:18080"
bash examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_v1.sh \
    data.train_files="$HOME/data/ocr/qwen_image/train.parquet" \
    data.val_files="$HOME/data/ocr/qwen_image/test.parquet" \
    trainer.logger='["console","rl_insight"]' \
    trainer.project_name=verl_omni \
    trainer.experiment_name=qwen_image_ocr_monitor \
    actor_rollout_ref.rollout.disable_log_stats=False \
    transfer_queue.metrics.enabled=True
```

The upstream script forwards overrides through `"$@"`. The V1 entrypoint enables TransferQueue itself; `transfer_queue.metrics.enabled` only exposes its metrics. Keep rollout statistics enabled: the integration rejects monitoring with disabled engine statistics.

For Ray job submission, propagate `RL_INSIGHT_SERVER_URL` to every worker using the job's runtime environment; see [Configuration](../../configuration.md). The framework manages `VERL_RL_INSIGHT_ENABLE`, so users do not need to set it manually.

## Verify verl-omni monitoring

1. Open Grafana, then **verl-omni** → `verl_omni_trainer_v1_with_vllm_omni_engine`.
2. Select project `verl_omni`, experiment `qwen_image_ocr_monitor`, and a recent time window.
3. Check step timing, reward statistics, rollout replicas, and TransferQueue metrics after training begins.
4. Use [Metric reference](metrics.md) to distinguish missing instrumentation from metrics that do not apply to the selected algorithm.

If trainer metrics are absent, check the logger and worker URL. If engine or queue metrics are absent, inspect Prometheus targets and their reachability from the monitoring host. If only traces are absent, check Tempo and rollout-worker runtime environments.

Remove `rl_insight` from the logger list and restart training to disable reporting. Stop the services with `rl-insight server stop` when no longer needed; data remains on disk.

## Related verl-omni guides

- [Unified configuration reference](../../configuration.md)
- [Upstream verl-omni integration guide](https://github.com/verl-project/verl-omni/blob/main/docs/start/rl_insight.md)
- [Hardware monitoring](../../hardware/index.md)
