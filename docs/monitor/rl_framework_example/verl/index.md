# Monitor a verl Training Run

This example adds RL-Insight to Qwen3-8B PPO training with FSDP and vLLM on GSM8K and MATH. Instructions are maintained here; `examples/monitor/verl/` links to this guide.

## Prepare verl and its data

Follow the [verl installation guide](https://verl.readthedocs.io/en/latest/start/install.html) to prepare a Linux GPU training environment. RL-Insight requires Python >= 3.10; verl and the inference engine may require a newer Python.

The command below follows the [upstream PPO script at revision 65ce76f](https://github.com/verl-project/verl/blob/65ce76f95d34da31f4812fcd8ae2c2617712fe81/examples/ppo_trainer/run_qwen3_8b_fsdp.sh). Its defaults use one node with eight GPUs, rollout TP=2, and Qwen3-8B actor and critic models. Memory requirements depend on the upstream batch and sequence settings. This is a real training workload, not a CPU demo.

The script and monitoring arguments were checked against source, not executed on training hardware. Use the referenced revision for the documented arguments, or review changes when using a newer checkout.

To prepare a separate checkout at that revision, run the following before following the installation guide in that checkout:

```bash
git clone https://github.com/verl-project/verl.git verl-monitor-example
cd verl-monitor-example
git checkout --detach 65ce76f95d34da31f4812fcd8ae2c2617712fe81
```

The remaining commands use Bash in the Linux training environment. The monitoring server can run on a separate supported Linux or Windows host.

Install this RL-Insight checkout into both training and monitoring-server environments:

```bash
python -m pip install -e /path/to/rl-insight
```

Activate the fully prepared verl environment. From the **verl repository root**, prepare both datasets:

```bash
mkdir -p "$HOME/data/gsm8k" "$HOME/data/math"
python3 examples/data_preprocess/gsm8k.py --local_save_dir "$HOME/data/gsm8k"
python3 examples/data_preprocess/math_dataset.py --local_save_dir "$HOME/data/math"
```

Both directories should contain `train.parquet` and `test.parquet`. Ensure training nodes can download `Qwen/Qwen3-8B`, or set `MODEL_PATH` and `CRITIC_MODEL_PATH` to accessible local weights.

Before launching, verify the active training environment and prepared data:

```bash
python3 -c "import ray, rl_insight, verl, vllm; print('Training and monitoring imports OK')"
test -s "$HOME/data/gsm8k/train.parquet" && test -s "$HOME/data/gsm8k/test.parquet"
test -s "$HOME/data/math/train.parquet" && test -s "$HOME/data/math/test.parquet"
```

## Start the verl example

On the **monitoring host**, follow [Server Installation](../../server_installation.md), then run:

```bash
rl-insight server install
rl-insight server start --detach
```

In the **training shell**, replace the server address with the URL printed at startup. From the **verl root**, launch the upstream recipe with monitoring overrides:

```bash
export RL_INSIGHT_SERVER_URL="http://<server-ip>:18080"
export VERL_USE_UV=0
export INFER_BACKEND=vllm
bash examples/ppo_trainer/run_qwen3_8b_fsdp.sh \
    trainer.logger='["console","rl_insight"]' \
    trainer.project_name=verl \
    trainer.experiment_name=qwen3_8b_ppo_monitor \
    actor_rollout_ref.rollout.disable_log_stats=False
```

`VERL_USE_UV=0` selects your active, already provisioned Python environment, where you installed RL-Insight. The script forwards overrides through `"$@"`; its model, optimizer, and resource settings remain defined by the upstream recipe. To use SGLang, install compatible dependencies and set `INFER_BACKEND=sglang` before launching.

For a TransferQueue-based trainer, also set `transfer_queue.metrics.enabled=True`. This example does not switch trainer modes or enable TransferQueue. For multi-node jobs, propagate the server URL through the Ray runtime environment; see [Configuration](../../configuration.md).

## Verify verl monitoring

1. Open the printed Grafana URL, then the **verl** folder.
2. Select `verl_trainer_v1_with_vllm_engine`, or the SGLang dashboard for that backend.
3. Select project `verl`, experiment `qwen3_8b_ppo_monitor`, and a time window covering the run.
4. After completed steps and successful scrapes, check training metrics and step timing. Inspect engine metrics while generation is active.

Panels requiring V1-specific instrumentation or TransferQueue can remain empty for other trainer modes. See [Metric reference](metrics.md) for sources, units, and interpretation.

If all panels are empty, check worker environment variables and Prometheus targets first. If only engine metrics are missing, check `disable_log_stats=False` and endpoint reachability. For missing traces, check Tempo and worker instrumentation; metrics and traces use different paths.

Remove `rl_insight` from `trainer.logger` and restart the job to disable framework reporting. `rl-insight server stop` stops the service stack and preserves collected data.

## Related verl guides

- [Unified configuration reference](../../configuration.md)
- [Upstream verl integration guide](https://github.com/verl-project/verl/blob/main/docs/advance/rl_insight.md)
- [Hardware monitoring](../../hardware/index.md)
