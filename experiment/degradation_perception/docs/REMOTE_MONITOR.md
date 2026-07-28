# Remote Monitor

The remote monitor incrementally reads UTF-8 JSON Lines through SSH from either
a remote file or Docker logs, then calls the degradation detector in-process.
It is a programmatic adapter, not a separate CLI command.

## Configure

Copy
[`monitor_config.example.yaml`](../monitor_config.example.yaml) to an
untracked local path and fill in the connection details:

```yaml
host: monitor.example.com
port: 22
username: rl-insight
key_filename: /path/to/read-only-key
connect_timeout: 10
source:
  type: file
  remote_path: /var/log/training/metrics.jsonl
offset_path: ./remote-monitor.offset.json
metrics:
  - timing_s/step
task_id: example-task
start_time: null
end_time: null
config_dir: ./config
```

For Docker logs, replace `source` with:

```yaml
source:
  type: docker
  container: verl-trainer
  tail_lines: 1000
```

Authentication may use `key_filename` or the environment variable named by
`password_env`. `known_hosts` can point to an additional local host-key file.
Do not place a password value directly in the YAML and do not commit the real
configuration.

## Remote Data

Each application line must be valid UTF-8 JSON:

```json
{"phase":"standard","timestamp":1,"metrics":{"timing_s/step":1.0}}
{"phase":"inference","timestamp":10,"metrics":{"timing_s/step":1.5}}
```

Docker output is read with Docker timestamps enabled; the payload after the
Docker timestamp must have the same JSON shape.

## Run One Cycle

```python
from experiment.degradation_perception.remote_monitor import run_remote_monitor

response = run_remote_monitor("path/to/monitor_config.yaml")
print(response)
```

For continuous polling, construct one `RemoteMonitor` and call `run()` on the
same instance. Reusing the instance preserves in-process history and the
standard threshold cache:

```python
from experiment.degradation_perception.remote_monitor import (
    RemoteMonitor,
    load_remote_monitor_config,
)

monitor = RemoteMonitor(load_remote_monitor_config("monitor_config.yaml"))
response = monitor.run()
```

The scheduling loop and polling interval belong to the caller.

## Standard Data and Cache Lifecycle

The first successful detection for a metric must include enough standard data.
After a standard model is validated, the same monitor instance can process
later inference-only batches.

The cache is rejected if the metric configuration changes. It is not written
to disk, so a service restart requires standard data again.

History confirmation is also process-local and isolated by `(task_id, metric)`.

## Offset Behavior

For a remote file, the offset store records the byte position and a short
file-head fingerprint. Truncation or rotation resets the cursor safely.
Incomplete trailing lines are retained for a later read.

For Docker, the cursor records the most recent Docker timestamp and hashes of
payloads at that timestamp to prevent duplicates.

The offset advances only after parsing and detection complete successfully.
Failed reads and failed detections do not skip data.

## Failure Response

Remote failures return state `2` and no degradation interval:

```json
{
  "states": {
    "timing_s/step": 2
  },
  "abnormalTimeRange": {
    "timing_s/step": []
  },
  "sourceStatus": "error",
  "sourceError": {
    "code": "REMOTE_CONNECTION_FAILED",
    "message": "..."
  }
}
```

A transport failure is operational failure, not evidence of performance
degradation.

## Safety and Testing

- system host keys are loaded and unknown host keys are rejected;
- SSH, SFTP, file, and command-stream resources are closed in cleanup paths;
- remote reads have size and timeout limits;
- Docker container names are strictly validated before command construction;
- offset files are updated atomically;
- tests use mocked transports rather than live hosts.

Run the remote adapter tests with:

```powershell
python -m pytest -q experiment/degradation_perception/tests/test_remote_monitor.py
```
