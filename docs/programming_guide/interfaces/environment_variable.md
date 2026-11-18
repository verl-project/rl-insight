# Environment Variable

This page lists environment variables in RL-Insight's public runtime interface. Test-only and CI-only variables are intentionally omitted.

| Variable | Scope | Purpose |
|---|---|---|
| `RL_INSIGHT_SERVER_URL` | Training client | RL-Insight server API URL, for example `http://<server-ip>:18080`. |
| `RL_INSIGHT_SERVER_BACKEND` | Training client | Trainer-side monitor client backend (`server.backend`), for example `ray`. Also applies to worker processes that call `rl_insight.init()` without a config. |
| `RL_INSIGHT_CONFIG` | Training client | Absolute path to a trainer-side YAML config file. |

```bash
export RL_INSIGHT_SERVER_URL=http://<server-ip>:18080
export RL_INSIGHT_SERVER_BACKEND=ray
export RL_INSIGHT_CONFIG=/absolute/path/to/client.yaml
```

For precedence and examples, see [Configuration](configuration.md).
