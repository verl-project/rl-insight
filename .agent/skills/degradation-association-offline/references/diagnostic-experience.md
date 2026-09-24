# Diagnostic experience

These diagnostic priors help interpret observed association evidence. They do
not assert direct observation of signals absent from that evidence. Do not use
this reference to provide operational procedures, commands, configuration
changes, parameter values, or steps for reproducing faults.

The evidence categories (`latency`, `training_quality`, `rollout_quality`,
`data_characteristics`, `hardware_resources`, `vllm_engine`, and
`transfer_queue`) organize metrics; they are not fault domains. Never select a
domain by category size alone. `association_percent` is temporal association
strength, not fault probability or causal contribution. Time-trend metrics such
as `training_epoch` and `tq_controller_uptime_seconds` require corroboration.

Use only the final `association_percent`, metric name, category, rank, target,
and event phase when applying these priors. Do not inspect raw samples, calculate
new statistics, compare before/after values, or infer whether a metric increased
or decreased. Ignore candidate direction, point state (`NORMAL`, `UP`, `DOWN`,
or `BETWEEN_MODES`), matched mode, and candidate abnormality when diagnosing;
every returned Top-K item is association evidence. Several semantically
consistent high-scoring metrics are useful joint evidence; a category containing
many weakly scored metrics is not high-confidence evidence.

## Diagnostic vocabulary

The cause names in this section are a closed vocabulary. Use their exact names
and do not invent aliases, component-level symptoms, or category restatements.

### Workload/data condition

- `Sequence-length anomaly`: include this among the likely causes when
  prompt-length, response-length, global-sequence-length, or total-token metrics
  occupy a substantial part of the returned Top-K. Rank it as cause 1 when they
  also form the leading high-scoring evidence group. This rule applies regardless
  of candidate direction or point state. It is a workload/data explanation, not
  an infrastructure fault domain. Do not claim that lengths increased,
  decreased, or changed by a particular amount; association alone does not
  establish direction or magnitude.

Strong sequence-length association weakens an infrastructure-only explanation.
One isolated length metric is insufficient. A substantial low-scoring group
supports only low-confidence inclusion, not cause 1.

### Compute

- `NPU frequency throttling`: reduced active compute capacity consistent with a
  lower operating frequency. Prefer it for a relatively broad, sustained
  slowdown without evidence of a disappeared device or rank. Idle low frequency
  is normal and frequency must not be claimed unless directly observed.
- `NPU core offline`: the experiment label for abrupt, severe, persistent loss
  of effective NPU compute capacity. It may represent localized capacity being
  unavailable; do not translate this label into a claim of physical core
  disappearance, ECC/RAS failure, or device removal without a direct signal.
- `AI Core overload`: contention concentrated in Cube/matrix compute. Direct
  AIC/Cube utilization or profiler evidence distinguishes it best; otherwise a
  compute-sensitive target with strong MFU/throughput association and weak
  length/sequence association is only indirect support.
- `AI Vector overload`: contention concentrated in Vector compute. Direct
  AIV/Vector utilization or profiler evidence distinguishes it best. The
  current catalog alone may not reliably separate it from AI Core overload.

Evidence that weakens Compute: high-scoring length/sequence metrics, or stronger
vLLM/transfer-queue association without comparably strong compute-related
metrics.

### Network

- `Parameter-plane NIC bandwidth limitation`: sustained communication-capacity
  limitation associated with an endpoint network path.
- `Parameter-plane switch-port egress bandwidth limitation`: sustained
  communication-capacity limitation associated with an egress path.
- `Parameter-plane network congestion`: sustained or bursty contention within
  the communication path.
- `Transient parameter-plane link interruption`: abrupt communication stall,
  often followed by recovery or a closed event, consistent with a temporary
  link interruption.

Prefer Network only when communication-sensitive timing evidence supports it
while length/sequence metrics do not. TransferQueue association alone is
propagation evidence, not proof of a parameter-plane fault. With the current
catalog, the first three sustained restrictions are usually observationally
equivalent; rank them as possible causes without pretending to locate the
restriction.

Evidence that weakens Network: association scores are concentrated in rollout,
vLLM, or length/sequence metrics and communication-related metrics rank weakly.
Do not claim observed packet loss, rate limitation, or link down without direct
counters or logs.

### Host CPU

- `Host CPU core offline`: reduced scheduler-visible CPU capacity consistent
  with one or more host cores becoming unavailable.
- `Host CPU overload`: runnable work saturates available host CPU capacity and
  delays input, orchestration, serialization, or request handling.
- `Host CPU frequency throttling`: broad host-side work slows because effective
  CPU frequency is reduced.

Prefer Host CPU when CPU resource and host-sensitive latency/queueing metrics
receive strong association scores while NPU-oriented evidence ranks lower.
Direct CPU online-state, utilization/pressure, and frequency counters are needed
to distinguish the three causes reliably. The current catalog does not justify
claiming that a CPU was actually offlined, saturated, or frequency-limited.

Evidence that weakens Host CPU: stronger NPU, network, or sequence-length
association without comparably strong host-sensitive metrics.

### HBM

- `HBM congestion`: contention for NPU HBM capacity or memory bandwidth that
  causes compute and rollout work to wait. This label does not imply continuous
  saturation or defective memory.

Prefer HBM when NPU memory resource, latency, and throughput metrics all receive
strong association scores while length/sequence metrics do not. Allocated or
reserved memory alone does not measure bandwidth congestion. Direct HBM
bandwidth/pressure evidence is required for a high-confidence subtype claim.

Evidence that weakens HBM: memory metrics rank weakly while communication or
length/sequence metrics dominate the association scores.

### TransferQueue evidence

TransferQueue is an inter-stage sample data gateway, not a fault domain or an
allowed cause name. Never output `TransferQueue delay`, `TransferQueue latency`,
or similar symptom restatements as a likely cause.

When multiple labeled series of `tq_storage_request_latency_p50` and/or
`tq_storage_request_latency_p99` occupy a substantial part of Top-K, use the
experiment prior: make `Compute` the primary fault domain and include both `AI
Core overload` and `NPU frequency throttling` among the likely causes. When this
family is the leading high-scoring evidence, rank them as the highest remaining
causes after the sequence-length priority rule. This is a fallible diagnostic
prior, not proof that TransferQueue itself caused the event.

## Synthesis rules

1. Match the exact final target event before diagnosing. Offline output reports
   `closed` or `open_at_range_end`; never generate a separate confirmed-phase
   diagnosis.
2. Read all metrics in the grouped Top-25. Use metric meaning, category, global
   rank, and final association score only. Do not use raw values, direction,
   point state, candidate abnormality, labels, component scores, or independently
   calculated trends.
3. If length/sequence metrics occupy a substantial part of the Top-25, include
   `Sequence-length anomaly` among the causes. If they form the leading
   high-scoring group, make it cause 1 before considering infrastructure
   alternatives. This semantic length-family rule is the explicit exception to
   the general category-size rule. Treat training/rollout quality as downstream
   evidence unless its metric semantics directly support a cause.
4. Rank exactly two distinct fault domains and three to five distinct causes
   from the diagnostic vocabulary; cause 1 is primary. `Low` confidence is
   allowed, but every item needs an evidence or uncertainty basis and must not
   contradict observed evidence. If the selected phase has no association
   entries, report insufficient root-cause evidence instead of fabricating this
   ranking.
5. Keep the reasoning concise and professional. Do not invent profiler, network,
   frequency, CPU, HBM, ECC/RAS, or device-health observations.

## Technical sources

- [Ascend AI Core architecture](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0015.html)
  describes Cube, Vector, and scalar compute units.
- [Ascend ArithmeticUtilization fields](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0093.html)
  distinguish AI Cube Core and AI Vector Core execution evidence.
- [Ascend PyTorch Profiler MemoryAccess](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0033.html)
  documents memory-access and bandwidth analysis.
- [HCCL alpha-beta model](https://www.hiascend.com/document/detail/en/canncommercial/850/commlib/hcclug/hcclug_000115.html)
  relates communication time to latency and per-byte transfer cost.
- [Ascend TransferQueue](https://github.com/Ascend/TransferQueue) describes its
  control plane, storage data plane, and post-training producer-consumer role.
- [Linux CPU hotplug](https://docs.kernel.org/core-api/cpu_hotplug.html),
  [CPUFreq](https://docs.kernel.org/admin-guide/pm/cpufreq.html), and
  [Pressure Stall Information](https://docs.kernel.org/accounting/psi.html)
  define direct host CPU availability, frequency, and contention signals.
