# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 11b. Google Cluster Traces Attempt / Episode Structure Characterization
#
# **Purpose.** Notebook 11 showed that the per-instance feature matrix leaks the
# label, mostly through lifecycle-total resubmission history, and that the
# per-instance terminal-outcome labelling produces a 78:1 imbalance that does
# not match the event-level ~3.39:1 rate. Both point at the same fix: model at
# the grain of a scheduled attempt (episode) rather than the whole instance.
#
# Before rebuilding the reconstruction, this notebook reads the real attempt
# structure off the trace so the segmentation rules are evidence-based rather
# than assumed (the "let the data shape the code" principle). It answers three
# questions:
#
# 1. How are repeated attempts delimited in the event stream (SUBMIT vs SCHEDULE
#    repetition)? Section 1 dumps full event sequences for multi-attempt
#    instances.
# 2. How common are multi-attempt instances, and how do schedule / fail / evict
#    counts distribute? Section 2 aggregates the per-instance lifecycle summary.
# 3. Under a SCHEDULE-delimited episode definition, are episodes well-formed
#    (one terminal each), and what is the per-episode FAIL_LOST imbalance?
#    Section 3 reconstructs episodes on a key-hashed sample and labels them per
#    the V01 definition (FAIL/LOST positive, FINISH negative, EVICT/KILL
#    excluded).
#
# **Inputs.** Cached `instance_events_labeled` (sentinel-filtered, label-
# augmented events) and `instance_lifecycle_summary` (per-instance summary).
#
# **Output.** `{OUTPUT_DIR}/tables/google_attempt_structure.csv` on Drive (the
# Section 2-3 summary numbers; persists beyond the runtime).
#
# Cross-references (`outputs/tables/eda_decisions.csv`): V01 (failure label),
# V08/V27 (EVICT/KILL exclusion), V09 (rapid-onset post-schedule crash), V10
# (resubmission dominance).

# %% [markdown]
# ---
# ## 0. Colab + BigQuery setup

# %%
# !pip install -q polars google-cloud-bigquery db-dtypes pyarrow

# %%
import os
import sys
from pathlib import Path

from google.colab import userdata

GITHUB_PAT = userdata.get('GITHUB_PAT')
REPO_OWNER = 'NU-Academics'
REPO_NAME = 'Dissertation-LebelN'
REPO_DIR = f'/content/{REPO_NAME}'
REPO_URL = f'https://{GITHUB_PAT}@github.com/{REPO_OWNER}/{REPO_NAME}.git'

if os.path.exists(REPO_DIR):
    # !cd {REPO_DIR} && git pull --quiet
    print(f"Repo already present at {REPO_DIR}; pulled latest.")
else:
    # !git clone --quiet {REPO_URL} {REPO_DIR}
    print(f"Cloned repo to {REPO_DIR}.")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# %%
from google.colab import auth

auth.authenticate_user()

# %%
import polars as pl

from utils.colab_setup import setup_drive, OUTPUT_DIR
from utils.bq_client import get_client, table_ref, DATASET, PROJECT_ID
from src.data.schemas import (
    EVENT_SUBMIT,
    EVENT_SCHEDULE,
    EVENT_EVICT,
    EVENT_FAIL,
    EVENT_FINISH,
    EVENT_KILL,
    EVENT_LOST,
)

# Mount Drive so outputs persist beyond the runtime; the cloned repo is
# recreated each session and is lost on termination.
setup_drive()
bq_client = get_client()

# Persistent Drive output store (mirrors notebooks 08 and 10), not the
# ephemeral cloned repo.
TABLES_DIR = OUTPUT_DIR / 'tables'
TABLES_DIR.mkdir(parents=True, exist_ok=True)
ATTEMPT_STRUCTURE_CSV = TABLES_DIR / 'google_attempt_structure.csv'

# Readable names for the event-type codes (for printing sequences).
TYPE_NAME = {
    EVENT_SUBMIT: "SUB", 1: "QUE", 2: "ENA", EVENT_SCHEDULE: "SCH",
    EVENT_EVICT: "EVI", EVENT_FAIL: "FAIL", EVENT_FINISH: "FIN",
    EVENT_KILL: "KILL", EVENT_LOST: "LOST", 9: "UPD_P", 10: "UPD_R",
}
TERMINAL_TYPES = (EVENT_EVICT, EVENT_FAIL, EVENT_FINISH, EVENT_KILL, EVENT_LOST)


def fqn(table: str) -> str:
    return table_ref(table)


def run_query(sql: str) -> pl.DataFrame:
    """Execute SQL and return a Polars DataFrame (small results only)."""
    job = bq_client.query(sql)
    df = pl.from_pandas(job.to_dataframe())
    print(f"  [BQ] bytes processed: {(job.total_bytes_processed or 0):,}")
    return df


print(f"Project: {PROJECT_ID}  Dataset: {DATASET}")

# %% [markdown]
# ---
# ## 1. Event sequences for multi-attempt instances
#
# Pull a handful of instances that were submitted more than once and scheduled
# at least once, then print their full event sequences. This shows directly
# whether a resubmission appears as a new SUBMIT, a repeated SCHEDULE, or both,
# which is the segmentation question. Keys are fetched first so the event scan
# is pruned by the table's clustering on `(collection_id, instance_index)`.

# %%
keys_df = run_query(f"""
SELECT collection_id, instance_index, submit_count, schedule_count,
       evict_count, fail_lost_count, outcome
FROM {fqn('instance_lifecycle_summary')}
WHERE submit_count > 1 AND schedule_count >= 1
  AND outcome IN ('FAIL_LOST', 'FINISH')
LIMIT 8
""")
print(keys_df)

# %%
# Build an explicit key predicate so BigQuery prunes via clustering.
pairs = [
    (int(r["collection_id"]), int(r["instance_index"]))
    for r in keys_df.to_dicts()
]
predicate = " OR ".join(
    f"(collection_id = {cid} AND instance_index = {iid})" for cid, iid in pairs
)

seq_df = run_query(f"""
SELECT collection_id, instance_index,
       STRING_AGG(CAST(type AS STRING), ',' ORDER BY time) AS type_seq,
       COUNT(*) AS n_events
FROM {fqn('instance_events_labeled')}
WHERE {predicate}
GROUP BY collection_id, instance_index
""")

# Render the numeric type sequences as readable names.
for row in seq_df.to_dicts():
    names = [TYPE_NAME.get(int(t), t) for t in row["type_seq"].split(",")]
    print(f"  ({row['collection_id']}, {row['instance_index']})  "
          f"{row['n_events']} events:  " + " -> ".join(names))

# %% [markdown]
# **What to look for.** A pattern like `SUB -> SCH -> FAIL -> SUB -> SCH -> FIN`
# confirms each scheduled run is a separable episode delimited by a SCHEDULE,
# with the resubmission marked by a fresh SUBMIT. If instead failures recur
# without an intervening SCHEDULE, or schedules repeat without a SUBMIT, the
# episode boundary needs adjusting before the rebuild.

# %% [markdown]
# ---
# ## 2. Population attempt-structure distribution
#
# Aggregate the per-instance lifecycle summary (cheap) to size how common
# multi-attempt instances are and how schedules / fails / evicts distribute.

# %%
structure_df = run_query(f"""
SELECT
    COUNT(*) AS n_instances,
    COUNTIF(submit_count > 1) AS n_multi_submit,
    COUNTIF(schedule_count > 1) AS n_multi_schedule,
    COUNTIF(schedule_count = 0) AS n_never_scheduled,
    COUNTIF(fail_lost_count > 0) AS n_any_fail_lost,
    COUNTIF(evict_count > 0) AS n_any_evict,
    SUM(submit_count) AS total_submits,
    SUM(schedule_count) AS total_schedules,
    SUM(fail_lost_count) AS total_fail_lost_events,
    SUM(evict_count) AS total_evict_events,
    COUNTIF(outcome = 'FINISH') AS n_terminal_finish,
    COUNTIF(outcome = 'FAIL_LOST') AS n_terminal_fail_lost,
    AVG(schedule_count) AS mean_schedules_per_instance,
    APPROX_QUANTILES(schedule_count, 100)[OFFSET(50)] AS median_schedules,
    APPROX_QUANTILES(schedule_count, 100)[OFFSET(99)] AS p99_schedules
FROM {fqn('instance_lifecycle_summary')}
""")
print(structure_df.transpose(include_header=True, header_name="metric", column_names=["value"]))

s = structure_df.to_dicts()[0]
n_inst = int(s["n_instances"])
print(f"\nMulti-submit instances:   {int(s['n_multi_submit']):,} "
      f"({100 * s['n_multi_submit'] / n_inst:.1f}%)")
print(f"Multi-schedule instances: {int(s['n_multi_schedule']):,} "
      f"({100 * s['n_multi_schedule'] / n_inst:.1f}%)")
print(f"Total scheduled runs:     {int(s['total_schedules']):,} "
      f"vs {n_inst:,} instances "
      f"({s['total_schedules'] / n_inst:.2f} schedules/instance)")

# Cheap per-episode imbalance proxy from the summary aggregates: every FAIL/LOST
# event is a failed scheduled run, and each finishing instance contributes one
# FINISH run. EVICT/KILL runs are excluded per V08/V27. This includes any
# pre-schedule failures, so Section 3 refines it to scheduled-only.
episodes_pos_proxy = int(s["total_fail_lost_events"])
episodes_neg_proxy = int(s["n_terminal_finish"])
ratio_proxy = episodes_neg_proxy / episodes_pos_proxy if episodes_pos_proxy else float("nan")
print(f"\nPer-episode imbalance proxy (FINISH : FAIL_LOST): {ratio_proxy:.2f}:1")
print(f"  vs per-instance terminal imbalance reported in notebook 11 (~78:1)")

# %% [markdown]
# ---
# ## 3. Episode reconstruction, reconciliation, and per-episode label (sampled)
#
# Define an episode by the running count of SCHEDULE events within an instance:
# every event carries `sched_seq`, the number of schedules seen up to and
# including it, and events with `sched_seq >= 1` belong to a scheduled run. An
# episode's terminal is the first terminal-type event in its `sched_seq` group.
#
# The sample is now drawn by hashing the **instance key** `(collection_id,
# instance_index)`, not `collection_id`. Collections vary in size by orders of
# magnitude, so a collection-level hash let a few giant collections dominate and
# skewed the previous run (the scaled FINISH-episode count came out far above
# the known FINISH-event total). Hashing per instance samples instances
# uniformly while keeping every event of a sampled instance together (the hash
# is constant within an instance), so the per-episode ratios are representative.
# Widen `SAMPLE_DENOM` to 1 for the full population.
#
# The sample is materialized once (a single scan of the source) and the three
# read-outs below run cheaply against it: (a) event-level reconciliation that
# splits FAIL_LOST events into pre-schedule vs post-schedule and counts FINISH
# events, (b) instance-level terminal counts, and (c) the per-episode label
# split. Together these reconcile the event-level, per-instance, and
# per-episode views of the imbalance.

# %%
SAMPLE_DENOM = 200  # ~1/200 of instances; set to 1 for the full trace.
SAMPLE_TABLE = 'attempt_sample_events'  # scratch table; safe to drop afterwards.

materialize_sql = f"""
CREATE OR REPLACE TABLE {fqn(SAMPLE_TABLE)}
CLUSTER BY collection_id, instance_index AS
SELECT
    collection_id, instance_index, time, type,
    COUNTIF(type = {EVENT_SCHEDULE}) OVER (
        PARTITION BY collection_id, instance_index
        ORDER BY time
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS sched_seq
FROM {fqn('instance_events_labeled')}
WHERE MOD(ABS(FARM_FINGERPRINT(
    CONCAT(CAST(collection_id AS STRING), '_', CAST(instance_index AS STRING))
)), {SAMPLE_DENOM}) = 0
"""
print(f"Materializing instance-key sample (1/{SAMPLE_DENOM}) with sched_seq ...")
job = bq_client.query(materialize_sql)
job.result()
print(f"  Sample table: {PROJECT_ID}.{DATASET}.{SAMPLE_TABLE}  "
      f"(bytes processed: {(job.total_bytes_processed or 0):,})")

# %% [markdown]
# ### 3.1 Event-level reconciliation (where do the FAIL_LOST events fall?)

# %%
recon_df = run_query(f"""
SELECT
    COUNT(*) AS n_events,
    COUNTIF(type IN ({EVENT_FAIL}, {EVENT_LOST})) AS fail_lost_events,
    COUNTIF(type IN ({EVENT_FAIL}, {EVENT_LOST}) AND sched_seq = 0) AS fail_lost_pre_schedule,
    COUNTIF(type IN ({EVENT_FAIL}, {EVENT_LOST}) AND sched_seq >= 1) AS fail_lost_post_schedule,
    COUNTIF(type = {EVENT_FINISH}) AS finish_events,
    COUNTIF(type = {EVENT_EVICT}) AS evict_events,
    COUNTIF(type = {EVENT_SCHEDULE}) AS schedule_events
FROM {fqn(SAMPLE_TABLE)}
""")
r = recon_df.to_dicts()[0]
fl_events = int(r["fail_lost_events"])
fl_pre = int(r["fail_lost_pre_schedule"])
fl_post = int(r["fail_lost_post_schedule"])
print(f"FAIL_LOST events: {fl_events:,}  "
      f"pre-schedule={fl_pre:,} ({100 * fl_pre / fl_events:.1f}%)  "
      f"post-schedule={fl_post:,} ({100 * fl_post / fl_events:.1f}%)")
print(f"FINISH events: {int(r['finish_events']):,}  "
      f"SCHEDULE events: {int(r['schedule_events']):,}")

# %% [markdown]
# ### 3.2 Instance-level terminal counts (on the same sample)

# %%
inst_df = run_query(f"""
WITH inst_terminal AS (
    SELECT collection_id, instance_index,
        ARRAY_AGG(type ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS terminal_type
    FROM {fqn(SAMPLE_TABLE)}
    GROUP BY collection_id, instance_index
)
SELECT
    COUNT(*) AS n_instances,
    COUNTIF(terminal_type = {EVENT_FINISH}) AS instances_finish,
    COUNTIF(terminal_type IN ({EVENT_FAIL}, {EVENT_LOST})) AS instances_fail_lost,
    COUNTIF(terminal_type = {EVENT_EVICT}) AS instances_evict,
    COUNTIF(terminal_type = {EVENT_KILL}) AS instances_kill
FROM inst_terminal
""")
i = inst_df.to_dicts()[0]
print(f"Instances: {int(i['n_instances']):,}  "
      f"terminal FINISH={int(i['instances_finish']):,}  "
      f"terminal FAIL_LOST={int(i['instances_fail_lost']):,}")

# %% [markdown]
# ### 3.3 Per-episode label split

# %%
episode_df = run_query(f"""
WITH episodes AS (
    SELECT
        collection_id, instance_index, sched_seq,
        COUNTIF(type IN ({EVENT_EVICT}, {EVENT_FAIL}, {EVENT_FINISH}, {EVENT_KILL}, {EVENT_LOST}))
            AS n_terminal,
        ARRAY_AGG(
            IF(type IN ({EVENT_EVICT}, {EVENT_FAIL}, {EVENT_FINISH}, {EVENT_KILL}, {EVENT_LOST}), type, NULL)
            IGNORE NULLS ORDER BY time LIMIT 1
        )[SAFE_OFFSET(0)] AS terminal_type
    FROM {fqn(SAMPLE_TABLE)}
    WHERE sched_seq >= 1
    GROUP BY collection_id, instance_index, sched_seq
)
SELECT
    COUNT(*) AS n_scheduled_episodes,
    COUNTIF(n_terminal = 1) AS n_single_terminal,
    COUNTIF(n_terminal = 0) AS n_no_terminal,
    COUNTIF(n_terminal > 1) AS n_multi_terminal,
    COUNTIF(terminal_type IN ({EVENT_FAIL}, {EVENT_LOST})) AS episodes_fail_lost,
    COUNTIF(terminal_type = {EVENT_FINISH}) AS episodes_finish,
    COUNTIF(terminal_type = {EVENT_EVICT}) AS episodes_evict,
    COUNTIF(terminal_type = {EVENT_KILL}) AS episodes_kill,
    COUNTIF(terminal_type IS NULL) AS episodes_open_no_terminal
FROM episodes
""")
print(episode_df)

# %%
e = episode_df.to_dicts()[0]
n_ep = int(e["n_scheduled_episodes"])
well_formed = int(e["n_single_terminal"]) / n_ep if n_ep else float("nan")
pos = int(e["episodes_fail_lost"])
neg = int(e["episodes_finish"])
ep_ratio = neg / pos if pos else float("nan")
modeled = pos + neg  # EVICT/KILL/open excluded from the primary label
pos_frac = pos / modeled if modeled else float("nan")

print(f"Scheduled episodes (sample 1/{SAMPLE_DENOM}): {n_ep:,}")
print(f"Well-formed (exactly one terminal):     {well_formed:.4f}")
print(f"  multi-terminal: {int(e['n_multi_terminal']):,}  "
      f"no-terminal/open: {int(e['n_no_terminal']):,}")
print(f"Per-episode label split (V01): FAIL_LOST={pos:,}  FINISH={neg:,}  "
      f"EVICT(excl)={int(e['episodes_evict']):,}  KILL(excl)={int(e['episodes_kill']):,}")
print(f"Scheduled-episode imbalance (FINISH : FAIL_LOST): {ep_ratio:.2f}:1 "
      f"(positive fraction {pos_frac:.4f})")

# %% [markdown]
# ### 3.4 Reconciliation across the three views
#
# These should now line up: post-schedule FAIL_LOST events feed the positive
# episodes, FINISH events should roughly equal FINISH episodes and finishing
# instances (one FINISH per finishing instance), and the per-episode imbalance
# is the honest modeling target.

# %%
print("Reconciliation (same instance-key sample):")
print(f"  FAIL_LOST  -> events post-schedule={fl_post:,}  episodes_fail_lost={pos:,}")
print(f"  FINISH     -> events={int(r['finish_events']):,}  "
      f"episodes={neg:,}  finishing_instances={int(i['instances_finish']):,}")
print(f"  Positive episodes are {100 * fl_post / fl_events:.1f}% of all FAIL_LOST events "
      f"(the rest fail before ever scheduling).")
fail_post_frac = fl_post / fl_events if fl_events else float("nan")

# %% [markdown]
# ### 3.5 Why does FINISH double? (multiple FINISH per finishing instance)
#
# FINISH events run ~2.1x the finishing-instance count, while failures do not
# double. Inspect the per-instance FINISH-count distribution and a few real
# sequences (rendered as `type@sched_seq`) to decide whether each scheduled
# FINISH run is a legitimate separate negative or a redundant record to dedupe.

# %%
finish_dist_df = run_query(f"""
WITH per_inst AS (
    SELECT collection_id, instance_index, COUNTIF(type = {EVENT_FINISH}) AS n_finish
    FROM {fqn(SAMPLE_TABLE)}
    GROUP BY collection_id, instance_index
    HAVING n_finish >= 1
)
SELECT
    COUNT(*) AS finishing_instances,
    COUNTIF(n_finish = 1) AS finish_x1,
    COUNTIF(n_finish = 2) AS finish_x2,
    COUNTIF(n_finish >= 3) AS finish_x3plus,
    AVG(n_finish) AS mean_finish,
    MAX(n_finish) AS max_finish
FROM per_inst
""")
fd = finish_dist_df.to_dicts()[0]
fin_inst = int(fd["finishing_instances"])
print(f"Finishing instances: {fin_inst:,}")
print(f"  exactly 1 FINISH: {int(fd['finish_x1']):,} ({100 * fd['finish_x1'] / fin_inst:.1f}%)")
print(f"  exactly 2 FINISH: {int(fd['finish_x2']):,} ({100 * fd['finish_x2'] / fin_inst:.1f}%)")
print(f"  3+ FINISH:        {int(fd['finish_x3plus']):,} ({100 * fd['finish_x3plus'] / fin_inst:.1f}%)")
print(f"  mean FINISH/finishing instance: {fd['mean_finish']:.2f}  max: {int(fd['max_finish'])}")

# %%
# Real sequences for the most-FINISH instances. type@sched_seq shows whether the
# repeated FINISHes sit in different scheduled runs (separate negatives) or the
# same sched_seq group (a redundant record to collapse).
multi_finish_seq_df = run_query(f"""
WITH mf AS (
    SELECT collection_id, instance_index, COUNTIF(type = {EVENT_FINISH}) AS n_finish
    FROM {fqn(SAMPLE_TABLE)}
    GROUP BY collection_id, instance_index
    HAVING n_finish > 1
    ORDER BY n_finish DESC
    LIMIT 8
)
SELECT
    s.collection_id, s.instance_index, mf.n_finish,
    STRING_AGG(CONCAT(CAST(s.type AS STRING), '@', CAST(s.sched_seq AS STRING)), ' ' ORDER BY s.time) AS seq
FROM {fqn(SAMPLE_TABLE)} s
JOIN mf USING (collection_id, instance_index)
GROUP BY s.collection_id, s.instance_index, mf.n_finish
ORDER BY mf.n_finish DESC
""")
for row in multi_finish_seq_df.to_dicts():
    toks = []
    for tok in row["seq"].split(" "):
        t, sq = tok.split("@")
        toks.append(f"{TYPE_NAME.get(int(t), t)}@{sq}")
    print(f"  ({row['collection_id']}, {row['instance_index']})  n_finish={row['n_finish']}:  "
          + " ".join(toks))

# %% [markdown]
# ---
# ## 4. Summary and read-out

# %%
summary_rows = [
    {"metric": "n_instances", "value": float(n_inst)},
    {"metric": "pct_multi_submit", "value": 100 * s["n_multi_submit"] / n_inst},
    {"metric": "pct_multi_schedule", "value": 100 * s["n_multi_schedule"] / n_inst},
    {"metric": "schedules_per_instance", "value": s["total_schedules"] / n_inst},
    {"metric": "per_episode_imbalance_proxy", "value": ratio_proxy},
    {"metric": "scheduled_episode_imbalance_sampled", "value": ep_ratio},
    {"metric": "scheduled_episode_pos_fraction_sampled", "value": pos_frac},
    {"metric": "fail_lost_post_schedule_fraction", "value": fail_post_frac},
    {"metric": "mean_finish_per_finishing_instance", "value": fd["mean_finish"]},
    {"metric": "episode_well_formed_fraction", "value": well_formed},
    {"metric": "episode_multi_terminal_fraction",
     "value": int(e["n_multi_terminal"]) / n_ep if n_ep else float("nan")},
    {"metric": "sample_denominator", "value": float(SAMPLE_DENOM)},
]
summary_df = pl.DataFrame(summary_rows)
summary_df.write_csv(str(ATTEMPT_STRUCTURE_CSV))
print(f"Summary written: {ATTEMPT_STRUCTURE_CSV}")
print(summary_df)

# %% [markdown]
# **Read-out for the rebuild.**
# - A high well-formed fraction confirms the SCHEDULE-delimited episode is the
#   right unit and the reconstruction can proceed on that basis.
# - The scheduled-episode imbalance is the honest modeling target. It will not
#   fall to the event-level ~3.39:1, because most failures land where the
#   reconciliation shows: a large share of FAIL_LOST events occur pre-schedule
#   and are excluded, and once scheduled most runs finish. Expect it to stay
#   highly imbalanced (the grain change fixes the leakage and gives more
#   positive examples than the per-instance label, but imbalance is intrinsic).
# - The reconciliation (Section 3.4) should line up: FINISH events, FINISH
#   episodes, and finishing instances within ~1x of each other. A large gap
#   means the segmentation still mis-counts and must be fixed before the rebuild.
# - A non-trivial `no-terminal/open` or `multi-terminal` count flags edge cases
#   (runs cut off at the trace boundary, or multiple terminals before the next
#   schedule) that the reconstruction will need an explicit rule for.
