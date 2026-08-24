#!/usr/bin/env python3
"""Seed macbook-air runtime evidence into AssistX projection."""
import time
from neo4j import GraphDatabase

NOW = int(time.time() * 1000)
EXP = NOW + 7 * 24 * 3600 * 1000
ALIAS = "qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled"

drv = GraphDatabase.driver("bolt://100.64.43.123:7687", auth=("neo4j", "knowledge_graph_2026"))
with drv.session(database="assistx") as s:
    s.run("""MERGE (r:RuntimeInstance {runtime_instance_id:'lmstudio-macbook-air'})
      SET r.node_id='scotts-macbook-air', r.runtime_kind='lmstudio',
          r.runtime_version='0.3.x', r.headless=true, r.admitted=true,
          r.status='ready', r.updated_at_ts=$now, r.expires_at_ts=$exp""",
          now=NOW, exp=EXP)
    s.run("""MERGE (m:LoadedModelInstance {model_instance_id:'qwen35-08b-claude-macbook'})
      SET m.model_key=$alias, m.provider_model=$alias, m.admitted=true,
          m.context_length=262144,
          m.capabilities_json='["chat","streaming","local_only"]',
          m.quantization='mlx_4bit',
          m.artifact_fingerprint='sha256:qwen35-08b-claude-macbook-mlx',
          m.updated_at_ts=$now, m.expires_at_ts=$exp
      WITH m MATCH (r:RuntimeInstance {runtime_instance_id:'lmstudio-macbook-air'})
      MERGE (r)-[:SERVES]->(m)""", alias=ALIAS, now=NOW, exp=EXP)
    s.run("""MERGE (p:AccessPath {runtime_instance_id:'lmstudio-macbook-air',
              base_url:'http://192.168.1.233:1234/v1'})
      SET p.transport='http', p.approved=true, p.preference=100,
          p.observed_at_ts=$now, p.expires_at_ts=$exp,
          p.approved_by='operator:scott', p.approval_id='macbook-16slot-rebench-20260824'""",
          now=NOW, exp=EXP)
    s.run("""MERGE (c:CapacityObservation {runtime_instance_id:'lmstudio-macbook-air'})
      SET c.parallel_slots=16, c.queue_limit=64, c.queue_timeout_seconds=600,
          c.approved=true, c.observed_at_ts=$now, c.expires_at_ts=$exp,
          c.approved_by='operator:scott', c.approval_id='macbook-16slot-rebench-20260824'""",
          now=NOW, exp=EXP)
    n = s.run("MATCH (n:RuntimeInstance) RETURN count(n) AS c").single()["c"]
    print(f"seeded; RuntimeInstance total: {n}")
drv.close()
EOF_MARKER_NOT_USED = None
