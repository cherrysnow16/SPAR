"""Most-recent-subgoal beliefs for the GraphGPO backend (SPAR v2).

Per training batch, one LLM call:
  1. each trajectory proposes up to 5 subgoals with achievement steps (self prompt, JSON schema);
     steps refer to the cleaned action list and are mapped back to original step indices
  2. subgoal strings of a uid-group are merged by LSH (MinHash + exact Jaccard check + union-find)
  3. belief at state j = merged label of the trajectory's subgoal achieved most recently before j
     (action o achieved it -> visible from state o+1)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from SPAR_v2.belief_pipeline import run_llm_batch
from SPAR_v2.subgoal_lsh import cluster_subgoals
from SPAR_v2.subgoal_variants import SCHEMAS, self_prompt


def _parse_self(resp: str, n_actions: int):
    try:
        items = json.loads(resp)["subgoals"]
    except Exception:
        return None
    out = []
    for x in items:
        text, st = str(x.get("subgoal", "")).strip(), x.get("step")
        if text:
            out.append((text, st if isinstance(st, int) and 0 <= st < n_actions else None))
    return out


def annotate_order_recent(trajs: list[dict], tokenizer, actor_rollout_wg, config) -> list[list[frozenset]]:
    """trajs[i]: {"group", "task", "won", "actions" (cleaned), "orig_idx" (cleaned idx -> original step), "n_states"}
    -> per trajectory, per original state position: frozenset() or frozenset({merged subgoal label})"""
    beliefs = [[frozenset()] * t["n_states"] for t in trajs]
    if not trajs:
        return beliefs
    prompts = [self_prompt(t["task"], t["actions"], t["won"], "json_constrained") for t in trajs]
    responses = run_llm_batch(prompts, tokenizer, actor_rollout_wg, config,
                              max_tokens=int(config.algorithm.get("spar_belief_max_tokens", 2048)),
                              json_schema=SCHEMAS["self"])
    own = [_parse_self(r, len(t["actions"])) for r, t in zip(responses, trajs)]

    by_group = defaultdict(list)
    for i, t in enumerate(trajs):
        by_group[t["group"]].append(i)
    thr = float(config.algorithm.spar.get("lsh_threshold", 0.6))
    stats = Counter(parse_fail=sum(o is None for o in own))
    for idx in by_group.values():
        texts = [s for i in idx if own[i] for s, _ in own[i]]
        if not texts:
            continue
        canon = cluster_subgoals(texts, threshold=thr)
        stats["labels"] += len(set(canon.values()))
        for i in idx:
            events = sorted((trajs[i]["orig_idx"][st], canon[s]) for s, st in own[i] or [] if st is not None)
            cur, e, row = frozenset(), 0, []
            for j in range(trajs[i]["n_states"]):
                while e < len(events) and events[e][0] < j:
                    cur = frozenset({events[e][1]})
                    e += 1
                row.append(cur)
            beliefs[i] = row
            stats["trajs_with_subgoal"] += bool(events)
    stats["trajs"] = len(trajs)
    print(f"[SPAR v2 recent] {dict(stats)}")
    return beliefs
