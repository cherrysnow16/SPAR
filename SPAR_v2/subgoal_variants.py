"""Subgoal-annotation variants, run side by side on the same rollouts (diagnostics only).

structure:
  self   - each trajectory proposes its own subgoals and their achievement steps (current SPAR)
  common - stage 1: one subgoal list per uid-group from all its trajectories;
           stage 2: each trajectory only assigns steps to that list
  group  - one call per uid-group: shared subgoal list + step assignments for every trajectory
fmt: text | json | json_constrained (vLLM structured outputs with a JSON schema)
budget: max generated tokens per call

Nothing here feeds training; run_all_variants dumps prompts/responses/parses and timings
to $SPAR_SUBGOAL_DUMP_DIR for offline analysis.
"""
from __future__ import annotations

import json
import os
import re
import time

from SPAR_v2.belief_pipeline import (
    _format_trajectory_actions,
    format_subgoal_prompt,
    run_llm_batch,
)

VARIANTS = [
    (structure, fmt, budget)
    for structure in ("self", "common", "group")
    for fmt, budget in (("text", 512), ("text", 2048), ("json", 512), ("json", 2048), ("json_constrained", 2048))
]

SUBGOAL_RULES = """Subgoal rules:
- Each subgoal must be a short phrase (at most 5 words).
- Subgoals must be mutually exclusive and collectively exhaustive.
- Ignore low-level navigation actions (e.g. "go to", "look", "open", "inventory").
- Focus only on subgoals that directly advance the task goal.
- Output at most 5 subgoals.
- Order them in the order they must occur to complete the task.
- The final subgoal must correspond to task completion."""

STEP_RULES = """Step assignment rules:
- The step number must be the exact "Step N" number from the trajectory it refers to.
- A subgoal is achieved only when the action at that step directly and completely fulfills it — not a preparatory or partial action.
- If multiple steps could qualify, use the earliest one.
- If a subgoal was never achieved, use {none}."""

_INT_OR_NULL = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
SCHEMAS = {
    "self": {"type": "object", "properties": {
        "reasoning": {"type": "string"},
        "subgoals": {"type": "array", "maxItems": 5, "items": {"type": "object", "properties": {
            "subgoal": {"type": "string"}, "step": _INT_OR_NULL}, "required": ["subgoal", "step"]}}},
        "required": ["reasoning", "subgoals"]},
    "common1": {"type": "object", "properties": {
        "reasoning": {"type": "string"},
        "subgoals": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string"}}},
        "required": ["reasoning", "subgoals"]},
    "common2": {"type": "object", "properties": {
        "reasoning": {"type": "string"},
        "steps": {"type": "array", "items": _INT_OR_NULL}},
        "required": ["reasoning", "steps"]},
    "group": {"type": "object", "properties": {
        "reasoning": {"type": "string"},
        "subgoals": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string"}},
        "attempts": {"type": "array", "items": {"type": "array", "items": _INT_OR_NULL}}},
        "required": ["reasoning", "subgoals", "attempts"]},
}

_JSON_EXAMPLES = {
    "self": '{"subgoals": [{"subgoal": "heat the mug", "step": 7}, {"subgoal": "put mug in cabinet", "step": null}]}',
    "common1": '{"subgoals": ["heat the mug", "put mug in cabinet"]}',
    "common2": '{"steps": [7, null]}',
    "group": '{"subgoals": ["heat the mug", "put mug in cabinet"], "attempts": [[7, 11], [4, null]]}',
}


def _output_section(kind: str, fmt: str, text_section: str) -> str:
    if fmt == "text":
        return text_section
    example = _JSON_EXAMPLES[kind]
    if fmt == "json":
        return ("First, reason step-by-step within <think> </think> tags.\n"
                "Then output ONLY a JSON object inside <json> </json> tags, shaped like this example:\n"
                f"<json>{example}</json>\nUse null for a subgoal that was never achieved.")
    # json_constrained: decoding is restricted to the schema, reasoning goes in a field
    with_reasoning = example.replace("{", '{"reasoning": "...", ', 1)
    return ("Respond with a single JSON object. Put your step-by-step reasoning in the \"reasoning\" field first, "
            f"then the answer fields, shaped like this example:\n{with_reasoning}\n"
            "Use null for a subgoal that was never achieved.")


def _attempts_block(trajs_actions, wons) -> str:
    blocks = []
    for n, (actions, won) in enumerate(zip(trajs_actions, wons), start=1):
        body = _format_trajectory_actions(actions) if actions else "(no effective actions)"
        blocks.append(f"Attempt {n} ({'SUCCEEDED' if won else 'FAILED'}):\n{body}")
    return "\n\n".join(blocks)


def self_prompt(task, actions, won, fmt):
    if fmt == "text":
        return format_subgoal_prompt(task, actions, won=won)  # current SPAR default prompt
    outcome = "SUCCESSFULLY completed" if won else "FAILED to complete"
    return (f"You are analyzing a trajectory of a goal-directed task.\n\nYour task is: {task}\n\n"
            f"Below is the full trajectory:\n{_format_trajectory_actions(actions)}\n\nThis trajectory {outcome} the task.\n\n"
            "Your job is to:\n(1) propose a minimal ordered list of high-level subgoals that decompose the task, AND\n"
            "(2) identify the step at which each proposed subgoal is first achieved in THIS trajectory.\n\n"
            f"{SUBGOAL_RULES}\n\n{STEP_RULES.format(none='null')}\n\n" + _output_section("self", fmt, ""))


def common1_prompt(task, trajs_actions, wons, fmt):
    text_out = ("First, reason step-by-step within <think> </think> tags.\n"
                "Then write the subgoals inside <subgoals> </subgoals> tags, one numbered subgoal per line, e.g.\n"
                "<subgoals>\n1. heat the mug\n2. put mug in cabinet\n</subgoals>")
    return (f"You are analyzing several attempts at the same goal-directed task.\n\nYour task is: {task}\n\n"
            f"Below are the trajectories of all attempts at this task:\n{_attempts_block(trajs_actions, wons)}\n\n"
            "Your job is to propose ONE minimal ordered list of high-level subgoals that decompose the task. "
            "The same list will be used to analyze every attempt, so it must describe the task itself, not any single attempt.\n\n"
            f"{SUBGOAL_RULES}\n\n" + _output_section("common1", fmt, text_out))


def common2_prompt(task, subgoals, actions, won, fmt):
    outcome = "SUCCESSFULLY completed" if won else "FAILED to complete"
    sg = "\n".join(f"{n}. {s}" for n, s in enumerate(subgoals, start=1))
    text_out = ("First, reason step-by-step within <think> </think> tags.\n"
                "Then write inside <assignments> </assignments> tags ONE line with the step numbers in the order of the "
                "subgoals above, separated by commas, using none for a subgoal that was never achieved, e.g.\n"
                "<assignments>7, none</assignments>")
    return (f"You are analyzing a trajectory of a goal-directed task.\n\nYour task is: {task}\n\n"
            f"The task is decomposed into the following ordered subgoals:\n{sg}\n\n"
            f"Below is the full trajectory:\n{_format_trajectory_actions(actions)}\n\nThis trajectory {outcome} the task.\n\n"
            "Your job is to identify the step at which each subgoal above is first achieved in THIS trajectory.\n\n"
            f"{STEP_RULES.format(none='none' if fmt == 'text' else 'null')}\n\n" + _output_section("common2", fmt, text_out))


def group_prompt(task, trajs_actions, wons, fmt):
    text_out = ("First, reason step-by-step within <think> </think> tags.\n"
                "Then write the subgoals inside <subgoals> </subgoals> tags, one numbered subgoal per line.\n"
                "Then write inside <assignments> </assignments> tags ONE line per attempt, in attempt order, with that "
                "attempt's step numbers in the order of the subgoals, separated by commas, using none for a subgoal it never "
                "achieved, e.g.\n<subgoals>\n1. heat the mug\n2. put mug in cabinet\n</subgoals>\n"
                "<assignments>\nAttempt 1: 7, 11\nAttempt 2: 4, none\n</assignments>")
    return (f"You are analyzing several attempts at the same goal-directed task.\n\nYour task is: {task}\n\n"
            f"Below are the trajectories of all attempts at this task:\n{_attempts_block(trajs_actions, wons)}\n\n"
            "Your job is to:\n(1) propose ONE minimal ordered list of high-level subgoals that decompose the task "
            "(the same list for every attempt), AND\n(2) for EACH attempt, identify the step at which each subgoal is "
            "first achieved in that attempt.\n\n"
            f"{SUBGOAL_RULES}\n\n{STEP_RULES.format(none='none' if fmt == 'text' else 'null')}\n\n"
            + _output_section("group", fmt, text_out))


# ---------------------------------------------------------------- parsing

def _load_json(resp: str, fmt: str):
    if fmt == "json":
        m = re.search(r"<json>(.*?)</json>", resp, re.S | re.I)
        cand = m.group(1) if m else resp[resp.rfind("{"):] if "{" in resp else ""
    else:
        cand = resp
    try:
        return json.loads(cand.strip())
    except Exception:
        # fall back to the outermost {...}
        a, b = resp.find("{"), resp.rfind("}")
        try:
            return json.loads(resp[a:b + 1]) if a != -1 and b > a else None
        except Exception:
            return None


def _step(v):
    if v is None:
        return None
    if isinstance(v, int):
        return v
    v = str(v).strip().lower()
    return int(v) if v.isdigit() else None


def parse_self(resp, fmt):
    """-> list[(subgoal_text, step|None)] or None"""
    if fmt == "text":
        m = re.search(r"<assignments>(.*?)</assignments>", resp, re.S | re.I)
        if not m:
            return None
        out = []
        for line in m.group(1).splitlines():
            line = line.strip()
            ms = re.search(r"STEP\s*:?\s*(\d+|none)", line, re.I)
            if not line or not ms:
                continue
            out.append((line.split("|")[0].strip(), _step(ms.group(1))))
        return out or None
    d = _load_json(resp, fmt)
    try:
        return [(str(x["subgoal"]), _step(x.get("step"))) for x in d["subgoals"]] or None
    except Exception:
        return None


def parse_common1(resp, fmt):
    if fmt == "text":
        m = re.search(r"<subgoals>(.*?)</subgoals>", resp, re.S | re.I)
        if not m:
            return None
        sg = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", l).strip() for l in m.group(1).splitlines()]
        return [s for s in sg if s][:5] or None
    d = _load_json(resp, fmt)
    try:
        return [str(s) for s in d["subgoals"]][:5] or None
    except Exception:
        return None


def _parse_step_list(text):
    return [_step(t) if t.strip().lower() != "none" else None for t in text.split(",") if t.strip()]


def parse_common2(resp, fmt):
    if fmt == "text":
        m = re.search(r"<assignments>(.*?)</assignments>", resp, re.S | re.I)
        return _parse_step_list(m.group(1).strip().splitlines()[0]) if m and m.group(1).strip() else None
    d = _load_json(resp, fmt)
    try:
        return [_step(v) for v in d["steps"]]
    except Exception:
        return None


def parse_group(resp, fmt):
    """-> (subgoals, list[list[step|None]] per attempt) or None"""
    if fmt == "text":
        sg = parse_common1(resp, "text")
        m = re.search(r"<assignments>(.*?)</assignments>", resp, re.S | re.I)
        if not sg or not m:
            return None
        rows = {}
        for line in m.group(1).splitlines():
            ma = re.match(r"\s*Attempt\s*(\d+)\s*[:|]\s*(.*)$", line, re.I)
            if ma:
                rows[int(ma.group(1)) - 1] = _parse_step_list(ma.group(2))
        if not rows:
            return None
        return sg, [rows.get(k) for k in range(max(rows) + 1)]
    d = _load_json(resp, fmt)
    try:
        return [str(s) for s in d["subgoals"]][:5], [[_step(v) for v in row] for row in d["attempts"]]
    except Exception:
        return None


# ---------------------------------------------------------------- runner

def _call(prompts, tokenizer, wg, config, fmt, budget, schema_key):
    t0 = time.time()
    resps = run_llm_batch(prompts, tokenizer, wg, config, max_tokens=budget,
                          json_schema=SCHEMAS[schema_key] if fmt == "json_constrained" else None)
    return resps, time.time() - t0


def run_variant(structure, fmt, budget, groups, tokenizer, wg, config):
    name = f"{structure}/{fmt}/{budget}"
    recs, timing = [], {}
    base = {"variant": name, "structure": structure, "fmt": fmt, "budget": budget}
    try:
        if structure == "self":
            idx = [(gi, i) for gi, g in enumerate(groups) for i in range(len(g["raw_action_list"]))]
            prompts = [self_prompt(groups[gi]["task_description"], groups[gi]["raw_action_list"][i],
                                   groups[gi]["episode_rewards"][i] > 0, fmt) for gi, i in idx]
            resps, timing["call"] = _call(prompts, tokenizer, wg, config, fmt, budget, "self")
            for (gi, i), p, r in zip(idx, prompts, resps):
                recs.append({**base, "stage": "self", "group": gi, "traj": i,
                             "won": bool(groups[gi]["episode_rewards"][i] > 0),
                             "actions": groups[gi]["raw_action_list"][i], "prompt": p, "response": r,
                             "parsed": parse_self(r, fmt)})
        elif structure == "common":
            p1 = [common1_prompt(g["task_description"], g["raw_action_list"],
                                 [x > 0 for x in g["episode_rewards"]], fmt) for g in groups]
            r1, timing["stage1"] = _call(p1, tokenizer, wg, config, fmt, budget, "common1")
            lists = [parse_common1(r, fmt) for r in r1]
            for gi, (p, r) in enumerate(zip(p1, r1)):
                recs.append({**base, "stage": "common1", "group": gi, "prompt": p, "response": r, "parsed": lists[gi]})
            idx = [(gi, i) for gi, g in enumerate(groups) if lists[gi] for i in range(len(g["raw_action_list"]))]
            p2 = [common2_prompt(groups[gi]["task_description"], lists[gi], groups[gi]["raw_action_list"][i],
                                 groups[gi]["episode_rewards"][i] > 0, fmt) for gi, i in idx]
            r2, timing["stage2"] = _call(p2, tokenizer, wg, config, fmt, budget, "common2") if p2 else ([], 0.0)
            for (gi, i), p, r in zip(idx, p2, r2):
                recs.append({**base, "stage": "common2", "group": gi, "traj": i,
                             "won": bool(groups[gi]["episode_rewards"][i] > 0), "subgoals": lists[gi],
                             "actions": groups[gi]["raw_action_list"][i], "prompt": p, "response": r,
                             "parsed": parse_common2(r, fmt)})
        else:  # group
            prompts = [group_prompt(g["task_description"], g["raw_action_list"],
                                    [x > 0 for x in g["episode_rewards"]], fmt) for g in groups]
            resps, timing["call"] = _call(prompts, tokenizer, wg, config, fmt, budget, "group")
            for gi, (p, r) in enumerate(zip(prompts, resps)):
                g = groups[gi]
                recs.append({**base, "stage": "group", "group": gi,
                             "wons": [bool(x > 0) for x in g["episode_rewards"]],
                             "actions_list": g["raw_action_list"], "prompt": p, "response": r,
                             "parsed": parse_group(r, fmt)})
    except Exception as e:  # keep going with the other variants
        print(f"[SPAR VARIANT] {name} failed: {e!r}")
        timing["error"] = repr(e)
    timing["total"] = sum(v for k, v in timing.items() if k in ("call", "stage1", "stage2"))
    print(f"[SPAR VARIANT] {name} time={timing['total']:.2f}s records={len(recs)}")
    return recs, {**base, **timing}


def run_all_variants(groups, tokenizer, wg, config):
    d = os.environ.get("SPAR_SUBGOAL_DUMP_DIR")
    if d:
        os.makedirs(d, exist_ok=True)
    for structure, fmt, budget in VARIANTS:
        recs, timing = run_variant(structure, fmt, budget, groups, tokenizer, wg, config)
        if d:
            with open(os.path.join(d, "variant_records.jsonl"), "a") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            with open(os.path.join(d, "variant_timing.jsonl"), "a") as f:
                f.write(json.dumps(timing) + "\n")
