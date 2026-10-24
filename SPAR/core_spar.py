from __future__ import annotations

import numpy as np
import torch

from verl import DataProto

from SPAR.advantage import episode_norm_reward, build_step_group, step_norm_reward
from SPAR.graph_utils import (
    to_hashable,
    extract_unique_states,
    build_trajectory,
    unique_trajectory,
)
from SPAR.propagation import propagate_reward_decay
from SPAR.belief_pipeline import (
    run_subgoal_pipeline,
    reset_llm_timing,
    get_llm_timing,
)


def to_hashable_belief(state, belief_set) -> tuple:
    if isinstance(belief_set, int):
        belief_set = frozenset()
    return (to_hashable(state), frozenset(belief_set))


def extract_unique_belief_states(state_list_with_beliefs: list[list[dict]]):
    belief_state_to_idx = {}
    idx_to_belief_state = {}
    counter = 0
    for traj in state_list_with_beliefs:
        for step in traj:
            key = to_hashable_belief(step["state"], step["belief"])
            if key not in belief_state_to_idx:
                belief_state_to_idx[key] = counter
                idx_to_belief_state[counter] = (step["state"], step["belief"])
                counter += 1
    return belief_state_to_idx, idx_to_belief_state


def build_belief_trajectory(
    state_list_with_beliefs: list[list[dict]],
    action_list: list[list[str]],
    belief_state_to_idx: dict,
) -> list[list[tuple]]:
    trajectory = []
    for i in range(len(state_list_with_beliefs)):
        traj = []
        state_seq = state_list_with_beliefs[i]
        for j in range(len(state_seq) - 1):
            src = state_seq[j]
            dst = state_seq[j + 1]
            src_key = to_hashable_belief(src["state"], src["belief"])
            dst_key = to_hashable_belief(dst["state"], dst["belief"])
            src_idx = belief_state_to_idx[src_key]
            dst_idx = belief_state_to_idx[dst_key]
            action_label = action_list[i][j] if j < len(action_list[i]) else ""
            traj.append((src_idx, action_label, dst_idx, src["reward"], dst["reward"]))
        trajectory.append(traj)
    return trajectory


def _compute_uid_groups(uid_list: list) -> list[tuple[int, int]]:
    if not uid_list:
        return []
    groups = []
    prev = uid_list[0]
    start = 0
    for idx in range(1, len(uid_list)):
        if uid_list[idx] != prev:
            groups.append((start, idx))
            start = idx
            prev = uid_list[idx]
    groups.append((start, len(uid_list)))
    return groups


def _get_task_description(envs, start_idx: int) -> str:
    try:
        tasks = envs.tasks
        return tasks[start_idx % len(tasks)]
    except (AttributeError, IndexError, TypeError):
        return "complete the task"


def _remap_beliefs(beliefs_list, selected_state_list, orig_indices_list):
    out = []
    for local_i, filtered_beliefs in enumerate(beliefs_list):
        orig_indices = orig_indices_list[local_i]
        num_states_orig = len(selected_state_list[local_i])
        remapped = []
        for j in range(num_states_orig):
            p = sum(1 for oi in orig_indices if oi < j)
            b = filtered_beliefs[p] if p < len(filtered_beliefs) else (
                filtered_beliefs[-1] if filtered_beliefs else frozenset())
            remapped.append(b)
        out.append(remapped)
    return out


def apply_spar(total_batch_list, config, tokenizer, envs, actor_rollout_wg):
    return apply_spar_propagation(total_batch_list, config, tokenizer, envs, actor_rollout_wg)


def apply_spar_propagation(
    total_batch_list: list,
    config,
    tokenizer,
    envs,
    actor_rollout_wg,
) -> list:
    reset_llm_timing()

    raw_state_list, raw_action_list = envs.state_preprocess(total_batch_list, tokenizer)

    uid_list = [total_batch_list[i][0]["uid"] for i in range(len(total_batch_list))]
    first_last_indices = _compute_uid_groups(uid_list)

    value_dict_list = []
    belief_state_to_idx_list = []
    all_beliefs: list[list[frozenset] | None] = [None] * len(total_batch_list)

    group_ctx = []
    pipeline_inputs = []
    for (start_idx, end_idx) in first_last_indices:
        selected_state_list = raw_state_list[start_idx:end_idx]
        selected_action_list = raw_action_list[start_idx:end_idx]

        episode_rewards = [sum(step["reward"] for step in raw_state_list[i])
                           for i in range(start_idx, end_idx)]

        task_description = _get_task_description(envs, start_idx)

        n_success = sum(1 for r in episode_rewards if r > 0)
        print(f"[SPAR] uid-group [{start_idx}:{end_idx}]  "
              f"trajs={end_idx - start_idx}  success={n_success}  "
              f"task: {task_description}")

        _, vanilla_state_to_idx, vanilla_idx_to_state = extract_unique_states(selected_state_list)
        vanilla_traj = build_trajectory(selected_state_list, selected_action_list, vanilla_state_to_idx)
        cleaned_vanilla = envs.clean_trajectory(vanilla_traj, vanilla_idx_to_state)

        filtered_action_list = []
        orig_indices_list = []
        for local_i, ctraj in enumerate(cleaned_vanilla):
            raw_actions = selected_action_list[local_i]
            f_actions, f_orig = [], []
            ptr = 0
            for (_, action, _, _, _) in ctraj:
                while ptr < len(raw_actions) and raw_actions[ptr] != action:
                    ptr += 1
                if ptr < len(raw_actions):
                    f_actions.append(action)
                    f_orig.append(ptr)
                    ptr += 1
            filtered_action_list.append(f_actions)
            orig_indices_list.append(f_orig)

        group_ctx.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "selected_state_list": selected_state_list,
            "selected_action_list": selected_action_list,
            "orig_indices_list": orig_indices_list,
            "remapped_beliefs_list": None,
        })
        pipeline_inputs.append({
            "task_description": task_description,
            "raw_state_list": selected_state_list,
            "raw_action_list": filtered_action_list,
            "episode_rewards": episode_rewards,
            "uid": uid_list[start_idx],
            "start_idx": start_idx,
        })

    if pipeline_inputs:
        beliefs_all = run_subgoal_pipeline(
            pipeline_inputs, tokenizer, actor_rollout_wg, config)
        for pos, ctx in enumerate(group_ctx):
            ctx["remapped_beliefs_list"] = _remap_beliefs(
                beliefs_all[pos], ctx["selected_state_list"], ctx["orig_indices_list"])

    for ctx in group_ctx:
        start_idx = ctx["start_idx"]
        selected_state_list = ctx["selected_state_list"]
        selected_action_list = ctx["selected_action_list"]
        remapped_beliefs_list = ctx["remapped_beliefs_list"]

        for local_i, beliefs in enumerate(remapped_beliefs_list):
            all_beliefs[start_idx + local_i] = beliefs

        state_list_with_beliefs = []
        for local_i, state_seq in enumerate(selected_state_list):
            beliefs = remapped_beliefs_list[local_i]
            augmented = []
            for j, step in enumerate(state_seq):
                b = beliefs[j] if j < len(beliefs) else beliefs[-1]
                augmented.append({
                    "state": step["state"],
                    "reward": step["reward"],
                    "belief": b,
                })
            state_list_with_beliefs.append(augmented)

        belief_state_to_idx, idx_to_belief_state = extract_unique_belief_states(state_list_with_beliefs)
        trajectory = build_belief_trajectory(
            state_list_with_beliefs, selected_action_list, belief_state_to_idx)
        idx_to_state_str = {idx: str(state) for idx, (state, _) in idx_to_belief_state.items()}
        cleaned_trajectory = envs.clean_trajectory(trajectory, idx_to_state_str)
        flat_unique_traj = unique_trajectory(cleaned_trajectory)

        G, value_dict = propagate_reward_decay(
            flat_unique_traj,
            gamma=config.algorithm.gamma,
            max_iter=1000,
        )

        for idx in range(len(idx_to_belief_state)):
            if idx not in value_dict:
                value_dict[idx] = 0.0

        value_dict_list.append(value_dict)
        belief_state_to_idx_list.append(belief_state_to_idx)

    traj_to_group = {}
    for group_idx, (start_idx, end_idx) in enumerate(first_last_indices):
        for i in range(start_idx, end_idx):
            traj_to_group[i] = group_idx

    new_reward_abs_list = []

    for i in range(len(total_batch_list)):
        batch_idx = traj_to_group[i]
        beliefs = all_beliefs[i] if all_beliefs[i] is not None else [frozenset()] * len(total_batch_list[i])

        initial_obs = total_batch_list[i][0]["anchor_obs"]
        initial_belief = beliefs[0] if beliefs else frozenset()
        initial_key = to_hashable_belief(initial_obs, initial_belief)

        state_idx = belief_state_to_idx_list[batch_idx].get(initial_key)
        prev_reward = value_dict_list[batch_idx].get(state_idx, 0.0) if state_idx is not None else 0.0

        new_reward_abs = []

        for j in range(len(total_batch_list[i]) - 1):
            if not total_batch_list[i][j]["active_masks"]:
                break

            next_state_pos = j + 1
            cur_belief = beliefs[next_state_pos] if next_state_pos < len(beliefs) else beliefs[-1]
            cur_obs = total_batch_list[i][j + 1]["anchor_obs"]
            cur_key = to_hashable_belief(cur_obs, cur_belief)

            state_idx = belief_state_to_idx_list[batch_idx].get(cur_key)
            if state_idx is not None:
                cur_reward = value_dict_list[batch_idx].get(state_idx, prev_reward)
            else:
                cur_reward = prev_reward

            new_reward_abs.append(cur_reward)
            prev_reward = cur_reward

        new_reward_abs_list.append(new_reward_abs)

    for i in range(len(total_batch_list)):
        beliefs = all_beliefs[i] if all_beliefs[i] is not None else [frozenset()] * len(total_batch_list[i])
        for j in range(len(total_batch_list[i])):
            belief = beliefs[j] if j < len(beliefs) else (beliefs[-1] if beliefs else frozenset())
            total_batch_list[i][j]["belief_state"] = belief
            if j < len(new_reward_abs_list[i]):
                total_batch_list[i][j]["step_rewards"] = new_reward_abs_list[i][j]
                total_batch_list[i][j]["step_rewards_abs"] = new_reward_abs_list[i][j]
            else:
                total_batch_list[i][j]["step_rewards"] = total_batch_list[i][j]["rewards"]
                total_batch_list[i][j]["step_rewards_abs"] = total_batch_list[i][j]["rewards"]

    _t = get_llm_timing()
    print(f"[SPAR TIMING] subgoal-pipeline LLM total: {_t['time_s']:.2f}s over "
          f"{_t['calls']} generate_sequences calls ({_t['prompts']} prompts) | "
          f"n_uid_groups={len(first_last_indices)}")
    print(f"[SPAR] length of total_batch_list: {len(total_batch_list)}")
    return total_batch_list


def compute_step_returns(batch: DataProto) -> torch.Tensor:
    rewards = batch.non_tensor_batch["step_rewards"].astype(np.float32)
    return torch.tensor(rewards, dtype=torch.float32, device=batch.batch["input_ids"].device)


def compute_spar_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    episode_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
):
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    step_group_uids = build_step_group(anchor_obs, index)
    step_advantages = step_norm_reward(
        step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    if episode_advantage_w != 0.0:
        episode_advantages = episode_norm_reward(
            token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)
        scores = step_advantage_w * step_advantages + episode_advantage_w * episode_advantages
    else:
        scores = step_advantage_w * step_advantages

    return scores, scores
