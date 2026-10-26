import numpy as np


def to_hashable(x):
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def extract_unique_states(state_list):
    unique_state_list = []
    unique_state_hashes = set()

    for traj in state_list:
        for step in traj:
            state_arr = step['state']
            state_hash = to_hashable(state_arr)
            if state_hash not in unique_state_hashes:
                unique_state_list.append(state_arr)
                unique_state_hashes.add(state_hash)

    state_to_idx = {to_hashable(state): i for i, state in enumerate(unique_state_list)}
    idx_to_state = {i: state for i, state in enumerate(unique_state_list)}
    return unique_state_list, state_to_idx, idx_to_state


def build_trajectory(state_list, action_list, state_to_idx):
    trajectory = []
    for i in range(len(state_list)):
        trajectory.append([])
        for j in range(len(state_list[i]) - 1):
            src_state = state_list[i][j]['state']
            dst_state = state_list[i][j + 1]['state']
            src_idx = state_to_idx[to_hashable(src_state)]
            dst_idx = state_to_idx[to_hashable(dst_state)]
            action_label = action_list[i][j]
            src_reward = state_list[i][j]['reward']
            dst_reward = state_list[i][j + 1]['reward']
            trajectory[i].append((src_idx, action_label, dst_idx, src_reward, dst_reward))
    return trajectory


def unique_trajectory(trajectory):
    deduped = []
    for i in range(len(trajectory)):
        deduped.append([])
        for item in trajectory[i]:
            if item not in deduped[i]:
                deduped[i].append(item)
    return [item for sublist in deduped for item in sublist]
