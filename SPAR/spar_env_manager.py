import os
import re
from typing import List
from functools import partial

from omegaconf import OmegaConf

from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory


def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile


def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        self.memory.reset(batch_size=len(text_obs))
        self.tasks = []
        self.extract_task(text_obs)

        text_location_holding, text_item_location = self.get_location_holding(infos)
        new_text_obs = []
        for text_obs_, text_location_holding_, text_item_location_ in zip(
                text_obs, text_location_holding, text_item_location):
            new_text_obs.append(f"{text_obs_} {text_location_holding_} {text_item_location_}")

        anchor_out = []
        for text_obs_, text_location_holding_, text_item_location_, commands in zip(
                text_obs, text_location_holding, text_item_location,
                self.envs.get_admissible_commands):
            anchor_out.append(
                f"{text_obs_} {text_location_holding_} {text_item_location_} "
                f"{','.join(sorted(commands))}")

        self.pre_text_obs = new_text_obs
        full_text_obs = self.build_text_obs(
            new_text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': anchor_out}, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        text_location_holding, text_item_location = self.get_location_holding(infos)

        new_text_obs = []
        for text_obs_, text_location_holding_, text_item_location_ in zip(
                text_obs, text_location_holding, text_item_location):
            new_text_obs.append(f"{text_obs_} {text_location_holding_} {text_item_location_}")

        anchor_out = []
        for text_obs_, text_location_holding_, text_item_location_, commands in zip(
                text_obs, text_location_holding, text_item_location,
                self.envs.get_admissible_commands):
            anchor_out.append(
                f"{text_obs_} {text_location_holding_} {text_item_location_} "
                f"{','.join(sorted(commands))}")

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = new_text_obs

        full_text_obs = self.build_text_obs(new_text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': anchor_out}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def get_inventory(self, batch_size: int) -> List[str]:
        text_obs, _, _, _, _ = self.envs.step(["inventory"] * batch_size)
        return text_obs

    def get_location_holding(self, infos):
        text_location_holding = []
        text_item_location = []
        for info in infos:
            items_info = info['items_info']
            obs_location = info['obs_location']
            obs_holding = info['obs_holding']
            item_location = info['item_location']
            holding_state_holding = ''
            for k, v in items_info.items():
                if k in obs_holding:
                    for s, ok in v.items():
                        if ok:
                            holding_state_holding += f"{s} "
            if holding_state_holding != '':
                holding_state_holding = f"({holding_state_holding.strip()})"
            elif obs_holding != 'nothing':
                holding_state_holding = f"(unprocessed)"
            obs_item_location = ''

            for obj, location in item_location.items():
                holding_state = ""
                if obj in items_info:
                    obj_state_dict = items_info[obj]
                    for s, ok in obj_state_dict.items():
                        if ok:
                            holding_state += f"{s} "
                if holding_state != "":
                    holding_state = f"({holding_state.strip()})"
                elif holding_state == "":
                    holding_state = f"(unprocessed)"

                if location['old_location'] != location['new_location']:
                    obs_item_location += f"{obj}"
                    obs_item_location += (
                        f"{holding_state} from {location['old_location']} "
                        f"move to {location['new_location']};")
                elif (holding_state != "(unprocessed)") and (obs_holding != obj):
                    obs_item_location += f"{obj}"
                    obs_item_location += (
                        f"{holding_state} from {location['old_location']} "
                        f"move to {location['new_location']};")

            if obs_item_location != "":
                obs_item_location = f"History moving list by you: {obs_item_location}"
            text_item_location.append(obs_item_location)
            text_location_holding.append(
                f"Location: {obs_location}. Items in hand (status): "
                f"{obs_holding}{holding_state_holding}.")
        return text_location_holding, text_item_location

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]],
                       init: bool = False) -> List[str]:
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action")

        for i in range(len(text_obs)):
            reformatted_admissible_actions = "\n ".join(
                f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def state_preprocess(self, total_batch_list, tokenizer):
        if not hasattr(self, '_compiled_patterns'):
            self._compiled_patterns = {
                'cool': re.compile(r"You cool the (.*?) using"),
                'heat': re.compile(r"You heat the (.*?) using"),
                'clean': re.compile(r"You clean the (.*?) using"),
                'object': re.compile(r"((?:[a-zA-Z]+ ?)+ \d+)"),
                'a_prefix': re.compile(r"\ba ([a-zA-Z]+ ?\d+)")
            }

        patterns = self._compiled_patterns
        success_state = "You Won!"
        state_template = "{} {}"
        object_state_template = "{} ({}ed)"

        all_responses = []
        response_positions = []
        for i, traj in enumerate(total_batch_list):
            for j in range(1, len(traj)):
                if not traj[j]['active_masks']:
                    all_responses.append(traj[j - 1]['responses'])
                    response_positions.append((i, j - 1, True))
                    break
                all_responses.append(traj[j - 1]['responses'])
                response_positions.append((i, j - 1, False))

        if all_responses:
            decoded_responses = tokenizer.batch_decode(all_responses)
            actions, valids = self.projection_f(
                decoded_responses, self.envs.get_admissible_commands)
        else:
            decoded_responses = []
            actions = []
            valids = []

        response_map = {}
        for idx, (i, j, is_final) in enumerate(response_positions):
            response_map[(i, j, is_final)] = str(actions[idx]) if idx < len(actions) else ""

        raw_state_list = []
        raw_action_list = []

        for i, traj in enumerate(total_batch_list):
            object_state_dict = {
                "cool": set(),
                "heat": set(),
                "clean": set(),
            }

            initial_state = state_template.format(
                traj[0]['anchor_obs'], traj[0]['inventory_state'])

            state_seq = [{
                "state": initial_state,
                "reward": 0,
                "active_masks": traj[0]['active_masks']
            }]

            total_batch_list[i][0]['anchor_obs'] = initial_state
            action_seq = []

            for j in range(1, len(traj)):
                if not traj[j]['active_masks']:
                    state_seq.append({
                        "state": success_state,
                        "reward": traj[j - 1]['rewards'],
                        "active_masks": traj[j - 1]['active_masks']
                    })

                    total_batch_list[i][j]['anchor_obs'] = success_state
                    action_seq.append(response_map.get((i, j - 1, True), ""))
                    break

                anchor_obs = traj[j]['anchor_obs']
                current_obj = None
                current_action = None

                for action, pattern in [('cool', patterns['cool']),
                                        ('heat', patterns['heat']),
                                        ('clean', patterns['clean'])]:
                    match = pattern.search(anchor_obs)
                    if match:
                        current_obj = match.group(1)
                        current_action = action
                        break

                if current_obj and current_action:
                    for other_action in object_state_dict:
                        if other_action != current_action:
                            object_state_dict[other_action].discard(current_obj)
                    object_state_dict[current_action].add(current_obj)

                inventory_state = traj[j]['inventory_state']

                obj_state_map = {}
                for action_key, obj_set in object_state_dict.items():
                    for obj in obj_set:
                        obj_state_map[obj] = action_key

                if obj_state_map:
                    inventory_state_no_a = patterns['a_prefix'].sub(r"\1", inventory_state)
                    objects = patterns['object'].findall(inventory_state_no_a)

                    if objects:
                        replace_map = {}
                        for obj in objects:
                            if obj in obj_state_map:
                                state = obj_state_map[obj]
                                replace_map[obj] = object_state_template.format(obj, state)

                        if replace_map:
                            def obj_replace(match):
                                return replace_map.get(match.group(0), match.group(0))
                            inventory_state = patterns['object'].sub(
                                obj_replace, inventory_state_no_a)

                traj[j]['inventory_state'] = inventory_state
                current_state = state_template.format(traj[j]['anchor_obs'], inventory_state)

                state_seq.append({
                    "state": current_state,
                    "reward": traj[j - 1]['rewards'],
                    "active_masks": traj[j - 1]['active_masks']
                })
                total_batch_list[i][j]['anchor_obs'] = current_state
                action_seq.append(response_map.get((i, j - 1, False), ""))

            raw_state_list.append(state_seq)
            raw_action_list.append(action_seq)
        return raw_state_list, raw_action_list

    def clean_trajectory(self, trajectory, idx_to_state):
        cleaned_trajectory = []
        for i in range(len(trajectory)):
            cleaned_trajectory.append([])
            cur_from_state = None
            j = 0
            while j < len(trajectory[i]):
                from_state, action, to_state, from_reward, to_reward = trajectory[i][j]
                if (action in ["inventory", "look"] or "examine" in action
                        or "Nothing happens" in idx_to_state[to_state]
                        or from_state == to_state):
                    cur_from_state = from_state
                    while j < len(trajectory[i]):
                        from_state, action, to_state, from_reward, to_reward = trajectory[i][j]
                        if (action in ["inventory", "look"] or "examine" in action
                                or "Nothing happens" in idx_to_state[to_state]
                                or cur_from_state == to_state):
                            j += 1
                        else:
                            cleaned_trajectory[i].append(
                                (cur_from_state, action, to_state, from_reward, to_reward))
                            j += 1
                            break
                else:
                    cleaned_trajectory[i].append(
                        (from_state, action, to_state, from_reward, to_reward))
                    j += 1

        return cleaned_trajectory

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)

                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]

        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


def make_envs(config):
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "alfworld" not in config.env.env_name.lower():
        raise ValueError(f"Unsupported environment: {config.env.env_name}")

    from SPAR.alfworld_envs import build_alfworld_envs
    from agent_system.environments.env_package.alfworld import alfworld_projection
    import agent_system.environments.env_package.alfworld as _alf_pkg

    _alf_pkg_dir = os.path.dirname(os.path.abspath(_alf_pkg.__file__))
    if config.env.env_name in ('alfworld/AlfredThorEnv', 'alfworld/AlfredTWEnv'):
        alf_config_path = os.path.join(_alf_pkg_dir, 'configs/config_tw.yaml')
    else:
        raise ValueError(f"Unsupported environment: {config.env.env_name}")

    env_kwargs = {
        'eval_dataset': OmegaConf.select(
            config, 'env.alfworld.eval_dataset', default='eval_in_distribution'),
    }
    _envs = build_alfworld_envs(
        alf_config_path, config.env.seed, config.data.train_batch_size, group_n,
        is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
    _val_envs = build_alfworld_envs(
        alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1,
        is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

    projection_f = partial(alfworld_projection)
    envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
    val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
    return envs, val_envs
