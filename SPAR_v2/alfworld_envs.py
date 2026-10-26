import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray
import re

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:

    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)
        self.env.seed(seed)
        self.history_items={}
        self.location="middle of a room"
        self.holding="nothing"
        self.item_location={}
        self.history_list=[]
        self.done=False

    def _update_item_location(self, old_holding):
        obj=old_holding
        if obj=='nothing':
            obj=self.holding
        assert obj !='nothing', \
            f"old_holding {old_holding}, self.holding {self.holding}, obj {obj}, self.location {self.location}, self.history_list\n {self.history_list}"

        if obj not in self.item_location:
            assert self.holding==obj
            self.item_location[obj]={'old_location':self.location,'new_location':self.location}
        else:
            if old_holding==obj:
                self.item_location[obj]['new_location']=self.location
            elif self.holding==obj:
                self.item_location[obj]['new_location']=self.item_location[obj]['old_location']


    def _update_item_state(self, action, obs):
        action = action.lower().strip()

        match = re.match(r"^(heat|cool|clean|slice)\s+([\w\s\d]+?)(?:\s+with\s+[\w\s\d]+)?$", action)
        if not match:
            return

        verb, obj = match.groups()
        obj = obj.strip()

        if verb not in obs:
            return
        if obj not in self.history_items:
            self.history_items[obj] = {
                "heated": False,
                "cooled": False,
                "cleaned": False,
                "slice": False
            }

        if verb == "heat":
            self.history_items[obj]["heated"] = True
        elif verb == "cool":
            self.history_items[obj]["cooled"] = True
        elif verb == "clean":
            self.history_items[obj]["cleaned"] = True
        elif verb == "toggle":
            prev = self.history_items[obj]["toggled"]
            self.history_items[obj]["toggled"] = not prev
        elif verb == "slice":
            self.history_items[obj]["sliced"] = True


    def _update_position(self, action,obs):
        action = action.lower().strip()

        match = re.match(r"^go to\s+(.+)$", action)
        if not match:
            return

        target_location = match.group(1).strip()
        assert target_location in obs, \
            f"target_location {target_location} not in obs {obs}; action {action}, self.location {self.location}, self.history_list\n {self.history_list}"

        self.location=target_location

    def _update_held_items(self, action,obs):
        action = action.lower().strip()

        old_holding=self.holding

        match_take = re.match(r"^take\s+(.+?)(?:\s+from\s+.+)?$", action)
        if match_take:
            obj = match_take.group(1).strip()
            assert self.holding=='nothing', \
                f"action {action}, obs {obs}, self.holding {self.holding}, self.location {self.location}, self.history_list\n {self.history_list}"
            if obj in obs:
                self.holding=obj

        match_drop = re.match(r"^drop\s+(.+)$", action)
        if match_drop:
            obj = match_drop.group(1).strip()
            if obj in obs:
                self.holding="nothing"

        match_put = re.match(r"^(put|place)\s+(.+?)\s+(?:in|on)\s+.+$", action)
        if match_put:
            obj = match_put.group(2).strip()
            if obj in obs:
                self.holding="nothing"

        match_move = re.match(r"^move\s+(.+?)(?:\s+to\s+.+)?$", action)
        if match_move:
            obj = match_move.group(1).strip()
            assert self.holding==obj, \
                f"action {action}, obs {obs}, self.holding {self.holding}, self.location {self.location}, self.history_list\n {self.history_list}"
            if obj in obs:
                self.holding="nothing"

        if old_holding != self.holding:
            self._update_item_location(old_holding)


    def step(self, action):
        actions = [action]
        obs, scores, dones, infos = self.env.step(actions)
        self.history_list.append((f"action:{action}",f"obs:{obs}",f"self.location: {self.location}",f"self.holding: {self.holding}",f"self.history_items: {self.history_items}",f"self.item_location: {self.item_location}"))
        if ("Nothing happens" not in obs[0]) and (not self.done):
            self._update_item_state(action,obs[0])
            self._update_position(action,obs[0])
            self._update_held_items(action,obs[0])
        self.done=dones[0]
        infos['observation_text'] = obs
        infos['obs_location']=[self.location]
        infos['obs_holding']=[self.holding]
        infos['items_info']=[self.history_items]
        infos['item_location']=[self.item_location]
        return obs, scores, dones, infos

    def reset(self):
        obs, infos = self.env.reset()
        self.history_items={}
        self.location="middle of a room"
        self.holding="nothing"
        self.item_location={}
        self.history_list=[]
        self.done=False
        infos['obs_location']=[self.location]
        infos['obs_holding']=[self.holding]
        infos['items_info']=[self.history_items]
        infos['item_location']=[self.item_location]
        infos['observation_text'] = obs
        return obs, infos

    def getobs(self):
        image = get_obs_image(self.env)
        image = image.cpu()
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        text_obs_list = []
        image_obs_list = []
        info_list = []

        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands

    def close(self):
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
