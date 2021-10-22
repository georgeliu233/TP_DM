import os
import time
import json
import math

import numpy as np
import tensorflow as tf

from cpprb import ReplayBuffer
from collections import deque
from utils import ZFilter

from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer, get_default_rb_dict
from tf2rl.misc.discount_cumsum import discount_cumsum
from tf2rl.envs.utils import is_discrete

class NeighbourBuffer(object):
    def __init__(self):
        self.buffer = dict()

    def add(self,ids,values):
        if ids not in self.buffer:
            self.buffer[ids] = list()
        self.buffer[ids].append(values)
    
    def clear(self):
        self.buffer = dict()
    
    def query(self,id_list,pad_length=0,pad_value=None):
        res = []
        for ids in id_list:

            if pad_length >0:

                line = self.buffer[ids]
                num = min(pad_length,len(line))

                if pad_value is None:
                    #None then pad the last value
                    v = line[-num:][0]
                else:
                    v = pad_value
                
                pad_line = [v]*(pad_length-num) + line[-num:]
                res.append(pad_line)
            else:
                res.append(self.buffer[ids])
        return res

class OnPolicyTrainer(Trainer):
    def __init__(self,ego_surr=False,surr_vehicles=5,save_name='./ppo', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_log = []
        self.eval_log = []
        self.step_log = []
        self.test_step = []
        self.success_rate=[]
        self.ego_surr=ego_surr
        self.surr_vehicles=surr_vehicles
        self.save_name=save_name
        print('normalized:',self._normalize_obs)

        if self.ego_surr:
            self.neighbours_buffer = NeighbourBuffer()
    
    def observation_adapter(self,env_obs):
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        # distance of vehicle from center of lane
        # closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))

        dist_from_centers = []
        angle_errors = []
        if len(wps)<3:
            for _ in range(3-len(wps)):
                dist_from_centers.append(-1)
                angle_errors.append(-1)
        for wp in wps:
            signed_dist_from_center = wp.signed_lateral_error(ego.position)
            lane_hwidth = wp.lane_width * 0.5
            dist_from_centers.append(signed_dist_from_center / lane_hwidth)
            angle_errors.append(wp.relative_heading(ego.heading))

        neighborhood_vehicles = env_obs.neighborhood_vehicle_states
        relative_neighbor_distance = [np.array([10, 10])]*3

        # no neighborhood vechicle
        if neighborhood_vehicles == None or len(neighborhood_vehicles) == 0:
            relative_neighbor_distance = [
                distance.tolist() for distance in relative_neighbor_distance]
        else:
            position_differences = np.array([math.pow(ego.position[0]-neighborhood_vehicle.position[0], 2) +
                                            math.pow(ego.position[1]-neighborhood_vehicle.position[1], 2) for neighborhood_vehicle in neighborhood_vehicles])

            nearest_vehicle_indexes = np.argsort(position_differences)
            for i in range(min(3, nearest_vehicle_indexes.shape[0])):
                relative_neighbor_distance[i] = np.clip(
                    (ego.position[:2]-neighborhood_vehicles[nearest_vehicle_indexes[i]].position[:2]), -10, 10).tolist()

        distances = [
                diff for diffs in relative_neighbor_distance for diff in diffs]
        observations =  np.array(
            dist_from_centers + angle_errors+ego.position[:2].tolist()+[ego.speed,ego.steering]+distances,
            dtype=np.float32,
        )
        assert observations.shape[-1]==16,observations.shape
        return observations
    
    def ego_surr_adapter(self,obs):
        # we only take (x,y,vx,vy,dis,psi) tuple for each state
        ego= obs.ego_vehicle_state
        neighbours = obs.neighborhood_vehicle_states

        ego_state = [
            ego.position[0],ego.position[1],
            0.0,float(ego.heading)
        ]
        dis_list = []
        min_ind = []
        id_list = []
        if len(neighbours)>0:
            
            for neighbour in neighbours:
                x,y= neighbour.position[0],neighbour.position[1]
                psi = float(neighbour.heading)
                dis = np.sqrt((ego_state[0]-x)**2 + (ego_state[1]-y)**2)
                self.neighbours_buffer.add(ids=neighbour.id, values=[x,y,dis,psi])
                dis_list.append(dis)
                id_list.append(neighbour.id)
            
            min_ind = np.argsort(dis_list)[:min(len(neighbours),self.surr_vehicles)]

        #ego
        res_ids = []
        for ids in min_ind:
            res_ids.append(id_list[ids])
        final_obs = [ego_state]

        # print(pos_diff)
        # print(ego_state)
        # print(dis_list)
        # print(min_ind)

        #neighbours sorted by distance
        # for i in range(len(min_ind)):
        #     final_obs.append(pos_diff[min_ind[i]])
        
        # #padding
        # for i in range(self.surr_vehicles-len(min_ind)):
        #     final_obs.append(ego_state)
        
        #(6,4)
        # print(np.reshape(final_obs,(-1)))
        # assert 1==0
        return np.reshape(final_obs,(-1)),res_ids

    def __call__(self):
        # Prepare buffer
        self.replay_buffer = get_replay_buffer(
            self._policy, self._env)
        kwargs_local_buf = get_default_rb_dict(
            size=self._policy.horizon, env=self._env)
        kwargs_local_buf["env_dict"]["logp"] = {}
        kwargs_local_buf["env_dict"]["val"] = {}
        if self.lstm:
            kwargs_local_buf["env_dict"]['mask'] = {}
        if is_discrete(self._env.action_space):
            kwargs_local_buf["env_dict"]["act"]["dtype"] = np.int32
        self.local_buffer = ReplayBuffer(**kwargs_local_buf)

        episode_steps = 0
        episode_return = 0
        episode_start_time = time.time()
        total_steps = np.array(0, dtype=np.int32)
        n_epoisode = 0
        obs = self._env.reset()
        init_pos = obs['Agent-LHC'].ego_vehicle_state.position[:2]
        if self.state_input:
            if self.surr_vehicles:
                obs,neighbor_ids = self.ego_surr_adapter(obs['Agent-LHC'])
            else:
                obs = self.observation_adapter(obs['Agent-LHC'])
        else:
            obs = obs['Agent-LHC'].top_down_rgb.data
        
        if self.lstm:
            buffer_queue = deque(maxlen=self.n_steps)
            # for _ in range(self.n_steps):
            buffer_queue.append(obs)
            
            if self.surr_vehicles:
                #[num_of_neighbors,time_step(pad_length),4]
                if len(neighbor_ids)>0:
                    neighbors = self.neighbours_buffer.query(neighbor_ids,pad_length=1)
                   
                    neighbors = np.reshape(np.transpose(neighbors,[1,0,2]),(1,-1))
                    # print(neighbors)
                    # print(np.array(list(buffer_queue)))
                    if len(neighbor_ids)<self.surr_vehicles:
                        arr = np.array([ list(np.array(list(buffer_queue))[-1])*1 ]*(self.surr_vehicles-len(neighbor_ids)))
                        out_obs = np.concatenate((np.array(list(buffer_queue)),
                                    neighbors,
                                    arr),
                                    axis=-1)
                    else:
                        out_obs = np.concatenate((np.array(list(buffer_queue)),
                                    neighbors),
                                    axis=-1)
                else:
                    out_obs = np.concatenate((np.array(list(buffer_queue)),
                                np.array([ list(np.array(list(buffer_queue))[-1])*1 ]*self.surr_vehicles)),
                                axis=-1)
            else:
                out_obs = np.array(list(buffer_queue))
            
            obs = np.concatenate(( out_obs,np.array([np.zeros_like(out_obs[0])]*(self.n_steps-1)) ),axis=0)
            mask = [1]+[0]*(self.n_steps-1)
        else:
            mask=None
        
        self.train_reward_scaler = ZFilter(shape=(),center=False,gamma=None)#self._policy.discount)

        tf.summary.experimental.set_step(total_steps)
        init_dis = 0
        while total_steps < self._max_steps:
            # Collect samples
            for _ in range(self._policy.horizon):
                if self._normalize_obs:
                    obs = self._obs_normalizer(obs, update=False)
                action, logp, val = self._policy.get_action_and_val(obs,np.expand_dims(mask,0))
                # if not is_discrete(self._env.action_space):
                #     env_act = np.clip(act, self._env.action_space.low, self._env.action_space.high)
                # else:
                #     env_act = act

                # next_obs, reward, done, _ = self._env.step(env_act)

                choice_action = []
                MAX_SPEED = 10
                choice_action.append((action[0]+1)/2*MAX_SPEED)
                if action[1]<= -1/3:
                    choice_action.append(-1)
                elif -1/3< action[1] <1/3:
                    choice_action.append(0)
                else:
                    choice_action.append(1)
                #print(choice_action)
                next_obs, reward, done, _ = self._env.step({
                "Agent-LHC":choice_action
                })
                # next_obs, reward, done, _ = self._env.step(action)
                done_events = next_obs["Agent-LHC"].events
                r = 0.0
                if done_events.reached_goal or (done["Agent-LHC"] and not done_events.reached_max_episode_steps):
                    r += 5.0
                if done_events.collisions !=[] or episode_steps==998:
                    r -= 10.0
                # r += next_obs['Agent-LHC'].ego_vehicle_state.speed*0.01

                r += next_obs['Agent-LHC'].ego_vehicle_state.speed/MAX_SPEED*0.1

                
                # curr_dis = next_obs['Agent-LHC'].distance_travelled
                curr_pos = next_obs['Agent-LHC'].ego_vehicle_state.position[:2]
                dis = np.linalg.norm(curr_pos-init_pos)
                # print(curr_dis-init_dis,dis)
                r += 0.2*(dis)

                # init_dis = curr_dis
                init_pos = curr_pos
                #self.memory.append(state, action, r, next_state, done["Agent-LHC"])
                episode_return += r

                if self.state_input:
                    if self.surr_vehicles:
                        next_obs,neighbor_ids = self.ego_surr_adapter(next_obs['Agent-LHC'])
                    else:
                        next_obs = self.observation_adapter(next_obs['Agent-LHC'])
                else:
                    next_obs = next_obs['Agent-LHC'].top_down_rgb.data

                if self._show_progress:
                    self._env.render()

                episode_steps += 1
                total_steps += 1
                # episode_return += reward

                done_flag = done["Agent-LHC"]
                if (hasattr(self._env, "_max_episode_steps") and
                    episode_steps == self._env._max_episode_steps):
                    done_flag = False
                
                if self.lstm:
                    buffer_queue.append(next_obs)
                    stp_num = np.clip(episode_steps+1,0,self.n_steps)
                    # next_obs = np.array(list(buffer_queue)+[np.zeros_like(next_obs)]*(self.n_steps-stp_num))
                    if self.surr_vehicles:
                        #[num_of_neighbors,time_step(pad_length),4]
                        if len(neighbor_ids)>0:
                            neighbors = self.neighbours_buffer.query(neighbor_ids,pad_length=stp_num)

                            neighbors = np.reshape(np.transpose(neighbors,[1,0,2]),(stp_num,-1))

                            if len(neighbor_ids)<self.surr_vehicles:
                                arr = np.array([ list(np.array(list(buffer_queue))[-1])*stp_num ]*(self.surr_vehicles-len(neighbor_ids)))
                                out_obs = np.concatenate((np.array(list(buffer_queue)),
                                            neighbors,
                                            arr),
                                            axis=-1)
                            else:
                                
                                out_obs = np.concatenate((np.array(list(buffer_queue)),
                                            neighbors),
                                            axis=-1)
                        else:
                            out_obs = np.concatenate((np.array(list(buffer_queue)),
                                        np.array([ list(np.array(list(buffer_queue))[-1])*stp_num ]*self.surr_vehicles)),
                                        axis=-1)
                    else:
                        out_obs = np.array(list(buffer_queue))
                    print(out_obs.shape)
                    if stp_num<self.n_steps:
                        next_obs = np.concatenate(( out_obs,np.array([np.zeros_like(out_obs[0])]*(self.n_steps-stp_num)) ),axis=0)
                    else:
                        next_obs = out_obs
                    mask = np.array([1]*(stp_num) + [0]*(self.n_steps-stp_num))
                    # print(next_obs.shape)
                    self.local_buffer.add(
                        obs=obs, act=action, next_obs=next_obs,
                        rew=self.train_reward_scaler(r), done=done_flag, logp=logp, val=val,mask=mask)
                else:
                    self.local_buffer.add(
                        obs=obs, act=action, next_obs=next_obs,
                        rew=self.train_reward_scaler(r), done=done_flag, logp=logp, val=val)
                obs = next_obs

                if done["Agent-LHC"] or episode_steps == self._episode_max_steps:
                    tf.summary.experimental.set_step(total_steps)
                    
                    self.finish_horizon()
                    obs = self._env.reset()
                    self.train_reward_scaler.reset()
                    init_dis = 0
                    init_pos = obs['Agent-LHC'].ego_vehicle_state.position[:2]

                    self.neighbours_buffer.clear()

                    if self.state_input:
                        if self.surr_vehicles:
                            obs,neighbor_ids = self.ego_surr_adapter(obs['Agent-LHC'])
                        else:
                            obs = self.observation_adapter(obs['Agent-LHC'])
                    else:
                        obs = obs['Agent-LHC'].top_down_rgb.data 
                    if self.lstm:
                        buffer_queue = deque(maxlen=self.n_steps)
                        # for _ in range(self.n_steps):
                        buffer_queue.append(obs)
                        if self.surr_vehicles:
                            #[num_of_neighbors,time_step(pad_length),4]
                            if len(neighbor_ids)>0:
                                neighbors = self.neighbours_buffer.query(neighbor_ids,pad_length=1)

                                neighbors = np.reshape(np.transpose(neighbors,[1,0,2]),(1,-1))
                                if len(neighbor_ids)<self.surr_vehicles:
                                    arr = np.array([ list(np.array(list(buffer_queue))[-1])*1 ]*(self.surr_vehicles-len(neighbor_ids)))
                                    out_obs = np.concatenate((np.array(list(buffer_queue)),
                                                neighbors,
                                                arr),
                                                axis=-1)
                                else:
                                    out_obs = np.concatenate((np.array(list(buffer_queue)),
                                                neighbors),
                                                axis=-1)
                            else:
                                out_obs = np.concatenate((np.array(list(buffer_queue)),
                                            np.array([ list(np.array(list(buffer_queue))[-1])*1 ]*self.surr_vehicles)),
                                            axis=-1)
                        else:
                            out_obs = np.array(list(buffer_queue))
                        
                        obs = np.concatenate(( out_obs,np.array([np.zeros_like(out_obs[0])]*(self.n_steps-1)) ),axis=0)
                        mask = [1]+[0]*(self.n_steps-1)
                        # obs = np.array(list(buffer_queue))
                    n_epoisode += 1
                    fps = episode_steps / (time.time() - episode_start_time)
                    self.logger.info(
                        "Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                            n_epoisode, int(total_steps), episode_steps, episode_return, fps))
                    tf.summary.scalar(name="Common/training_return", data=episode_return)
                    tf.summary.scalar(name="Common/training_episode_length", data=episode_steps)
                    tf.summary.scalar(name="Common/fps", data=fps)
                    self.return_log.append(episode_return)
                    self.step_log.append(int(total_steps))
                    with open('/home/haochen/TPDM_transformer/'+self.save_name+'.json','w',encoding='utf-8') as writer:
                        writer.write(json.dumps([self.return_log,self.step_log],ensure_ascii=False,indent=4))
                    episode_steps = 0
                    episode_return = 0
                    episode_start_time = time.time()

                if total_steps % self._test_interval == 0:
                    avg_test_return, avg_test_steps ,success_rate= self.evaluate_policy(total_steps)
                    self.eval_log.append(avg_test_return)
                    self.test_step.append(avg_test_steps)
                    self.success_rate.append(success_rate)
                    
                    with open('/home/haochen/TPDM_transformer/'+self.save_name+'_test.json','w',encoding='utf-8') as writer:
                        writer.write(json.dumps([self.eval_log,self.success_rate,self.test_step],ensure_ascii=False,indent=4))
                    self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f},success rate:{3}, over {2: 2} episodes".format(
                        total_steps, avg_test_return, self._test_episodes,success_rate))
                    tf.summary.scalar(
                        name="Common/average_test_return", data=avg_test_return)
                    tf.summary.scalar(
                        name="Common/average_test_episode_length", data=avg_test_steps)
                    self.writer.flush()

                if total_steps % self._save_model_interval == 0:
                    self.checkpoint_manager.save()

            self.finish_horizon(last_val=val)

            tf.summary.experimental.set_step(total_steps)

            # Train actor critic
            if self._policy.normalize_adv:
                samples = self.replay_buffer.get_all_transitions()
                self.mean_adv = np.mean(samples["adv"])
                self.std_adv = np.std(samples["adv"])
                # Update normalizer
                if self._normalize_obs:
                    self._obs_normalizer.experience(samples["obs"])
            with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                for _ in range(self._policy.n_epoch):
                    if False:
                        self.slide_window_lstm()
                    else:
                        samples = self.replay_buffer._encode_sample(
                            np.random.permutation(self._policy.horizon))
                        if self._normalize_obs:
                            samples["obs"] = self._obs_normalizer(samples["obs"], update=False)
                        if self._policy.normalize_adv:
                            adv = (samples["adv"] - self.mean_adv) / (self.std_adv + 1e-8)
                        else:
                            adv = samples["adv"]
                        # print('training',int(self._policy.horizon / self._policy.batch_size))
                        # print(adv.shape)
                        
                        # print(samples["obs"].shape)
                        # print(samples["act"].shape)
                        # print(samples["logp"].shape)
                        # print(samples["ret"].shape)
                        if 'mask' not in samples.keys():

                            for idx in range(int(self._policy.horizon / self._policy.batch_size)):
                                target = slice(idx * self._policy.batch_size,
                                            (idx + 1) * self._policy.batch_size)
                                # print(idx)
                                # print(samples["obs"][target])
                                # print(np.sum(np.isnan(samples["obs"][target])))
                                # print(np.sum(np.isnan(samples["act"][target])))
                                # print(np.sum(np.isnan(samples["logp"][target])))
                                # print(np.sum(np.isnan(samples["ret"][target])))
                                # print(np.sum(np.isnan(adv[target])))
                                # np.savez(
                                #     '/home/haochen/TPDM_transformer/test.npz',
                                #     states=samples["obs"][target],
                                #     actions=samples["act"][target],
                                #     advantages=adv[target],
                                #     logp_olds=samples["logp"][target],
                                #     returns=samples["ret"][target]
                                # )

                                self._policy.train(
                                    states=samples["obs"][target],
                                    actions=samples["act"][target],
                                    advantages=adv[target],
                                    logp_olds=samples["logp"][target],
                                    returns=samples["ret"][target])
                        else:                
                            for idx in range(int(self._policy.horizon / self._policy.batch_size)):
                                target = slice(idx * self._policy.batch_size,
                                            (idx + 1) * self._policy.batch_size)
                                np.save('/home/haochen/TPDM_transformer/obs_test.npy',samples["obs"][target])
                                assert 1==0
                                self._policy.train(
                                    states=samples["obs"][target],
                                    actions=samples["act"][target],
                                    advantages=adv[target],
                                    logp_olds=samples["logp"][target],
                                    returns=samples["ret"][target],
                                    mask=np.array(samples['mask'][target]))


        tf.summary.flush()

    def slide_window_lstm(self):
        samples =self.replay_buffer.get_all_transitions()
        new_sample = {
            'obs':[],
            'act':[],
            'adv':[],
            'logp':[],
            'ret':[]
        }
        for i in range(self._policy.horizon-self.n_steps):
            new_sample['obs'].append(samples['obs'][i:i+self.n_steps])
            new_sample['ret'].append(samples['ret'][i:i+self.n_steps])
            new_sample['act'].append(samples['act'][i:i+self.n_steps])
            new_sample['logp'].append(samples['logp'][i:i+self.n_steps])

            new_sample['adv'].append(samples['adv'][i:i+self.n_steps])
        
        if self._normalize_obs:
            obs = self._obs_normalizer(np.array(new_sample["obs"]), update=False)
        else:
            obs = np.array(new_sample['obs'])
        if self._policy.normalize_adv:
            adv = (np.array(new_sample["adv"]) - self.mean_adv) / (self.std_adv + 1e-8)
        else:
            adv = np.array(new_sample["adv"])
        
        for idx in range(int(obs.shape[0] / self._policy.batch_size)):
            target = slice(idx * self._policy.batch_size,
                            (idx + 1) * self._policy.batch_size)
            # print(target)
            self._policy.train(
                states=obs[target],
                actions=np.array(new_sample['act'])[target],
                advantages=adv[target],
                logp_olds=np.array(new_sample["logp"])[target],
                returns=np.array(new_sample["ret"])[target])
        

        

    def finish_horizon(self, last_val=0):
        self.local_buffer.on_episode_end()
        
        samples = self.local_buffer._encode_sample(
            np.arange(self.local_buffer.get_stored_size()))
        rews = np.append(samples["rew"], last_val)
        vals = np.append(samples["val"], last_val)

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self._policy.discount * vals[1:] - vals[:-1]
        if self._policy.enable_gae:
            advs = discount_cumsum(deltas, self._policy.discount * self._policy.lam)
        else:
            advs = deltas

        # Rewards-to-go, to be targets for the value function
        rets = discount_cumsum(rews, self._policy.discount)[:-1]
        self.replay_buffer.add(
            obs=samples["obs"], act=samples["act"], done=samples["done"],
            ret=rets, adv=advs, logp=np.squeeze(samples["logp"]),mask=samples['mask'])
        self.local_buffer.clear()
        

    def evaluate_policy(self, total_steps):
        avg_test_return = 0.
        avg_test_steps = 0
        success_time = 0
        self.eval_reward_scaler = ZFilter(shape=(),center=False)
        if self._save_test_path:
            replay_buffer = get_replay_buffer(
                self._policy, self._test_env, size=self._episode_max_steps)
        for i in range(self._test_episodes):
            episode_return = 0.
            episode_time = 0
            frames = []
            obs = self._test_env.reset()
            init_pos = obs['Agent-LHC'].ego_vehicle_state.position[:2]
            if self.state_input:
                if self.surr_vehicles:
                    obs = self.ego_surr_adapter(obs['Agent-LHC'])
                else:
                    obs = self.observation_adapter(obs['Agent-LHC'])
            else:
                obs = obs['Agent-LHC'].top_down_rgb.data
            
            if self.lstm:
                buffer_queue = deque(maxlen=self.n_steps)
                # for _ in range(self.n_steps):
                buffer_queue.append(obs)
                # obs = np.array(list(buffer_queue))
                obs = np.array(list(buffer_queue)+[np.zeros_like(obs)]*(self.n_steps-1))
                mask = [1]+[0]*(self.n_steps-1)

            avg_test_steps += 1
            flag=False
            init_dis=0
            for _ in range(self._episode_max_steps):
                if self._normalize_obs:
                    obs = self._obs_normalizer(obs, update=False)
                act, _ = self._policy.get_action(obs, test=True,mask=np.expand_dims(mask,0))
                # act = (act if is_discrete(self._env.action_space) else
                #        np.clip(act, self._env.action_space.low, self._env.action_space.high))

                # next_obs, reward, done, _ = self._test_env.step(act)
                choice_action = []
                MAX_SPEED = 10
                choice_action.append((act[0]+1)/2*MAX_SPEED)
                if act[1]<= -1/3:
                    choice_action.append(-1)
                elif -1/3< act[1] <1/3:
                    choice_action.append(0)
                else:
                    choice_action.append(1)
                #print(choice_action)
                next_obs, reward, done, _ = self._test_env.step({
                "Agent-LHC":choice_action
                })
                # next_obs, reward, done, _ = self._env.step(action)
                done_events = next_obs["Agent-LHC"].events
                r = 0.0
                if done_events.reached_goal or (done["Agent-LHC"] and not done_events.reached_max_episode_steps):
                    r += 5.0
                if done_events.collisions !=[] or episode_time==998:
                    r -= 10.0
                    flag =True
                r += next_obs['Agent-LHC'].ego_vehicle_state.speed/MAX_SPEED*0.1


                # curr_dis = next_obs['Agent-LHC'].distance_travelled
                curr_pos = next_obs['Agent-LHC'].ego_vehicle_state.position[:2]
                dis = np.linalg.norm(curr_pos-init_pos)
                # diff = curr_dis-init_dis
                r += 0.2*(dis)

                # init_dis = curr_dis
                init_pos = curr_pos
                #self.memory.append(state, action, r, next_state, done["Agent-LHC"])
                # episode_return += r

                if self.state_input:
                    if self.surr_vehicles:
                        next_obs = self.ego_surr_adapter(next_obs['Agent-LHC'])
                    else:
                        next_obs = self.observation_adapter(next_obs['Agent-LHC'])
                else:
                    next_obs = next_obs['Agent-LHC'].top_down_rgb.data

                avg_test_steps += 1
                episode_time+=1
                if self.lstm:
                    buffer_queue.append(next_obs)
                    stp_num = np.clip(episode_time+1,0,self.n_steps)
                    next_obs = np.array(list(buffer_queue)+[np.zeros_like(next_obs)]*(self.n_steps-stp_num))
                    mask = np.array([1]*(stp_num) + [0]*(self.n_steps-stp_num))

                if self._save_test_path:
                    replay_buffer.add(
                        obs=obs, act=act, next_obs=next_obs,
                        rew=reward, done=done)

                if self._save_test_movie:
                    frames.append(self._test_env.render(mode='rgb_array'))
                elif self._show_test_progress:
                    self._test_env.render()

                episode_return += r
                obs = next_obs
                if done['Agent-LHC']:
                    # done_events = next_obs["Agent-LHC"].events
                    obs = self._test_env.reset()
                    self.eval_reward_scaler.reset()
                    init_pos = obs['Agent-LHC'].ego_vehicle_state.position[:2]
                    obs = obs['Agent-LHC'].top_down_rgb.data

                    if not flag:
                        success_time+=1
                    init_dis=0
                    break
            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(
                total_steps, i, episode_return)
            if self._save_test_path:
                save_path(replay_buffer.sample(self._episode_max_steps),
                          os.path.join(self._output_dir, prefix + ".pkl"))
                replay_buffer.clear()
            if self._save_test_movie:
                frames_to_gif(frames, prefix, self._output_dir)
            avg_test_return += episode_return
        if self._show_test_images:
            images = tf.cast(
                tf.expand_dims(np.array(obs).transpose(2, 0, 1), axis=3),
                tf.uint8)
            tf.summary.image('train/input_img', images, )
        return avg_test_return / self._test_episodes, avg_test_steps / self._test_episodes, success_time/self._test_episodes
