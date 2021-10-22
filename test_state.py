from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec,Agent
import os
import gym
import numpy as np
import matplotlib.pyplot as plt

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.LanerWithSpeed, max_episode_steps=1000,neighborhood_vehicles=True),
    agent_builder=None
)
agent_specs={
    'Agent-LHC':agent_spec
}
env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/left_turn"],
    agent_specs=agent_specs,
    headless=True
)

obs = env.reset()

def get_state(env_obs):
    ego = env_obs.ego_vehicle_state
    neighborhood_vehicles = env_obs.neighborhood_vehicle_states
    road_wp = env_obs.road_waypoints
    path = env_obs.waypoint_paths
    cloud = env_obs.lidar_point_cloud
    print(dir(ego))
    print(float(ego.heading))
    print(len(neighborhood_vehicles))
    print(neighborhood_vehicles[0])
    # print(road_wp)
    # print(path)
    # print(cloud)
    # cloud_printer(cloud)
    # print(road_wp)
    # print(road_wp)
    print(dir(road_wp))
    # print(road_wp.lanes.keys())
    # print(road_wp.route_waypoints)
    # waypoint_printer(road_wp)
    # print(path)
    path_printer(path)


def cloud_printer(cloud):
    cloud_info,hit,ray = cloud
    print(len(cloud_info))
    plt.figure()
    color = ['blue','red']
    for pt,h in zip(ray,hit):
        if h==True:
            plt.scatter(pt[0],pt[1],c=color[1])
        else:
            plt.scatter(pt[0],pt[1],c=color[0])
    plt.savefig('/home/haochen/TPDM_transformer/cloud.png')

def waypoint_printer(waypoint):
    plt.figure()
    for k,wps in waypoint.lanes.items():
        x,y = [],[]
        print(k,len(wps))
        for wp_list in wps:
            print(len(wp_list))
            for wp in wp_list:
                x.append(wp.pos[0])
                y.append(wp.pos[1])
        plt.scatter(x, y)
        plt.plot(x,y)
    plt.savefig('/home/haochen/TPDM_transformer/wp.png')

def path_printer(paths):
    plt.figure()
    for path in paths:
        x,y = [],[]
        for wp in path:
            x.append(wp.pos[0])
            y.append(wp.pos[1])
        plt.scatter(x, y)
        plt.plot(x,y)
    plt.savefig('/home/haochen/TPDM_transformer/path.png')

get_state(obs['Agent-LHC'])

