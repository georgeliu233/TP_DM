import json
import matplotlib.pyplot as plt
import numpy as np

def smooth(scalar,weight=0.85):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_comp(weight=0.99):
    abs_path = '/home/haochen/TPDM_transformer/'
    # json_list = [
    #     "log_loop_fusioned",
    #     'log_loop_state',
    #     'log_loop_cnn'
    # ]
    json_list = ['ppo_neighbor']
    # json_list = ['log_ppo']
    data_list = []
    for path in json_list:
        with open(abs_path+path+'.json','r',encoding='utf-8') as reader:
            r,t = json.load(reader)
            data_list.append([r,t])
    
    plt.figure()
    for data in data_list:
        plt.plot(data[1][:],smooth(data[0][:],weight))
    plt.savefig('/home/haochen/TPDM_transformer/res_ppo_9.png')

def plot_comp_test(weight=0.5):
    abs_path = '/home/haochen/TPDM_transformer/'
    # json_list = [
    #     "log_loop_fusioned",
    #     'log_loop_state',
    #     'log_loop_cnn'
    # ]
    json_list = ['log_test_ppo','log_test_ppo_2']
    # json_list = ['log_ppo']
    data_list = []
    for path in json_list:
        with open(abs_path+path+'.json','r',encoding='utf-8') as reader:
            _,t,_ = json.load(reader)
            data_list.append([t,np.linspace(5000,5000*len(t),len(t))])
    
    plt.figure()
    for data in data_list:
        plt.plot(data[1][:],smooth(data[0][:],weight))
    plt.savefig('/home/haochen/TPDM_transformer/res_ppo_9_test.png')

plot_comp()
# plot_comp_test()