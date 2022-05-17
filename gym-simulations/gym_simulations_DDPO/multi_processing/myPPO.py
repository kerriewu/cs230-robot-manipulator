#!/usr/bin/python
# -*- coding: utf-8 -*-
from PPOModel import *
import gym
import multiprocessing
from draw import Painter
from copy import deepcopy
from time import time
import random
from Arm_env import ArmEnv

def setup_seed(seed):
    """Set the random number seed function"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def main():
    """
    1. Multiple processes do not train the network, just get the network of the main process to explore the environment, and pass the transition back to the main process through pipe
    2. The main process packs the transitions of all child processes into a buffer for network training
    3. Pass the updated net to the child process and return to 1
    """

    """arm env"""
    env = ArmEnv(mode='hard')
    obs_dim = env.state_dim
    act_dim = env.action_dim
    net = GlobalNet(obs_dim, act_dim)
    """End"""

    ppo = AgentPPO(deepcopy(net))
    process_num = 6
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num) for pipe1, pipe2 in (multiprocessing.Pipe(),))
    child_process_list = []
    for i in range(process_num):
        pro = multiprocessing.Process(target=child_process, args=(pipe_dict[i][1],))
        child_process_list.append(pro)
    [pipe_dict[i][0].send(net) for i in range(process_num)]
    [p.start() for p in child_process_list]

    rewardList = list()
    timeList = list()
    begin = time()
    MAX_EPISODE = 150
    batch_size = 128
    max_reward = -np.inf
    for episode in range(MAX_EPISODE):
        reward = 0
        buffer_list = list()
        for i in range(process_num):
            # With the function of synchronizing the child process, it will not go to the statement after the for without receiving the data of the child process
            receive = pipe_dict[i][0].recv()        
            data = receive[0]
            buffer_list.append(data)
            reward += receive[1]
        ppo.update_policy_mp(batch_size,8,buffer_list)
        net.act.load_state_dict(ppo.act.state_dict())
        net.cri.load_state_dict(ppo.cri.state_dict())
        [pipe_dict[i][0].send(net) for i in range(process_num)]

        reward /= process_num
        rewardList.append(reward)
        timeList.append(time() - begin)
        print(f'episode:{episode}  reward:{reward} time:{timeList[-1]}')

        if reward > max_reward and episode > MAX_EPISODE*2/3:
            max_reward = reward
            torch.save(net.act.state_dict(), '../trained_models/act.pkl')

    [p.terminate() for p in child_process_list]

    painter = Painter(load_csv=False)
    painter.addData(rewardList, 'MP-PPO')
    painter.drawFigure()


def child_process(pipe):
    setup_seed(0)
    # can change mode
    env = ArmEnv(mode='hard')

    env.reset()
    while True:
        # Receive the net parameters of the main thread, this sentence also has the function of synchronization
        net = pipe.recv() 
        ppo = AgentPPO(net,if_explore=True)
        rewards, steps = ppo.update_buffer(env, 500, 1)
        transition = ppo.buffer.sample_all()
        r = transition.reward
        m = transition.mask
        a = transition.action
        s = transition.state
        log = transition.log_prob
        data = (r,m,s,a,log)
        # pipe cannot directly transfer the buffer back to the main process. It may be that there is a transition in the buffer, so the data is taken out and packaged and sent back.
        pipe.send((data,rewards))



if __name__ == "__main__":
    main()


