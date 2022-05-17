from Arm_env import ArmEnv

if __name__ == "__main__":
    env = ArmEnv()
    o = env.reset()
    sys.path.append('./multi_processing_ppo')
    from PPO.multi_processing_ppo.PPOModel import *
    net = GlobalNet(env.state_dim,env.action_dim)
    net.act.load_state_dict(torch.load('./trained_models/act.pkl'))
    while 1:
        env.render()
        a = net.act(torch.tensor(o, dtype=torch.float32, device="cpu")).detach().numpy()
        o2,r,d,_ = env.step(a)
        o = o2