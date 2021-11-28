from env.multirunner import MultiRunner as MEnv
from env.multirunner import WrappedRunner as WEnv
from learning.wrapper import wrapper_policy
from learning.model import resnet18

if __name__ == "__main__":
    policy = wrapper_policy(resnet18(use_bn=True)).to('cuda:0')
    other_policy = wrapper_policy(resnet18(use_bn=True)).to('cuda:0')

    env = WEnv()
    done = False
    obs = env.reset()
    while not done:
        state, other_state = obs
        action = policy(state) if state is not None else -1
        other_action = other_policy(other_state) if other_state is not None else -1
        obs, rew, done, info = env.step((action, other_action))


    env = MEnv(policy, other_policy, n_env_parallel=3, n_torch_parallel=2, max_env_num=100, max_batch_size=5000, max_step=300000)
    env.collect()