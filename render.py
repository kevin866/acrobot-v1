import gym
import time
from policy import Policy
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder

torch.manual_seed(0) # set random seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state_dict = torch.load('checkpoint.pth')

policy = Policy()
policy.load_state_dict(state_dict)
policy = policy.to(device)

def show_smart_agent():
    env = gym.make('Acrobot-v1')
    recorder = VideoRecorder(env, path='./video.mp4', enabled=True)
    state = env.reset()
    cum_rew = 0

    for t in range(1000):
        recorder.capture_frame()
        action, _ = policy.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        cum_rew+=1
        if done:
            print(cum_rew)
            break 
        time.sleep(0.1)

    env.close()

show_smart_agent()