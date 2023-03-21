import matplotlib.pyplot as plt
import torch
"""
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())"""
def plot(score, avg_score, episodes, avg_episodes):
    plt.title('Result of DDQN')
    plt.plot(episodes, score, label = "reward of each episodes")
    plt.plot(avg_episodes, avg_score, label = "average reward of 100 episodes")
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.legend()
    plt.show()
