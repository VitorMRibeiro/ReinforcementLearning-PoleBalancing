import math
from random import random
import json
import signal
import sys
import time
import gym
env = gym.make('CartPole-v1')

INITIAL_EPSILON = 0.3 
FINAL_EPSILON = 0.001
ALPHA = 0.02
INTERVAL_SIZE = 0.2 # size of intervals in which the continuous observations will breaked in.
RANGE = 3           # range of observation values considered

# file to which the learned action-values will be saved
file_name = 'cartpole-v2.txt'

N = (int)(RANGE * 2)/INTERVAL_SIZE

number_of_states = (int)(pow(N, 4))
number_of_actions = 2

policy = []
action_value = []

# try to open the file with the action-values saved from previous runs
try:
    values_file = open(file_name)
except:
    values_file = ''

if values_file != '':
    action_value = json.load(values_file)
    # calculate the best policy based on the action values
    for s in range(number_of_states):
        best_action = -1
        best_action_value = -999999
        for a in range(number_of_actions):
            if action_value[s][a] > best_action_value:
                best_action_value = action_value[s][a]
                best_action = a
        policy.append(best_action)
else:
    # initializes the policy with random actions, and the action values with 0
    for i in range(number_of_states):
        policy.append(env.action_space.sample())
        action_value.append([0] * number_of_actions)


def discretize(obs):
    """takes a continuous observations and convertes into a discrete state."""
    f_1 = math.floor((min(obs[0], RANGE) + RANGE) / INTERVAL_SIZE) 
    f_2 = math.floor((min(obs[1], RANGE) + RANGE) / INTERVAL_SIZE)
    f_3 = math.floor((min(obs[2], RANGE) + RANGE) / INTERVAL_SIZE)
    f_4 = math.floor((min(obs[3], RANGE) + RANGE) / INTERVAL_SIZE) 
    state = (int)(f_1 * N * N * N + f_2 * N * N + f_3 * N + f_4)
    return state

# when interupted the program will save the its current action values in a file
def signal_handler(sig, frame):
    f = open(file_name, 'w')
    json.dump(action_value, f)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

average_total_reward = 0
epsilon = INITIAL_EPSILON

for i_episode in range(900000):
    epsilon = max(FINAL_EPSILON, epsilon - 0.00001)
    observation = env.reset()
    history = []
    t = 0
    if i_episode % 1000 == 0:
        print(f"episode {i_episode}")
    while(1):
        if i_episode % 1000 == 0:
            env.render()
            time.sleep(0.03)
        
        state = discretize(observation)
        history.append({'state': state})

        if random() < epsilon:
            # exploring
            action = 0 if policy[state] == 1 else 1
        else:
            # exploiting
            action = policy[state]

        observation, reward, done, info = env.step(action)

        history[t]['action'] = action
        history[t]['reward'] = reward
        t += 1
        if done:
            partial_return = 0
            Q = dict()
            for i_history in reversed(history):
                i_state = i_history['state']
                i_action = i_history['action']
                i_reward = i_history['reward']

                partial_return += i_reward
                Q[i_state * number_of_actions + i_action] = partial_return

            for i, q in Q.items():
                s, a = divmod(i, number_of_actions)

                action_value[s][a] += ALPHA * (q - action_value[s][a])
                if action_value[s][a] > action_value[s][policy[s]]:
                    policy[s] = a

            average_total_reward = average_total_reward + ALPHA * (t - average_total_reward)
            print(f"reward: {t}, average: {average_total_reward}")
            break

f = open(file_name, 'w')
json.dump(action_value, f)