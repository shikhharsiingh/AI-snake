import numpy as np
import rnadom
import torch
from environment import SnakeGameAI, Direction, Point
from collections import deque

MAX_MEM = 100000
BATCH_SiZE = 1000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0                        #randomness
        self.gamma = 0                          #discount rate
        self.memory = deque(maxlen = MAX_MEM)


    def get_state(self, env):
        pass

    def remember(self, state, action, reward, nxt_state, done):
        pass

    def get_action(self, state):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self):
        pass

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    environment = SnakeGameAI()
    while True:
        old_state = agent.get_state(environment)

if __name__ == '__main__':
    train()

