import numpy as np
import random
from environment import SnakeGameAI, Direction, Point
from collections import deque
import torch

MAX_MEM = 100000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0                        #randomness
        self.gamma = 0                          #discount rate
        self.memory = deque(maxlen = MAX_MEM)
        self.model = None
        self.trainer = None

    def get_state(self, env):
        head = env.snake[0]

        #defining danger points
        dpt_u = Point(head.x, head.y - 20)
        dpt_r = Point(head.x + 20, head.y)
        dpt_l = Point(head.x - 20, head.y)
        dpt_d = Point(head.x, head.y + 20)

        #checking direction booleans
        dir_l = env.direction == Direction.LEFT
        dir_r = env.direction == Direction.RIGHT
        dir_u = env.direction == Direction.UP
        dir_d = env.direction == Direction.DOWN

        state = [
            #danger straight ahead
            (dir_l and env.iscollision(dpt_l)) or
            (dir_u and env.iscollision(dpt_u)) or
            (dir_r and env.iscollision(dpt_r)) or
            (dir_d and env.iscollision(dpt_d)),

            #danger left
            (dir_l and env.iscollision(dpt_d)) or
            (dir_u and env.iscollision(dpt_l)) or
            (dir_r and env.iscollision(dpt_u)) or
            (dir_d and env.iscollision(dpt_r)),

            #danger right
            (dir_l and env.iscollision(dpt_u)) or
            (dir_u and env.iscollision(dpt_r)) or
            (dir_r and env.iscollision(dpt_d)) or
            (dir_d and env.iscollision(dpt_l)),

            #Move taken
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food location
            env.food.x < env.head.x,
            env.food.x > env.head.x,
            env.food.y < env.head.y,
            env.food.y > env.head.y
        ]

        return np.array(state, dtype = int)

    def remember(self, state, action, reward, nxt_state, done):
        self.memory.append((state, action, reward, nxt_state, done))  #will pop from start, if memory exceeds MAX_LENGTH

    def get_action(self, state):
        #tradeoff between exploration and exploitation
        #in the start we need random moves and explore the environment
        #but as we get better, we want to exploit the agent

        self.eplison = 80 - self.n_games    #degree of randomness, as the number of games increase degree of randomness decreases
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:   #if degree of randomness is less, less times, random moves will be used
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item()
            
        final_move[move] = 1
        return final_move

    def train_long_memory(self):
            if len(self.memory) > BATCH_SIZE:
                mini_sample = random.sample(self.memory, BATCH_SIZE)
            else:
                mini_sample = self.memory

            states, actions, rewards, nxt_states, dones = zip(*mini_sample)        
            self.trainer.train_step(states, actions, rewards, nxt_states, dones)


    def train_short_memory(self, state, action, reward, nxt_state, done):
        self.trainer.train_step(state, action, reward, nxt_state, done)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    environment = SnakeGameAI()
    while True:
        old_state = agent.get_state(environment)            #get the old state

        final_move = agent.get_action(old_state)            #get the action w.r.t the old state

        reward, done, score = environment.play_step(final_move) #play the game and get the results with the action taken

        new_state = agent.get_state(environment)            #get the new state of the game

        agent.train_short_memory(old_state, final_move, reward, new_state, done)    #train the short memory

        agent.remember(old_state, final_move, reward, new_state, done)              #remember the stuff

        if done:                                            #if game over 
            environment.reset()                             #reset the game
            agent.n_games += 1                              #increase number of games played
            agent.train_long_memory()                       #train long memory

        if score > record:
            record = score                                  #set record score

        print("Game:", agent.n_games, "Score:", score, "Record:", record) 

        #plot some stuff
        #TODO: plot

if __name__ == '__main__':
    train()