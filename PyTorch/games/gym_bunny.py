from PyTorch.games.gym_handler import gym_trainer
import numpy as np
import random

class ReplayMemory:
    def __init__(self,capacity,target=500):
        self.target = target # ideal replay length
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self,history):
        score = np.array(history['reward']).sum()*len(history['action'])/self.target
        self.memory.append((score,history))
        if len(self.memory)>self.capacity:
            self.memory.sort()
            if np.random.randint(2):
                self.memory.pop(0)
            else:
                self.memory.pop(np.random.randint(self.capacity))

    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)

class gym_bunny:
    def __init__(self, game="BipedalWalker-v2", mem_cap = 50):
        self.trainer = gym_trainer(game)
        self.memory = ReplayMemory(mem_cap)
        for i in range(mem_cap):
            self.memory.add(self.trainer.random_run())
