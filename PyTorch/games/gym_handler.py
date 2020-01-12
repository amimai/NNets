import gym

class gym_trainer:
    def __init__(self, enviro="CartPole-v1"):
        self.env = gym.make(enviro)

    def interact(self,action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    # execute some random actions
    def random_run(self,show=False, limit=500):
        done = False
        hist = self.history_lib()
        i = 0
        observation = self.env.reset()
        while not done and i<limit:
            i += 1
            if show: self.env.render()
            action = self.env.action_space.sample()  # your agent here (this takes random actions)
            observation, reward, done, info = self.interact(action)
            hist['action'].append(action)
            hist['observation'].append(observation)
            hist['reward'].append(reward)
            hist['done'].append(done)
            hist['info'].append(info)

        observation = self.env.reset()
        self.env.close()
        return hist

    # make history library
    def history_lib(self):
        hist = {
            'action':[],
            'observation':[],
            'reward':[],
            'done':[],
            'info':[]
        }
        return hist


if __name__ == '__main__':
    trainer = gym_trainer("BipedalWalker-v2")
    hist = trainer.random_run(show=True)
    import numpy as np
    print(np.array(hist['action']).shape)