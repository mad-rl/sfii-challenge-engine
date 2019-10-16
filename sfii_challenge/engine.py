import os
import retro

from core.mad_rl import MAD_RL

class Engine:

    def __init__(self):
        self.config = MAD_RL.config()
        self.env = retro.make(os.getenv('GAME', 'StreetFighterIISpecialChampionEdition-Genesis'))
        self.agent = MAD_RL.agent()

    def train(self):

        for episode in range(self.config["episodes"]):
            game_finished = False
            self.env.reset()

            self.agent.start_episode(episode)

            step = 0
            while not game_finished:
                for step in range(self.config["steps_per_episode"]):
                    self.agent.start_step(step)

                    observation = self.env.render(mode='rgb_array')
                    action = self.agent.get_action(observation)

                    next_observation, reward, game_finished, info = self.env.step(action)

                    self.agent.add_experience(observation, reward, action, next_observation, info=info)
                    self.agent.end_step(step)

                    step = step + 1

                    if game_finished:
                        break

                self.agent.train()

            self.agent.end_episode(episode)

    def test(self):

        for episode in range(self.config["episodes"]):
            game_finished = False
            self.env.reset()

            self.agent.start_episode(episode)

            step = 0
            while not game_finished:
                self.agent.start_step(step)

                observation = self.env.render(mode='rgb_array')
                action = self.agent.get_action(observation)

                next_observation, reward, game_finished, _info = self.env.step(action)

                self.agent.end_step(step)

                step = step + 1

                if game_finished:
                    break

            self.agent.end_episode(episode)
