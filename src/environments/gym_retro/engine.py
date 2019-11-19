import os
import subprocess

import numpy as np
import retro
import torch

from shutil import copyfile

from core.mad_rl import MAD_RL


class Engine:
    def __init__(self, engine_parameters, agent_parameters, shared_agent):
        self.agent_parameters = agent_parameters
        self.engine_parameters = engine_parameters
        self.shared_agent = shared_agent

        game = os.getenv(
            'GAME', 'StreetFighterIISpecialChampionEdition-Genesis')

        self.game_folder = self.engine_parameters['game_folder']
        self.game_character = self.engine_parameters['character']
        self.load_state()

        self.replay = self.engine_parameters['replay']
        testing_agent = int(self.engine_parameters['num_processes'])
        if self.replay and testing_agent:
            self.env = retro.make(game=game, record="./replays/")
        else:
            self.env = retro.make(game=game)

    def load_state(self):
        src_path = 'states/' + self.game_character + '.state'
        dst_path = self.game_folder + self.game_character + '.state'
        copyfile(src_path, dst_path)

        filename = self.game_folder + "metadata.json"
        with open(filename, "w") as f:
            f.write('{"default_state": "' + self.game_character + '"}')

    def train(self, seed):
        torch.manual_seed(seed)
        self.agent = MAD_RL.agent(self.agent_parameters)
        self.agent.initialize_optimizer(self.shared_agent)
        self.agent.get_model().train()

        # First state
        observation = self.env.render(mode='rgb_array')
        state = self.agent.get_state(None, observation)

        episodes = self.engine_parameters['episodes_training']
        for episode in range(int(episodes)):
            game_finished = False
            self.env.reset()

            self.agent.start_episode(episode)
            while not game_finished:
                # Sync with the shared model
                self.agent.load_model(self.shared_agent.get_model())
                action, _ = self.agent.get_action(state)

                for step in range(self.engine_parameters['delay_frames']):
                    self.agent.start_step(step)

                    observation, reward, game_finished, info = self.env.step(
                        action)

                    if game_finished:
                        self.env.reset()
                        observation = self.env.render(mode='rgb_array')

                    next_state = self.agent.get_state(state, observation)
                    self.agent.add_experience(
                        state, action, reward, next_state, info)

                    state = next_state

                    if game_finished:
                        break

                    self.agent.end_step(step)

                self.agent.train(game_finished, self.shared_agent)
                self.agent.end_episode(episode)

    def test(self, seed):
        torch.manual_seed(seed)
        self.agent = MAD_RL.agent(self.agent_parameters)
        self.agent.initialize_optimizer(self.shared_agent)
        self.agent.get_model().eval()

        # First state
        observation = self.env.render(mode='rgb_array')
        state = self.agent.get_state(None, observation)

        rewards = []
        values = []
        reward_sum = 0
        episode = 0
        episode_steps = 0

        episodes = self.engine_parameters['episodes_testing']
        for episode in range(int(episodes)):
            # Sync with the shared model
            self.agent.load_model(self.shared_agent.get_model())

            game_finished = False
            self.env.reset()

            self.agent.start_episode(episode)
            while not game_finished:
                action, value = self.agent.get_action(state)

                for step in range(self.engine_parameters['delay_frames']):
                    self.agent.start_step(step)

                    observation, reward, game_finished, info = self.env.step(
                        action)

                    rewards.append(reward)
                    values.append(value.data[0, 0])
                    episode_steps += 1
                    reward_sum += reward

                    if game_finished:
                        episode += 1

                        dict_info = {}
                        dict_info["episode"] = episode
                        dict_info["episode_steps"] = (str(episode_steps))
                        dict_info["episode_reward"] = reward_sum
                        dict_info["reward_min"] = np.min(rewards)
                        dict_info["reward_max"] = np.max(rewards)
                        dict_info["reward_avg_by_step"] = np.mean(rewards)
                        dict_info["value_avg"] = np.mean(values)

                        print(dict_info)

                        rewards = []
                        reward_sum = 0
                        episode_steps = 0

                        self.env.reset()
                        observation = self.env.render(mode='rgb_array')

                    next_state = self.agent.get_state(state, observation)
                    state = next_state

                    if game_finished:
                        break

                    self.agent.end_step(step)
                self.agent.end_episode(episode)

        if self.replay:
            replay_path = (
                "replays/StreetFighterIISpecialChampionEdition-Genesis-" +
                self.game_character +
                "-000000.bk2")
            subprocess.run(
                ['python3.7', '-m', 'retro.scripts.playback_movie',
                 replay_path])
