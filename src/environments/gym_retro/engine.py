import os
import subprocess

import numpy as np
import retro
import torch

from pathlib import Path
from shutil import copyfile

from core.mad_rl import MAD_RL


class Engine:
    def __init__(self, engine_parameters, agent_parameters, shared_agent):
        self.agent_parameters = agent_parameters
        self.engine_parameters = engine_parameters
        self.shared_agent = shared_agent

        self.game = os.getenv(
            'GAME', 'StreetFighterIISpecialChampionEdition-Genesis')

        self.game_folder = self.engine_parameters['game_folder']
        self.game_character = self.engine_parameters['character']
        self.load_state()

    def load_state(self):
        src_path = 'states/' + self.game_character + '.state'
        dst_path = self.game_folder + self.game_character + '.state'
        copyfile(src_path, dst_path)

        filename = self.game_folder + "metadata.json"
        with open(filename, "w") as f:
            f.write('{"default_state": "' + self.game_character + '"}')

    def train(self, seed):
        torch.manual_seed(seed)

        env = retro.make(game=self.game)
        self.agent_parameters['seed'] = seed
        self.agent = MAD_RL.agent(self.agent_parameters)
        self.agent.initialize_optimizer(self.shared_agent)
        self.agent.get_model().train()

        # First state
        env.reset()
        observation = env.render(mode='rgb_array')
        state = self.agent.get_state(None, observation)

        episodes = self.engine_parameters['episodes_training']
        for episode in range(int(episodes)):
            game_finished = False
            self.agent.start_episode(episode)
            while not game_finished:
                # Sync with the shared model
                self.agent.load_model(self.shared_agent.get_model())
                action, _ = self.agent.get_action(state)

                for step in range(self.engine_parameters['delay_frames']):
                    self.agent.start_step(step)

                    observation, reward, game_finished, info = env.step(
                        action)

                    if game_finished:
                        env.reset()
                        observation = env.render(mode='rgb_array')

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

        self.replay = self.engine_parameters['replay']
        if self.replay:
            env = retro.make(game=self.game, record="./replays/")
            episodes_testing = int(self.engine_parameters['episodes_testing'])
            episode_replay = int(episodes_testing / 10)
            if episodes_testing < 10:
                episode_replay = 1
        else:
            env = retro.make(game=self.game)

        self.agent = MAD_RL.agent(self.agent_parameters)
        self.agent.get_model().eval()

        # First state
        env.reset()
        observation = env.render(mode='rgb_array')
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

            self.agent.start_episode(episode)
            while not game_finished:
                action, value = self.agent.get_action(state)

                for step in range(self.engine_parameters['delay_frames']):
                    self.agent.start_step(step)

                    observation, reward, game_finished, info = env.step(
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

                        observation = env.render(mode='rgb_array')

                    next_state = self.agent.get_state(state, observation)
                    state = next_state

                    if game_finished:
                        env.reset()

                        if self.replay and episode % episode_replay == 0:
                            replay_path = (
                                "replays/" +
                                "StreetFighterIISpecialChampionEdition-" +
                                "Genesis-" +
                                self.game_character +
                                "-" + str(episode - 1).zfill(6) + ".bk2")  # index formed of 6 digits
                            subprocess.run(
                                ['python3.7', '-m',
                                 'retro.scripts.playback_movie',
                                 replay_path],
                                stdout=open(os.devnull, 'w'),
                                stderr=subprocess.STDOUT)
                        break

                self.agent.end_step(step)
            self.agent.end_episode(episode)

        # Remove .bk files
        folder = Path('replays/')
        files = folder.glob('*.bk2')
        for f in files:
            os.remove(f)
