# MAD_RL_ SFII CHALLENGE ENGINE

This is an engine in a docker for the Street Fighter Challenge that MAD_RL_ is launching in order to make a Reinforcement Learning competition.

## Basic Usage

Clone the [challenge repository](https://github.com/mad-rl/sfii-challenge) and use the following command in order to use it.

```
docker run -v $PWD/sfii_agent_base:/sfii-challenge/mad-rl-framework/src/sfii_agent -v $PWD/roms:/sfii-challenge/roms/ sfii_challenge_engine
```
