import os

import torch
import torch.multiprocessing as mp

from core.mad_rl import MAD_RL


# Do not use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''


ENGINE_PARAMETERS = {
    'episodes_training': os.getenv("EPISODES_TRAINING", 10),
    'episodes_testing': os.getenv("EPISODES_TESTING", 5),
    'num_processes': os.getenv("NUM_PROCESSES", 5),
    'output_models_path': os.getenv("OUTPUT_MODELS_PATH", "models"),
    'delay_frames': os.getenv("DELAY_FRAMES", 50),
    'replay': os.getenv("REPLAY", False),
    'module': os.getenv("ENGINE_MODULE", "src.environments.gym_retro.engine"),
    'class': os.getenv("ENGINE_CLASS", "Engine")
}

print("Engine parameters: ", ENGINE_PARAMETERS)

AGENT_PARAMETERS = {
    'frames': 16,
    'cnn_channels': 32,
    'n_outputs': 49,
    'screen_height': 256,
    'screen_width': 200,
    'width': 80,
    'height': 80,
    'start_from_model': os.getenv("START_FROM_MODEL", "models/sf2_a3c.pth"),
    'module': os.getenv("AGENT_MODULE",
                        "src.environments.gym_retro.my_agent.agent"),
    "class": os.getenv("AGENT_CLASS", "Agent")
}

print("Agent parameters: ", AGENT_PARAMETERS)

if __name__ == '__main__':
    torch.manual_seed(42)

    shared_agent = MAD_RL.agent(AGENT_PARAMETERS)
    shared_agent.get_model().share_memory()

    num_processes = int(ENGINE_PARAMETERS['num_processes'])
    processes = []

    engine = MAD_RL.engine(ENGINE_PARAMETERS, AGENT_PARAMETERS, shared_agent)

    # Test process
    p = mp.Process(target=engine.test, args=[num_processes])
    p.start()
    processes.append(p)

    # Training processes
    for rank in range(0, num_processes - 1):
        p = mp.Process(target=engine.train, args=[rank])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
