import yaml

import learning.ppo_agent as ppo_agent
import learning.awr_agent as awr_agent
import learning.amp_agent as amp_agent
import learning.ase_agent as ase_agent
import learning.add_agent as add_agent
from util.logger import Logger

def build_agent(agent_file, env, device):
    agent_config = load_agent_file(agent_file)
    
    agent_name = agent_config["agent_name"]
    Logger.print("Building {} agent".format(agent_name))

    if (agent_name == ppo_agent.PPOAgent.NAME):
        agent = ppo_agent.PPOAgent(config=agent_config, env=env, device=device)
    elif (agent_name == awr_agent.AWRAgent.NAME):
        agent = awr_agent.AWRAgent(config=agent_config, env=env, device=device)
    elif (agent_name == amp_agent.AMPAgent.NAME):
        agent = amp_agent.AMPAgent(config=agent_config, env=env, device=device)
    elif (agent_name == ase_agent.ASEAgent.NAME):
        agent = ase_agent.ASEAgent(config=agent_config, env=env, device=device)
    elif (agent_name == add_agent.ADDAgent.NAME):
        agent = add_agent.ADDAgent(config=agent_config, env=env, device=device)
    else:
        assert(False), "Unsupported agent: {}".format(agent_name)

    num_params = agent.calc_num_params()
    Logger.print("Total parameter count: {}".format(num_params))

    return agent

def load_agent_file(file):
    with open(file, "r") as stream:
        agent_config = yaml.safe_load(stream)
    return agent_config
