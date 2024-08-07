import json
import argparse
import torch

from envs.frozen_lake import FrozenLake
from agents.EL.q_learning_agent import QLearningAgent
from agents.EL.actor_critic_agent import ActorCriticAgent
from agents.EL.sarsa_agent import SarsaAgent
from agents.MM.dyna_agent import DynaAgent

from envs.catcher import Catcher
from agents.FN.deep_q_network_agent import DeepQNetworkAgent
from agents.FN.advantage_actor_critic_agent import AdvantageActorCriticAgent

from envs.cart_pole import CartPole
from agents.FN.value_function_agent import ValueFunctionAgent
from agents.FN.policy_gradient_agent import PolicyGradientAgent
from agents.FN.advantage_actor_critic_agent import AdvantageActorCriticNetAgent

def select_env(env_type):
    if env_type == 'FrozenLake':
        return FrozenLake()
    elif env_type == 'Catcher':
        return Catcher()
    elif env_type == 'CartPole':
        return CartPole()
    else:
        raise ValueError(f'Unknown environment type: {env_type}')

def select_agent(agent_type, env, device):
    if agent_type == 'QLearningAgent':
        return QLearningAgent(env.actions)
    elif agent_type == 'ActorCriticAgent':
        return ActorCriticAgent(env.actions)
    elif agent_type == 'SarsaAgent':
        return SarsaAgent(env.actions)
    elif agent_type == 'DeepQNetworkAgent':
        return DeepQNetworkAgent(actions=env.actions, device=device)
    elif agent_type == 'ValueFunctionAgent':
        return ValueFunctionAgent(env.actions)
    elif agent_type == 'PolicyGradientAgent':
        return PolicyGradientAgent(env.actions)
    elif agent_type == 'AdvantageActorCriticNetAgent':
        return AdvantageActorCriticNetAgent(actions=env.actions)
    elif agent_type == 'AdvantageActorCriticAgent':
        return AdvantageActorCriticAgent(actions=env.actions, device=device)
    elif agent_type == 'DynaAgent':
        return DynaAgent(actions=env.actions)
    else:
        raise ValueError(f'Unknown agent type: {agent_type}')

def load_agent(agent_type, model_path, env):
    if agent_type == 'DeepQNetworkAgent':
        return DeepQNetworkAgent.load(env.actions, model_path)
    elif agent_type == 'ValueFunctionAgent':
        return ValueFunctionAgent.load(env.actions, model_path)
    elif agent_type == 'PolicyGradientAgent':
        return PolicyGradientAgent.load(env.actions, model_path)
    elif agent_type == 'AdvantageActorCriticNetAgent':
        return AdvantageActorCriticNetAgent.load(actions=env.actions, model_path=model_path)
    elif agent_type == 'AdvantageActorCriticAgent':
        return AdvantageActorCriticAgent.load(actions=env.actions, model_path=model_path, device="mps")
    else:
        raise ValueError(f'Unknown agent type: {agent_type}')

def start_server(env_type, agent_type, device):
    env = select_env(env_type)
    agent = select_agent(agent_type, env, device)

    for e in range(agent.episode_count):
        done = False
        current_world_time = -1

        while not done:
            addr, state, reward, done, world_time = env.receive()

            if world_time <= current_world_time:
                continue
            else:
                current_world_time = world_time

            action = agent.policy(state)
            agent.learn(state, action, reward, done)

            env.send(addr, action, current_world_time)

        if e != 0 and agent.training and e % agent.report_interval == 0:
            agent.show_reward_log(episode=e)

        if agent.save_interval and agent.training and e % agent.save_interval == 0:
            agent.save(f'./models/{agent_type}_{env_type}_{e}.pth')
    
    print('Finished training!')

def play(env_type, model_type, model_path, episode_cnt=100, report_interval=50):
    env = select_env(env_type)
    agent = load_agent(model_type, model_path, env)

    for e in range(episode_cnt):
        done = False
        current_world_time = -1

        while not done:
            addr, state, reward, done, world_time = env.receive()

            if world_time <= current_world_time:
                continue
            else:
                current_world_time = world_time

            action = agent.policy(state)

            env.send(addr, action, current_world_time)

        if e != 0 and e % report_interval == 0:
            agent.show_reward_log(episode=e)

    print('Finished play!')

# default_env, default_agent, default_model_path = 'FrozenLake', 'QLearningAgent', None
# default_env, default_agent, default_model_path = 'FrozenLake', 'ActorCriticAgent', None
# default_env, default_agent, default_model_path = 'FrozenLake', 'SarsaAgent', None
# default_env, default_agent, default_model_path = 'FrozenLake', 'DynaAgent', None
# default_env, default_agent, default_model_path = 'Catcher', 'DeepQNetworkAgent', None
# default_env, default_agent, default_model_path = 'Catcher', 'AdvantageActorCriticAgent', None
# default_env, default_agent, default_model_path = 'CartPole', 'ValueFunctionAgent', None
# default_env, default_agent, default_model_path = 'CartPole', 'PolicyGradientAgent', None
default_env, default_agent, default_model_path = 'CartPole', 'AdvantageActorCriticNetAgent', None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start server with specified environment and agent.")
    parser.add_argument('--env', type=str, default=default_env, help='Type of environment')
    parser.add_argument('--agent', type=str, default=default_agent, help='Type of agent')
    parser.add_argument('--model_path', type=str, default=default_model_path, help='model path to load')

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    args = parser.parse_args()
    isPlay = False if args.model_path == None else True
    print(f'Play: {isPlay}, Server started, waiting for data...(Environment: {args.env}, Agent: {args.agent}), device: {device}')
    
    if isPlay: 
        play(env_type=args.env, model_type=args.agent, model_path=args.model_path)
    else:
        start_server(env_type=args.env, agent_type=args.agent, device=device)
    