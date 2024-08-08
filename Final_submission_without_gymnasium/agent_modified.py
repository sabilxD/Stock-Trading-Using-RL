import importlib
import sys
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from utils import *

class Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.softmax(x)
        return x

class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        super().__init__(balance=balance)
        self.model_type = 'DQN'
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=100)
        self.buffer_size = 60

        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995 # decrease exploration rate as the agent becomes good at trading
        self.is_eval = is_eval
        self.model = torch.load('models/{}.pt'.format(model_name)) if is_eval else self.build_model()

        self.writer = SummaryWriter(log_dir='./logs/DQN_tensorboard')
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def build_model(self):
        return Model(self.state_dim, self.action_dim)

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0 #reset exploration rate

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            options = self.model(state).cpu().numpy()
        return np.argmax(options)

    def experience_replay(self):
        if len(self.memory) < self.buffer_size:
            return 0
        mini_batch = [self.memory[i] for i in range(len(self.memory) - self.buffer_size, len(self.memory))]
        
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q_values = self.model(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        loss = self.criterion(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.writer.add_scalar('Loss/train', loss.item())
        
        return loss.item()

model_name = 'DQN'
stock_name = input('Stock Name: ')
window_size = 3
num_episode = 2
initial_balance = 10000

stock_prices = stock_close_prices(stock_name)

trading_period = len(stock_prices) - 1
returns_across_episodes = []
num_experience_replay = 0
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}


agent = Agent(state_dim=window_size + 3, balance=initial_balance)

def hold(actions):
    #encourage selling for profit and liquidity
    next_probable_action = np.argsort(actions)[1]
    if next_probable_action == 2 and len(agent.inventory) > 0:
        max_profit = stock_prices[t] - min(agent.inventory)
        if max_profit > 0:
            sell(t)
            actions[next_probable_action] = 1 # reset this action's value to the highest
            return 'Hold', actions

def buy(t):
    if agent.balance > stock_prices[t]:
        agent.balance -= stock_prices[t]
        agent.inventory.append(stock_prices[t])
        return 'Buy: ${:.2f}'.format(stock_prices[t])

def sell(t):
    if len(agent.inventory) > 0:
        agent.balance += stock_prices[t]
        bought_price = agent.inventory.pop(0)
        profit = stock_prices[t] - bought_price
        global reward
        reward = profit
        return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)

start_time = time.time()

for episode in range(1 , num_episode+1):
    print(f'Episode : {episode}')
    
    agent.reset()
    state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))

    for t in range(1, trading_period + 1):
        if t % 100 == 0:
            print(f'\n-------------------Period: {t}/{trading_period}-------------------')

        reward = 0
        next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
        previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        
        actions = agent.model(torch.FloatTensor(state).unsqueeze(0)).cpu().detach().numpy()[0]
        action = agent.act(state)

        print('Step: {}\tHold signal: {:.4} \tBuy signal: {:.4} \tSell signal: {:.4}'.format(t, actions[0], actions[1], actions[2]))

        if action != np.argmax(actions): print(f"\t\t'{action_dict[action]}' is an exploration.")
        if action == 0: # hold
            execution_result = hold(actions)
        if action == 1: # buy
            execution_result = buy(t)      
        if action == 2: # sell
            execution_result = sell(t)        

        if isinstance(execution_result, tuple): # if execution_result is 'Hold'
                actions = execution_result[1]
                execution_result = execution_result[0]
                print(execution_result)
                
        current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        unrealized_profit = current_portfolio_value - agent.initial_portfolio_value
        reward += unrealized_profit

        agent.portfolio_values.append(current_portfolio_value)
        agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
        
        done = True if t == trading_period else False
        agent.remember(state, actions, reward, next_state, done)
        
        state = next_state

        #experience replay 
        if len(agent.memory) > agent.buffer_size:
            num_experience_replay += 1
            loss = agent.experience_replay()
            print('Episode: {}\tLoss: {:.2f}\tAction: {}\tReward: {:.2f}\tBalance: {:.2f}\tNumber of Stocks: {}'.format(episode, loss, action_dict[action], reward, agent.balance, len(agent.inventory)))
            agent.writer.add_scalar('Loss/train', loss, num_experience_replay)

        if done:
            portfolio_return = agent.portfolio_values[-1] - agent.initial_portfolio_value
            returns_across_episodes.append(portfolio_return)

        if episode % 2 == 0:
            if model_name == 'DQN':
                torch.save(agent.model.state_dict(), 'models/DQN_ep' + str(episode) + '.pt')
                
print('total training time: {0:.2f} min'.format((time.time() - start_time)/60))
plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)
