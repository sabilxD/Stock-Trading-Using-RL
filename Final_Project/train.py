import importlib
import sys
import time
import numpy as np

from utils import *

import random
from collections import deque

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


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
        self.model = load_model('models/{}.h5'.format(model_name)) if is_eval else self.model()

        self.tensorboard = TensorBoard(log_dir='./logs/DQN_tensorboard', update_freq=90)
        self.tensorboard.set_model(self.model)


    def model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_dim, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        return model

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0 #reset exploration rate

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        options = self.model.predict(state)
        return np.argmax(options[0])


    def experience_replay(self):
        # retrieve recent buffer_size long memory
        mini_batch = [self.memory[i] for i in range(len(self.memory) - self.buffer_size + 1, len(self.memory))]

        for state, actions, reward, next_state, done in mini_batch:
            if not done:
                Q_target_value = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                Q_target_value = reward
            next_actions = self.model.predict(state)
            next_actions[0][np.argmax(actions)] = Q_target_value
            history = self.model.fit(state, next_actions, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0]





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


# model = importlib.import_module(f'agents.{model_name}')
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
    
    agent.reset() #see
    state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))

    for t in range(1, trading_period + 1):
        if t % 100 == 0:
            print(f'\n-------------------Period: {t}/{trading_period}-------------------')

        reward = 0
        next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
        previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        
        actions = agent.model.predict(state)[0]
        action = agent.act(state)

        print('Step: {}\tHold signal: {:.4} \tBuy signal: {:.4} \tSell signal: {:.4}'.format(t, actions[0], actions[1], actions[2]))

        if action != np.argmax(actions): print(f"'{action_dict[action]}' is an exploration.")
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
        if len(agent.memory) > agent.buffer_size: #see
            num_experience_replay += 1
            loss = agent.experience_replay()
            print('Episode: {}\tLoss: {:.2f}\tAction: {}\tReward: {:.2f}\tBalance: {:.2f}\tNumber of Stocks: {}'.format(episode, loss, action_dict[action], reward, agent.balance, len(agent.inventory)))
            agent.tensorboard.on_batch_end(num_experience_replay, {'loss': loss, 'portfolio value': current_portfolio_value}) 

        if done:
            portfolio_return = agent.portfolio_values[-1] - agent.initial_portfolio_value
            returns_across_episodes.append(portfolio_return)

        if episode % 2 == 0:
            if model_name == 'DQN':
                agent.model.save('models/DQN_ep' + str(episode) + '.h5')
                
print('total training time: {0:.2f} min'.format((time.time() - start_time)/60))
plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)


                        
