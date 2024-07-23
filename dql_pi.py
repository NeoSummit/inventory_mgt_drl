import time
from env_pi import Env # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Hyper parameters
memory_size = 1000
hidden_dim = 64
learning_rate = 0.001
batch_size = 64
gamma = 0.99
epsilon_0 = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

# Training parameters
num_episodes = 3000
evaluation_interval = 50
max_time_steps = 100

# Environment parameters
num_hospitals = 3
max_inventory = 50
max_demand = max_inventory // 10
max_demand_warehouse = max_inventory // 10
max_demand_local = max_inventory // 20
sub_action_size_1 = max_demand_warehouse + 1
sub_action_size_2 = max_demand_local + 1 
previous_demand_size = 8
env = Env(num_hospitals, max_inventory, max_demand, max_time_steps, previous_demand_size)
inventory_size = env.num_hospitals
action_dim = int(env.num_hospitals + env.num_hospitals * env.num_hospitals)

# Neural Network for Q-learning
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        # MLP for Inventory
        self.inventory_fc1 = nn.Linear(inventory_size, hidden_dim)
        self.inventory_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # LSTM for Demand
        self.lstm = nn.LSTM(previous_demand_size, hidden_size= hidden_dim , batch_first=True)
        self.lstm_fc = nn.Linear(hidden_dim, hidden_dim)

        input_dql_size = self.lstm_fc.out_features * num_hospitals + self.inventory_fc2.out_features
        self.fc1 = nn.Linear(input_dql_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Sorties pour chaque dimension de l'action
        self.fc_outs = nn.ModuleList([nn.Linear(hidden_dim, sub_action_size_1) for _ in range(env.num_hospitals)] + 
                                     [nn.Linear(hidden_dim, sub_action_size_2) for _ in range(action_dim - env.num_hospitals)]) 
       
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.float()
        inventory = x[:, :inventory_size]
        demand = x[:, inventory_size:]

        # Process inventory with MLP
        inventory_out = torch.relu(self.inventory_fc1(inventory))
        inventory_out = torch.relu(self.inventory_fc2(inventory_out))

        # Process demand with LSTM
        lstm_out_tensor = torch.zeros(batch_size, num_hospitals, self.lstm_fc.out_features)
        demand = demand.reshape(batch_size, num_hospitals, previous_demand_size)
        for i in range(1, num_hospitals):
            demand_i = demand[:, i, :] 
            lstm_out, _ = self.lstm(demand)
            lstm_out = lstm_out[:, -1, :]  # Get the last output from LSTM
            lstm_out = torch.relu(self.lstm_fc(lstm_out))
            lstm_out_tensor[:, i, :] = lstm_out

        flattened_lstm_out = lstm_out_tensor.reshape(batch_size, num_hospitals * self.lstm_fc.out_features)

        # Concatenate inventory and demand outputs
        combined = torch.cat((inventory_out, flattened_lstm_out), dim=1)

        # Further processing with fully connected layers
        combined_out = torch.relu(self.fc1(combined))
        combined_out = torch.relu(self.fc2(combined_out))

        # Return Q-values for each action dimension
        return [fc_out(combined_out) for fc_out in self.fc_outs] # liste de tenseur de taille (batch_size, max_demand_warehouse + 1) ou (batch_size, max_demand_local + 1)

class Train_DQL:
    def __init__(self):
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon_0

        self.indexAction_to_actionValue_1 = {i : i for i in range(sub_action_size_1)}
        self.indexAction_to_actionValue_2 = {i : i for i in range(sub_action_size_2)}
        self.actionValue_to_indexAction_1 = {i: i for i in range(sub_action_size_1)}
        self.actionValue_to_indexAction_2 = {i: i for i in range(sub_action_size_2)}
        
        self.policy_net = QNetwork().float()
        self.target_net = QNetwork().float()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.counter_inner_replay = 0
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
            action = [torch.argmax(q_value).item() for q_value in q_values]
            # torch.argmax(values) : Trouve l'indice du maximum dans chaque tenseur
            # liste d'entier où chaque entier représente l'action choisie pour une dimension spécifique.
        action = [self.indexAction_to_actionValue_1[action[i]] for i in range(env.num_hospitals)] + [self.indexAction_to_actionValue_2[action[i]] 
                                                                                               for i in range(env.num_hospitals, len(action))]
        return action
    
    def act_greedy(self, state):
        """
        Choose the greedy action based on the current state. Exploration is done using epsilon-greedy.

        Parameters:
        state (list): The current state of the environment.

        Returns:
        list: The greedy action to take based on the current state.
        """
        if np.random.rand() <= self.epsilon:
            action_1 = np.random.randint(0, max_demand_warehouse, size=env.num_hospitals)
            action_2 = env.create_random_correct_matrix(env.num_hospitals, max_demand_local).reshape(-1)
            return np.concatenate((action_1, action_2))
        return self.act(state)
    
    def replay(self):
        """
        Replay the memory and train the agent by sampling a random batch of experiences.

        Parameters:
        None

        Returns:
        None
        """
        if len(self.memory) < batch_size:
            return
        self.counter_inner_replay += 1
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Conversion des valeurs des actions aux indices correspondants
        actions = [[self.actionValue_to_indexAction_1[k] for k in actions[i][:env.num_hospitals]] + 
                            [self.actionValue_to_indexAction_2[k] for k in actions[i][env.num_hospitals:]] 
                            for i in range(batch_size)]

        

        # Convertir les listes de numpy.ndarrays en numpy.ndarray for speed
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        states = torch.tensor(states).float() # shape (batch_size, num_hospitals)
        actions = torch.tensor(actions).long() # shape (batch_size, num_hospitals + num_hospitals * num_hospitals), long for indexing
        rewards = torch.tensor(rewards).float() # shape (batch_size,)
        next_states = torch.tensor(next_states).float() # shape (batch_size, num_hospitals)
        dones = torch.tensor(dones).float() # shape (batch_size,)
        
        # Calculer les Q-valeurs actuelles et suivantes pour chaque dimension
        q_values = self.policy_net(states)  # liste de tenseur de taille (batch_size, max_order + 1)
        next_q_values = self.target_net(next_states) 
        expected_q_values_list = []
        q_values_list = []
        copy_actions = actions.clone()

        loss = 0
        for dim in range(action_dim):
            q_values_dim = q_values[dim].gather(1, actions[:, dim].unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values[dim].max(1)[0]
            expected_q_values = rewards + gamma * next_q_value * (1 - dones)
            
            loss += self.criterion(q_values_dim, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def evaluate_agent(self, episodes=10):
        """
        Evaluate the agent over multiple episodes.

        Parameters:
        env (Env): The environment to evaluate the agent on.

        Returns:
        float: The average reward over the evaluation episodes.
        """
        total_rewards = 0
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:  
                action = self.act(state)
                next_state, reward, done, *rest = env.step(action)
                episode_reward += reward
                state = next_state.copy()
            total_rewards += episode_reward
            # print(f"Action: {100 * env.count_correct_form / env.count_in_step_function}")
        avg_reward = total_rewards / episodes
        return avg_reward
    
    def train(self):
        """
        Train the agent on the environment. 

        Parameters:
        None

        Returns:
        None
        """
        for e in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.act_greedy(state) 
                next_state, reward, done, *rest = env.step(action)
                # list, dict et tableaux sont des objets mutables, donc on doit les copier pour juste enregistrer leurs valeurs
                # reward et done sont non mutables car entier et booléen respectivement
                self.remember(state.copy(), action.copy(), reward, next_state.copy(), done)
                state = next_state.copy()
                self.replay()
            
            if e % evaluation_interval == 0:
                self.update_target_network()
                avg_reward = self.evaluate_agent()
                print(f"Evaluation after episode {e}: Average Reward: {avg_reward:.2f}")

# Main
if __name__ == "__main__":
    agent = Train_DQL()
    agent.train()




