
# DDQN+DQN+ Prioritized Replay
# Actor-Critic - tensorflow1.15
from collections import deque
import random
import gym
import numpy as np
from tensorflow.keras import models, layers, optimizers
from prioritized_memory import SumTree, Memory
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time as t


class DoubleDQN(object):
    def __init__(self,replay_size, memory_size=10000, prioritized=False):
        self.step = 0
        self.replay_size = replay_size 
        self.replay_queue = deque(maxlen=self.replay_size)
        self.memory_size = memory_size
        self.tau = 1e-2 #MountainCar-v0
        self.model = self.create_model()
        self.prioritized = prioritized
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)



    def create_model(self):
        
        STATE_DIM, ACTION_DIM = 2, 3
        model = models.Sequential([
            layers.Dense(100, input_dim=STATE_DIM, activation='relu'),
            layers.Dense(ACTION_DIM, activation="linear")
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.001))
        return model

    def act(self, s, epsilon=0.1):
        
        #
        if np.random.uniform() < epsilon - self.step * 0.0002:
            return np.random.choice([0, 1, 2])
        return np.argmax(self.model.predict(np.array([s]))[0])

    def save_model(self, file_path='MountainCar-v0-Ddqn.h5'):
        print('model saved')
        self.model.save(file_path)

        
    def store_transition(self, s, a, r, s_, dd):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r], s_, dd)) # transition -> 7x1
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:
            #self.replay_queue.append((s, [a, r], s_, dd))
            transition = np.hstack((s, [a, r], s_, dd)) # transition -> 7x1
            self.replay_queue.append(transition)

    def expReplay(self, batch_size=64, lr=1, factor=0.95):
                
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(batch_size)
        else:
            batch_memory = random.sample(self.replay_queue, batch_size)
        
        s_batch = np.array([replay[[0,1]] for replay in batch_memory])
        a = np.array([replay[[2]] for replay in batch_memory])
        r = np.array([replay[[3]] for replay in batch_memory])
        next_s_batch = np.array([replay[[4,5]] for replay in batch_memory])
        d = np.array([replay[[6]] for replay in batch_memory])

        Q = self.model.predict(s_batch)
        Q_next = self.model.predict(next_s_batch)
        Q_targ = self.target_model.predict(next_s_batch)

        #update Q value
        td_error = np.zeros((d.shape[0],), dtype=float)
        for i in range(d.shape[0]):
            old_q = Q[i, int(a[i])]
            if int(d[i]) == 1:
                Q[i, int(a[i])] = r[i]
            else:
                next_best_action = np.argmax(Q_next[i,:])
                Q[i, int(a[i])] = r[i] + factor * Q_targ[i, next_best_action]


            if self.prioritized:
                td_error[i] = abs(old_q - Q[i, int(a[i])])


        if self.prioritized:
            self.memory.batch_update(tree_idx, td_error)
        
        
        self.model.fit(s_batch, Q, verbose=0)

    def transfer_weights(self):
        """ Transfer Weights from Model to Target at rate Tau
        """
        W = self.model.get_weights()
        tgt_W = self.target_model.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.target_model.set_weights(tgt_W) 

class DQN(object):
    def __init__(self,replay_size, memory_size=10000, prioritized=False):
        self.step = 0
        self.replay_size = replay_size 
        self.replay_queue = deque(maxlen=self.replay_size)
        self.memory_size = memory_size
        self.model = self.create_model()
        self.prioritized = prioritized
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)


    def create_model(self):
       
        STATE_DIM, ACTION_DIM = 2, 3
        model = models.Sequential([
            layers.Dense(100, input_dim=STATE_DIM, activation='relu'),
            layers.Dense(ACTION_DIM, activation="linear")
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.001))
        return model

    def act(self, s, epsilon=0.1):

        if np.random.uniform() < epsilon - self.step * 0.0002:
            return np.random.choice([0, 1, 2])
        return np.argmax(self.model.predict(np.array([s]))[0])

    def save_model(self, file_path='MountainCar-v0-dqn.h5'):
        print('model saved')
        self.model.save(file_path)

    def remember(self, s, a, next_s, reward):
        #
        if next_s[0] >= 0.4:
            reward += 1
        self.replay_queue.append((s, a, next_s, reward))
        
    def store_transition(self, s, a, r, s_, dd):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r], s_, dd)) # transition -> 7x1
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:
            #self.replay_queue.append((s, [a, r], s_, dd))
            transition = np.hstack((s, [a, r], s_, dd)) # transition -> 7x1
            self.replay_queue.append(transition)

    def expReplay(self, batch_size=64, lr=1, factor=0.95):
                
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(batch_size)
        else:
            batch_memory = random.sample(self.replay_queue, batch_size)
        
        s_batch = np.array([replay[[0,1]] for replay in batch_memory])
        a = np.array([replay[[2]] for replay in batch_memory])
        r = np.array([replay[[3]] for replay in batch_memory])
        next_s_batch = np.array([replay[[4,5]] for replay in batch_memory])
        d = np.array([replay[[6]] for replay in batch_memory])

        Q = self.model.predict(s_batch)
        Q_next = self.model.predict(next_s_batch)

        td_error = np.zeros((d.shape[0],), dtype=float)
        for i in range(d.shape[0]):
            old_q = Q[i, int(a[i])]
            if int(d[i]) == 1:
                Q[i, int(a[i])] = r[i]
            else:
                next_best_action = np.argmax(Q_next[i,:])
                Q[i, int(a[i])] = r[i] + factor * Q_next[i, next_best_action]
                #q_target = r[i] + factor * Q_next[i, next_best_action]


            if self.prioritized:
                td_error[i] = abs(old_q - Q[i, int(a[i])])
                #td_error[i] = abs(q_target - old_q)

        if self.prioritized:
            self.memory.batch_update(tree_idx, td_error)

        self.model.fit(s_batch, Q, verbose=0)
        
class A2CAgent:
    
    def __init__(self, replay_size,memory_size=10000, prioritized=False, load_models = False, actor_model_file = '', critic_model_file = '', is_eval = False):
        self.state_size = 2
        self.action_size = 3
        self.step = 0
        self.replay_size = replay_size  
        self.replay_queue = deque(maxlen=self.replay_size)
        self.memory_size = memory_size
        self.prioritized = prioritized
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        

        # Hyper parameters for learning
        self.value_size = 1
        self.layer_size = 16
        self.discount_factor = 0.99
        self.actor_learning_rate = 0.0005
        self.critic_learning_rate = 0.005
        self.is_eval = is_eval

        # Create actor and critic neural networks
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        #self.actor.summary()


        if load_models:
            if actor_model_file:
                self.actor.load_weights(actor_model_file)
            if critic_model_file:
                self.critic.load_weights(critic_model_file)
    
    # The actor takes a state and outputs probabilities of each possible action
    def build_actor(self):
        
        layer1 = Dense(self.layer_size, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform')
        layer2 = Dense(self.layer_size, input_dim=self.layer_size, activation='relu',
                        kernel_initializer='he_uniform')
        # Use softmax activation so that the sum of probabilities of the actions becomes 1
        layer3 = Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform') # self.action_size = 2
        
        actor = Sequential(layers = [layer1, layer2, layer3]) 
        
        # Print a summary of the network
        actor.summary()
        
        # We use categorical crossentropy loss since we have a probability distribution
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_learning_rate))
        return actor

    # The critic takes a state and outputs the predicted value of the state
    def build_critic(self):
        
        layer1 = Dense(self.layer_size, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform')
        layer2 = Dense(self.layer_size, input_dim=self.layer_size, activation='relu',
                         kernel_initializer='he_uniform')
        layer3 = Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform') # self.value_size = 1
        
        critic = Sequential(layers = [layer1, layer2, layer3])
        
        # Print a summary of the network
        critic.summary()
        
        critic.compile(loss='mean_squared_error', optimizer=Adam(lr=self.critic_learning_rate))
        return critic    

    def act(self, state):
        # Get probabilities for each action
        policy = self.actor.predict(np.array([state]), batch_size=1).flatten()

        # Randomly choose an action
        if not self.is_eval:
            return np.random.choice(self.action_size, 1, p=policy).take(0)
        else:
            return np.argmax(policy) # 20191117- for evaluation
    
        
    def store_transition(self, s, a, r, s_, dd):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r], s_, dd)) 
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:
            #self.replay_queue.append((s, [a, r], s_, dd))
            transition = np.hstack((s, [a, r], s_, dd)) 
            self.replay_queue.append(transition)
            
    def expReplay(self, batch_size=64, lr=1, factor=0.95):
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(batch_size)
        else:
            batch_memory = random.sample(self.replay_queue, batch_size)

        s_prevBatch = np.array([replay[[0,1]] for replay in batch_memory])
        a = np.array([replay[[2]] for replay in batch_memory])
        r = np.array([replay[[3]] for replay in batch_memory])
        s_currBatch = np.array([replay[[4,5]] for replay in batch_memory])
        d = np.array([replay[[6]] for replay in batch_memory])

        td_error = np.zeros((d.shape[0],), dtype=float)   
        for i in range(d.shape[0]):
            q_prev = self.critic.predict(np.array([s_prevBatch[i,:]]))
            q_curr = self.critic.predict(np.array([s_currBatch[i,:]]))
            if int(d[i]) == 1:
                q_curr = r[i]
            q_realP = r[i] + factor * q_curr
            advantages = np.zeros((1, self.action_size))
            advantages[0, int(a[i])] = q_realP - q_prev

            if self.prioritized:
                td_error[i] = abs(advantages[0, int(a[i])])
                
            self.actor.fit(np.array([s_prevBatch[i,:]]), advantages, epochs=1, verbose=0)
            self.critic.fit(np.array([s_prevBatch[i,:]]), reshape(q_realP), epochs=1, verbose=0)
            
        if self.prioritized:
            self.memory.batch_update(tree_idx, td_error)

# Reshape array for input into keras       
def reshape(state):
    return np.reshape(state, (1, -1))


env = gym.make('MountainCar-v0')
env = env.unwrapped
episodes = 100  #
replay_size = 2000
score_list = []
train_list = []

#agent = DQN(replay_size, memory_size=10000, prioritized=True)
#agent = DoubleDQN(replay_size, memory_size=10000, prioritized=True)
agent = A2CAgent(replay_size, memory_size=10000, prioritized=True)


print('Start: ', t.strftime("%H:%M:%S", t.gmtime()))
for i in range(episodes):
    s = env.reset()
    score = 0
    while True:
        a = agent.act(s)
        next_s, reward, done, _ = env.step(a)
        if done:  reward = 10
        agent.store_transition(s, a, reward, next_s, done)

     
        if agent.step > replay_size:
            agent.expReplay(batch_size=64)
            #agent.transfer_weights()
        #agent.train()
        score += reward
        s = next_s
        agent.step += 1
        if agent.step % 1000 == 0:
            print('Finished: ',agent.step,' step')
        if done:
            score_list.append(score)
            train_list.append(agent.step)
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break

    if np.mean(score_list[-10:]) > -120:
        agent.save_model()
        break
env.close()
print('episode:', i, 'lastTEN:', np.mean(score_list[-10:]))
import matplotlib.pyplot as plt
print('Stop: ', t.strftime("%H:%M:%S", t.gmtime()))
plt.plot(train_list, color='green')
plt.show()
