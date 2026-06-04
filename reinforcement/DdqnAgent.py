from tensorflow import keras
import numpy as np
import random
from Util import Util
from tensorflow.keras.callbacks import TensorBoard
import time
from collections import deque

from tensorflow.python.keras.saving.save import load_model


class DoubleDeepQNetwork():

    # Initializes the Double Deep Q-Network (DDQN) agent with the specified configuration and environment.
    # Sets up hyperparameters like epsilon for exploration, gamma for discounting, and batch size for training.
    # Initializes the neural network models for both the primary and target Q-networks.
    # If prefilled actions are provided, they are loaded for controlled execution.
    #
    # FIX: learning_rate and gamma are now properly forwarded from Main.py (including
    #      the values suggested by Optuna in tune mode). Previously lr was accepted as
    #      a parameter but the default 0.01 was always used because Main.py never passed
    #      a value — meaning Optuna's lr suggestions had zero effect on training.
    def __init__(self, config, env, http_client,
                 is_controlled, is_prefilled_actions,
                 gamma=0.95, learning_rate=0.01, batch_size=128):

        self.ACTIONS             = None
        self.config              = config
        self.env                 = env
        self.http_client         = http_client
        self.is_controlled       = is_controlled
        self.is_prefilled_actions = is_prefilled_actions
        self.nS                  = int(self.env.INPUT_SHAPE)
        self.nA                  = int(self.env.OUTPUT_SHAPE)
        self.gamma               = gamma
        self.epsilon             = 1.0
        self.epsilon_min         = 0.01
        self.epsilon_decay       = config.epsilon_decay
        self.learning_rate       = learning_rate
        self.batch_size          = batch_size
        self.memory_size         = 100000  # buffer generoso per server con RAM abbondante (Xeon E5-2660 v3)
        self.memory              = deque(maxlen=self.memory_size)
        self.update_target_each  = 100  # steps

        self.model        = self.build_model()
        self.model_target = self.build_model()
        self.update_target_from_model()

        self.loss         = []
        self.episode_loss = []

        import os as _tb_os
        _trial_id = _tb_os.environ.get("TRIAL_ID", "default").replace("_", "")
        self.tensorboard = TensorBoard(
            log_dir=f"logs/{_trial_id}_{int(time.time())}",
            histogram_freq=0,
            write_graph=False
        )
        self.tensorboard.set_model(self.model)

        if is_prefilled_actions:
            self.prefilled_actions = self.read_lines_from_file(
                config.prefilled_actions_file
            )
            print("<------> Actions are prefilled:")
            for a in self.prefilled_actions:
                print(f"---------> {a}")

    # Reads a list of actions from a specified file.
    # Each action is expected to be defined on a separate line; empty lines are ignored.
    def read_lines_from_file(self, file_path):
        with open(file_path, 'r') as file:
            return [line.strip() for line in file if line.strip()]

    # Constructs the neural network model for the DDQN agent using Keras.
    # The model consists of multiple dense layers with ReLU and Sigmoid activations.
    # The final layer uses linear activation to represent Q-values.
    # FIX: Adam's 'lr' keyword was deprecated in newer Keras versions — replaced
    #      with 'learning_rate' to avoid the deprecation warning.
    def build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(self.nS,
                               input_dim=self.nS, activation='relu'),
            keras.layers.Dense(2 * self.nS,
                               activation='relu'),
            keras.layers.Dense(4 * (self.nS + self.nA) + 2,
                               activation='relu'),
            keras.layers.Dense(2 * self.nS + 2,
                               activation='sigmoid'),
            keras.layers.Dense(self.nA,
                               activation='linear'),
        ])
        model.compile(
            loss='mean_squared_error',
            # FIX: 'lr' is deprecated since Keras 2.x — use 'learning_rate'
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model

    # Copies the weights from the primary model to the target model.
    def update_target_from_model(self):
        self.model_target.set_weights(self.model.get_weights())
        print('<------> Target Model Updated')

    # Selects an action based on the current state and epsilon-greedy policy.
    # Returns the selected action and a flag indicating whether it was predicted.
    def action(self, step, state):
        print('<--------------------------------------------------------->\n')
        if self.is_controlled:
            return self.do_controlled_prompt()
        if self.is_prefilled_actions:
            return self.do_action_from_prefilled(step)

        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.nA)
            print(f'<------> Taking action randomly: {action}')
            return action, False

        action_vals = self.model.predict(
            np.reshape(state, [1, self.nS]), verbose=0
        )
        action = np.argmax(action_vals[0])
        print(f'<------> Taking action with predict: {action}')
        return action, True

    def do_controlled_prompt(self):
        action = -1
        while action < 0:
            print("Available actions:")
            for i, ACTION in enumerate(self.ACTIONS):
                parts = ACTION.split(':')
                if parts[0] == "redirect":
                    info = self.get_controlled_redirect_action_with_dist(parts)
                    print(f" - {i}: {ACTION} (redirect from {info[0]} to {info[1]})")
                else:
                    print(f" - {i}: {ACTION}")
            action = int(input("Enter action index: "))
            if 0 <= action < self.nA:
                return action, True
            print(f'<------> Action ({action}) not recognised, try again.')
            action = -1

    # Executes a predefined action from the prefilled list.
    def do_action_from_prefilled(self, step):
        custom_action = self.get_step_index_action_or_nothing(step)
        action = -1
        if custom_action == Util.nothing_action():
            action = len(self.ACTIONS) - 1
        elif custom_action.startswith("bw"):
            action = self.ACTIONS.index(custom_action)
        elif custom_action.startswith("redirect"):
            parts  = custom_action.split(':')
            parsed = Util.redirect_action(parts[1], parts[3])
            print(f'---------> {custom_action} -> {parsed}')
            action = self.ACTIONS.index(parsed)
        if action == -1:
            action = len(self.ACTIONS) - 1
        print(f'<------> Taking action with prefilled: {action}')
        return action, True

    def get_step_index_action_or_nothing(self, step):
        idx = step - 1
        if idx < len(self.prefilled_actions):
            return self.prefilled_actions[idx]
        return Util.nothing_action()

    def get_controlled_redirect_action_with_dist(self, parts):
        return parts[1], parts[3]   # host_name, dst_switch

    def test_action(self, state):
        action_vals = self.model.predict(
            np.reshape(state, [1, self.nS]), verbose=0
        )
        return np.argmax(action_vals[0])

    # Stores an experience tuple into the replay memory.
    def store(self, state, action, reward, nstate, done):
        self.memory.append((state, action, reward, nstate, done))

    # Trains the model using a random batch of experiences (Double DQN).
    def experience_replay(self, batch_size):
        minibatch       = random.sample(self.memory, batch_size)
        minibatch_array = np.array(minibatch, dtype=object)

        st  = np.stack([minibatch_array[i, 0] for i in range(len(minibatch_array))]).reshape(batch_size, self.nS)
        nst = np.stack([minibatch_array[i, 3] for i in range(len(minibatch_array))]).reshape(batch_size, self.nS)

        st_predict          = self.model.predict(st,  verbose=0)
        nst_predict         = self.model.predict(nst, verbose=0)
        nst_predict_target  = self.model_target.predict(nst, verbose=0)

        x, y = [], []
        for index, (state, action, reward, nstate, done) in enumerate(minibatch):
            x.append(state)
            if done:
                target = reward
            else:
                best_action = np.argmax(nst_predict[index])   # online model picks action
                target = reward + self.gamma * nst_predict_target[index][best_action]  # target model evaluates it
                print(f"<------> reward: {reward:.4f} | future: {nst_predict_target[index][best_action]:.4f}")
            target_f          = st_predict[index].copy()
            target_f[action]  = target
            y.append(target_f)

        x_reshape = np.array(x).reshape(batch_size, self.nS)
        y_reshape = np.array(y)

        hist = self.model.fit(
            x_reshape, y_reshape,
            epochs=1, verbose=0,
            callbacks=[self.tensorboard]
        )
        self.loss.append(hist.history['loss'][0])
        self.episode_loss.append(hist.history['loss'][0])

    def decay_epsilon(self):
        """Decay epsilon while respecting the minimum epsilon floor."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save_model(self, filename):
        self.model_target.save(filename)

    def load_model(self, filename):
        self.model_target = load_model(filename)

    def set_actions(self, ACTIONS):
        self.ACTIONS = ACTIONS