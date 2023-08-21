import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class FirmEnvironment:
    def __init__(self, num_firms, num_periods):
        self.num_firms = num_firms
        self.num_periods = num_periods
        self.endowments = np.random.uniform(0, 10, num_firms)
        self.avg_productivity = 0.5
        self.productivity = np.random.normal(0.5, 0.2, num_firms)

    def step(self, actions):
        rewards = np.zeros(self.num_firms)
        self.avg_productivity = np.mean(self.productivity) / self.num_firms
        for i in range(self.num_firms):
            productivity = self.productivity[i]

            if actions[i] == 0:  # Do not produce invest in the risk free
                rewards[i] = 0.1

            else:  # Produce
                if productivity > self.avg_productivity:
                    rewards[i] = 1
                else:
                    rewards[i] = -0.2
            self.endowments[i] += rewards[i]

        self.firm_action = actions  # Update the action of the firm
        # Update average productivity
        return rewards, self.endowments.copy()


# Modify the get_action function


def get_action(state, avg_productivity, epsilon=0.4, model):
    if random.random() < epsilon:
        return random.choice([0, 1])

    state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
    avg_productivity_tensor = tf.convert_to_tensor(avg_productivity, dtype=tf.float32)
    # Reshape tensors to have compatible shapes for concatenation
    state_tensor = tf.reshape(state_tensor, (1, -1))
    avg_productivity_tensor = tf.reshape(avg_productivity_tensor, (1, -1))

    combined_input = tf.concat([state_tensor, avg_productivity_tensor], axis=1)

    logits = model(combined_input)
    action_probs = tf.nn.softmax(logits)
    chosen_action = tf.argmax(action_probs, axis=1).numpy()

    return chosen_action


# Modify the AIModel class
class AIModel(tf.keras.Model):
    def __init__(self):
        super(AIModel, self).__init__()
        self.concat_layer = tf.keras.layers.Concatenate(
            axis=-1
        )  # Concatenate along the last axis
        self.dense1 = tf.keras.layers.Dense(16, activation="relu")
        self.dense2 = tf.keras.layers.Dense(2, activation="softmax")

    def call(self, inputs):
        x = self.concat_layer([inputs])  # Wrap the input tensor in a list
        x = self.dense1(x)
        return self.dense2(x)

