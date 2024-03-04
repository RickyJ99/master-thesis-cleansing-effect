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


def get_action(state, avg_productivity, epsilon=0.4):
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


# Modify the update_model function
def update_model(firm_index, state, action, rewards):
    with tf.device("/GPU:0"):
        with tf.GradientTape() as tape:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            avg_productivity_tensor = tf.convert_to_tensor(
                env.avg_productivity, dtype=tf.float32
            )
            # Reshape tensors to have compatible shapes for concatenation
            state_tensor = tf.reshape(state_tensor, (1, -1))
            avg_productivity_tensor = tf.reshape(avg_productivity_tensor, (1, -1))
            combined_input = tf.concat([state_tensor, avg_productivity_tensor], axis=1)
            logits = firm_models[firm_index](
                tf.convert_to_tensor(combined_input, dtype=tf.float32)
            )
            loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            loss = loss_fn([action], logits)
            loss = tf.reduce_mean(loss * rewards)
        grads = tape.gradient(loss, firm_models[firm_index].trainable_variables)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.01
        )  # Create a new optimizer instance
        optimizer.apply_gradients(
            zip(grads, firm_models[firm_index].trainable_variables)
        )


# Training the AI Firm
num_episodes = 100

for episode in range(num_episodes):
    # Reinforcement Learning Setup
    env = FirmEnvironment(num_firms=10, num_periods=100)
    model = AIModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Define a list of neural network models
    firm_models = [AIModel() for _ in range(env.num_firms)]

    for period in range(env.num_periods):  # Loop over periods first
        out_of_market_counts = 0
        actions = []
        state = env.endowments.copy()  # Define the state before the loop over firms

        for firm_index in range(env.num_firms):  # Loop over firms within each period
            action = get_action(
                state[firm_index], env.avg_productivity
            )  # Compute action for this firm
            actions.append(action)

        rewards, next_state = env.step(
            actions
        )  # All firms take their actions, compute rewards
        for firm_index in range(env.num_firms):
            update_model(
                firm_index, state[firm_index], actions[firm_index], rewards[firm_index]
            )

            state[firm_index] = next_state[
                firm_index
            ]  # Update state for the next period

            if actions[firm_index] == 0:
                out_of_market_counts += 1
        # computing the total reward
        total_reward = np.sum(rewards)

        # compute the avarage reward

        avg_reward = np.mean(rewards)

        # Calculate share of firms staying out of the market
        share_out_of_market = out_of_market_counts / env.num_firms

        print(
            f"Episode: {episode} Period: {period} Total Reward: {total_reward:.2f} \t Avg. reward: {avg_reward:.2f} \t Share of firms out of market: {share_out_of_market:.2f} "
        )

# Save the trained model using SavedModel format
model.save("trained_model")

# Save individual firm models using SavedModel format
for firm_index in range(env.num_firms):
    firm_models[firm_index].save(f"firm_model_{firm_index}")

