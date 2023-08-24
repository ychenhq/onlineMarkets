import numpy as np
import matplotlib.pyplot as plt
# import random

class RepeatedBimatrixGame:
    def __init__(self, player1_payoffs, player2_payoffs):
        self.player1_payoffs = player1_payoffs
        self.player2_payoffs = player2_payoffs
        self.n_actions = player1_payoffs.shape[0]
        
    def play(self, player1_action, player2_action):
        return (self.player1_payoffs[player1_action, player2_action],
                self.player2_payoffs[player1_action, player2_action])
        
    def random_play(self):
        player1_action = np.random.randint(self.n_actions)
        player2_action = np.random.randint(self.n_actions)
        return self.play(player1_action, player2_action)



def no_regret_algorithm(t, x_t, y_t, learning_rate):
    # Implement the no-regret algorithm
    x_t_new = ...
    y_t_new = ...
    return x_t_new, y_t_new

def play_game(num_rounds, learning_rate1, learning_rate2):
    x_history = np.zeros((num_rounds,))
    y_history = np.zeros((num_rounds,))

    for t in range(num_rounds):
        x_t = x_history[t]
        y_t = y_history[t]
        x_t_new, y_t_new = no_regret_algorithm(t, x_t, y_t, learning_rate1)
        x_history[t+1] = x_t_new
        y_t_new, y_t_new = no_regret_algorithm(t, x_t, y_t, learning_rate2)
        y_history[t+1] = y_t_new

    return x_history, y_history

def run_simulation(num_rounds, learning_rate1, learning_rate2, num_simulations):
    x_history_avg = np.zeros((num_rounds,))
    y_history_avg = np.zeros((num_rounds,))

    for i in range(num_simulations):
        x_history, y_history = play_game(num_rounds, learning_rate1, learning_rate2)
        x_history_avg += x_history
        y_history_avg += y_history

    x_history_avg /= num_simulations
    y_history_avg /= num_simulations

    return x_history_avg, y_history_avg

def plot_results(x_history_avg, y_history_avg):
    plt.plot(x_history_avg, label='Learning Algorithm 1')
    plt.plot(y_history_avg, label='Learning Algorithm 2')
    plt.legend()
    plt.xlabel('Round')
    plt.ylabel('Payoff')
    plt.show()


# Define the payoffs for player 1 and player 2
player1_payoffs = np.array([[3, 0], [5, 1]])
player2_payoffs = np.array([[3, 5], [0, 1]])

# Create a repeated bimatrix game
game = RepeatedBimatrixGame(player1_payoffs, player2_payoffs)

num_rounds  = 1000
learning_rate1 = ...
learning_rate2 = ...
num_simulations = ...

# Play the game num_rounds times
for i in range(num_rounds):
    payoffs = game.random_play()
    print(f"Play {i}: Player 1 payoff = {payoffs[0]}, Player 2 payoff = {payoffs[1]}")

# x_history_avg, y_history_avg = run_simulation(num_rounds, learning_rate1, learning_rate2, num_simulations)
# plot_results(x_history_avg, y_history_avg)

