import numpy as np
import matplotlib.pyplot as plt


# Define the game of repeated bimatrix games
class BimatrixGame:
    def __init__(self, p1, p2):
        self.player1_payoffs = p1
        self.player2_payoffs = p2
        self.n_actions = len(p1)
        # print("n_actions", self.n_actions)

    def play(self, p1_action, p2_action):
        # print("player1_action", player1_action)
        # print("player2_action", player2_action)
        return self.player1_payoffs[p1_action, p2_action], self.player2_payoffs[p1_action, p2_action]



class LearningAlgorithm:

    def chooseAction(self):
        pass

    def updateProbs(self, action_payoffs, round_number):
        pass


class Hedge(LearningAlgorithm):
    # run choose action for each round
    # afterwards update with updateProbs
    # optimal lr =  (2kTlog(n)/2)^(2/3) i think?
    def __init__(self, actions_len, lr):
        self.probs = np.divide(np.ones(actions_len), actions_len)
        self.regret = 0
        self.lr = lr
        self.actions_len = actions_len
        self.best_hindsight_matrix =[]

    def chooseAction(self):
        action = np.random.choice(self.actions_len, p=self.probs)
        return action

    def updateProbs(self, action_payoffs, round_number):
        best_hindsight_matrix.append(action_payoffs)
        best_hindsight_val = best_hindsight(self.best_hindsight_matrix)
        self.regret += best_hindsight_val[0] - np.max(action_payoffs)
        sum = 0
        for action, action_payoff in enumerate(action_payoffs):
            if self.regret == 0:
                self.probs[action] = self.probs[action] * (action_payoff/(round_number+1))
            else:
                self.probs[action] = self.probs[action] * (action_payoff / (action_payoff + self.regret / (round_number + 1)))
            sum += self.probs[action]

        self.probs = self.probs/sum

class FollowPerturbedLeader(LearningAlgorithm):
    def __init__(self, actions_len, scale):
        self.actions_len = actions_len
        self.weights = np.divide(np.ones(actions_len), actions_len)
        self.scale = scale
        self.weights = np.add(self.weights, np.random.gumbel(scale= self.scale , size= self.actions_len))

    def chooseAction(self):
        return np.argmax(self.weights)
    
    def updateProbs(self, action_payoffs, round_number):
        self.weights = np.add(self.weights, action_payoffs)
        self.weights = np.add(self.weights, np.random.gumbel(scale= self.scale , size= self.actions_len))

class ExponentialWeights(LearningAlgorithm):
    def __init__(self, actions_len, lr, h):
        self.probs = np.divide(np.ones(actions_len), actions_len)
        self.lr = lr
        self.actions_len = actions_len
        self.summed_weights = np.zeros(actions_len)
        self.h = h

    def chooseAction(self):
        action = np.random.choice(self.actions_len, p=self.probs)
        return action

    def updateProbs(self, action_payoffs, round_number):
        summed_weights = self.calculateSummedWeights(action_payoffs)
        out = []
        sum = 0
        for ele in summed_weights:
            numerator = np.power((1 + self.lr), ele / self.h)
            sum += numerator
            out.append(numerator)

        self.probs = np.divide(out, sum)

    def calculateSummedWeights(self, round):
        self.summed_weights = np.add(self.summed_weights, round)
        return self.summed_weights


def best_hindsight(matrix):
    summed = np.sum(matrix, axis=0)
    argmax_payoff = np.argmax(summed)
    max_payoff = np.max(summed)

    return max_payoff, argmax_payoff

best_hindsight_matrix = []

def runGame(p1, p2, num_rounds):
    avg_p1_payoff = 0
    avg_p2_payoff = 0
    for i in range(num_rounds):
        player_1 = p1.chooseAction()
        player_2 = p2.chooseAction()
        payoffs = game.play(player_1, player_2)
        possible_payoffs_1 = game.player1_payoffs[:, player_2]
        possible_payoffs_2 = game.player2_payoffs[player_1, :] 
        best_hindsight_matrix.append(possible_payoffs_2)

        p1.updateProbs(possible_payoffs_1, i)
        p2.updateProbs(possible_payoffs_2, i)

        avg_p1_payoff += payoffs[0]
        avg_p2_payoff += payoffs[1]
    return avg_p1_payoff/num_rounds, avg_p2_payoff/num_rounds


def plotAlgorithm(p1, p2, p1_name, p2_name):
    rounds = [1, 10, 100, 1000, 5000]
    p1_arr = []
    p2_arr = []
    for num_rounds in rounds:
        p1 = Hedge(len(player2_payoffs[0]),0.1)
        p2 = FollowPerturbedLeader(len(player2_payoffs), 0.1)
        p1_payoff, p2_payoff = runGame(p1, p2, num_rounds)
        p1_arr.append(p1_payoff)
        p2_arr.append(p2_payoff)

    plt.plot(rounds, p1_arr, label=p1_name)
    plt.plot(rounds, p2_arr, label=p2_name)
    plt.title(f"{p1_name} vs {p2_name} Average Payoff")
    plt.legend()
    plt.xlabel("Number of Rounds")
    plt.ylabel("Average Payoff Over Rounds")
    plt.show()


def countMultipleNash(p1,p2):
    num_rounds = 100
    iterations = 100
    count = 0
    for i in range(iterations):
        # p1 = Hedge(len(player2_payoffs[0]),0.1)
        p1 = ExponentialWeights(len(player1_payoffs[0]), 0.1, h=np.max(player1_payoffs))
        p2 = FollowPerturbedLeader(len(player2_payoffs), 0.1)
        p1_payoff, p2_payoff = runGame(p1, p2, num_rounds)
        if np.abs(1-p1_payoff) <0.5:
            count += 1
    return count/iterations

def generatePayoffs(player1_val, player2_val, step, auction):
    if auction == "fpa":
        player2_payoffs = []
        player1_payoffs = []

        for i in range(0 , 100, step):
            player2 = np.arange(0, 100, step)
            for idx, ele in enumerate(player2):
                if ele ==i:
                    player2[idx] = (player2_val - ele)/2
                elif ele<i:
                    player2[idx] = 0
                elif ele > i:
                    player2[idx] = player2_val - ele

            player2_payoffs.append(player2)
            player1 = np.arange(0, 100, step)
            for idx, ele in enumerate(player1):
                if ele ==i:
                    player1[idx] = (player1_val - ele)/2
                elif ele<i:
                    player1[idx] = 0
                elif ele > i:
                    player1[idx] = player1_val - ele
            player1_payoffs.append(player1)
        player1_payoffs = np.array(player1_payoffs)
        player2_payoffs = np.array(player2_payoffs)
        return player1_payoffs, player2_payoffs
    
    if auction == "gspa":
        player1_payoffs = []
        player2_payoffs = []
        for i in range(0, 100, step):
            player2 = np.arange(0, 100, step)
            for idx, ele in enumerate(player2):
                if ele ==i:
                    player2[idx] = (player2_val*w_0 + (player1_val - ele) * w_1)/2
                elif ele<i:
                    player2[idx] = player2_val * w_0
                elif ele > i:
                    player2[idx] = (player2_val - i) * w_1
            player2_payoffs.append(player2)
            player1 = np.arange(0, 100, step)
            for idx, ele in enumerate(player1):
                if ele ==i:
                    player1[idx] = (player1_val*w_0 + (player2_val - ele) * w_1)/2
                elif ele<i:
                    player1[idx] = player1_val * w_0
                elif ele > i:
                    player1[idx] = (player1_val - i) * w_1

            player1_payoffs.append(player1)
        player1_payoffs = np.array(player1_payoffs)
        player2_payoffs = np.array(player2_payoffs)
        return player1_payoffs, player2_payoffs
    

w_0 = 0.1
w_1 = 1
player1_payoffs, player2_payoffs = generatePayoffs(30, 90, 1, "fpa")
print(player1_payoffs[1])
print(player2_payoffs[1])
# player1_payoffs, player2_payoffs = generatePayoffs(50, 70, 1, "gspa")
game = BimatrixGame(player1_payoffs, player2_payoffs)
num_rounds = 1000
hedge = Hedge(len(player2_payoffs[0]),0.1)
lrs = np.arange(0,0.11,0.01)
ftpl = FollowPerturbedLeader(len(player2_payoffs), 0.1)
util1= []
util2 = []
for lr in lrs:
    exp_weights = ExponentialWeights(len(player1_payoffs[0]), lr, h=np.max(player1_payoffs))
    exp_weights_2 = ExponentialWeights(len(player2_payoffs[0]), lr, h=np.max(player2_payoffs))
    payoff1, payoff2 = runGame(exp_weights, exp_weights_2, num_rounds)
    print(payoff1, payoff2)
    util1.append(payoff1)
    util2.append(payoff2)

plt.plot(lrs, util1, label="Player 1")
plt.plot(lrs, util2, label="Player 2")
plt.legend()
plt.xlabel("Player2_LR")
plt.ylabel("Average Payoff per Round")
plt.title("Varying Player LR Avg. Payoff for Player1 LR = 0.1")
plt.show()

# print(countMultipleNash(hedge, ftpl))
# runGame(hedge, ftpl, 100)
# plotAlgorithm(hedge, ftpl, "Hedge(lr = 0.1)", "FTPL(scale = 0.1)")

# def searchexploitLearning(p2, num_rounds):
#     avg_p1_payoff = 0
#     avg_p2_payoff = 0
#     count =0
#     thresholds = [5]
#     vals = range(0,30,1)
#     cur_max = -1
#     best_val_threshold = None
#     util = []
#     for val in vals:
#         for threshold in thresholds:
#             count = 0
#             avg_p1_payoff = 0
#             avg_p2_payoff = 0
#             p2 = ExponentialWeights(len(player1_payoffs[0]), 0.1, h=np.max(player2_payoffs))
#             for i in range(num_rounds):
#                 if count == threshold:
#                     count = 0
#                     player_1 = val
#                 else:
#                     player_1 = 0
#                     count+=1

#                 player_2 = p2.chooseAction()
#                 payoffs = game.play(player_1, player_2)
#                 possible_payoffs_1 = game.player1_payoffs[:, player_2]
#                 possible_payoffs_2 = game.player2_payoffs[player_1, :]

#                 p2.updateProbs(possible_payoffs_2, i)

#                 avg_p1_payoff += payoffs[0]
#                 avg_p2_payoff += payoffs[1]
#             util.append(avg_p1_payoff)
#             if avg_p1_payoff > cur_max:
#                 print(avg_p1_payoff/num_rounds)
#                 best_val_threshold = (val, threshold)
#                 cur_max = avg_p1_payoff/num_rounds


#     plt.plot(vals, util)
#     plt.xlabel("Activation Values")
#     plt.ylabel("Average Payoff per Round")
#     plt.title("Varying Activation Values on Avg Utili for threshold = 5")
#     plt.show()
#     return best_val_threshold, cur_max


# def exploitLearning(p2, num_rounds):
#     avg_p1_payoff = 0
#     avg_p2_payoff = 0
#     count =0
    
#     for i in range(num_rounds):
#         if count == 3:
#             count = 0
#             player_1 = 17
#         else:
#             player_1 = 0
#             count+=1
        
#         player_2 = p2.chooseAction()
#         payoffs = game.play(player_1, player_2)
#         possible_payoffs_1 = game.player1_payoffs[:, player_2]
#         possible_payoffs_2 = game.player2_payoffs[player_1, :]

#         p2.updateProbs(possible_payoffs_2, i)

#         avg_p1_payoff += payoffs[0]
#         avg_p2_payoff += payoffs[1]
        
#     return avg_p1_payoff/num_rounds, avg_p2_payoff/num_rounds

# # print(searchexploitLearning(exp_weights, 1000))
# # print(exploitLearning(exp_weights, 100))

# def exploitGSPA(p2, num_rounds):
#     avg_p1_payoff = 0
#     avg_p2_payoff = 0
#     count =0
    
#     for i in range(num_rounds):
#         if count == 10000:
#             count = 0
#             player_1 = 2
#         else:
#             player_1 = 64
#             count+=1
        
#         player_2 = p2.chooseAction()
#         payoffs = game.play(player_1, player_2)
#         possible_payoffs_1 = game.player1_payoffs[:, player_2]
#         possible_payoffs_2 = game.player2_payoffs[player_1, :]

#         p2.updateProbs(possible_payoffs_2, i)

#         avg_p1_payoff += payoffs[0]
#         avg_p2_payoff += payoffs[1]
        
#     return avg_p1_payoff/num_rounds, avg_p2_payoff/num_rounds


# def searchexploitGSPA(p2, num_rounds):
#     avg_p1_payoff = 0
#     avg_p2_payoff = 0
#     vals = range(0,100)
#     util = []
#     util2 = []
#     for val in vals:
#         avg_p1_payoff = 0
#         avg_p2_payoff = 0
#         p2 = ExponentialWeights(len(player1_payoffs[0]), 0.1, h=np.max(player2_payoffs))
#         for i in range(num_rounds):
#             player_1 = val
#             player_2 = p2.chooseAction()
#             payoffs = game.play(player_1, player_2)
#             possible_payoffs_1 = game.player1_payoffs[:, player_2]
#             possible_payoffs_2 = game.player2_payoffs[player_1, :]
#             p2.updateProbs(possible_payoffs_2, i)
#             avg_p1_payoff += payoffs[0]
#             avg_p2_payoff += payoffs[1]
#         util.append(avg_p1_payoff/num_rounds)
#         util2.append(avg_p2_payoff/num_rounds)
#     plt.plot(vals, util, label="Player 1")
#     plt.plot(vals, util2, label="Player 2")
#     plt.legend()
#     plt.xlabel("Player 1 Bid Values")
#     plt.ylabel("Average Payoff per Round")
#     plt.title("Varying Bid on Payoff for P1 = 50, P2 = 70")
#     plt.show()

#     return avg_p1_payoff/num_rounds, avg_p2_payoff/num_rounds

# searchexploitGSPA(exp_weights, 1000)
# # print(exploitGSPA(exp_weights, 100))