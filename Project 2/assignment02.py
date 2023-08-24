import math
import random
import numpy as np

#let k be the decisions
k = 2
#n be the number of total trials
n = 5
actions_arr = np.random.uniform(low=0, high=1, size=(n,k))
#theoretical learning rate = e = √(ln k / n)
epsilons = [0,1,math.sqrt((math.log(k))/n),1000]
print("original actions_payoff\n",actions_arr)
# print("original actions_payoff\n".join('{}'.format(*k) for k in enumerate(actions_arr)))
def totalPayoffPerAction(arr):
    return sum(arr)

def calculateTotalPayOff(e, l):
    total_payoff = 0
    for actions in range(n):
        total_payoff += math.pow((1+e),adversial_actions_payoff[actions][l-1])
    return total_payoff

#adversial fair payoffs
adversial_actions_payoff = actions_arr
#assign random payoff for all decisions in each loop
for e in epsilons:
    weights = [ [1]*k for i in range(n)]
    for l in range(k):
        x = random.randrange(0,1)
        min_action = 0
        min_payoff = totalPayoffPerAction(adversial_actions_payoff[0][0:l])
        #print(min_payoff)
        #accumulated payoffs for each action 
        for action in range(n):
            total_payoff =0
            for c in range(l):
                total_payoff += adversial_actions_payoff[action][c]  
            #not sure what if multiple actions have the smallest payoff at the same time
            if total_payoff < min_payoff :
                min_action = action
        #Assign this payoff to the action j* that has the smallest total payoff so far, i.e., j* = argminj Vji-1 where Vji = Σir=1 vji.  
        for i in range(n):
            adversial_actions_payoff[i][l] = adversial_actions_payoff[i][l]+round(random.uniform(0,1),3) if(min_action == i) else adversial_actions_payoff[i][l]
            total_payoff = calculateTotalPayOff(e,l)
            weights[i][l] = math.pow((1+e),adversial_actions_payoff[i][l-1]) / total_payoff
    print("\nweight = ",e)
    print(*weights, sep='\n')
# print("adversarial fair payoffs\n",adversial_actions_payoff)
print("\nadversarial fair payoffs",*adversial_actions_payoff, sep='\n')
#Bernoulli fair payoffs

bernoulli_actions_payoff = [[0]*k for i in range(n)]
bernoulli_actions_prob = []
for action in range(n):
    row = []
    for c in range(k):
        #row.append(round(random.uniform(0,0.5),3))
        row.append(random.choice([0,0.5]))
    bernoulli_actions_prob.append(row) 


#adding payoffs by bernoulli distribution
for col in range(k):
    total = 0
    #sum up the action probabilities
    for actions in range(n):
        if bernoulli_actions_prob[actions][col] == 0.5: 
            total += 1
    for actions in range(n):
        bernoulli_actions_payoff[actions][col] = total/n if(bernoulli_actions_prob[actions][col] == 0.5) else 1- total/n
    
# print("\n\nBernoulli fair payoffs\noriginal payoff\n",bernoulli_actions_payoff)
print("\n\nBernoulli fair payoffs\noriginal payoff\n",*bernoulli_actions_payoff, sep='\n')

#in this part of the project you are to look for other data source 

