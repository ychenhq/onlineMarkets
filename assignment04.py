import random
import numpy as np
import matplotlib.pyplot as plt

# set values of bidder values
def value_distribution(n):
#   print( [random.uniform(0, 1) for i in range(n)])
  return [random.uniform(0, 1) for i in range(n)]

# calculate virtual welfare given current reserve price
def virtual_welfare(reserve_price, value_distribution):
    return sum([value - reserve_price for value in value_distribution if value > reserve_price])

# calculate expected revenue given current reserve price
def expected_revenue(reserve_price, value_distribution):
    n = len(value_distribution)
    vw = sum([value - reserve_price for value in value_distribution if value > reserve_price])
    return (1/n) * vw

# implementation of exponentially weighted average algorithm
def exponential_weighted_average(reserve_price, value, tau, alpha):
    return (1 - alpha) * reserve_price + alpha * (value - (1 / tau))

# calculate optimal reserve price for the distribution
def optimal_reserve_price(value_distribution):
    return sum(value_distribution) / len(value_distribution)

# compare performance of learning algorithm to optimal reserve price
def compare_performance(vd, tau, alpha, num_round):
    # initialize reserve price
    reserve_price = 0

    # calculate optimal reserve price
    # optimal_reserve = optimal_reserve_price(value_distribution)
    optimal_reserve = 0.5

    # calculate optimal expected revenue
    # optimal_expected_revenue = expected_revenue(optimal_reserve, value_distribution)
    optimal_expected_revenue = 5/12

    # initialize list to store expected revenue over time
    expected_revenue_list = []

    # calculate expected revenue over time
    for i in range(num_round):
        # update reserve price using exponentially weighted average algorithm
        arr = vd
        vd.sort(reverse=True)
        reserve_price = exponential_weighted_average(reserve_price, vd[1], tau, alpha)
        # print("reserve_price", reserve_price)
        expected_revenue_list.append(expected_revenue(reserve_price, arr))
        vd =value_distribution(num_bidders)
        # print("value_distribution", vd)
    # print("count", count)
    return expected_revenue_list, optimal_expected_revenue

# number of bidders
num_bidders = 3
num_rounds = [10, 100, 1000, 3000]


vd =value_distribution(num_bidders)
# set the range of values for tau and alpha
taus = [0.5, 1, 2, 4]
alphas = [0.2, 0.3, 0.4]

# initialize dictionary to store expected revenue for each tau and alpha
revenues = {}

# compare performance of learning algorithm to optimal reserve price for each tau and alpha
for num_round in num_rounds:
    for tau in taus:
        for alpha in alphas:
            expected_revenue_list, optimal_expected_revenue = compare_performance(vd, tau, alpha, num_round)
            plt.plot(expected_revenue_list, label=f"tau={tau}, alpha={alpha}")
            plt.xlabel(str(num_round) +" Rounds")
            plt.ylabel("Expected Revenue")
            plt.legend()
            revenues[(tau, alpha)] = expected_revenue_list
        plt.plot( [optimal_expected_revenue]*num_round, 'k--', label='optimal expected revenue')
        plt.show()
    