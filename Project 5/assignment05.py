import random
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt

class Advertiser:
    def __init__(self, value):
        self.value = value
        self.bid = 0
        self.total_bid = 0
        self.num_wins = 0
        self.rationalizable_set = []
        self.regret_set = []
    
    def update_bid(self, winner_index, price_paid, epsilon, alpha, t):
        """Update the bid of the advertiser based on the outcome of the previous auction."""
        if winner_index == self.index:
            self.num_wins += 1
        self.total_bid += self.bid
        self.bid = max(0, self.bid + alpha * (self.value - price_paid) / (epsilon + price_paid))

        # Compute the rationalizable set
        max_bid = max([a.bid for a in advertisers if a != self])
        for i in range(t):
            eps = self.get_epsilon(i + 1)
            for v in [0.1, 0.3, 0.5, 0.7, 0.9]:
                bids = [a.bid if a != self else v for a in advertisers]
                regret = max(0, v * (max_bid - self.bid) - eps)
                if regret == 0:
                    self.rationalizable_set.append((v, eps))

    def get_epsilon(self, t):
        """Compute the value of epsilon for the current round."""
        return 1 / (t ** 0.5)
    

def quality_weighted_first_price_auction(demographic_info, qualities, bids):
    """Given the demographic information of a user, qualities of ads, and bids from advertisers,
    return the winning bidder and the corresponding price paid in a quality-weighted first-price auction.
    """
    # Filter the ad qualities and bids based on the demographic information of the user
    ad_qualities = [qualities[i] for i in demographic_info]
    ad_bids = [bids[i] for i in demographic_info]
    
    # Compute the quality-weighted bids
    weighted_bids = [q * b for q, b in zip(ad_qualities, ad_bids)]
    
    # Find the index of the highest quality-weighted bid
    winner_index = demographic_info[weighted_bids.index(max(weighted_bids))]
    
    # Return the winning bidder and the corresponding price paid
    return winner_index, ad_bids[winner_index]


def quality_weighted_first_price_auction_round(advertisers, t, alpha):
    """Run a single round of the quality-weighted first-price auction with regret minimization."""
    # Generate the qualities of the ads for the current round
    qualities = [random.uniform(0, 1) for _ in range(len(advertisers))]
    
    # Compute the bids of the advertisers for the current round
    bids = [advertiser.bid for advertiser in advertisers]
    
    # Run the auction
    winner_index, price_paid = quality_weighted_first_price_auction(range(len(advertisers)), qualities, bids)
    
    # Update the bids of the advertisers based on the outcome of the auction
    for advertiser in advertisers:
        advertiser.rationalizable_set = []
        advertiser.update_bid(winner_index, price_paid, advertiser.get_epsilon(t), alpha, t)
    
    # Return the winner and the price paid in the auction
    return winner_index, price_paid

# Let's say there are three users
# Suppose there are 3 advertisers with endowed values
advertisers = [Advertiser(0.8), Advertiser(0.6), Advertiser(0.7)]

# Set the learning rate and the initial bid to 0
alpha = 0.05
for advertiser in advertisers:
    advertiser.bid = 0
    advertiser.index = advertisers.index(advertiser)

# Run 1000 rounds of the auction
t = 1
for _ in range(100):
    winner_index, price_paid = quality_weighted_first_price_auction_round(advertisers, t, alpha)
    print("Round", t, "- The winner is advertiser", winner_index, "with a bid of", price_paid)
    
    # Print the regrets of the advertisers
    total_regret = 0
    for advertiser in advertisers:
        v = advertiser.value
        eps = advertiser.get_epsilon(t)
        bid = advertiser.bid
        max_bid = max([a.bid for a in advertisers if a != advertiser])
        regret = max(0, v * (max_bid - bid) - eps)
        advertiser.regret_set.append(regret)
        total_regret += regret
        print("Advertiser", advertisers.index(advertiser), "- Regret:", regret)
        
    # Compute the average regret and print it
    avg_regret = total_regret / len(advertisers)
    print("Average regret:", avg_regret)
    
    # Increment the round counter
    t += 1

def compute_point_prediction(rationalizable_sets):
    """Compute the point prediction from the intersection of the rationalizable sets."""
    weighted_values = []
    total_weight = 0
    for rationalizable_set in rationalizable_sets:
        if len(rationalizable_set) > 0:
            min_epsilon = min([r[1] for r in rationalizable_set])
            for r in rationalizable_set:
                if r[1] <= min_epsilon + 0.01:
                    weight = math.exp(-r[1] ** 2 / 2)
                    weighted_values.append((r[0], weight))
                    total_weight += weight
    if total_weight == 0:
        return None
    else:
        v_prime = sum([w[0] * w[1] / total_weight for w in weighted_values])
        return v_prime

# Print the final bids and total bids of the advertisers
for advertiser in advertisers:
    print("Advertiser", advertisers.index(advertiser), "- Final bid:", advertiser.bid, "- Total bid:", advertiser.total_bid)
    # print("Advertiser", advertisers.index(advertiser), "- Rationalizable Set:", advertiser.rationalizable_set)
    # compute point prediction
    v_prime = compute_point_prediction(advertiser.rationalizable_set)
    if v_prime is None:
        print("No point prediction can be made.")
    else:
        print(f"Point prediction: {v_prime:.3f}")

#plot the rationalizable set of each advertiser
# for advertiser in advertisers:
#     # for rationalizable_set in advertiser.rationalizable_set:
#     plt.plot(advertiser.rationalizable_set, label = "advertiser"+str(advertisers.index(advertiser)))
# plt.title("Rationalizable Set")
# plt.xlabel("Value")
# plt.ylabel("Epsilon")
# plt.show()
# plt.close()

#plot the regret of each advertiser
# for advertiser in advertisers:
#     #plot the regret set of each advertiser
#     advertiser.regret_set.reverse()
#     print(advertiser.regret_set)
#     plt.plot(advertiser.regret_set, label = "advertiser"+str(advertisers.index(advertiser)))
# plt.title("Regret")
# plt.xlabel("Round")
# plt.ylabel("Regret")
# plt.show()