from Q_learning_agent import QLearningAgent
from simulation_model import Inventory
import simpy
import numpy as np
import matplotlib.pyplot as plt


# Set global constants
PERIODIC_CHECK = 1 #check inventory each day
START_INV = 0
MAX_INVENTORY = 30
D_MEAN = 2 # simulate normally distributted demand
D_STD = 1 
MIN_L = 2 # lead time is uniformly distributed (but in this project constant lead time is used to reduce complexity)
MAX_L = 2 
HOLDING_COST = 50
ORDERING_COST = 5000
SHORTAGE_COST = 1000
NUM_PERIODS = 100
TEST_SIMULATIONS = 1000
np.random.seed(0)

num_policies = 0
# there are "1+2+3+..+30" number of possible policies for max capapity of 30
for i in range(MAX_INVENTORY):
    num_policies += i+1
print("Number of possible actions: " , num_policies)

# mapping action index to (s,S) policy
def map_action_to_s_S(policy_index):
    count = 0
    for i in range(MAX_INVENTORY):
        for j in range(MAX_INVENTORY):
            if i <= j:
                if count == policy_index:
                    return i+1, j+1,policy_index  # Return immediately when (s,S) policy for action index is found
                count += 1
    return None  # Handle cases where action index is out of range


def run_episode_with_s_S(agent, num_periods, s, S):
    """Run a single inventory management episode.
    
       Output "Total cost" for each episode """
       
    env = simpy.Environment()
    inv = Inventory(
        env, D_MEAN, D_STD, MIN_L, MAX_L, HOLDING_COST, ORDERING_COST,
        SHORTAGE_COST, START_INV, MAX_INVENTORY,
        PERIODIC_CHECK, False
    )
    
    Grand_total_cost = 0
    
    for i in range(num_periods):
        start_state = inv.inv_pos


        agent_action = agent.policy_evaluate_action_generate(start_state ,s,S )
        order_quantity = agent_action   
        
            
        inv.order_quantity = order_quantity

        # Run Simulation
        env.run(until = (i+1)* PERIODIC_CHECK )
            
        Grand_total_cost += inv.total_cost
        
    return Grand_total_cost

# Optimal Policy search
print("Searching...")
cost_each_policy = []
for policy_index in range (num_policies):
    avg_total_cost = 0 #avg total cost for each policy
    s,S,policy_index = map_action_to_s_S(policy_index) # map policy index to associate (s,S) policy

    agent = QLearningAgent(MAX_INVENTORY, 0.99) #agent is not for learning, just for orderquantity generation
    
    for _ in range(TEST_SIMULATIONS):
        total_cost = run_episode_with_s_S(agent, NUM_PERIODS,s,S)
        avg_total_cost += total_cost / TEST_SIMULATIONS
    cost_each_policy.append(avg_total_cost)
         
print("Search Completed.")


# Plotting cost of each policy
plt.figure(figsize=(20, 4))
plt.plot(range(len(cost_each_policy)), cost_each_policy, marker='o', linestyle='-')
plt.xlabel('Policy_index')
plt.ylabel('Total Cost')
plt.title('Total Cost per Episode')
plt.grid(True)
plt.show()

best_policy = cost_each_policy.index(np.min(cost_each_policy))
print(best_policy)
s,S,index = map_action_to_s_S(best_policy)
print(s,S)
print(np.min(cost_each_policy))