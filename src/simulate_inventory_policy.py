from Q_learning_agent import QLearningAgent
from simulation_model import Inventory
import simpy
import numpy as np
from scipy import stats


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
MAX_INV_POS = 60 # capped inventory position
MIN_INV_POS = 0
np.random.seed(0)


def run_episode_with_s_S(agent, num_periods, s, S):
    """Run a single inventory episode.
    
       Output "Total cost" for each episode """
       
    env = simpy.Environment()
    inv = Inventory(
        env, D_MEAN, D_STD, MIN_L, MAX_L, HOLDING_COST, ORDERING_COST,
        SHORTAGE_COST, START_INV, MAX_INVENTORY,
        PERIODIC_CHECK, MAX_INV_POS, MIN_INV_POS , False
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
        shortage_prob = inv.shortage_day/num_periods
        
    return Grand_total_cost,shortage_prob

total_cost_array = []
shortage_prob_array = []
for _ in range (TEST_SIMULATIONS):
    agent = QLearningAgent(MAX_INVENTORY, 0.99) #agent is not for learning, just for orderquantity generation
    total_cost,shortage_prob =  run_episode_with_s_S(agent, NUM_PERIODS,4,21)
    total_cost_array.append(total_cost)
    shortage_prob_array.append(shortage_prob)
    
mean_cost = np.mean(total_cost_array)
std_cost = np.std(total_cost_array)
print(f"Average Cost = {mean_cost:.2f}") 
print(f"Cost_STD = {std_cost:.2f}")

mean_shortage_prob = np.mean(shortage_prob_array)
std_shortage_prob = np.std(shortage_prob_array)
print(f"Average Cost = {mean_shortage_prob:.2f}") 
print(f"Cost_STD = {std_shortage_prob:.2f}")
