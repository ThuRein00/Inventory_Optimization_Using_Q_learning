from Q_learning_agent import QLearningAgent
from simulation_model import Inventory
import simpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
MAX_INV_POS = 60
MIN_INV_POS = 0
NUM_PERIODS = 100
EPISODES = 10000
NUM_TRAIN = 50 # agent trains independent training rounds to get trends on 
TEST_SIMULATIONS = 1000

# Hyperparameters
EPSILON0 = 0.9
E_DECAY = 5e-5
ALPHA0 = 0.9
A_DECAY = 1e-5

np.random.seed(0)

# Global optimal Cost
GLOBAL_OPTIMAL_COST = 95287 # lowest simulated cost 

def run_episode(agent, num_periods, count=0 , avg_q_matrix = np.zeros((MAX_INV_POS+1,MAX_INVENTORY+1))  ,training=False, policy_evaluation = False ,performance_evaluation = False):
    """Run a single inventory management episode. There are three modes 
    1) Training with Q agent
    2) Policy Evaluation with Q agent
    3) Performance Evaluation to test agent lernt policy
    
    Note: avg_q_matrix is needed only for policy evaluation
    
Output 1) "Total cost" for each episode
       2) "count" is for hyperparameter decay 
       3) daily inventory level is to keep track of daily on hand inventory per episode
       4) shortage probability"""
       
    env = simpy.Environment()
    inv = Inventory(
        env, D_MEAN, D_STD, MIN_L, MAX_L, HOLDING_COST, ORDERING_COST,
        SHORTAGE_COST, START_INV, MAX_INVENTORY,
        PERIODIC_CHECK,MAX_INV_POS,MIN_INV_POS,False
    )
    
    Grand_total_cost = 0 #total cost of an episode
    
    for i in range(num_periods):
        
        agent.alpha = ALPHA0 / (1 + count * A_DECAY)
        agent.epsilon = EPSILON0 / (1 + count* E_DECAY)
        
        start_state = inv.inv_pos
        # print(f"start_inv : {start_inv}, start_state : {start_state}")
        if i < num_periods-1:
            final_state = False
        else:
            final_state = True # indicates final state
         

        if training : agent_action = agent.choose_action(start_state) 
        if policy_evaluation: agent_action = agent.policy_evaluate_action_generate(start_state ,5,25 ) # you can change policy you want to evaluate(Only (s,S) policy)
        if performance_evaluation: agent_action =np.argmax(avg_q_matrix[start_state]) 
        order_quantity = agent_action   

        inv.order_quantity = order_quantity
    
    
        # Run Simulation
        env.run(until = (i+1)* PERIODIC_CHECK )

        reward  = -inv.total_cost
        # print(f"reward : {reward}")
        next_state = inv.inv_pos
        # print(f"start{start_state} ")
        # print(f"next{next_state }")
        if training : agent.learn( start_state ,agent_action, reward, next_state, final_state)
        if policy_evaluation :agent.policy_evaluate( start_state ,agent_action, reward, next_state, final_state)
        # print("agent_learned ")

        count+=1
            
        Grand_total_cost += inv.total_cost
        shortage_prob = inv.shortage_day/num_periods
        daily_inv_level = inv.daily_inv
        
    
    return Grand_total_cost,count,daily_inv_level,shortage_prob


# Training
print("Starting Training...")
last_cost_array = [] # cost of last episode of each training round is stored
last_shortage_prob_array = [] # shortage probability of each training round is stored
total_square_error_per_episode = np.zeros((EPISODES,)) # square error of each episode is stored
q_matrix_store  = [] #store q table of different simulation rounds
for _ in range (NUM_TRAIN):
    agent = QLearningAgent(MAX_INVENTORY, 0.99)
    square_error_per_episode = []
    count = 0
    for episode in range(EPISODES):
        total_cost,count,_,shortage = run_episode(agent, NUM_PERIODS,count,training=True)
        square_error_per_episode.append((total_cost - GLOBAL_OPTIMAL_COST )**2)
    
    total_square_error_per_episode += np.array(square_error_per_episode)
    last_cost_array.append(total_cost)
    last_shortage_prob_array.append(shortage)
    q_matrix_store.append(agent.q_table)
         
print("Training Completed.")

# training performance (average over independent training rounds)
mean_cost = np.mean(last_cost_array)
std_cost = np.std(last_cost_array)
mean_short = np.mean(last_shortage_prob_array)
std_short = np.std(last_shortage_prob_array)
print(f"learned policy Average Cost = {mean_cost:.2f}") 
print(f"Cost_STD = {std_cost:.2f}")
print(f"learned policy Shortage Probability = {mean_short:.4f}")
print(f"Shortage Probability STD = {std_short:.4f}")

# Learning Curve
plt.figure(figsize=(20, 4))
plt.plot(range(EPISODES), np.sqrt(total_square_error_per_episode/NUM_TRAIN), linestyle='-') # mean sqare error 
plt.xlabel('Episode Number')
plt.ylabel('RMSE (over 50 runs)')
plt.title('Learning Curve')
plt.grid(True)
plt.show()


# Plot avg Q-table heatmap for independent simulation rounds
#avg q values
q_matrix = np.array([agent.q_table[state] for state in sorted(agent.q_table.keys())] )
avg_q_matrix = np.zeros((q_matrix.shape))
for i in range (NUM_TRAIN):
    q_table = q_matrix_store[i]
    q_matrix = np.array([q_table[state] for state in sorted(q_table.keys())] )
    avg_q_matrix += q_matrix/NUM_TRAIN

# Plot heat map
plt.figure(figsize=(50, 30))
sns.heatmap(
    np.abs(avg_q_matrix),
    annot=False,
    cmap="viridis",
    yticklabels=[f"State {s}" for s in sorted(agent.q_table.keys())],
    xticklabels=[f"Action {a}" for a in range(MAX_INVENTORY)] # max allowed order agent can make is Max_inventory number of orders
)
plt.title("Q-Table Heatmap")
plt.xlabel("Actions")
plt.ylabel("States")
plt.xticks(rotation=45)
plt.show()

# Show avg best actions
print("Avg Learned Q-table:")
rows = q_matrix.shape[0]
for i in range (rows):
    print(f"State {i}: {np.argmax(avg_q_matrix[i])}")
    
# Evaluation of learned policy (Evaluate learned policy (avg_Q_values)")
print("\nRunning Evaluation...")

total_daily_inv_eva = np.zeros(NUM_PERIODS)
grand_total_costs_eva = []
avg_shortage_prob_eva = 0


for _ in range(TEST_SIMULATIONS):
    total_cost_eva, _ , daily_inv_eva, shortage_prob_eva = run_episode(agent, NUM_PERIODS,avg_q_matrix = avg_q_matrix, performance_evaluation=True) 
    grand_total_costs_eva.append(total_cost_eva)
    total_daily_inv_eva += np.array(daily_inv_eva)
    avg_shortage_prob_eva += shortage_prob_eva / TEST_SIMULATIONS

# Calculate evaluation metrics
avg_daily_inv_eva = total_daily_inv_eva / TEST_SIMULATIONS
mean_cost = np.mean(grand_total_costs_eva)
std_cost = np.std(grand_total_costs_eva)

print(f"Simulated Average Cost = {mean_cost:.2f}")
print(f"Simulated STD = {std_cost:.2f}")
print(f"Shortage Probability = {avg_shortage_prob_eva:.4f}")

# Plot avg inventory levels
plt.figure(figsize=(20, 4))
plt.plot(range(1, len(avg_daily_inv_eva) + 1), avg_daily_inv_eva)
plt.xticks(range(10, len(avg_daily_inv_eva)+1, 10))
plt.xlabel('Day')
plt.ylabel('Average Inventory Level')
plt.title('Daily Inventory Level')
plt.grid(True)
plt.tight_layout()
plt.show()