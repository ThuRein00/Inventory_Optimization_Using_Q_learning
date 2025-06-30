import numpy as np
from typing import List, Tuple
import simpy


class Inventory:
    def __init__(self, 
                 env: simpy.Environment,              # SimPy simulation environment
                 d_mean: float,                       # Mean demand per day
                 d_std: float,                        # Standard deviation of demand
                 min_l: int,                          # Minimum lead time (days)
                 max_l: int,                          # Maximum lead time (days)
                 holding_cost: float,                 # Cost per unit per day for holding inventory
                 ordering_cost: float,                # Fixed cost per order placed
                 shortage_cost: float,                # Cost per unit per day for shortage
                 initial_inv: int,                    # Starting inventory level
                 max_inventory: int,                  # Maximum inventory capacity
                 period: int,                         # Number of days to simulate per period (one inventory is dones at one period)
                 max_inv_pos: int,                   # Inventory position is capped to reduce number of states
                 min_inv_pos : int,
                 print_output: bool,                 # Whether to print debug information
                 ) -> None:
        self.env: simpy.Environment = env
        self.inv_on_hand: int = initial_inv  # initial inventory amount
        self.inv_pos : int = initial_inv #on_hand + pending order - back log
        self.back_log : int = 0
        self.order_on_the_way: bool = False
        
        # Demand is normally distributed
        self.demand: int = 0
        self.d_mean: float = d_mean
        self.d_std: float = d_std
        
        # Lead time is uniformly distributed
        self.min_l: int = min_l
        self.max_l: int = max_l
        
        # Max inventory
        self.max_inv: int = max_inventory
        
        # Capped inventory position
        self.max_inv_pos : int = max_inv_pos
        self.min_inv_pos : int = min_inv_pos
        
        # Costs
        self.holding_cost: float = holding_cost  # Holding cost per day
        self.ordering_cost: float = ordering_cost
        self.shortage_cost: float = shortage_cost
        
        self.total_cost: float = 0
        self.print: bool = print_output  # if True, outputs are shown
        self.period: int = period
        self.shortage_day: float = 0 # set as float to find probability
        self.order_quantity: int = 1
        
        # Process and data storage
        self.env.process(self.run())
        self.pending_order: List[Tuple[int, int]] = []  # stores (order quantity, pending order)
        self.daily_inv: List[int] = [] # float to find avg inventory

    def check_inventory(self)-> None:
        """
        Check inventory and place order if needed.
        
        Places an order if order_quantity > 0, calculates random lead time,
        and adds the order to pending_order list with arrival day.
        
        """
        # order if order quantity is not 0
        if self.order_quantity != 0: 
            
            if self.print: print(f"Order is placed: {self.order_quantity}")
            self.total_cost += self.ordering_cost # ordering cost incurred
            
            #Lead time
            Lead_time = np.random.randint(self.min_l, self.max_l+1)
            if self.print == True: print(f"Lead time is: {Lead_time}") 
            self.pending_order.append((self.order_quantity,Lead_time+self.env.now))
            self.order_on_the_way = True
            if self.print == True :print(self.pending_order)
            
        else:
            if self.print == True: print("Order is placed: 0")
            self.order_on_the_way = False
        
        total_pending: int = 0 # keep track of pending order that is not arrived yet
        
        for order in self.pending_order:
            quantity: int
            quantity,_ = order
            total_pending += quantity
        
        self.inv_pos = self.inv_on_hand + total_pending - self.back_log
        # Inventory Position is limit
        if self.inv_pos >= self.max_inv_pos:
            self.inv_pos = self.max_inv_pos
        if self.inv_pos < self.min_inv_pos:
            self.inv_pos = self.min_inv_pos
                    
    def inv_replenish(self):    
        """
        Replenish inventory when orders arrive.
        
        Checks pending orders and adds quantity to inventory when
        the current simulation day matches the order arrival day.
        """
        for order in self.pending_order:
            quantity: int
            arrival_day: int
            quantity,arrival_day = order
            if self.env.now == arrival_day:
                self.inv_on_hand += quantity
                self.pending_order.remove((quantity,arrival_day))
                self.order_on_the_way = False
                if self.print == True: print("inv replenish")
            
    def inventory_shortage(self) -> None:
        """
        Handle inventory shortage.
        
        When inventory goes negative, calculates shortage cost,
        adds to total cost, increments shortage days, and resets
        inventory to 0 (no backorders allowed).
        """
        
        # check for shortage
        if self.inv_on_hand < 0:
            #shortage cost incurred
            shortage: float = -self.inv_on_hand * self.shortage_cost
            self.total_cost += shortage
            self.shortage_day += 1
            
            # shortages are backlogged
            self.back_log += abs(self.inv_on_hand) 
            if self.print == True: print(f"  → Inventory Shortage occurred. Back log quantity = {self.back_log}")
            self.inv_on_hand = 0
            
    def handle_back_log(self) -> None:
        """ Handle Back Log
            if there is stock avaliable after demand is met, it met back logged order.
            There are two cases for back log
            1) back log is more than demand
            2) demand is more than back log.
            
        """
        if self.inv_on_hand > 0:
            over_demand  = self.inv_on_hand # on hand inv left afer meeting demand
            if self.back_log >= over_demand:
                self.back_log -= over_demand # backlog is met with over demand
                self.inv_on_hand = 0
            else:  
                self.inv_on_hand -= self.back_log
                self.back_log = 0
            
            
    def inventory_over_run(self) -> None:
        """
        Handle inventory overrun.
        
        Caps inventory at maximum capacity when it exceeds max_inventory.
        Excess inventory is lost (no additional cost).
        """
        if self.inv_on_hand > self.max_inv:
            if self.print == True: print(f"  → Inventory Overrun = {self.inv_on_hand}")            
            # self.inv_on_hand = self.max_inv 
            self.inv_on_hand = self.inv_on_hand 
            

    def run(self):
        """
        Main simulation loop - SimPy process generator.
        
        Runs continuous simulation periods, each containing multiple days.
        For each day: replenishes orders, generates demand, handles shortages/overruns,
        calculates costs, and places new orders when none are pending.
        
        """
        while True:
            # reset total cost to get immediate reward
            self.total_cost = 0             
                    
            # run for one period
            for j in range(self.period):
            
                self.inv_replenish()
                yield self.env.timeout(0)
    
                    
                # print day start condition
                if self.print == True:
                    print(f"Day {self.env.now+1}")
                    print(f"Day Start on hand Inventory {self.inv_on_hand}")
                    print(f"Day Start Inventory position {self.inv_pos}")
                    print(f"Day Start Backlog {self.back_log}")
                
                # simulate demand
                self.demand = max(0, int(np.random.normal(loc=self.d_mean, scale=self.d_std, size=1)))
                # self.demand = np.random.poisson(self.d_mean)
                if self.print == True: print(f"Demand = {self.demand}" )
             
                # subtract demand 
                self.inv_on_hand -= self.demand
                
                #Check inventory Shortege
                self.inventory_shortage()
                yield self.env.timeout(0) # make sure it is inventory_shortage to fully finished
                
                self.handle_back_log()
                yield self.env.timeout(0) # fuifil back order

                self.inventory_over_run()
                yield self.env.timeout(0) # make sure it is inventory_over_run to fully finished
                
                #Holding Cost
                Total_Holding_cost = self.inv_on_hand * self.holding_cost  # total holding cost per day
                self.total_cost += Total_Holding_cost
                
                # save each daily inventory
                self.daily_inv.append(self.inv_on_hand)
                  
                if j < self.period:
                    self.check_inventory()
                    yield self.env.timeout(0) # make sure it is check_inventory to fully finished
                    
                # print Day end condition
                if self.print == True :
                    print(f"Day End on hand Inventory  {self.inv_on_hand}")   
                    print(f"Day End Inventory position {self.inv_pos}")
                    print(f"Day End Backlog  {self.back_log}") 
                    print(f"Total Cost {self.total_cost} ")
                    
                # advance one day
                yield self.env.timeout(1)
                 
                if self.print == True: print("Day End \n") 

# Example usage (commented out):
# env: simpy.Environment = simpy.Environment()
# inv: Inventory = Inventory(env, 2, 1, 2, 3, 1, 2, 1, 5, 5, True)
# env.run(until=15)