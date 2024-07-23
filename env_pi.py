from typing import Dict, List, Tuple, Union
import numpy as np

class Env:
    def __init__(self, num_hospitals: int, max_inventory: int, max_demand: int, max_steps: int, previous_demand_size):
        # Attributs fixes
        self.num_hospitals = num_hospitals
        self.max_inventory = max_inventory
        self.max_demand = max_demand
        self.max_steps = max_steps
        self.previous_demand_size = previous_demand_size
        
    def reset(self)-> np.ndarray:
        # Attributs non mutables
        self.current_step = 0
        self.count_in_step_function = 0
        self.count_correct_form = 0  

        # Generate new demands for the t current step for each hospital
        self.set_demands(np.random.randint(0, self.max_demand, size=self.num_hospitals))
        
        # Generate initial inventory levels
        inventory_levels = np.random.randint(self.max_inventory//3, self.max_inventory, 
                                                  size=self.num_hospitals)
        self.set_inventory_levels(inventory_levels)
        # Generate initial previous demands 
        list_demand = []
        for i in range(self.num_hospitals):
            list_demand.append(np.ones(self.previous_demand_size) * np.random.randint(0, self.max_demand//2))
        demand = np.concatenate(list_demand)

        self.set_previous_demand(demand)

        return np.concatenate((self.get_inventory_levels(), self.get_previous_demand()))

    def step(self, action: Union[np.ndarray, List[int]]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Perform a step in the environment.

        Parameters:
        action (np.ndarray): Action to take.

        Returns:
        Tuple[np.ndarray, float, bool, Dict]: Next state, reward, done flag, additional information.
        """

        self.current_step += 1
        self.count_in_step_function += 1
        # Check if the action is valid
        if not isinstance(action, (np.ndarray, list)) or len(action) != self.num_hospitals + self.num_hospitals ** 2:
            raise TypeError(f"Expected action to be a vector of length {self.num_hospitals + self.num_hospitals ** 2}, got {type(action)} with length {len(action)}")
        
        # Convert action to a numpy array if it's a list
        if isinstance(action, list):
            action = np.array(action)

        orders = action[:self.num_hospitals]
        order_warehouse = sum(orders)
        for i in range(self.num_hospitals):
            self.set_one_inventory_levels(i, self.get_inventory_levels()[i] - orders[i])

        D = action[self.num_hospitals:] # D[i, j] quantité demandée en dépannage par l'hôpital i à l'hôpital j
        D = D.reshape(self.num_hospitals, self.num_hospitals) 
        np.fill_diagonal(D, 0)  
        is_correct_form = self.is_matrix_correct_form(D)
        self.count_correct_form += is_correct_form 
        penalty_constraints_form = 1000 * (1 - is_correct_form)

        supplier  = { j for j in range(self.num_hospitals) if np.any(D[:, j] > 0) }
        # D, supplier = self.process_demands_random_order(D) # opération transparente si la matrice est déjà dans la forme correcte
        
        # Process exchanges between hospitals
        reel_exchange, over_command = self.process_exchanges(D, supplier)
        # reel_exchange[i][j] quantité réelle que l'hôpital i transfère à l'hôpital j
        exchange_total = sum([sum(reel_exchange[i].values()) for i in supplier]) 

        # Update inventory levels based on exchanges
        for i in supplier:
            for j in reel_exchange[i]:
                self.set_one_inventory_levels(j, self.get_inventory_levels()[j] + reel_exchange[i][j])
                self.set_one_inventory_levels(i, self.get_inventory_levels()[i] - reel_exchange[i][j])
        

        # Calculate unmet demand and overstock
        unmet_demand, over_stock_quantity = self.calculate_demand_overstock()

        # Define the costs
        order_warehouse_cost = 3  # Cost for each unit of product ordered from the warehouse    
        over_command_cost = 4     # Cost for each unit of product over ordered from another hospital
        exchanges_cost = 2  # Cost for each unit of product exchanged between hospitals
        unmet_demand_cost = 8  # Cost for each unit of unmet demand
        overstock_cost = 4      # Cost for each unit of overstock
        holding_cost = 1        # Cost for each unit of inventory held

        total_penalty = (
            unmet_demand_cost * unmet_demand +
            holding_cost * sum(self.get_inventory_levels()) +
            penalty_constraints_form
        )
        reward = -total_penalty 

        done = self.current_step >= self.max_steps

        self.update_previous_demand(self.get_demands(), self.get_previous_demand())

        return np.concatenate((self.get_inventory_levels(), self.get_previous_demand())), reward, done, {}
    
    def update_previous_demand(self, demands: np.ndarray, previous_demand: np.ndarray) -> None:
        """
        Update the previous demands for the next step.

        Parameters:
        demands (np.ndarray): Current demands.
        previous_demand (np.ndarray): Previous demands.

        Returns:
        np.ndarray: Updated demands.
        """
        list_new_previous_demand = []
        for i in range(self.num_hospitals):
            previous_demand_i = previous_demand[i * self.previous_demand_size : (i + 1) * self.previous_demand_size]
            new_previous_demand_i = np.roll(previous_demand_i, 1)
            new_previous_demand_i[0] = demands[i]
            list_new_previous_demand.append(new_previous_demand_i)

        new_previous_demand = np.concatenate(list_new_previous_demand)

        self.set_previous_demand(new_previous_demand)    

    def render(self):
        print(f'Inventory Levels: {self.get_inventory_levels()}')
        print(f'Demands: {self.get_demands()}')

    def random_distribution_dict(self, indices: List[int], total_value: int) -> Dict[int, int]:
        """
        Répartit de manière aléatoire la total_value entre les indices de la liste.

        Parameters:
        indices (list): Liste des indices.
        total_value (int): Valeur totale à répartir (doit être un entier).

        Returns:
        dict: Un dictionnaire où chaque clé est un indice et chaque valeur est la partie attribuée de total_value sous forme entière.
        """
        # Générer des fractions aléatoires qui somment à 1
        random_fractions = np.random.dirichlet(np.ones(len(indices)), size=1)[0]
        
        # Calculer les valeurs attribuées à chaque indice
        distributed_values = total_value * random_fractions
        
        # Convertir les valeurs en entiers tout en préservant la somme totale
        integer_values = np.floor(distributed_values).astype(int)
        remainder = total_value - np.sum(integer_values)
    
        # Distribuer le reste
        for i in range(int(remainder)):
            integer_values[i % len(indices)] += 1
        
        # Créer le dictionnaire de distribution
        distribution = {index: value for index, value in zip(indices, integer_values)}
        
        return distribution

    def process_exchanges(self, exchanges: np.ndarray, supplier: set) -> Tuple[Dict[int, Dict[int, int]], int]:
        """
        Process exchanges between hospitals.

        Parameters:
        exchanges (np.ndarray): Exchange matrix.
        supplier (set): Set of hospitals that are suppliers.

        Returns:
        Dict[int, Dict[int, int]]: Real exchanges between hospitals.
        """
        exchanges = exchanges.T # Transpose the exchanges matrix
        # exchanges[i, j] quantité potentielle que i va transférer à j
        reel_exchange = {}
        over_command = 0
        for i in supplier:
            list_index_i = [] 
            total_value_i = 0
            for j in range(self.num_hospitals):
                if exchanges[i, j] > 0: 
                    list_index_i.append(j)
                    total_value_i += exchanges[i, j] 
            over_command_i = total_value_i - min(self.get_inventory_levels()[i], total_value_i) 
            over_command += over_command_i
            if over_command_i > 0:
                total_value_i = min(self.get_inventory_levels()[i], total_value_i)
                reel_exchange[i] = self.random_distribution_dict(list_index_i, total_value_i)
            else:
                reel_exchange[i] = {j: exchanges[i, j] for j in range(self.num_hospitals) if exchanges[i, j] > 0}
                # reel_exchange[i][j] quantité réelle que l'hôpital i transfère à l'hôpital j
        return reel_exchange, over_command
    
    def calculate_demand_overstock(self) -> Tuple[int, int]:
        """
        Calculate the unmet demand and overstock quantities.

        Returns:
        Tuple[int, int]: Unmet demand and overstock quantities.
        """
        unmet_demand = 0
        over_stock_quantity = 0
        for i in range(self.num_hospitals):
            unmet_demand_i = max(0, self.get_demands()[i] - self.get_inventory_levels()[i])
            unmet_demand += unmet_demand_i
            if unmet_demand_i > 0:
                self.set_one_inventory_levels(i, 0)
            else:
                self.set_one_inventory_levels(i, self.get_inventory_levels()[i] - self.get_demands()[i]) 
            over_stock_quantity_i = max(self.get_inventory_levels()[i] - self.max_inventory, 0)
            over_stock_quantity += over_stock_quantity_i
            if over_stock_quantity_i > 0:
                self.set_one_inventory_levels(i, self.max_inventory)
            
        return unmet_demand, over_stock_quantity
    
    def process_demands_random_order(self, D: np.ndarray) -> np.ndarray:
        """
        La méthode corrige la forme de la matrice avec un ordre de traitement aléatoire.

        Parameters:
        D (np.ndarray): Matrix to process.

        Returns:
        np.ndarray: Processed matrix.
        """

        num_hospitals = D.shape[0]
        # Generate a random order for processing
        # hospital_order = [i for i in range(num_hospitals)] # pour le test unitaire
        hospital_order = np.random.permutation(num_hospitals)
        supplier = set()
        for i in hospital_order:
            if not i in supplier:
                if np.any(D[i, :] > 0): # i est dépanné

                    # Si i est dépanné il ne peut être dépanneur
                    D[:, i] = 0 

                    # Les dépanneurs de i ne peuvent être dépannés
                    supplier_of_i = {j for j in range(num_hospitals) if D[i,j] > 0} 
                    for j in supplier_of_i:
                        D[j, :] = 0
                    supplier = supplier.union(supplier_of_i)

        return D, supplier
    
    def is_matrix_correct_form(self, D: np.ndarray) -> bool:
        """
        Check if the matrix is in the correct form for processing.

        Parameters:
        D (np.ndarray): Matrix to check.

        Returns:
        bool: True if the matrix is in the correct form, False otherwise.
        """
        num_hospitals = D.shape[0]
        for i in range(num_hospitals):
            if np.any(D[i, :] > 0): # i est dépanné 
                if np.any(D[:, i] > 0): # i est dépanneur
                    return False
                supplier_of_i = {j for j in range(num_hospitals) if D[i,j] > 0} 
                # les dépanneurs de i et ne peuvent pas être dépannés
                for j in supplier_of_i:
                    if np.any(D[j, :] > 0): 
                        return False
        return True
    
    def create_random_correct_matrix(self, num_hospitals: int, max_demand_local: int) -> np.ndarray:
        """
        Create a random matrix in the correct form for processing.

        Parameters:
        num_hospitals (int): Number of hospitals.
        max_demand_local (int): Maximum demand that can be fulfilled by a hospital.

        Returns:
        np.ndarray: Random matrix in the correct form.
        """
        D = np.zeros((num_hospitals, num_hospitals), dtype=int)
        
        # Randomly select indices of suppliers
        suppliers = np.random.choice(range(num_hospitals), size=np.random.randint(0, num_hospitals), replace=False)
        non_suppliers = [i for i in range(num_hospitals) if i not in suppliers]

        for j in suppliers:
            # Select random indices of hospitals to be helped (excluding suppliers)
            recipient_j = np.random.choice(non_suppliers, size=np.random.randint(1, len(non_suppliers) + 1), replace=False)
            for i in recipient_j:
                D[i, j] = np.random.randint(1, max_demand_local)
        # D[i, j] quantité demandée en dépannage par l'hôpital i à l'hôpital j
        return D
    
    def get_inventory_levels(self) -> np.ndarray:
        return self.inventory_levels.copy()
    def set_inventory_levels(self, inventory_levels: np.ndarray):
        self.inventory_levels = inventory_levels.copy()
    def set_one_inventory_levels(self, index: int, new_value: int):
        self.inventory_levels[index] = new_value
    def get_previous_demand(self) -> np.ndarray:
        return self.previous_demand.copy()
    def set_previous_demand(self, previous_demand: np.ndarray):
        self.previous_demand = previous_demand.copy()
    def get_demands(self) -> np.ndarray:
        return self.demands.copy()
    def set_demands(self, demands: np.ndarray):
        self.demands = demands.copy()

    


# # # Tests unitaires


# # Tests unitaires


# # Création d'un tableau 1D initial
# original_array = np.arange(16)  # par exemple un tableau de forme (16,)
# print("Original array (16,):")
# print(original_array)    
# # Reshape en une forme 2D (4, 4)
# reshaped_array = original_array.reshape(4, 4)
# print("Reshaped array (4x4):")
# print(reshaped_array)

# # Inverse du reshape (4, 4) -> (16,)
# inverse_reshaped_array = reshaped_array.reshape(-1)
# print("\nInverse reshaped array (16,):")
# print(inverse_reshaped_array)


# # Example usage
# env = Env(6, 100, 25, 200)
# num_hospitals = env.num_hospitals
# max_demand_local = 10
# for _ in range(1):
#     D = env.create_random_correct_matrix(num_hospitals, max_demand_local)
#     print(D.shape)
#     print(D)
#     D = D.reshape(-1)
#     print(D.shape)
#     D = D.reshape(num_hospitals, num_hospitals)
#     print(D)
#     # Validate the matrix
#     is_correct = env.is_matrix_correct_form(D)
#     print("Matrix is correct:", is_correct)


# # Tests unitaires 

# # ## Test de la méthode random_distribution_dict

# # D = np.array([
# #     [0, 0, 0, 2],
# #     [0, 0, 0, 0],
# #     [0, 2, 0, 3],
# #     [0, 0, 0, 0]
# # ])
# env = Env(3, 100, 25, 200)
# # is_correct_form = env.is_matrix_correct_form(D)
# # print("Is Correct Form:", is_correct_form)
# # processed_D, supplier = env.process_demands_random_order(D)
# # print("Original D:")
# # print(D)
# # print("Processed D:")
# # print(processed_D)
# # print("Supplier:")
# # print(supplier)

# ## Test de la méthode calculate_demand_overstock

# distribution = env.random_distribution_dict([3, 1, 8], 49)

# print("Random Distribution:")
# print(distribution)

