import platform
import subprocess
import numpy as np


import torch

a = 1
print(id(a))
a += 1
print(id(a))

# ensemble = {1, 2, 3}
# ensemble.add(5)
# print(ensemble)  # Affichera {1, 2, 3, 4}


# # Supposons que q_values soit un tenseur de valeurs Q de forme (batch_size, num_actions)
# q_values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

# # Supposons que copy_actions soit un tenseur d'actions sélectionnées de forme (batch_size, num_dims)
# copy_actions = torch.tensor([[0, 1], [1, 2], [2, 0]])

# dim = 1

# # Étape par étape
# selected_q_values = q_values.gather(1, copy_actions[:, dim].unsqueeze(1)) # Sélectionnez les valeurs Q pour les actions prises (batch_size)

# print("q_values:")
# print(q_values)
# print("\ncopy_actions:")
# print(copy_actions)
# print("\ncopy_actions[:, dim]:")
# print(copy_actions[:, dim])
# print("\ncopy_actions[:, dim].unsqueeze(1):")
# print(copy_actions[:, dim].unsqueeze(1))
# print("\nq_values.gather(1, copy_actions[:, dim].unsqueeze(1)):")
# print(q_values.gather(1, copy_actions[:, dim].unsqueeze(1)))
# print("\nselected_q_values:")
# print(selected_q_values)

