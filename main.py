import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from misc import generate_corrective_field, generate_fields_matrix, generate_powers_vector
np.set_printoptions(edgeitems=20, linewidth=200)

grid_dimensions = (10, 10)
detectors_positions = [(0,0), (0,9), (0,4), (9,0), (9,9)]
n_detectors = len(detectors_positions)
target_position = (5, 8)

# Se generan las intensidades que llegan a cada detector.
# Se generan a partir de la distancia euclidiana y se calcula su inversa
# para emular un escenario real.
powers = 1/generate_powers_vector(detectors_positions, target_position)

# Se genera una matriz de [filasGrid] x [columnasGrid] x [numeroDetectores]
# donde cada capa de la matriz contiene el campo para cada detector
fields = generate_fields_matrix(
    grid_dimensions[0],
    grid_dimensions[1],
    detectors_positions
    )

# Se muestra la suma de los campos de todos los detectores
plt.scatter(
    [x[1] for x in detectors_positions],
    [y[0] for y in detectors_positions],
    s=100,
    color="red"
    )
plt.scatter(target_position[1], target_position[0], marker="+", color="yellow")
plt.imshow(fields.sum(axis=2))
plt.show()

# Se calcula el campo corrector de la suma de los campos anteriores.
corrective_field = generate_corrective_field(fields)

# Se muestra la matriz de corrección de la suma de los campos.
plt.imshow(corrective_field)
plt.show()

weighted_fields =  powers * fields
sum = weighted_fields.sum(axis=2)
# final = sum/sum.max()
final = sum + 0.9*corrective_field

plt.scatter(
    [x[1] for x in detectors_positions],
    [y[0] for y in detectors_positions],
    s=100,
    color="red"
    )
plt.scatter(target_position[1], target_position[0], marker="+", color="yellow")
plt.imshow(final)
plt.show()