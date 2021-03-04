import numpy as np

def euclidean_distance(p1, p2):
    x = p1[0] - p2[0]
    y = p1[1] - p2[1]
    return np.sqrt(x*x + y*y)


def generate_fields_matrix(rows, cols, detectors_positions):
    n_detectors = len(detectors_positions)

    detectors_fields = np.empty(
    (rows, cols, n_detectors),
    np.float64
)
    for detector_idx in range(n_detectors):
        for row in range(rows):
            for col in range(cols):
                distance = euclidean_distance(
                    (row, col),
                    detectors_positions[detector_idx]
                    )
                detectors_fields[row, col, detector_idx] = distance
    
    return detectors_fields

def generate_corrective_field(fields):
    accumulated_fields = fields.sum(axis=2)
    max = accumulated_fields.max()
    return max - accumulated_fields

def generate_powers_vector(detectors_positions, target_position):
    n_detectors = len(detectors_positions)
    powers = np.empty((1, 1, n_detectors))
    for detector_idx in range(n_detectors):
        powers[0][0][detector_idx] = euclidean_distance(
            detectors_positions[detector_idx],
            target_position
            )
    return np.log10(powers)