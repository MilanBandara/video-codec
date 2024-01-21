import numpy as np
from itertools import chain

def zigzag_iteration(matrix):
    rows, cols = matrix.shape
    result = []

    for i in range(rows + cols - 1):
        if i % 2 == 0:  # Even rows
            for j in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                result.append(matrix[j, i - j])
        else:  # Odd rows
            for j in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                result.append(matrix[j, i - j])

    return result

def run_length_coding_binary(result):
    run_length_code = []
    current_value = None
    current_length = 0

    for value in result:
        if value == current_value:
            current_length += 1
        else:
            if current_value is not None:
                binary_value = format(current_value, '08b')  # Convert to binary with 8 bits
                binary_length = format(current_length, '08b')  # Convert run length to binary with 8 bits
                run_length_code.append((binary_value, binary_length))
            current_value = value
            current_length = 1

    if current_value is not None:
        binary_value = format(current_value, '08b')  # Convert to binary with 8 bits
        binary_length = format(current_length, '08b')  # Convert run length to binary with 8 bits
        run_length_code.append((binary_value, binary_length))

    return run_length_code

def decode_run_length(bit_sequence):

    remaining_bits = bit_sequence
    result = []

    while len(remaining_bits)>0:
        value,count = int(remaining_bits[0:8],2) , int(remaining_bits[8:16],2)
        result_list = [value] * count
        result.append(result_list)
        remaining_bits = remaining_bits[16:]
    
    result = list(chain(*result))
    return result

def recreate_matrix_from_zigzag(result, rows, cols):
    matrix = np.zeros((rows, cols), dtype=int)

    index = 0

    for i in range(rows + cols - 1):
        if i % 2 == 0:  # Even rows
            for j in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                matrix[j, i - j] = result[index]
                index += 1
        else:  # Odd rows
            for j in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                matrix[j, i - j] = result[index]
                index += 1

    return matrix


def get_bit_sequence(run_length_code_binary):
    result_string = ''

    for i in run_length_code_binary:
        result_string = result_string + i[0] + i[1]

    return result_string