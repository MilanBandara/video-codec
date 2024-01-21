import numpy as np
from itertools import chain
from scipy.fft import dctn,dct,idct

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

# def run_length_coding_binary(result):
#     run_length_code = []
#     current_value = None
#     current_length = 0

#     for value in result:
#         if value == current_value:
#             current_length += 1
#         else:
#             if current_value is not None:
#                 binary_value = format(current_value, '08b')  # Convert to binary with 8 bits
#                 binary_length = format(current_length, '08b')  # Convert run length to binary with 8 bits
#                 run_length_code.append((binary_value, binary_length))
#             current_value = value
#             current_length = 1

#     if current_value is not None:
#         binary_value = format(current_value, '08b')  # Convert to binary with 8 bits
#         binary_length = format(current_length, '08b')  # Convert run length to binary with 8 bits
#         run_length_code.append((binary_value, binary_length))

#     return run_length_code

def run_length_coding_binary(result):
    run_length_code = []
    current_value = None
    current_length = 0

    for value in result:
        if value == current_value:
            current_length += 1
        else:
            if current_value is not None:
                if current_value>=0:
                    binary_value = format(current_value, '08b')  # Convert to binary with 8 bits
                    binary_length = format(current_length, '08b')  # Convert run length to binary with 8 bits
                    run_length_code.append((binary_value, binary_length))
                else:
                    binary_value = format(current_value, '09b')  # Convert to binary with 9 bits
                    binary_length = format(current_length, '08b')  # Convert run length to binary with 8 bits
                    run_length_code.append((binary_value, binary_length))
            current_value = value
            current_length = 1

    if current_value is not None:
        if current_value>=0:
            binary_value = format(current_value, '08b')  # Convert to binary with 8 bits
            binary_length = format(current_length, '08b')  # Convert run length to binary with 8 bits
            run_length_code.append((binary_value, binary_length))
        else:
            binary_value = format(current_value, '09b')  # Convert to binary with 9 bits
            binary_length = format(current_length, '08b')  # Convert run length to binary with 8 bits
            run_length_code.append((binary_value, binary_length))

    return run_length_code

# def decode_run_length(bit_sequence):

#     remaining_bits = bit_sequence
#     result = []

#     while len(remaining_bits)>0:
#         if remaining_bits[0] == '-':
#             value,count = int(remaining_bits[1:9],2) , int(remaining_bits[9:17],2)
#             result_list = [value] * count
#             result.append(result_list)
#             remaining_bits = remaining_bits[17:]
#         else:
#             value,count = int(remaining_bits[0:8],2) , int(remaining_bits[8:16],2)
#             result_list = [value] * count
#             result.append(result_list)
#             remaining_bits = remaining_bits[16:]
    
#     result = list(chain(*result))
#     return result

def decode_run_length(bit_sequence):

    remaining_bits = bit_sequence
    result = []

    while len(remaining_bits)>0:
        if remaining_bits[0] == '-':
            value,count = -1*int(remaining_bits[1:9],2) , int(remaining_bits[9:17],2)
            result_list = [value] * count
            result.append(result_list)
            remaining_bits = remaining_bits[17:]
        else:
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

def compress_image(image):
    result = zigzag_iteration(image)
    run_length_code_binary = run_length_coding_binary(result)
    bit_sequence = get_bit_sequence(run_length_code_binary)
    return bit_sequence
    
# def compress_image(block):
#     print(block.shape)

def decompress_bit_sequence(bit_sequence):
    result = decode_run_length(bit_sequence)
    decoded = np.array(recreate_matrix_from_zigzag(result, 8, 8))
    return decoded

def compress_run_length_blocks(blocks,location):
    result_string = ''
    for block in blocks:
        # print(block.shape)
        bit_sequence = compress_image(block)
        result_string = result_string + bit_sequence + "\n"
        # print(result_string)
        
    with open(location, 'w') as f:
        f.write(result_string)
    return result_string


def decompress_run_length_blocks(location):

    with open(location, 'r') as file:
        text_string = file.read().strip()

    block_strings = text_string.split("\n")

    result_blocks = np.empty((0, 8,8))

    for index,string in enumerate(block_strings):
        decompressed_block = decompress_bit_sequence(string)
        result_blocks = np.concatenate((result_blocks, [decompressed_block]), axis=0)
    
    result_blocks = result_blocks.astype(int)
    return result_blocks



######################################################################33

def get_blocks(image_array):

    height, width = image_array.shape
    pad_height = (8 - height % 8) % 8
    pad_width = (8 - width % 8) % 8

    print(height, width)
    print(pad_height,pad_width)

    # Pad the image with zeros
    padded_image = np.pad(image_array, ((0, pad_height), (0, pad_width)), 'constant')

    # Calculate the number of blocks in each dimension
    num_blocks_h = padded_image.shape[0] // 8
    num_blocks_w = padded_image.shape[1] // 8

    blocks = np.empty((0, 8, 8))  # Initialize with the expected final shape


    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            start_x = j*8
            end_x = start_x + 8
            start_y = i*8
            end_y = start_y+8
            block = padded_image[start_x:end_x,start_y:end_y]
            blocks = np.concatenate((blocks, [block]), axis=0)
    return blocks

def get_dct_blocks(blocks):
    #perform DCT on blocks

    dct_blocks = np.empty((0, 8,8))  # Initialize with the expected final shape

    for index,block in enumerate(blocks):
        dct_image = dct(dct(block.astype(float), axis=0), axis=1)/2
        # dct_image = np.round(dct_image)
        dct_blocks = np.concatenate((dct_blocks, [dct_image]), axis=0)
    return dct_blocks

def quantize_blocks(dct_blocks,level): 

    quantization_matrix = np.array([[16,  11,  10,  16,  24,  40,  51,  61], 
                                [12,  12,  14,  19,  26,  58,  60,  55],
                                [14,  13,  16,  24,  40,  57,  69,  56], 
                                [14,  17,  22,  29,  51,  87,  80,  62], 
                                [18,  22,  37,  56,  68, 109, 103,  77], 
                                [24,  35,  55,  64,  81, 104, 113,  92], 
                                [49,  64,  78,  87, 103, 121, 120, 101], 
                                [72,  92,  95,  98, 112, 100, 103,  99]])
                                
    if level == "high":
        quantization_matrix = quantization_matrix*1
    elif level == "mid":
        quantization_matrix = quantization_matrix*10
    else:
        quantization_matrix = quantization_matrix*100
    
    
    quntized_blocks = np.empty((0, 8,8))  # Initialize with the expected final shape

    for index,block in enumerate(dct_blocks):
        quntized = block/quantization_matrix
        quntized = np.round(quntized)
        quntized_blocks = np.concatenate((quntized_blocks, [quntized]), axis=0)

    quntized_blocks = quntized_blocks.astype(int)
    return quntized_blocks