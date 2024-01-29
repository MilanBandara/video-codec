import numpy as np
from scipy.fft import dctn,dct,idctn
from huffman_code import *

def get_blocks(image_array):

    height, width = image_array.shape
    pad_height = (8 - height % 8) % 8
    pad_width = (8 - width % 8) % 8

    # print(height, width)
    # print(pad_height,pad_width)

    # Pad the image with zeros
    padded_image = np.pad(image_array, ((0, pad_height), (0, pad_width)), 'constant')

    # Calculate the number of blocks in each dimension
    num_blocks_h = padded_image.shape[0] // 8
    num_blocks_w = padded_image.shape[1] // 8

    blocks = np.zeros((6400, 8, 8))  # Initialize with the expected final shape

    itter = 0
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            start_x = j*8
            end_x = start_x + 8
            start_y = i*8
            end_y = start_y+8
            block = padded_image[start_x:end_x,start_y:end_y]
            blocks[itter] = block
            itter = itter + 1
    return blocks


def reconstruct_from_blocks_test(result_blocks):

    idct_blocks_low = result_blocks  # Initialize with the expected final shape

    reconstructed = np.zeros((640,640))

    block_number = 0
    for i in range(80):
        for j in range(80):
            start_x = j*8
            end_x = start_x + 8
            start_y = i*8
            end_y = start_y+8
            reconstructed[start_x:end_x,start_y:end_y] = idct_blocks_low[block_number]
            block_number = block_number + 1
    
    return reconstructed

def dct_for_one_block(block):
    dct_image = dctn(block, type=2, norm='ortho')
    return dct_image

def get_dct_blocks(blocks):
    #perform DCT on blocks

    dct_blocks = np.zeros((6400, 8,8))  # Initialize with the expected final shape

    for index,block in enumerate(blocks):
        dct_image = dctn(block, type=2, norm='ortho')
        dct_blocks[index] = dct_image
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
    if level == "mid":
        quantization_matrix = quantization_matrix*5
    if level == "low":
        quantization_matrix = quantization_matrix*20

    quntized_blocks = dct_blocks/quantization_matrix
    quntized_blocks = quntized_blocks.astype(int)
    return quntized_blocks

def dequntize_blocks(result_blocks,level):
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
        quantization_matrix = quantization_matrix*5
    elif level == "low":
        quantization_matrix = quantization_matrix*20
    else:
        quantization_matrix = quantization_matrix*level

    dequntized_blocks = result_blocks*quantization_matrix

    return dequntized_blocks
    

def quantize_blocks_custom(dct_blocks,scaler): 

    quantization_matrix = np.array([[16,  11,  10,  16,  24,  40,  51,  61], 
                                [12,  12,  14,  19,  26,  58,  60,  55],
                                [14,  13,  16,  24,  40,  57,  69,  56], 
                                [14,  17,  22,  29,  51,  87,  80,  62], 
                                [18,  22,  37,  56,  68, 109, 103,  77], 
                                [24,  35,  55,  64,  81, 104, 113,  92], 
                                [49,  64,  78,  87, 103, 121, 120, 101], 
                                [72,  92,  95,  98, 112, 100, 103,  99]])
                                
    
    quantization_matrix = quantization_matrix*scaler

    # quntized_blocks = np.empty((0, 8,8))  # Initialize with the expected final shape

    # for index,block in enumerate(dct_blocks):
    #     quntized = block/quantization_matrix
    #     quntized = np.round(quntized)
    #     quntized_blocks = np.concatenate((quntized_blocks, [quntized]), axis=0)
    quntized_blocks = dct_blocks/quantization_matrix
    # quntized_blocks = dct_blocks

    quntized_blocks = quntized_blocks.astype(int)
    return quntized_blocks

def compress(quntized_blocks,location):
    result_string = ''
    huffman_codes_list = []
    for index,block in enumerate(quntized_blocks):
        uniques_low,counts_low = np.unique(block,return_counts = True)
        probabilities_low = counts_low/np.sum(counts_low)
        huffman_code_low = generate_huffman_code(uniques_low,probabilities_low)
        huffman_codes_list.append(huffman_code_low)
        bit_sequence_low = compress_image(huffman_code_low,block)
        result_string = result_string + str(bit_sequence_low) + "\n"
    
    with open(location, 'w') as f:
        f.write(result_string)

    return huffman_codes_list


def decode_blocks(location,huffman_codes_list):
    # Load the text file
    with open(location, 'r') as file:
        text_string = file.read().strip()

    block_strings = text_string.split("\n")

    result_blocks = np.empty((0, 8,8))

    for index,string in enumerate(block_strings):
        tree = build_tree(huffman_codes_list[index])
        decoded_values = np.array(huffman_decoding_tree(string, tree))
        decoded_image = np.reshape(decoded_values,(8,8)).astype(int)
        result_blocks = np.concatenate((result_blocks, [decoded_image]), axis=0)
    result_blocks = result_blocks.astype(int)

    return result_blocks

def reconstruct_from_blocks(result_blocks):

    idct_blocks_low = np.zeros((6400, 8,8))  # Initialize with the expected final shape

    for index,block in enumerate(result_blocks):
        idct_image = idctn(block, type=2, norm='ortho')
        # idct_blocks_low = np.concatenate((idct_blocks_low, [idct_image]), axis=0)
        idct_blocks_low[index] = idct_image

    # idct_blocks_low = idct_blocks_low*quantization_matrix

    reconstructed = np.zeros((640,640))

    block_number = 0
    for i in range(80):
        for j in range(80):
            start_x = j*8
            end_x = start_x + 8
            start_y = i*8
            end_y = start_y+8
            reconstructed[start_x:end_x,start_y:end_y] = idct_blocks_low[block_number]
            block_number = block_number + 1
    
    return reconstructed

def calculate_psnr(original_image, reconstructed_image):
    # Assuming images are numpy arrays with the same shape and data type
    mse = np.mean((original_image - reconstructed_image) ** 2)
    max_pixel_value = np.max(original_image)
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr



################################################################################################

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

    result_blocks = np.zeros((6400, 8,8))

    for index,string in enumerate(block_strings):
        decompressed_block = decompress_bit_sequence(string)
        # result_blocks = np.concatenate((result_blocks, [decompressed_block]), axis=0)
        result_blocks[index] = decompressed_block
    
    result_blocks = result_blocks.astype(int)
    return result_blocks


def compress_complete_image(complete_image,scaler,location):
    blocks = get_blocks(complete_image)
    dct_blocks = get_dct_blocks(blocks)
    high_quntized_blocks = quantize_blocks_custom(dct_blocks,scaler)
    result_string_high = compress_run_length_blocks(high_quntized_blocks,location)

    return result_string_high

def decompress_complete_image(location,level):
    result_blocks = decompress_run_length_blocks(location)
    #add the dequantization step here
    result_blocks = dequntize_blocks(result_blocks,level)
    reconstructed_image = reconstruct_from_blocks(result_blocks)

    return reconstructed_image
