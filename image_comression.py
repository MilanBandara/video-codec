import numpy as np
from scipy.fft import dctn,dct,idct
from huffman_code import *

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
    idct_blocks_low = np.empty((0, 8,8))  # Initialize with the expected final shape

    for index,block in enumerate(result_blocks):
        idct_image = idct(idct(block, axis=1), axis=0)
        idct_blocks_low = np.concatenate((idct_blocks_low, [idct_image]), axis=0)

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