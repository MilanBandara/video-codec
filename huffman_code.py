import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

class TreeNode:
    #This class defines a node object in the huffman tree
    def __init__(self,data,probability,left=None, right=None):
        self.data = data
        self.probability = probability
        self.children = []
        self.parent = None
        self.bit = None
        self.bit_sequence = False
        self.left = left
        self.right = right

    def add_child(self,child):
        child.parent = self
        self.children.append(child)

    def assign_bits(self):
        
        if self.children:
            self.children[0].bit = "1"
            self.children[1].bit = "0"
            for i in self.children:
                i.assign_bits()
                
    def assign_bit_sequence(self):
        if not self.parent:
            for i in self.children:
                    i.assign_bit_sequence()
        if self.parent:
            self.bit_sequence = self.parent.bit_sequence + self.bit
            if self.children:
                for i in self.children:
                        i.assign_bit_sequence()
            else:
                self.bit_sequence = self.bit_sequence[1:]

    def get_leaf_values(self):

        if not self.children:
            #a leaf node
            return self.data,self.bit_sequence

        left_values = self.children[0].get_leaf_values()
        right_values = self.children[1].get_leaf_values()
        self.children[0].get_leaf_values()
        self.children[1].get_leaf_values()

        return left_values,right_values

def build_tree(codebook):
    root = TreeNode(None,None)

    for symbol, code in codebook.items():
        node = root

        for bit in code:
            if bit == "0":
                if node.left is None:
                    node.left = TreeNode(None,None)
                node = node.left
            else:
                if node.right is None:
                    node.right = TreeNode(None,None)
                node = node.right

        node.data = symbol

    return root

def huffman_decoding_tree(bit_sequence, root):
    decoded_values = []
    current_node = root
    
    for bit in bit_sequence:
        if bit == "0":
            current_node = current_node.left
        else:
            current_node = current_node.right
        if current_node.data is not None:
            decoded_values.append(current_node.data)
            current_node = root

    return decoded_values

def quantinze(number,cropped):

    image = cropped
    #quantice a number
    number_of_ranges = 7
    max_ = np.max(image)
    min_ = np.min(image)
    range_ = max_ - min_
    range_length = math.ceil(range_/number_of_ranges)
    updated_range = range_length*number_of_ranges
    value_to_add_for_range = updated_range - range_
    if (max_+ number_of_ranges) <= 255:
        max_ = max_ + value_to_add_for_range
    else:
        min_ = min_ - value_to_add_for_range
    quantized = None
    #find in which range the number is in
    #assign the quantized value 
    middle = min_ + (math.ceil((number-min_)/range_length)-1)*range_length + range_length/2 #middle value of the range it belongs
    if number<min_:
        quantized = min_
    elif number > max_:
        quantized = max_
    elif number < middle:
        quantized = min_ + (math.ceil((number-min_)/range_length)-1)*range_length
    else:
        quantized = min_ + (math.ceil((number-min_)/range_length))*range_length
    return quantized

def quantize_image(image,reference):
    uniques = np.unique(image)
    new_arr = image
    for i in uniques:
        quantized_value = quantinze(i,reference)

        new_arr = np.where(new_arr == i, quantized_value, new_arr)
    return new_arr

def flatten_tuple(nested_tuple):
    flattened_list = []
    for element in nested_tuple:
        if isinstance(element, tuple):
            flattened_list.extend(flatten_tuple(element))
        else:
            flattened_list.append(element)
    tuples_list = []
    return flattened_list

def build_huffman_tree_from_leaves(leaves_):
    #creating leaf nodes for each symbol
    sorted_leaves = sorted(leaves_, key=lambda node: node.probability, reverse=True)
    # #now itteratively create the tree
    count = 1
    while len(sorted_leaves)>1:
        new_node_probability = sorted_leaves[-1].probability + sorted_leaves[-2].probability
        new_node = TreeNode(data= f"inter_{count}",probability=new_node_probability)
        new_node.add_child(sorted_leaves[-2])
        new_node.add_child(sorted_leaves[-1]) # does this order matter
        sorted_leaves = sorted_leaves[:-2]
        sorted_leaves.append(new_node)
        sorted_leaves = sorted(sorted_leaves, key=lambda node: node.probability, reverse=True)
        count = count + 1
    sorted_leaves[-1].bit_sequence = "X" 
    sorted_leaves[-1].assign_bits()
    sorted_leaves[-1].assign_bit_sequence()
    # sorted_leaves[-1].print_tree()
    flattened_list = flatten_tuple(sorted_leaves[-1].get_leaf_values())
    tuple_list = []
    for i in range(0, len(flattened_list), 2):
        tuple_list.append((flattened_list[i], flattened_list[i + 1]))
    flattened_list = tuple_list
    
    # print(flattened_list)
    return flattened_list

def generate_huffman_code(uniques,probabilities):

    symbols_objects = []
    for data,probabiliy in zip(uniques.tolist(),probabilities):
        leaf = TreeNode(str(data),probabiliy)
        symbols_objects.append(leaf)

    codes = build_huffman_tree_from_leaves(symbols_objects)
    result = {}

    for item in codes:
        result[item[0]] = item[1]

    return result

def compress_image(huffman_code_for_cropped,image):
    #this image should be quantized using the range in the cropped image rgb channel
    array_to_compress = image.flatten()
    bit_sequence = ""
    for i in array_to_compress:
        symbol = huffman_code_for_cropped[str(i)]
        bit_sequence = bit_sequence + symbol

    return bit_sequence

def save_string_to_file(string, filename):
    #saves a given string to a text file as the given file name
    with open(filename, 'w') as f:
        f.write(string)

def get_entropy(image):
    uniques,counts = np.unique(image,return_counts = True)
    probabilities = counts/np.sum(counts)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def psnr(image1, image2):   
# Convert the images to grayscale
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate the mean squared error (MSE)
    mse = np.mean((image1 - image2) ** 2)

    # Check if the MSE is zero
    if mse == 0:
        return float('inf')

     # Calculate the PSNR
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr

def average_length(image,result):
    uniques,counts = np.unique(image,return_counts = True)
    probabilities = counts/np.sum(counts)
    average_length = 0
    for unique,prob in zip(uniques,probabilities):
        code_length = len(result[str(unique)])
        average_length = average_length + code_length*prob
    return average_length