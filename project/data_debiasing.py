import random

def augment_dataset(orig_data_path, target_size, save_path):
    '''
    Augment the dataset to resample until the desired (larger) size is achieved.
    Input arguments:
        orig_data_path : 
            tsv file path which contains the data for either aae or wae 
            data, toxic or nontoxic data (1 or 4 possibilities). First and only column 
            contains text data.
        target_size:
            a number representing how big the final file should be after augmenting
            (target size).
        save_path:
            the path to save the new (augmented) data to.
    '''
    # First, convert the tsv entries into a list 
    lines = []
    with open(orig_data_path) as f:
        lines = f.readlines()
        
    lines = list(lines)

    # Make sure this function is being for augmenting, not reducing
    assert target_size >= len(lines)

    # From the list, resampling randomly until the desired size is reached
    added_lines = []

    while len(lines) + len(added_lines) < target_size:
        new_line = random.choice(lines)
        added_lines.append(new_line)

    lines = added_lines + lines

    # Save the new dataset to the specified save_path location
    with open(save_path, 'w') as f:
        for line in lines:
            f.write(line)

def reduce_dataset(orig_data_path, target_size, save_path=None):
    '''
    Reduce the dataset by sampling from it until the desired (smaller) size is achieved.
    Input arguments:
        orig_data_path : 
            tsv file path which contains the data for either aae or wae 
            data, toxic or nontoxic data (1 or 4 possibilities). First and only column 
            contains text data.
        target_size:
            a number representing how small the final file should be after reducing
            (target size).
        save_path:
            the path to save the new (augmented) data to.
    '''
    # First, convert the tsv entries into a list
    lines = []
    with open(orig_data_path) as f:
        lines = f.readlines()
        
    lines = list(lines)

    # Make sure this function is being for reducing, not augmenting
    assert target_size <= len(lines)

    # Randomly select target_size examples from the list
    lines = random.sample(lines, target_size)

    return lines

    """
    # Save the new dataset to the specified save_path location
    with open(save_path, 'w') as f:
        for line in lines:
            f.write(line)
    """

def augment_dataset_LM(orig_data_path, target_size, save_path):
    '''
    Augment the dataset until the desired (larger) size is achieved. Instead of resampling,
    prompt a language model (LM) to generate synthetic data for the class. 
    Input arguments:
        orig_data : 
            tsv file path which contains the data for either aae or wae 
            data, toxic or nontoxic data (1 or 4 possibilities). First and only column 
            contains text data.
        target_size:
            a number representing how big the final file should be after augmenting
            (target size).
        save_path:
            the path to save the new (augmented) data to.
    '''
    # First, convert the tsv entries into a list 
    lines = []
    with open(orig_data_path) as f:
        lines = f.readlines()
        
    lines = list(lines)

    # Make sure this function is being for augmenting, not reducing
    assert target_size >= len(lines)

    # Prompt a language model (LM) to generate synthetic data for the class
    added_lines = []
    lines = added_lines + lines

    # Save the new dataset to the specified save_path location
    with open(save_path, 'w') as f:
        for line in lines:
            f.write(line)

def make_dataset(na, ta, nw, tw, save_path):
    # For D3 dataset
    nontoxic_aae = reduce_dataset("predictions/nontoxic_aae.tsv", na)
    toxic_aae = reduce_dataset("predictions/toxic_aae.tsv", ta)
    nontoxic_wae = reduce_dataset("predictions/nontoxic_wae.tsv", nw)
    toxic_wae = reduce_dataset("predictions/toxic_wae.tsv", tw)

    # Convert lists to tuples with the first value representing 0 for nontoxic
    # and 1 for toxic
    lines = []
    for item in nontoxic_aae:
        lines.append(("0", item))
    for item in nontoxic_wae:
        lines.append(("0", item))
    for item in toxic_aae:
        lines.append(("1", item))
    for item in toxic_wae:
        lines.append(("1", item))

    # Randomize the order
    random.shuffle(lines)

    # Save to the csv
    with open(save_path, 'w') as f:
        for toxicity, line in lines:
            f.write(toxicity + "\t" + line)

if __name__ == "__main__":
    # NUMBER ORDER: na, ta, nw, tw
    # make_dataset(148, 17, 8817, 1018, "../data/debiased/D3/train.tsv")
    # make_dataset(145, 20, 8819, 1016, "../data/debiased/D4/train.tsv")
    make_dataset(4399, 601, 4484, 516, "../data/debiased/D5/train.tsv")
    

