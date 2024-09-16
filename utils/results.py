import json

def get_correct_entries(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    correct_entries = [1 if question['correct'] == 1 else 0 for question in data['questions']]
    
    return correct_entries


def get_mismatch_indices(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length")
    
    mismatch_indices = [i for i, (val1, val2) in enumerate(zip(list1, list2)) if val1 == 0 and val2 == 1]
    
    return mismatch_indices

def get_improvement_entries(improvement_indices, file_path):
    # Load the JSON data
    with open(file_path) as f:
        data = json.load(f)

    # Get the entries by index
    entries = [data['questions'][i] for i in improvement_indices]

    return entries

def compare_indices(list1, list2):
    # Convert lists to sets
    set1 = set(list1)
    set2 = set(list2)

    # Find common elements
    common_elements = set1 & set2
    num_common_elements = len(common_elements)

    # Find uncommon elements
    uncommon_elements = (set1 - set2) | (set2 - set1)
    num_uncommon_elements = len(uncommon_elements)

    # Calculate proportions
    proportion_common = num_common_elements / (num_common_elements + num_uncommon_elements)

    return num_common_elements, proportion_common

