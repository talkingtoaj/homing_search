def add_to_log(text, test_fname='test_data'):
    with open(test_fname, "a+") as write_file:
        write_file.write(text)
    print(text)

def blank_log():
    with open('test_data', "w") as write_file:
        write_file.write("")

def moving_average(values:List[float])->float:
    """ A very simple function to average a time-series of values favouring later values """
    avg = values[0]
    for val in values:
        avg = avg*0.4 + val*0.6
    return avg     

def sort_dict_keys_alphabetically (_dict:dict):
    _dict = dict(sorted(_dict.items()))

def remove_nan_results(results):
    return {key:val for key,val in results.items() if str(key) != 'nan'}

def unique_pairs(n):
    """ generates tuples of all pair combinations of integers up to n """
    for i in range(n):
        for j in range(i+1, n):
            yield i, j
