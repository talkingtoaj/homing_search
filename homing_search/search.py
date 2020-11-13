import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Concatenate, DenseFeatures
from tensorflow.keras.optimizers import Adagrad, Adam, schedules, RMSprop, SGD, Adadelta
from tensorflow.keras.layers.experimental.preprocessing import Normalization, StringLookup, IntegerLookup, CategoryEncoding
from matplotlib import pyplot
from tensorflow.keras.metrics import MeanSquaredError, CosineSimilarity, MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from typing import List, Tuple, Dict, Any
import random
import time
import math
from .utils import add_to_log, blank_log, moving_average, sort_dict_keys_alphabetically, remove_nan_results, unique_pairs
from .data import KerasAdaptor, SklearnAdaptor

class HomingSearchKeras():
    def __init__(self, **kwargs):
        self.build_obj = kwargs['build_obj']
        if kwargs['ml_library'] in ['keras', 'sklearn']:
            if kwargs['ml_library'] == 'keras':
                self.interface = KerasAdaptor(**kwargs)
            elif kwargs['ml_library'] == 'sklearn':
                self.interface = SklearnAdaptor(**kwargs)
        self.interface.prepare_data()

    def start(self, params:Dict[List], repeats=2, epochs=100, time_limit=5):
        """ 
            Starts homing search 

            params: is a dictionary of parameters used by build_fn and fit.
                Each parameter key should be assigned a List object of possible combinations.
                learning_rate and batch_size will be set to defaults of 0.1 and 256 if not provided.
                To set a single value, provide a list with a single element
                e.g. 
                    params = {
                        'g1dim': [5,20,40], 'g2dim': [1,2,4], 
                        'ga':['relu', 'tanh', 'softplus', 'elu'],
                        'optimizer':['Adagrad','SGD','RMSprop','adam', 'Adadelta'], 
                        'learning_rate':[0.1, 0.01], 
                        'batch_size': [256], # batch size will not vary
                    }   
            time_limit: measured in minutes 
        """
        blank_log()
        # percentage of all results obtained to consult in determining params for subsequent rounds, will go down with each round
        self.best_fraction = 0.3 
        self.new_params = params
        self.repeats = repeats
        self.epochs = epochs
        self.all_results = {} # key=metric, value=responsible parameters
        sort_dict_keys_alphabetically(self.new_params)
        add_to_log(f"starting params:\n{self.new_params}\n")
        self.finish_time = time.time() + 60 * time_limit
        while self.finish_time > time.time():
            self.calc_round_finish_time()
            options = self.create_options()
            add_to_log(f"time remaining: {round((self.finish_time-time.time())/60)} minutes\nGenerated up to {len(options)} options to examine\n")
            if len(options) == 0:
                # TODO: run final test for results
                break
            results = self.fit_all_options(options)
            self.all_results.update(results)
            self.determine_new_param_range()
            add_to_log(f"\nnew param range:\n {(self.new_params)}\n")
            if self.final_params_reached():
                break
        self.present_findings()

    def calc_round_finish_time(self, fraction_of_total_per_round=0.33):
        self.round_alloted_time = (self.finish_time - time.time()) * fraction_of_total_per_round

    def present_findings(self):
        sorted_results = sorted(self.all_results.items()) # List[Tuple[float,dict]]
        top_num = min(3, len(sorted_results))
        add_to_log(f"\n\nTop {top_num} results are:\n")
        for i in range(top_num):
            add_to_log(f"i: {sorted_results[i]}\n")  

    def fit_all_options(self, options:List[dict]):
        round_finish_time = self.round_alloted_time + time.time()
        results = {} # key:metric score, value: params
        for i, parameter_option in enumerate(options):
            parameter_option = dict(sorted(parameter_option.items()))
            if round_finish_time:
                if time.time() > round_finish_time:
                    break
                remaining_time = round_finish_time - time.time()
                pc = (1-remaining_time/self.round_alloted_time)*100
                add_to_log(f"time remaining: {round((remaining_time)/60)} minutes. {pc:.0f}% progress - attempting param combination: {parameter_option}\n")
            else:                        
                add_to_log(f"{i+1} of {len(options)} - attempting param combination: {parameter_option}\n")
            results.update(
                self.fit_option(parameter_option, save_best=False)
            )
        self.all_results.update(results)

    def fit_option(self, parameter_option, save_best=False) -> dict[float:Any]:
        """ compiles a model for parameter_option, trains and reports the performance metric """
        parameter_option = dict(sorted(parameter_option.items()))
        add_to_log(f"attempting combination:\n{parameter_option}\n")
        o = parameter_option
        scores = []
        for i in range(self.repeats):
            # need to rebuild model each time, or weights perpetuate from previous attempt            
            score = self.interface.fit(
                    model = self.build_obj.build(**o),
                    metric = 'val_mean_absolute_percentage_error',
                    batch_size = o['batch_size'],
                    learning_rate = o['learning_rate'],
                    save_best = save_best,
                )
            add_to_log(f"score achieved: {score:.2f}%\n") 
            scores.append(score)
        mean_score = sum(scores)/len(scores)
        add_to_log(f"mean score: {mean_score:.2f}%\n\n")
        return {mean_score:parameter_option}

    def determine_new_param_range(self):
        """ updates self.new_params to reflect appearance in the top results so far """
        self.all_results = remove_nan_results(self.all_results)
        best_results = self.select_top_results()
        unique_param_values = self.unique_param_values(best_results)

        old_params = self.new_params
        new_params = {}
        for key,value_list in unique_param_values.items():
            _type = type(value_list[0])
            if _type in [int, float]:
                old_range = max(old_params[key]) - min(old_params[key])
                new_range = old_range*0.3
                _mean = sum(value_list)/len(value_list)
                _min = _type(_mean - new_range/2)
                old_range_went_neg = min(old_params[key]) < 0
                old_range_below_one = min(old_params[key]) < 1
                if not old_range_went_neg:
                    _min = _type(max(_min, 0.0))
                _max = _type(_mean + new_range/2)
                if _type is int and not old_range_below_one:
                    _min = max(1,_min)
                    _max = max(1,_max)
                _range = _max - _min

                # simplify range if possible
                if _range == 0:
                    new_params[key] = [_min]
                else:
                    _mid = _type((_min+_max)/2)
                    # if range is insignificant, discontinue ranging
                    if _range/_mid < 0.03:
                        new_params[key] = [_mid]
                    else:
                        if _mid == _min or _mid == _max:
                            new_params[key] = [_min, _max]
                        else:
                            new_params[key] = [_min, _mid, _max]
                    # if we are stuck in a loop, repeating the same values
                    if old_params[key] == new_params[key]:
                        # then only include the best value(s) found
                        new_params[key] = list(set(value_list))
            else:                
                """ we take a middle road between featuring each unique value at least once and 
                    overloading the parameter range with duplicates to reflect the frequency of 
                    each value in best_results
                """
                random.shuffle(value_list)
                unique_values = list(set(value_list))
                value_list = value_list[:len(unique_values) * 3]
                for value in unique_values:
                    # ensure every value is represented at least once
                    if value not in value_list:
                        value_list.append(value)                
                new_params[key] = value_list 
        new_params = dict(sorted(new_params.items()))
        self.new_params = new_params

    def unique_param_values(self, result_set: List[Tuple[float,dict]]) -> dict[str:list]:
        param_range = {}        
        for score, params in result_set:
            for param_name, param_value in params.items():
                values = param_range.get(param_name,[])
                values.append(param_value)
                param_range[param_name] = values
        return param_range

    def final_params_reached(self):
        """ if there is only one value left per param range """
        for value_list in self.new_params.values():
            if len(value_list) > 1:
                return False
        if self.new_params.values():
            return {key:value[0] for key,value in self.new_params.items()}
        else:
            raise AttributeError(f"params dict was empty")

    def select_top_results(self) -> List[Tuple[float,dict]]:
        """ 
            Each round will less than double the number of stored results. 
            By halving the percentage of results we consider each round, 
            we narrow in on only the very best results
        """        
        sorted_results = sorted(self.all_results.items()) # List[Tuple[float,dict]]
        top_fraction = max(3,math.ceil(len(sorted_results) * self.best_fraction))
        self.best_fraction *= 0.5 
        return sorted_results[:int(top_fraction)]

    def create_options(self):            
        options = self.random_selections()
        options = self.remove_repeats(options)
        return options

    def random_selections(self, max_selections=200):
        def add_essential_params(option):
            defaults = {
                'learning_rate': 0.1, 
                'batch_size': 256,
            }
            for key, default in defaults.items():
                if key not in option.keys():                
                    option[key] = default

        options = []
        for i in range(max_selections):
            option = {}
            for key,value_list in self.new_params.items():
                option[key] = random.choices(value_list)[0]
            options.append(option)
            add_essential_params(option)        
        return options

    def remove_repeats(self, options):
        """ removes repeats due to previous runs or random selection """
        previous_options = [option for option in self.all_results.values()]
        refined_list = [option for option in options if option not in previous_options]

        # remove clone repeats, due to random selection
        clone_idxs = []
        for i, j in unique_pairs(len(refined_list)):
            found_difference = False
            i_entry = refined_list[i]
            j_entry = refined_list[j]
            for key,value in i_entry.items():
                _type = type(i_entry[key])
                if _type is float:
                    if not math.isclose(i_entry[key], j_entry[key], rel_tol=1e-5):
                        found_difference = True
                        break # is different, find a new unique_pair
                else:
                    if i_entry[key] != j_entry[key]:
                        found_difference = True
                        break # is different, find a new unique_pair
            if not found_difference: 
                clone_idxs.append(i)
        clone_idxs = sorted(list(set(clone_idxs)), reverse=True)
        for idx in clone_idxs:
            del refined_list[idx]
        return refined_list