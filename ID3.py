import pandas
import math
from sklearn.model_selection import train_test_split
import pprint
MAX_LAYERS = 4

def train_and_test(database: pandas.DataFrame, to_predict: str, determine_function):
    pass

def _predict_value(row: pandas.Series, decision_tree: dict):
    a = {}
    b = decision_tree
    while True:
            if a == decision_tree:
                pprint.pprint(decision_tree)
                pprint.pprint(row)
                print("error")
                exit(1)
            if type(decision_tree) is dict:
                a = decision_tree
                for field, interval in decision_tree:
                    if row[field] in interval:
                        decision_tree = decision_tree[(field, interval)]
                        break
            else:
                return decision_tree

def generate_decision_tree(database: pandas.DataFrame, to_predict: str, determine_function):
    return determine_layer(database, to_predict, determine_function, 0)

def determine_layer(database: pandas.DataFrame, to_predict: str, determine_function, layer_index: int):
    layer = choose_field(database, to_predict, determine_function)
    if layer == "" or layer_index >= MAX_LAYERS:
        return database[to_predict].array[0]
    res = {}
    for value in sorted(database[layer].unique()):
        filtered_db = database[database[layer] == value].drop(layer, axis=1)
        res[(layer, value)] = determine_layer(filtered_db, to_predict, determine_function, layer_index + 1)
    return res

def choose_field(database: pandas.DataFrame, to_predict: str, determine_function) -> str:
    grades_db = {}
    for col in database.columns:
        if col != to_predict:
            grades_db[col] = determine_function(database, col, to_predict)
    
    max_key = max(grades_db, key=lambda k: grades_db[k])
    if (grades_db[max_key] == 0):
        return "" # in this case, neither option gains us any certainty
    
    return max_key

def gain_function(database: pandas.DataFrame, filter_column: str, to_predict: str) -> float:
    return calculate_single_entropy(database[to_predict]) - calculate_double_entropy(database[[filter_column, to_predict]], to_predict)

def calculate_single_entropy(database: pandas.DataFrame):
    predictions_db = {}
    for field in database:
        if field in predictions_db.keys():
            predictions_db[field] += 1
        else:
            predictions_db[field] = 1
    
    return sum([(-1) * p * math.log2(p) for p in [val / len(database) for val in predictions_db.values()]])

def calculate_double_entropy(database: pandas.DataFrame, to_predict: str):
    cols = [col for col in database.columns]
    cols.remove(to_predict)
    lookup_col = cols[0]

    entropy = 0
    for value in sorted(database[lookup_col].unique()):
        filtered_db = database[database[lookup_col] == value][to_predict]
        entropy += (calculate_single_entropy(filtered_db) * len(filtered_db) / len(database))

    return entropy
