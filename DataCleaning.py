import pandas
import math
import ID3
import numpy as np

MISSING_THRESHOLD = 5
MIN_MAX_OFFSET = 1

id_col = "id"
missing_values = ("?",)
ranges_dict = {"blood pressure" : (50, 180), "blood glucose random": (70, 490), "serum creatinine": (0.4, 10), "sodium" : (104, 163), "potassium": (2.5, 7.6), "hemoglobin": (6, 17.8), "white blood cell count": (2200, 16700), "red blood cell count": (2.1, 6.5)}
numeric_fields = ("age", "blood pressure", "specific gravity", "albumin", "sugar", "blood glucose random", "serum creatinine", "sodium", "potassium", "hemoglobin", "packed cell volume", "red blood cell count", "white blood cell count")
categoric_fields = ("red blood cells", "pus cell", "pus cell clumps", "bacteria", "diabetes mellitus", "coronary artery disease", "appetite", "pedal edema", "anemia", "classification")
positive_values = ("yes", "normal", "present", "good", "ckd")

def remove_out_of_range(db : pandas.DataFrame) -> pandas.DataFrame:
    for field, range in ranges_dict.items():
        db[field] = pandas.to_numeric(db[field], errors='coerce').fillna(range[0]-1)
        db.loc[(db[field] > range[1]) | (db[field] < range[0]), field] = '?'
    return db

def remove_missing_info_lines(db : pandas.DataFrame) -> pandas.DataFrame:
    count = db.astype(str).apply(lambda x: x.str.count('\?')).sum(axis=1)
    return db[count < MISSING_THRESHOLD]

def fill_in_numerical_missing_info(db : pandas.DataFrame) -> pandas.DataFrame:
    for field in numeric_fields:
        mean = pandas.to_numeric(db[field], errors='coerce').mean()
        db["m_"+field] = db[field].map(lambda x: mean if x in missing_values else x)
        db = db.drop(field, axis=1)
        db = db.rename(columns={"m_"+field: field})
    return db

def do_binning(db : pandas.DataFrame) -> pandas.DataFrame:
    for field in numeric_fields:
        col = pandas.to_numeric(db[field], errors='coerce')
        bins = np.linspace(col.min() - MIN_MAX_OFFSET, col.max() + MIN_MAX_OFFSET, int(math.sqrt(len(db))))
        db[field] = pandas.cut(col, bins=bins, include_lowest=True)
    return db

def convert_categories_to_binary(db: pandas.DataFrame) -> pandas.DataFrame:
    for field in categoric_fields:
        db["b_"+field] = db[field].map(lambda x: 1 if x in positive_values else 0)
        db = db.drop(field, axis=1)
        db = db.rename(columns={"b_"+field: field})
    return db

def fill_in_binary_missing_info(db: pandas.DataFrame) -> pandas.DataFrame:
    ID3.MAX_LAYERS = 2
    for field in categoric_fields:

        col = db[(db[field] == '?')]
        if (len(col) > 0):
            db = _fill_in_col(db, field)
    return db
    
def _fill_in_col(db: pandas.DataFrame, replace_col: str) -> pandas.DataFrame:
    learn_db = db.drop([id_col], axis=1)
    learn_db = learn_db[learn_db[replace_col] != '?']
    learn_db = do_binning(learn_db)
    table = ID3.generate_decision_tree(learn_db, replace_col, ID3.gain_function)
    missing_db = db[db[replace_col] == '?']
    for id in missing_db[id_col]:
        db = _fill_by_id(db, table, id, replace_col)
    return db

def _fill_by_id(db: pandas.DataFrame, replace_table : dict, id: int, replace_col : str) -> pandas.DataFrame:
    while True:
        if type(replace_table) is dict:
            for field, interval in replace_table:
                if db.loc[(db[id_col] == id), field].iloc[0] in interval:
                    replace_table = replace_table[(field, interval)]
                    break
        else:
            db.loc[(db[id_col] == id), replace_col] = replace_table
            break
    return db