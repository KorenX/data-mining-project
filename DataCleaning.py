import pandas
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier

MISSING_THRESHOLD = 5
MIN_MAX_OFFSET = 1

id_col = "id"
missing_values = ("?",)
ranges_dict = {"blood pressure" : (50, 180), "blood glucose random": (70, 490), "serum creatinine": (0.4, 10), "sodium" : (104, 163), "potassium": (2.5, 7.6), "hemoglobin": (6, 17.8), "white blood cell count": (2200, 16700), "red blood cell count": (2.1, 6.5)}
numeric_fields = ["age", "blood pressure", "specific gravity", "albumin", "sugar", "blood glucose random", "serum creatinine", "sodium", "potassium", "hemoglobin", "packed cell volume", "red blood cell count", "white blood cell count"]
categoric_fields = ["red blood cells", "pus cell", "pus cell clumps", "bacteria", "diabetes mellitus", "coronary artery disease", "appetite", "pedal edema", "anemia", "classification"]
positive_values = ["yes", "normal", "present", "good", "ckd"]

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
    for field in categoric_fields:
        col = db[(db[field] == '?')]
        if (len(col) > 0):
            db = _fill_in_col(db, field)
    return db
    
def _fill_in_col(db: pandas.DataFrame, replace_col: str) -> pandas.DataFrame:
    remove_lines = categoric_fields + [id_col]
    remove_lines.remove(replace_col)
    learn_db = db.drop(remove_lines, axis=1)
    learn_db = learn_db[learn_db[replace_col] != '?']

    tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    tree_model.fit(learn_db.drop(replace_col, axis=1), learn_db[replace_col])

    fill_db = db[db[replace_col] == '?']
    fill_db = fill_db.drop(remove_lines + [replace_col], axis=1)
    filled_col = tree_model.predict(fill_db)
    db.loc[db[replace_col] == '?', replace_col] = filled_col
    return db
