import pandas
import DataCleaning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import graphviz

def clean_db(db_name: str, db_new_name: str):
    db = pandas.read_excel(db_name,  index_col=False)
    db = DataCleaning.remove_out_of_range(db)
    db = DataCleaning.remove_missing_info_lines(db)
    db = DataCleaning.fill_in_numerical_missing_info(db)
    db = DataCleaning.fill_in_binary_missing_info(db)
    db = DataCleaning.convert_categories_to_binary(db)
    
    db.to_excel(db_new_name, index=False)

def print_scores(t_test, t_prediction):
    print(f"accuracy: {accuracy_score(t_test, t_prediction)}")
    print(f"precision: {precision_score(t_test, t_prediction)}")
    print(f"recall: {recall_score(t_test, t_prediction)}")
    print(f"f1: {f1_score(t_test, t_prediction)}")
    print(f"roc auc: {roc_auc_score(t_test, t_prediction)}")

#GAIN DECISION TREE
def gain_tree(f_train, f_test, t_train, t_test):
    tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=None)
    tree_model.fit(f_train, t_train)

    graph_data = export_graphviz(tree_model, out_file=None, feature_names=f_train.columns, class_names=['0', '1'], filled=True)
    graph = graphviz.Source(graph_data)
    graph.format = "png"
    graph.render('decision_tree_gain', view=False)

    t_prediction = tree_model.predict(f_test)
    print(f"###Gain Scores:")
    print_scores(t_test, t_prediction)

#GINI DECISION TREE
def gini_tree(f_train, f_test, t_train, t_test):
    tree_model = DecisionTreeClassifier()
    tree_model.fit(f_train, t_train)

    graph_data = export_graphviz(tree_model, out_file=None, feature_names=f_train.columns, class_names=['0', '1'], filled=True)
    graph = graphviz.Source(graph_data)
    graph.format = "png"
    graph.render('decision_tree_gini', view=False)

    t_prediction = tree_model.predict(f_test)
    print(f"###Gini Scores:")
    print_scores(t_test, t_prediction)

#ADABOOST DECISION TREE
def adaboost_tree(f_train, f_test, t_train, t_test):
    tree_model = DecisionTreeClassifier()

    ada_model = AdaBoostClassifier(base_estimator=tree_model)
    ada_model.fit(f_train, t_train)

    graph_data = export_graphviz(ada_model.estimators_[0], out_file=None, feature_names=f_train.columns, class_names=['0', '1'], filled=True)
    graph = graphviz.Source(graph_data)
    graph.format = "png"
    graph.render('decision_tree_adaboost', view=False)

    t_prediction = ada_model.predict(f_test)
    print(f"###AdaBoost Scores:")
    print_scores(t_test, t_prediction)

#RANDOM DECISION FOREST
def random_forest(f_train, f_test, t_train, t_test):
    forest_model = RandomForestClassifier(criterion="entropy", max_depth=None, n_estimators=10)
    forest_model.fit(f_train, t_train)

    graph_data = export_graphviz(forest_model.estimators_[0], out_file=None, feature_names=f_train.columns, class_names=['0', '1'], filled=True)
    graph = graphviz.Source(graph_data)
    graph.format = "png"
    graph.render('decision_forest_gain', view=False)

    t_prediction = forest_model.predict(f_test)
    print(f"###Random Forest Scores:")
    print_scores(t_test, t_prediction)

def train_models(db_name: str):
    db = pandas.read_excel(db_name,  index_col=False).drop("id", axis=1)
    
    features = db.drop("classification", axis=1)
    target = db["classification"]

    f_train, f_test, t_train, t_test = train_test_split(features, target, test_size=0.33, random_state=42)
    
    gain_tree(f_train, f_test, t_train, t_test)
    gini_tree(f_train, f_test, t_train, t_test)
    adaboost_tree(f_train, f_test, t_train, t_test)
    random_forest(f_train, f_test, t_train, t_test)


if __name__ == "__main__":
    clean_db("OriginalDB.xlsx", "FilteredDB.xlsx")
    train_models("FilteredDB.xlsx")