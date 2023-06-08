import pandas
import DataCleaning
import graphviz
import numpy
import sys, os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def clean_db(db_name: str, db_new_name: str):
    db = pandas.read_excel(db_name,  index_col=False)
    db = DataCleaning.remove_out_of_range(db)
    db = DataCleaning.remove_missing_info_lines(db)
    db = DataCleaning.fill_in_numerical_missing_info(db)
    db = DataCleaning.fill_in_binary_missing_info(db)
    db = DataCleaning.convert_categories_to_binary(db)
    
    db.to_excel(db_new_name, index=False)

def print_binary_scores(t_test, t_prediction):
    print(f"accuracy: {accuracy_score(t_test, t_prediction)}")
    print(f"precision: {precision_score(t_test, t_prediction)}")
    print(f"recall: {recall_score(t_test, t_prediction)}")
    print(f"f1: {f1_score(t_test, t_prediction)}")
    print(f"roc auc: {roc_auc_score(t_test, t_prediction)}")

def print_continuous_scores(t_test, t_prediction):
    print(f"mae: {mean_absolute_error(t_test, t_prediction)}")
    print(f"mse: {mean_squared_error(t_test, t_prediction)}")
    print(f"rmse: {numpy.sqrt(mean_squared_error(t_test, t_prediction))}")
    print(f"r2: {r2_score(t_test, t_prediction)}")

#GAIN DECISION TREE
def gain_tree(f_train, f_test, t_train, t_test):
    tree_model = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=1, max_depth=None, max_leaf_nodes=None)
    tree_model.fit(f_train, t_train)

    graph_data = export_graphviz(tree_model, out_file=None, feature_names=f_train.columns, class_names=['0', '1'], filled=True)
    graph = graphviz.Source(graph_data)
    graph.format = "png"
    graph.render('decision_tree_gain', view=False)

    t_prediction = tree_model.predict(f_test)
    print(f"###Gain Scores:")
    print_binary_scores(t_test, t_prediction)

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
    print_binary_scores(t_test, t_prediction)

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
    print_binary_scores(t_test, t_prediction)

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
    print_binary_scores(t_test, t_prediction)

#KNN CLASSIFICATION MODULE
def k_nearest_neighbors(f_train, f_test, t_train, t_test, k):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(f_train, t_train)

    t_prediction = knn_model.predict(f_test)
    print(f"###K Nearest Neighbors Scores (k={k}):")
    print_binary_scores(t_test, t_prediction)

#NAIVE BAYES CLASSIFICATION MODULE
def naive_bayes(f_train, f_test, t_train, t_test):
    nb_model = GaussianNB()
    nb_model.fit(f_train, t_train)

    t_prediction = nb_model.predict(f_test)
    print(f"###Naive Bayes Scores:")
    print_binary_scores(t_test, t_prediction)

#DBSCAN CLUSTERING MODULE
def dbscan_clustering(features: pandas.DataFrame, max_dist, min_samples):
    scaler_model = StandardScaler()
    normalized_features = scaler_model.fit_transform(features)

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(normalized_features)

    dbscan_model = DBSCAN(eps=max_dist, min_samples=min_samples)
    labels = dbscan_model.fit_predict(reduced_features)
    unique_clusters = numpy.unique(labels[labels != -1])

    for cluster in unique_clusters:
        cluster_points = reduced_features[labels==cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')


    noise_points = reduced_features[labels == -1]
    plt.scatter(noise_points[:, 0], noise_points[:, 1], color='black', label='Noise')

    plt.xlabel("Normalized Vector 1")
    plt.ylabel("Normalized Vector 2")
    plt.title('DBSCAN Clustering')
    plt.legend()

    plt.savefig(f'dbscan_{max_dist}_{min_samples}.png')
    plt.clf()

#FFNN CLASSIFICATION MODULE
def feed_forward_neural_network(f_train, f_test, t_train, t_test, layer_size, hidden_layers_amount, verbose=False):
    if not verbose:
        sys.stdout = open(os.devnull, 'w')
    ffnn_model = Sequential()
    ffnn_model.add(Dense(layer_size, activation='relu', input_dim=f_train.shape[1]))
    for _ in range(hidden_layers_amount):
        ffnn_model.add(Dense(layer_size, activation='relu'))
    ffnn_model.add(Dense(1, activation='sigmoid'))

    ffnn_model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mean_squared_error'])

    f_array = numpy.array(f_train)
    f_u = f_array.mean(axis=0, keepdims=True)
    f_std = f_array.std(axis=0, keepdims=True)
    f_array = (f_array - f_u) / f_std

    ffnn_model.fit(f_array, t_train, epochs=10, batch_size=32)

    f_test = (f_test - f_u) / f_std
    t_prediction = ffnn_model.predict(f_test)
    t_prediction = numpy.round(t_prediction)
    ctr = 0
    for t in t_test:
        if (t != t_prediction[ctr]):
            print(f"misclassifed index: {t_test.index[ctr]}")
        ctr+=1

    if not verbose:
        sys.stdout = sys.__stdout__
    print(f"###FeedForward NeuralNetwork Scores ({hidden_layers_amount} hidden layers, {layer_size} neurons per):")
    print_continuous_scores(t_test, t_prediction)

def train_models(db_name: str):
    db = pandas.read_excel(db_name,  index_col=False).drop("id", axis=1)
    
    features = db.drop("classification", axis=1)
    target = db["classification"]

    f_train, f_test, t_train, t_test = train_test_split(features, target, test_size=0.33, random_state=42)
    
    gain_tree(f_train, f_test, t_train, t_test)
    gini_tree(f_train, f_test, t_train, t_test)
    adaboost_tree(f_train, f_test, t_train, t_test)
    random_forest(f_train, f_test, t_train, t_test)
    naive_bayes(f_train, f_test, t_train, t_test)
    k_nearest_neighbors(f_train, f_test, t_train, t_test, 1)
    k_nearest_neighbors(f_train, f_test, t_train, t_test, 5)
    dbscan_clustering(features, 0.5, 5)
    dbscan_clustering(features, 0.75, 4)

    db = DataCleaning.convert_numeric_to_continuous(db)

    features = db.drop("classification", axis=1)
    target = db["classification"]
    
    f_train, f_test, t_train, t_test = train_test_split(features, target, test_size=0.33, random_state=42)

    feed_forward_neural_network(f_train, f_test, t_train, t_test, 64, 2, verbose=True)
    feed_forward_neural_network(f_train, f_test, t_train, t_test, 15, 1)
    feed_forward_neural_network(f_train, f_test, t_train, t_test, 30, 2)
    feed_forward_neural_network(f_train, f_test, t_train, t_test, 30, 3)


if __name__ == "__main__":
    # clean_db("OriginalDB.xlsx", "FilteredDB.xlsx")
    train_models("FilteredDB.xlsx")