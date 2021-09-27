import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
from sklearn import svm
from sklearn import neighbors
from sklearn import utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

seed = 69

def generate_decision_tree_curves(train_X, test_X, train_Y, test_Y):
    # Creating initial classifier with no pruning
    dtClassifer = tree.DecisionTreeClassifier(random_state=seed)
    dtClassifer.fit(train_X, train_Y)

    predict_Y = dtClassifer.predict(train_X)

    print("Unpruned Decision Tree accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = dtClassifer.predict(test_X)

    print("Unpruned Decision Tree accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))


    # Create decision tree with pruning

    # tuning for max_leaf_nodes
    max_leaf_nodes_range = np.arange(10, 501, 10)
    train_scores, test_scores = validation_curve(tree.DecisionTreeClassifier(random_state=seed), train_X, train_Y, param_name="max_leaf_nodes", param_range=max_leaf_nodes_range, cv = 5, scoring = "precision", n_jobs=4)
    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    best_index = np.argmax(avg_test_scores)
    best_max_leaf_nodes = max_leaf_nodes_range[best_index]
    print("Best validation score for max_leaf_nodes: ", avg_test_scores[best_index])
    print("Best max_leaf_nodes: ", best_max_leaf_nodes)

    plt.title("Decision tree precision when tuning max_leaf_nodes")
    plt.xlabel("max_leaf_nodes")
    plt.ylabel("Precision")
    plt.plot(max_leaf_nodes_range, avg_train_scores, label="Training precision")
    plt.plot(max_leaf_nodes_range, avg_test_scores, label="Cross-validation precision")
    plt.legend()
    plt.savefig('figures/wine_dt_validation_curve_max_leaf_nodes.png')
    plt.clf()

    # Tuning for max_depth
    max_depth_range = np.arange(1, 51)
    train_scores, test_scores = validation_curve(tree.DecisionTreeClassifier(random_state=seed), train_X, train_Y, param_name="max_depth", param_range=max_depth_range, cv = 5, scoring = "precision", n_jobs=4)
    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    best_index = np.argmax(avg_test_scores)
    best_max_depth = max_depth_range[best_index]
    print("Best validation score for max_depth: ", avg_test_scores[best_index])
    print("Best max_depth: ", best_max_depth)

    plt.title("Decision tree precision when tuning max_depth")
    plt.xlabel("max_depth")
    plt.ylabel("precision")
    plt.plot(max_depth_range, avg_train_scores, label="Training precision")
    plt.plot(max_depth_range, avg_test_scores, label="Cross-validation precision")
    plt.legend()
    plt.savefig('figures/wine_dt_validation_curve_max_depth.png')
    plt.clf()

    # Random search for best parameters
    param_dict = {"max_depth": max_depth_range, "max_leaf_nodes": np.arange(50, 201, 10), "ccp_alpha": [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]}

    clf = tree.DecisionTreeClassifier(random_state=seed)
    random_search = RandomizedSearchCV(clf, param_dict, random_state=seed, scoring='precision', cv=5, n_iter=50, n_jobs=4)
    random_search = random_search.fit(train_X, train_Y)

    params = random_search.best_params_

    print(params)
    
    # plot learning curve
    clf = tree.DecisionTreeClassifier(random_state=seed, max_leaf_nodes=params['max_leaf_nodes'], max_depth=params['max_depth'], ccp_alpha=params['ccp_alpha'])
    train_sizes, train_scores, test_scores = learning_curve(clf, train_X, train_Y, cv=5, train_sizes=np.arange(0.05, 1.01, 0.05), n_jobs=4, scoring='accuracy')

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    plt.title("Decision tree learning curve")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.plot(train_sizes, avg_train_scores, label="Training accuracy")
    plt.plot(train_sizes, avg_test_scores, label="Cross-validation accuracy")
    plt.legend()
    plt.savefig('figures/wine_dt_learning_curve.png')
    plt.clf()

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Pruned Decision Tree accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Pruned Decision Tree accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    print(confusion_matrix(test_Y, predict_Y))

def generate_neural_net_curves(train_X, test_X, train_Y, test_Y):
    # Scale data for all features

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    # Creating initial classifier with no hyper parameter tuning
    clf = neural_network.MLPClassifier(random_state=seed, solver='sgd', max_iter=2000, learning_rate='adaptive', learning_rate_init=0.02, early_stopping=True)

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Unoptimized Neural Network accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Unoptimized Neural Network accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    # Tuning hyperparameters
    num_hidden_units_range = []
    hidden_units_to_plot = []
    for i in range(3, 31, 3):
        num_hidden_units_range.append((i,i))
        hidden_units_to_plot.append(i)

    clf = neural_network.MLPClassifier(random_state=seed, solver='sgd', max_iter=2000, learning_rate='adaptive', learning_rate_init=0.02, early_stopping=True)

    train_scores, test_scores = validation_curve(clf, train_X, train_Y, param_name="hidden_layer_sizes", param_range=num_hidden_units_range, cv = 5, scoring = "precision", n_jobs=4)

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    best_index = np.argmax(avg_test_scores)
    best_num_hidden_units = num_hidden_units_range[best_index]
    print("Best validation score for number of hidden units: ", avg_test_scores[best_index])
    print("Best number of hidden units: ", best_num_hidden_units)

    plt.title("Neural Network precision when tuning number of hidden units")
    plt.xlabel("Number of hidden units")
    plt.ylabel("Precision")
    plt.plot(hidden_units_to_plot, avg_train_scores, label="Training precision")
    plt.plot(hidden_units_to_plot, avg_test_scores, label="Cross-validation precision")
    plt.legend()
    plt.savefig('figures/wine_nn_validation_curve_num_hidden_units.png')
    plt.clf()
    
    # Tuning for alpha
    alpha_range = [0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

    train_scores, test_scores = validation_curve(neural_network.MLPClassifier(random_state=seed, solver='sgd', max_iter=2000, learning_rate='adaptive', learning_rate_init=0.02, early_stopping=True, hidden_layer_sizes=best_num_hidden_units), train_X, train_Y, param_name="alpha", param_range=alpha_range, cv = 5, scoring = "precision", n_jobs=4)

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    best_index = np.argmax(avg_test_scores)
    best_alpha = alpha_range[best_index]
    print("Best validation score for alpha: ", avg_test_scores[best_index])
    print("Best alpha: ", best_alpha)

    plt.title("Neural Network precision when tuning alpha")
    plt.xlabel("alpha")
    plt.xscale("log")
    plt.ylabel("precision")
    plt.plot(alpha_range, avg_train_scores, label="Training precision")
    plt.plot(alpha_range, avg_test_scores, label="Cross-validation precision")
    plt.legend()
    plt.savefig('figures/wine_nn_validation_curve_alpha.png')
    plt.clf()

    # Random search for best parameters
    param_dict = {"hidden_layer_sizes": num_hidden_units_range, "alpha": alpha_range}

    clf = neural_network.MLPClassifier(random_state=seed, solver='sgd', max_iter=2000, learning_rate='adaptive', learning_rate_init=0.02, early_stopping=True)
    random_search = RandomizedSearchCV(clf, param_dict, random_state=seed, scoring='precision', cv=5, n_iter=50, n_jobs=4)
    random_search = random_search.fit(train_X, train_Y)

    params = random_search.best_params_

    print(params)
    
    # Plot learning curve
    clf = neural_network.MLPClassifier(random_state=seed, solver='sgd', max_iter=2000, learning_rate='adaptive', early_stopping=True, hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=0.02, alpha=params['alpha'])
    train_sizes, train_scores, test_scores = learning_curve(clf, train_X, train_Y, cv=5, train_sizes=np.arange(0.05, 1.01, 0.05), n_jobs=4, scoring='accuracy')

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    plt.title("Neural Network learning curve")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.plot(train_sizes, avg_train_scores, label="Training accuracy")
    plt.plot(train_sizes, avg_test_scores, label="Cross-validation accuracy")
    plt.legend()
    plt.savefig('figures/wine_nn_learning_curve.png')
    plt.clf()

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Optimized Neural Network accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Optimized Neural Network accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    print(confusion_matrix(test_Y, predict_Y))

def generate_boosting_curves(train_X, test_X, train_Y, test_Y):
    # Creating initial classifier with no hyper parameter tuning
    clf = ensemble.AdaBoostClassifier(random_state=seed)
    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Unoptimized Adaboost accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Unoptimized Adaboost accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    # Tuning hyperparameters

    # Tuning for decision tree max depth

    max_depth_range = np.arange(1, 21)

    test_scores = []
    train_scores = []

    for max_depth in max_depth_range:
        clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(random_state=seed, max_depth=max_depth), random_state=seed)
        scores = cross_validate(clf, train_X, train_Y, cv = 5, scoring = "precision", return_train_score=True)
        train_scores.append(scores['train_score'])
        test_scores.append(scores['test_score'])

    test_scores = np.array(test_scores)
    train_scores = np.array(train_scores)

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    best_index = np.argmax(avg_test_scores)
    best_max_depth = max_depth_range[best_index]
    print("Best validation score for max_depth: ", avg_test_scores[best_index])
    print("Best max_depth: ", best_max_depth)

    plt.title("Adaboost precision when tuning decision tree max_depth")
    plt.xlabel("Decision tree max_depth")
    plt.ylabel("Precision")
    plt.plot(max_depth_range, avg_train_scores, label="Training precision")
    plt.plot(max_depth_range, avg_test_scores, label="Cross-validation precision")
    plt.legend()
    plt.savefig('figures/wine_adaboost_validation_curve_max_depth.png')
    plt.clf()

    # Tuning for n_estimators
    n_estimators_range = np.arange(10, 201, 10)
    clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(random_state=seed, max_depth=best_max_depth), random_state=seed)
    train_scores, test_scores = validation_curve(clf, train_X, train_Y, param_name="n_estimators", param_range=n_estimators_range, cv = 5, scoring = "precision", n_jobs=4)

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    best_index = np.argmax(avg_test_scores)
    best_n_estimators = n_estimators_range[best_index]
    print("Best validation score for n_estimators: ", avg_test_scores[best_index])
    print("Best n_estimators: ", best_n_estimators)

    plt.title("Adaboost precision when tuning number of estimators")
    plt.xlabel("Number of estimators")
    plt.ylabel("Precision")
    plt.plot(n_estimators_range, avg_train_scores, label="Training precision")
    plt.plot(n_estimators_range, avg_test_scores, label="Cross-validation precision")
    plt.legend()
    plt.savefig('figures/wine_adaboost_validation_curve_n_estimators.png')
    plt.clf()

    # Random search for best parameters
    param_dict = {"base_estimator__max_depth": np.arange(1, 21), "base_estimator__ccp_alpha": [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]}

    clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(random_state=seed), random_state=seed)
    random_search = RandomizedSearchCV(clf, param_dict, random_state=seed, scoring='precision', cv=5, n_iter=50, n_jobs=4)
    random_search = random_search.fit(train_X, train_Y)

    params = random_search.best_params_

    print(params)

    # plot learning curve
    clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(random_state=seed, max_depth=params['base_estimator__max_depth'], ccp_alpha=params['base_estimator__ccp_alpha']), random_state=seed)
    train_sizes, train_scores, test_scores = learning_curve(clf, train_X, train_Y, cv=5, train_sizes=np.arange(0.05, 1.01, 0.05), n_jobs=4)

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    plt.title("Adaboost learning curve")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.plot(train_sizes, avg_train_scores, label="Training accuracy")
    plt.plot(train_sizes, avg_test_scores, label="Cross-validation accuracy")
    plt.legend()
    plt.savefig('figures/wine_adaboost_learning_curve.png')
    plt.clf()

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Optimized Adaboost accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Optimized Adaboost accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    print(confusion_matrix(test_Y, predict_Y))

def generate_knn_curves(train_X, test_X, train_Y, test_Y):
    # Scale data
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    clf = neighbors.KNeighborsClassifier()

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Unoptimized KNN accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Unoptimized KNN accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    # Tuning hyperparameters

    # Tuning for n_neighbors

    n_neighbors_range = np.arange(1, 21)

    clf = neighbors.KNeighborsClassifier()

    train_scores, test_scores = validation_curve(clf, train_X, train_Y, param_name="n_neighbors", param_range=n_neighbors_range, cv = 5, scoring = "precision", n_jobs=4)

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    best_index = np.argmax(avg_test_scores)
    best_n_neighbors = n_neighbors_range[best_index]
    print("Best validation score for n_neighbors: ", avg_test_scores[best_index])
    print("Best n_neighbors: ", best_n_neighbors)

    plt.title("KNN precision when tuning number of neighbors")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Precision")
    plt.plot(n_neighbors_range, avg_train_scores, label="Training precision")
    plt.plot(n_neighbors_range, avg_test_scores, label="Cross-validation precision")
    plt.legend()
    plt.savefig('figures/wine_knn_validation_curve_n_neighbors.png')
    plt.clf()

    # Tuning for n_neighbors with distance weights
    n_neighbors_range = np.arange(1, 21)

    clf = neighbors.KNeighborsClassifier(weights='distance')

    train_scores, test_scores = validation_curve(clf, train_X, train_Y, param_name="n_neighbors", param_range=n_neighbors_range, cv = 5, scoring = "precision", n_jobs=4)

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    best_index = np.argmax(avg_test_scores)
    best_n_neighbors = n_neighbors_range[best_index]
    print("Best validation score for n_neighbors: ", avg_test_scores[best_index])
    print("Best n_neighbors: ", best_n_neighbors)

    plt.title("KNN precision when tuning number of neighbors with distance weights")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Precision")
    plt.plot(n_neighbors_range, avg_train_scores, label="Training precision")
    plt.plot(n_neighbors_range, avg_test_scores, label="Cross-validation precision")
    plt.legend()
    plt.savefig('figures/wine_knn_validation_curve_n_neighbors_dist_weights.png')
    plt.clf()

    # plot learning curve
    clf = neighbors.KNeighborsClassifier(n_neighbors=2)
    train_sizes, train_scores, test_scores = learning_curve(clf, train_X, train_Y, cv=5, train_sizes=np.arange(0.05, 1.01, 0.05), n_jobs=4)

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    plt.title("KNN learning curve")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.plot(train_sizes, avg_train_scores, label="Training accuracy")
    plt.plot(train_sizes, avg_test_scores, label="Cross-validation accuracy")
    plt.legend()
    plt.savefig('figures/wine_knn_learning_curve.png')
    plt.clf()

    clf = neighbors.KNeighborsClassifier(n_neighbors=2)
    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Optimized KNN accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Optimized KNN accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    print(confusion_matrix(test_Y, predict_Y))

def generate_svm_curves(train_X, test_X, train_Y, test_Y):
    # Scale data
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    clf = svm.SVC(random_state=seed)

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Unoptimized SVM accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Unoptimized SVM accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    # Tuning hyper parameters

    # Tuning for polynomial degree
    clf = svm.SVC(random_state=seed, kernel='poly')

    degree_range = np.arange(11)

    train_scores, test_scores = validation_curve(clf, train_X, train_Y, param_name="degree", param_range=degree_range, cv = 5, scoring = "precision", n_jobs=4)

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    best_index = np.argmax(avg_test_scores)
    best_degree = degree_range[best_index]
    print("Best validation score for degree: ", avg_test_scores[best_index])
    print("Best degree: ", best_degree)

    plt.title("SVM precision when tuning polynomial degree")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Precision")
    plt.plot(degree_range, avg_train_scores, label="Training precision")
    plt.plot(degree_range, avg_test_scores, label="Cross-validation precision")
    plt.legend()
    plt.savefig('figures/wine_svm_validation_curve_degree.png')
    plt.clf()

    # Tuning for C
    clf = svm.SVC(random_state=seed, kernel='rbf')

    C_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

    train_scores, test_scores = validation_curve(clf, train_X, train_Y, param_name="C", param_range=C_range, cv = 5, scoring = "precision", n_jobs=4)

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    best_index = np.argmax(avg_test_scores)
    best_C = C_range[best_index]
    print("Best validation score for C: ", avg_test_scores[best_index])
    print("Best C: ", best_C)

    plt.title("SVM precision when tuning C")
    plt.xlabel("C")
    plt.ylabel("Precision")
    plt.xscale("log")
    plt.plot(C_range, avg_train_scores, label="Training precision")
    plt.plot(C_range, avg_test_scores, label="Cross-validation precision")
    plt.legend()
    plt.savefig('figures/wine_svm_validation_curve_C.png')
    plt.clf()

    # plot learning curve
    clf = svm.SVC(random_state=seed, kernel='rbf')
    train_sizes, train_scores, test_scores = learning_curve(clf, train_X, train_Y, cv=5, train_sizes=np.arange(0.05, 1.01, 0.05), n_jobs=4, scoring = "accuracy")

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    plt.title("SVM rbf learning curve")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.plot(train_sizes, avg_train_scores, label="Training accuracy")
    plt.plot(train_sizes, avg_test_scores, label="Cross-validation accuracy")
    plt.legend()
    plt.savefig('figures/wine_svm_rbf_learning_curve.png')
    plt.clf()

    clf = svm.SVC(random_state=seed, kernel='poly', degree=3)
    train_sizes, train_scores, test_scores = learning_curve(clf, train_X, train_Y, cv=5, train_sizes=np.arange(0.05, 1.01, 0.05), n_jobs=4, scoring = "accuracy")

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    plt.title("SVM polynomial degree 3 learning curve")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.plot(train_sizes, avg_train_scores, label="Training accuracy")
    plt.plot(train_sizes, avg_test_scores, label="Cross-validation accuracy")
    plt.legend()
    plt.savefig('figures/wine_svm_poly_3_learning_curve.png')
    plt.clf()
    
    # Random search for best parameters
    param_dict = {"C": C_range, "gamma": ['scale', 'auto', 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]}

    clf = svm.SVC(random_state=seed, kernel='rbf')
    random_search = RandomizedSearchCV(clf, param_dict, random_state=seed, scoring='precision', cv=5, n_iter=50, n_jobs=4)
    random_search = random_search.fit(train_X, train_Y)

    params = random_search.best_params_

    print(params)

    clf = svm.SVC(random_state=seed, kernel='rbf', C=params["C"], gamma=params["gamma"])

    train_sizes, train_scores, test_scores = learning_curve(clf, train_X, train_Y, cv=5, train_sizes=np.arange(0.05, 1.01, 0.05), n_jobs=4, scoring = "accuracy")

    avg_train_scores = np.average(train_scores, axis=1)
    avg_test_scores = np.average(test_scores, axis=1)

    plt.title("SVM rbf learning curve")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.plot(train_sizes, avg_train_scores, label="Training accuracy")
    plt.plot(train_sizes, avg_test_scores, label="Cross-validation accuracy")
    plt.legend()
    plt.savefig('figures/wine_svm_rbf_final_learning_curve.png')
    plt.clf()

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Optimized SVM accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Optimized SVM accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    print(confusion_matrix(test_Y, predict_Y))
    

if (__name__ == '__main__'):
    features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]

    df_wine = pd.read_csv('data/winequality-white.csv')
    df_wine['good'] = df_wine['quality'] > 6.5
    df_wine['good'] = df_wine['good'].astype(int)

    # Splitting dataset into train, test sets
    train_df, test_df = train_test_split(df_wine, test_size=0.2, random_state=seed)
    
    # Balancing the training set

    train_good_df = train_df[train_df['good']==1]
    train_df = pd.concat([train_df[train_df['good']==0].head(len(train_good_df)), train_good_df])
    train_df = utils.shuffle(train_df, random_state=seed)

    train_X = train_df[features]
    test_X = test_df[features]
    train_Y = train_df['good']
    test_Y = test_df['good']

    generate_decision_tree_curves(train_X, test_X, train_Y, test_Y)
    generate_neural_net_curves(train_X, test_X, train_Y, test_Y)
    generate_boosting_curves(train_X, test_X, train_Y, test_Y)
    generate_knn_curves(train_X, test_X, train_Y, test_Y)
    generate_svm_curves(train_X, test_X, train_Y, test_Y)