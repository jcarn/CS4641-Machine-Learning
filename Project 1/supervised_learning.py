# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 15:37:47 2019

@author: John
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import ShuffleSplit, GridSearchCV, train_test_split, cross_val_score
import graphviz 
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn import tree, neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import validation_curve
from sklearn.multiclass import OneVsOneClassifier


import operator, time
import warnings
import my_plots
import my_data


def grid_search_model(m, data, cvgrid_params, is_multi = True, standardize = False):
    if is_multi:
        X_train, X_test, y_train, y_test, labels = my_data.ready_multiwine_data(data)
        cv_splits = 2
    else:
        X_train, X_test, y_train, y_test, labels = my_data.ready_singlewine_data(data)
        cv_splits = 5

    
    if standardize:
        scaler = StandardScaler(with_mean = False)  
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)  
        X_test = scaler.transform(X_test)  
    
    print(X_train.shape)
    print(m)
    m.fit(X_train, y_train)        
#    acc = OneVsOneClassifier(m).fit(X_train, y_train).predict(X_test)

    print("start learning:")
    
    grid = GridSearchCV(m, param_grid=cvgrid_params, scoring='f1_macro', cv=cv_splits, verbose = 1, n_jobs = 3)
    grid.fit(X_train, y_train)
#    print(grid.cv_results_)
    print(grid.best_params_)
    
    comparison = pd.DataFrame({'actual':y_test.values, 'predicted':m.predict(X_test)})
##    print(accuracy_score(comparison.actual, comparison.predicted))
##
    print('Accuracy Score:',accuracy_score(comparison.actual, comparison.predicted)*100,"%")
    print('F1 Score:', f1_score(comparison.actual, comparison.predicted, average='macro'))

#    comparison.head(5)
#    return models
    
    
def final_run_tree(data, model_params, is_multi, cv_splits):
    print("DT Final Training")
    print("Multiwine" if is_multi else "Single Wine")
    print()
    if is_multi:
        X_train, X_test, y_train, y_test, labels = my_data.ready_multiwine_data(data)
    else:
        X_train, X_test, y_train, y_test, labels = my_data.ready_singlewine_data(data)
        
    title = "Pruned DT Max Depth"
    tree_model = tree.DecisionTreeClassifier(min_samples_leaf = model_params[1])
    my_plots.plot_validation_curve(title, tree_model, X_train, y_train, "max_depth", range(1, 20))
    
    title = "Pruned DT Min Samples per Leaf"
    tree_model = tree.DecisionTreeClassifier(max_depth = model_params[0])
    my_plots.plot_validation_curve(title, tree_model, X_train, y_train, "min_samples_leaf", range(1, 20))

    tree_model = tree.DecisionTreeClassifier(min_samples_leaf = model_params[1], max_depth = model_params[0])

    title = "Training Set (Pruned DT)"
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
    my_plots.plot_learning_curve(tree_model, title, X_train, y_train, ylim=(0.1, 1.01), cv=cv, n_jobs=3)
#    plt.show()
    
    title = "Test Set (Pruned DT)"
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
    my_plots.plot_learning_curve(tree_model, title, X_test, y_test, ylim=(0.1, 1.01), cv=cv, n_jobs=3)
#    plt.show()
   
    t0 = time.time()
    tree_model.fit(X_train, y_train)
    print("done in %0.3fs" % (time.time() - t0))
    
    comparison = pd.DataFrame({'actual':y_test.values, 'predicted':tree_model.predict(X_test)})
    print('Accuracy Score:',accuracy_score(comparison.actual, comparison.predicted)*100,"%")
    print('F1 Score:', f1_score(comparison.actual, comparison.predicted, average='macro'))
    
    
def final_run_boost_tree(data, model_params, is_multi, cv_splits):
    print("Boosted Tree Final Training")
    print("Multiwine" if is_multi else "Single Wine")
    print()

    if is_multi:
        X_train, X_test, y_train, y_test, labels = my_data.ready_multiwine_data(data)
    else:
        X_train, X_test, y_train, y_test, labels = my_data.ready_singlewine_data(data)
    
    title = "Boosted DT Estimators"
    boost_tree_model = GradientBoostingClassifier(max_depth = model_params[1], learning_rate = model_params[2])
    my_plots.plot_validation_curve(title, boost_tree_model, X_train, y_train, 'n_estimators', range(50, 201, 50))
    
    title = "Boosted DT Max Depth"
    boost_tree_model = GradientBoostingClassifier(n_estimators = model_params[0], learning_rate = model_params[2])
    my_plots.plot_validation_curve(title, boost_tree_model, X_train, y_train, 'max_depth', range(1, 11, 2))

    title = "Boosted DT Learning Rate"
    boost_tree_model = GradientBoostingClassifier(n_estimators = model_params[0], max_depth = model_params[1])
    my_plots.plot_validation_curve(title, boost_tree_model, X_train, y_train, 'learning_rate', np.logspace(-2, 1, num = 10))
    
    boost_tree_model = GradientBoostingClassifier(n_estimators = model_params[0], max_depth = model_params[1],
                                            learning_rate = model_params[2])

    title = "Training Set (Boosted DT)"
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
    my_plots.plot_learning_curve(boost_tree_model, title, X_train, y_train, ylim=(0.1, 1.01), cv=cv, n_jobs=3)
#    plt.show()
    
    title = "Test Set (Boosted DT)"
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
    my_plots.plot_learning_curve(boost_tree_model, title, X_test, y_test, ylim=(0.1, 1.01), cv=cv, n_jobs=3)
    
    t0 = time.time()
    boost_tree_model.fit(X_train, y_train)
    print("done in %0.3fs" % (time.time() - t0))
    
    comparison = pd.DataFrame({'actual':y_test.values, 'predicted':boost_tree_model.predict(X_test)})
    print('Accuracy Score:',accuracy_score(comparison.actual, comparison.predicted)*100,"%")
    print('F1 Score:', f1_score(comparison.actual, comparison.predicted, average='macro'))
    
    
def final_run_net(data, model_params, is_multi, cv_splits):
    print("Final Run Net")
    print("Multiwine" if is_multi else "Single Wine")
    print()
    
    if is_multi:
        X_train, X_test, y_train, y_test, labels = my_data.ready_multiwine_data(data)
    else:
        X_train, X_test, y_train, y_test, labels = my_data.ready_singlewine_data(data) 
        
    scaler = StandardScaler(with_mean = False)  
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test) 

    print(hidden_layer_params)
    title = "Neural Net Hidden Layers"   
    net_model = MLPClassifier(learning_rate_init = model_params[1])
#    my_plots.plot_validation_curve(title, net_model, X_train, y_train, "hidden_layer_sizes", hidden_layer_params)
    
    title = "Neural Net Learning Rate"
    net_model = MLPClassifier(hidden_layer_sizes = model_params[0])
    my_plots.plot_validation_curve(title, net_model, X_train, y_train, "learning_rate_init", np.logspace(-4, -2, 10))
    
    title = "Neural Net Iterations"
    net_model = MLPClassifier(hidden_layer_sizes = model_params[0], learning_rate_init = model_params[1])
    my_plots.plot_validation_curve(title, net_model, X_train, y_train, "max_iter", [20*i for i in range(1, 10)])
    
    net_model = MLPClassifier(hidden_layer_sizes = model_params[0], learning_rate_init = model_params[1])
    
    title = "Training Set (Neural Net)"
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
    my_plots.plot_learning_curve(net_model, title, X_train, y_train, ylim=(0.1, 1.01), cv=cv, n_jobs=3)
#    plt.show()
    
    title = "Test Set (Neural Net)"
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
    my_plots.plot_learning_curve(net_model, title, X_test, y_test, ylim=(0.1, 1.01), cv=cv, n_jobs=3)
#    plt.show()
    
    t0 = time.time()
    net_model.fit(X_train, y_train)
    print("done in %0.3fs" % (time.time() - t0))
    
    comparison = pd.DataFrame({'actual':y_test.values, 'predicted':net_model.predict(X_test)})
    print('Accuracy Score:',accuracy_score(comparison.actual, comparison.predicted)*100,"%")
    print('F1 Score:', f1_score(comparison.actual, comparison.predicted, average='macro'))

def final_run_knn(data, model_params, is_multi, cv_splits):
    print("KNN Final Training")
    print("Multiwine" if is_multi else "Single Wine")
    print()

    if is_multi:
        X_train, X_test, y_train, y_test, labels = my_data.ready_multiwine_data(data)
    else:
        X_train, X_test, y_train, y_test, labels = my_data.ready_singlewine_data(data) 
  
    knn_model = neighbors.KNeighborsClassifier(n_neighbors = model_params[0], weights = model_params[1])

    title = "KNN N-Neighbors"
    knn_model = neighbors.KNeighborsClassifier(weights = model_params[1])
    my_plots.plot_validation_curve(title, knn_model, X_train, y_train, "n_neighbors",  [i for i in range(2, 11)])
    
    knn_model = neighbors.KNeighborsClassifier(n_neighbors = model_params[0], weights = model_params[1])
    
    title = "Training Set (KNN)"
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
    my_plots.plot_learning_curve(knn_model, title, X_train, y_train, ylim=(0.1, 1.01), cv=cv, n_jobs=3)
#    plt.show()
    
    title = "Test Set (KNN)"
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
    my_plots.plot_learning_curve(knn_model, title, X_test, y_test, ylim=(0.1, 1.01), cv=cv, n_jobs=3)
#    plt.show()
    
    t0 = time.time()
    knn_model.fit(X_train, y_train)
    print("done in %0.3fs" % (time.time() - t0))
    
    comparison = pd.DataFrame({'actual':y_test.values, 'predicted':knn_model.predict(X_test)})
    print('Accuracy Score:',accuracy_score(comparison.actual, comparison.predicted)*100,"%")
    print('F1 Score:', f1_score(comparison.actual, comparison.predicted, average='macro'))
    
def final_run_rbfsvm(data, model_params, is_multi, cv_splits):
    print("RBF SVM Final Training")
    print("Multiwine" if is_multi else "Single Wine")
    print()

    if is_multi:
        X_train, X_test, y_train, y_test, labels = my_data.ready_multiwine_data(data)
    else:
        X_train, X_test, y_train, y_test, labels = my_data.ready_singlewine_data(data) 

    title = "RBF SVM, Iterations"   
    bfsvm_model = OneVsOneClassifier(SVC(kernel = 'rbf', gamma = 'auto'))
#    print(bfsvm_model.get_params().keys())
    my_plots.plot_validation_curve(title, bfsvm_model, X_train, y_train, "estimator__max_iter", [40 * i for i in range(5, 15)])

    title = "RBF SVM, Gamma"   
    bfsvm_model = OneVsOneClassifier(SVC(kernel = 'rbf', gamma = 'auto'))
    my_plots.plot_validation_curve(title, bfsvm_model, X_train, y_train, "estimator__gamma", np.logspace(-4,-1,num=4))
        
    bfsvm_model = OneVsOneClassifier(SVC(kernel = 'rbf', gamma = 'auto'))
    
    title = "Training Set (RBF SVM)"
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
    my_plots.plot_learning_curve(bfsvm_model, title, X_train, y_train, ylim=(0.1, 1.01), cv=cv, n_jobs=3)
#    plt.show()
    
    title = "Test Set (RBF SVM)"
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
    my_plots.plot_learning_curve(bfsvm_model, title, X_test, y_test, ylim=(0.1, 1.01), cv=cv, n_jobs=3)
#    plt.show()
    
    bfsvm_model = OneVsOneClassifier(SVC(kernel = 'rbf', gamma = 'auto'))
    t0 = time.time()
    bfsvm_model.fit(X_train, y_train)
    print("done in %0.3fs" % (time.time() - t0))
    
    comparison = pd.DataFrame({'actual':y_test.values, 'predicted':bfsvm_model.predict(X_test)})
    print('Accuracy Score:',accuracy_score(comparison.actual, comparison.predicted)*100,"%")
    print('F1 Score:', f1_score(comparison.actual, comparison.predicted, average='macro'))
    
    
def final_run_lsvm(data, model_params, is_multi, cv_splits):
    print("Linear SVM Final Training")
    print()

    if is_multi:
        X_train, X_test, y_train, y_test, labels = my_data.ready_multiwine_data(data)
    else:
        X_train, X_test, y_train, y_test, labels = my_data.ready_singlewine_data(data) 

    title = "Linear SVM Iterations"   
    linearsvm_model = LinearSVC(multi_class = 'ovr')
    my_plots.plot_validation_curve(title, linearsvm_model, X_train, y_train, "max_iter", [500*i for i in range(1, 15)])
    
    linearsvm_model = LinearSVC(max_iter = model_params[0], multi_class = 'ovr')
    
    title = "Training Set (Linear SVM)"
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
    my_plots.plot_learning_curve(linearsvm_model, title, X_train, y_train, ylim=(0.1, 1.01), cv=cv, n_jobs=3)
#    plt.show()
    
    title = "Test Set (Linear SVM)"
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)
    my_plots.plot_learning_curve(linearsvm_model, title, X_test, y_test, ylim=(0.1, 1.01), cv=cv, n_jobs=3)
#    plt.show()
    
    t0 = time.time()
    linearsvm_model.fit(X_train, y_train)
    print("done in %0.3fs" % (time.time() - t0))
    
    comparison = pd.DataFrame({'actual':y_test.values, 'predicted':linearsvm_model.predict(X_test)})
    print('Accuracy Score:',accuracy_score(comparison.actual, comparison.predicted)*100,"%")
    print('F1 Score:', f1_score(comparison.actual, comparison.predicted, average='macro'))
    
    
warnings.filterwarnings('ignore')
multiwine_d = my_data.get_multiwine_data()
multiwine_ds = multiwine_d.sample(frac = .2)
#print(multiwine_ds.shape)


singlewine_d = my_data.get_singlewine_data()
singlewine_ds = singlewine_d.sample(frac = 1.0)

##############################################################################
#Decision Tree
##############################################################################

tree_model = tree.DecisionTreeClassifier()
tree_param = {
    'min_samples_leaf': [2*i for i in range(1, 10)],
    'max_depth': [i for i in range(5,15)]
}
#grid_search_model(tree_model, singlewine_ds, tree_param, is_multi = False)
#grid_search_model(tree_model, multiwine_ds,tree_param,  is_multi = True)
#final_run_tree(singlewine_ds, (10, 2), False, 25) 
#final_run_tree(multiwine_ds, (14, 6), True, 5)

##############################################################################
#                                   Boosted Tree
##############################################################################

boost_tree_model = GradientBoostingClassifier()
boost_param = {
    'n_estimators': [50*i for i in range(1, 4)],
    'max_depth': list(range(1,6)),
    'learning_rate': np.logspace(-2, 1, num = 10)
}
#grid_search_model(boost_tree_model, singlewine_ds, boost_param, is_multi = False)
#grid_search_model(boost_tree_model, multiwine_ds,boost_param,  is_multi = True)
#final_run_boost_tree(singlewine_ds, (50, 5, .464), False, 25) 
#final_run_boost_tree(multiwine_ds, (150, 2, .215), True, 5)



##############################################################################
#                                   Neural Net
##############################################################################

net_model = MLPClassifier(learning_rate_init = .001)
hidden_layer_params = list()
hidden_layer_params.append((100))
for i in range(1, 3):
    hidden_layer_params.append(tuple(50 for i in range(1, i + 1)))
for i in range(3, 5):
    hidden_layer_params.append(tuple(20 for i in range(1, i + 1)))
for i in range(5, 10):
    hidden_layer_params.append(tuple(10 for i in range(1, i + 1)))

net_param = {
    'hidden_layer_sizes': hidden_layer_params,
#    'learning_rate_init': np.logspace(-5, -2, 10)
}
#grid_search_model(net_model, singlewine_ds, net_param, is_multi = False, standardize = True)
#grid_search_model(net_model, multiwine_ds,net_param,  is_multi = True, standardize = True)
#final_run_net(singlewine_ds, ((11, 11), 0.001), False, 5)
#final_run_net(multiwine_ds, ((20, 20, 20), .0008), True, 5)


##############################################################################
#                            K-Nearest-Neighbors
##############################################################################
knn_model = neighbors.KNeighborsClassifier()
knn_param = {
    'n_neighbors': [i for i in range(2, 20)],
    'weights': ['uniform', 'distance'],
    }
#grid_search_model(knn_model, singlewine_ds, knn_param, is_multi = False)
grid_search_model(knn_model, multiwine_ds,knn_param,  is_multi = True)
#final_run_knn(singlewine_ds, (3, 'distance'), False, 25) 
#final_run_knn(multiwine_ds, (6, 'distance'), True, 5)

##############################################################################
#                                Linear Kernel SVM
##############################################################################
linearsvm_model = LinearSVC(multi_class = 'ovr')
linearsvm_param = {
    'max_iter': [400 * i for i in range(1, 30)]
}
#grid_search_model(linearsvm_model, singlewine_ds, linearsvm_param, is_multi = False)
#grid_search_model(linearsvm_model, multiwine_ds,linearsvm_param,  is_multi = True)
#final_run_lsvm(singlewine_ds, [7000], False, 25) 
#final_run_lsvm(multiwine_ds, [3600], True, 5) 

##############################################################################
#                           RBF Kernel SVM
##############################################################################
rbfsvm_model = OneVsOneClassifier(SVC(kernel = 'rbf'))
rbfsvm_param = {
    'max_iter': [400 * i for i in range(1, 30)],
    'gamma': np.logspace(-2,0,num=10)
}
#grid_search_model(rbfsvm_model, singlewine_ds, rbfsvm_param, is_multi = False)
#grid_search_model(rbfsvm_model, multiwine_ds,rbfsvm_param,  is_multi = True)
#final_run_rbfsvm(singlewine_ds, (.01, 4000), False, 25) 
#final_run_rbfsvm(multiwine_ds, (.001, 1200), True, 5)
