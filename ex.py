"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

from sklearn import datasets, metrics, svm
from utils import preprocess_data, split_train_dev_test, train_model, read_digits, predict_and_eval, get_hyperparameter_combinations, tune_hparams
from itertools import product
from joblib import dump, load
import pandas as pd

X, y = read_digits()

classifier_param_dict = {}

#SVM
gamma = [0.001, 0.01, 0.1, 1, 10, 100]
C_range = [0.1, 1, 2, 5, 10]
h_params={}
h_params['gamma'] = gamma
h_params['C'] = C_range
h_parameters = get_hyperparameter_combinations(h_params)
classifier_param_dict['svm'] = h_parameters

#decision tree
max_depth_list = [5, 10, 15, 20, 50, 100]
h_params_tree = {}
h_params_tree['max_depth'] = max_depth_list
h_parameters = get_hyperparameter_combinations(h_params_tree)
classifier_param_dict['tree'] = h_parameters

#logistic regression classifier
solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
h_params_logistic = {}
h_params_logistic['solver'] = solver
h_parameters = get_hyperparameter_combinations(h_params_logistic)
classifier_param_dict['logistic'] = h_parameters

#hyper parameter tuning
#h_parameters = dict(product(gamma, C_range,repeat=1))
# h_parameters=list(product(gamma, C_range))

# dataset_combination = list(product(test_range, dev_range))

t_sizes =  [0.2]
d_sizes  =  [0.2]

for test_size in t_sizes:
    for dev_size in d_sizes:
        train_size = 1 - test_size - dev_size

        X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_size=test_size, dev_size=dev_size)
        
        X_train = preprocess_data(X_train)
        X_test = preprocess_data(X_test)
        X_dev = preprocess_data(X_dev)

        binary_prediction = {}
        model_preds = {}
        for model_type in classifier_param_dict:
            current_hparams = classifier_param_dict[model_type]
            best_hparams, best_model_path, best_accuracy  = tune_hparams(X_train, y_train, X_dev, 
            y_dev, current_hparams, model_type)        

            best_model = load(best_model_path) 

            test_acc, test_f1, predicted_y = predict_and_eval(best_model, X_test, y_test)
            train_acc, train_f1, _ = predict_and_eval(best_model, X_train, y_train)
            dev_acc = best_accuracy

            print("{}\ttest_size={:.2f} dev_size={:.2f} train_size={:.2f} train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}, test_f1={:.2f}".format(model_type, test_size, dev_size, train_size, train_acc, dev_acc, test_acc, test_f1))

            binary_prediction[model_type] = y_test == predicted_y
            model_preds[model_type] = predicted_y
            
            print("{}-Ground Truth Confusion metrics".format(model_type))
            print(metrics.confusion_matrix(y_test, predicted_y))


print("Confusion metric".format())
print(metrics.confusion_matrix(model_preds['svm'], model_preds['tree']))

print("binary predictions")
print(metrics.confusion_matrix(binary_prediction['svm'], binary_prediction['tree'], labels=[True, False]))
print("binary predictions -- normalized over true labels")
print(metrics.confusion_matrix(binary_prediction['svm'], binary_prediction['tree'], labels=[True, False] , normalize='true'))
print("binary predictions -- normalized over pred  labels")
print(metrics.confusion_matrix(binary_prediction['svm'], binary_prediction['tree'], labels=[True, False] , normalize='pred'))