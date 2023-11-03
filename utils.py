from sklearn import datasets, metrics, svm, tree
from sklearn.model_selection import train_test_split
from joblib import dump, load

def split_train_dev_test(X, y, test_size, dev_size):
    train_size = 1 - dev_size - test_size
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_size, random_state=1)
    new_test_size = test_size / (1 - train_size)    
    X_dev, X_test, y_dev, y_test = train_test_split(X_rem, y_rem, test_size=new_test_size, random_state=1)  
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)    
    # f"Classification report for classifier {model}:\n"
    # f"{metrics.classification_report(y_test, predicted)}\n")

    # ###############################################################################
    # # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
    # # true digit values and the predicted digit values.

    # disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    # disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}")

    # #plt.show()

    # ###############################################################################
    # # If the results from evaluating a classifier are stored in the form of a
    # # :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
    # # `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
    # # as follows:


    # # The ground truth and predicted lists
    # y_true = []
    # y_pred = []
    # cm = disp.confusion_matrix

    # # For each cell in the confusion matrix, add the corresponding ground truths
    # # and predictions to the lists
    # for gt in range(len(cm)):
    #     for pred in range(len(cm)):
    #         y_true += [gt] * cm[gt][pred]
    #         y_pred += [pred] * cm[gt][pred]

    # print(
    #     "Classification report rebuilt from confusion matrix:\n"
    #     f"{metrics.classification_report(y_true, y_pred)}\n"
    # )

    accuracy = metrics.accuracy_score(y_test, predicted)
    f1_score = metrics.f1_score(y_test, predicted, average="macro")
    return accuracy,f1_score,predicted

def train_model(X_train, y_train, parameters, model_type="svm"):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC
    if model_type == "tree":
        # Create a classifier: a decision tree classifier
        clf = tree.DecisionTreeClassifier
    model = clf(**parameters)
    # train the model
    model.fit(X_train, y_train)
    return model

def tune_hparams(X_train, y_train, X_dev, y_dev, hyper_parameters, model_type="svm"):
    optimal_accuracy=-1
    optimal_model=None
    for params in hyper_parameters:
        #print("Current Gamma value={} and Current C value={}".format(current_gamma,C_current))

        #train model on different hyper parameters
        current_model = train_model(X_train, y_train, params, model_type=model_type)

        #predict
        current_accuracy,_,_ = predict_and_eval(current_model, X_dev, y_dev)
        if current_accuracy > optimal_accuracy:
            #print("new optimal accuracy", current_accuracy)
            optimal_accuracy = current_accuracy
            optimal_params = params
            
            optimal_model_path = "./models/{}_".format(model_type) +"_".join(["{}:{}".format(k,v) for k,v in params.items()]) + ".joblib"
            optimal_model = current_model
    
    # save the best_model    
    dump(optimal_model, optimal_model_path) 

    return optimal_params, optimal_model_path, optimal_accuracy

def get_combinations(param_name, param_values, base_combinations):    
    new_combinations = []
    for value in param_values:
        for combination in base_combinations:
            combination[param_name] = value
            new_combinations.append(combination.copy())    
    return new_combinations

def get_hyperparameter_combinations(dict_of_param_lists):    
    base_combinations = [{}]
    for param_name, param_values in dict_of_param_lists.items():
        base_combinations = get_combinations(param_name, param_values, base_combinations)
    return base_combinations

def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y 

def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data