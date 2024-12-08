import os
import time
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    print("You need to install pandas")
    exit()

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
except ImportError:
    print("You need to install scikit-learn")
    exit()

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
except ImportError:
    print("You need to install mlflow")
    exit()

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    from hyperopt.pyll.base import scope
except ImportError:
    print("You need to install hyperopt")
    exit()

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Predictions")

features = pd.read_csv(os.path.abspath('data/features_cleaned.csv'))
labels = pd.read_csv(os.path.abspath('data/labels_cleaned.csv'))

# Scale data
scaler = MinMaxScaler()
features[features.columns] = scaler.fit_transform(features[features.columns])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print("Data prepared!")

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    # "SVM": SVC(),
}

param_spaces = {
    "Logistic Regression": {
        'C': hp.loguniform('C', -4, 4)
    },
    "Random Forest": {
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 200, 1)),
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 20, 1))
    },
    "KNN": {
        'n_neighbors': scope.int(hp.quniform('n_neighbors', 1, 20, 1)),
        'weights': hp.choice('weights', ['uniform', 'distance']),
        'p': scope.int(hp.quniform('p', 1, 5, 1))
    },
    "Decision Tree": {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 20, 1))
    },
    # "SVM": {
    #     'C': hp.loguniform('C', -4, 4),
    #     'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    #     'degree': scope.int(hp.quniform('degree', 2, 5, 1)),
    #     'gamma': hp.loguniform('gamma', -4, 4)
    # }
}

evalData = X_test.copy()
evalData['label'] = y_test

best_model = None
best_score = float('-inf')
best_params = None

for name, model in models.items():
    print("Tuning hyperparameters for model", name)
    
    def objective(params):
        model.set_params(**params)
        model.fit(X_train, y_train.values.ravel())
        accuracy = accuracy_score(y_test, model.predict(X_test))
        return {'loss': -accuracy, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(fn=objective, space=param_spaces[name], algo=tpe.suggest, max_evals=50, trials=trials)
    
    model.set_params(**best)
    model.fit(X_train, y_train.values.ravel())
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mse = mean_squared_error(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    
    signature = infer_signature(X_train, model.predict(X_train))
    
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(best)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        model_uri = mlflow.get_artifact_uri("model")
        
        result = mlflow.evaluate(
            model=model_uri,
            data=X_test,
            targets='label',
            model_type='classifier',
            evaluators=['default'],
        )
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_params = best

# Save the best model's parameters in a separate mlflow experiment
mlflow.set_experiment("BestModelParams")
with mlflow.start_run(run_name="Best Model") as run:
    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", best_score)
