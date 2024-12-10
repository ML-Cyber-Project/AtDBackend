import os

try:
    import pandas as pd
except ImportError:
    print("You need to install pandas")
    exit()

try:
    import numpy as np
except ImportError:
    print("You need to install numpy")
    exit()

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, classification_report
    from sklearn.model_selection import StratifiedKFold
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
    import optuna
except ImportError:
    print("You need to install optuna")
    exit()

NUM_TRIALS = 5
NUM_FOLDS = 4
DEBUG = False

mlflow.set_tracking_uri("https://mlflow.docsystem.xyz")
mlflow.set_experiment("Predictions")

features = pd.read_csv(os.path.abspath('data/features_cleaned.csv'))
labels = pd.read_csv(os.path.abspath('data/labels_cleaned.csv'))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=25)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert to DataFrame
X_train = pd.DataFrame(X_train, columns=features.columns)
X_test = pd.DataFrame(X_test, columns=features.columns)

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=34)
skf.get_n_splits(X_train, y_train)

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
}

def objective(trial, model_name, X_train_fold, y_train_fold, X_val_fold, y_val_fold):
    if model_name == "Logistic Regression":
        params = {
            'C': trial.suggest_loguniform('C', 1e-10, 1e10),
        }
    elif model_name == "Random Forest":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
        }
    elif model_name == "KNN":
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 5)
        }
    elif model_name == "Decision Tree":
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
        }

    model = models[model_name]
    model.set_params(**params)
    y_pred = model.fit(X_train_fold, y_train_fold.values.ravel()).predict(X_val_fold)
    accuracy = accuracy_score(y_val_fold, y_pred)
    if DEBUG:
        print(classification_report(y_val_fold, y_pred))
    
    return accuracy

# reset index
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

evalData = X_test.copy()
evalData['label'] = y_test

best_model = None
best_score = float('-inf')
best_params = None

for name in models.keys():
    print("Tuning hyperparameters for model", name)
    
    study = optuna.create_study(direction='maximize')
    
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        study.optimize(lambda trial: objective(trial, name, X_train_fold, y_train_fold, X_val_fold, y_val_fold), n_trials=NUM_TRIALS)
    
    best = study.best_params
    model = models[name]
    model.set_params(**best)
    model.fit(X_train, y_train.values.ravel())
    
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    mse = mean_squared_error(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    
    signature = infer_signature(X_train, model.predict(X_train))
    
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(best)
        mlflow.log_metric("accuracy", test_accuracy)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        model_uri = mlflow.get_artifact_uri("model")
        
        mlflow.evaluate(
            model=model_uri,
            data=evalData,
            targets='label',
            model_type='classifier',
            evaluators=['default'],
        )
        
        if test_accuracy > best_score:
            best_score = test_accuracy
            best_model = model
            best_params = best

# Save the best model's parameters in a separate mlflow experiment
mlflow.set_experiment("BestModelParams")
with mlflow.start_run(run_name="Best Model") as run:
    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", best_score)
