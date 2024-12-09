import os

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
    import optuna
except ImportError:
    print("You need to install optuna")
    exit()

mlflow.set_tracking_uri("https://mlflow.docsystem.xyz")
mlflow.set_experiment("Predictions")

features = pd.read_csv(os.path.abspath('data/features_cleaned.csv'))
labels = pd.read_csv(os.path.abspath('data/labels_cleaned.csv'))

# Scale data
scaler = MinMaxScaler()
features[features.columns] = scaler.fit_transform(features[features.columns])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

print("Data prepared!")

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
}

def objective(trial, model_name):
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
    model.fit(X_train, y_train.values.ravel())
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return accuracy

evalData = X_test.copy()
evalData['label'] = y_test

best_model = None
best_score = float('-inf')
best_params = None

for name in models.keys():
    print("Tuning hyperparameters for model", name)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, name), n_trials=20)
    
    best = study.best_params
    model = models[name]
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
            data=evalData,
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
