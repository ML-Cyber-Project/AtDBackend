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
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, f1_score
    # from sklearn.model_selection import StratifiedKFold
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

NUM_TRIALS = 5
NUM_FOLDS = 4
DEBUG = True

IS_PROD = True
mlflow.set_tracking_uri("https://mlflow.docsystem.xyz" if IS_PROD else "http://127.0.0.1:8080")
mlflow.set_experiment("RandomSearch")

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

# skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=34)
# skf.get_n_splits(X_train, y_train)

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
}

param_distributions = {
    "Logistic Regression": {
        'C': np.logspace(-10, 10, 100),
    },
    "Random Forest": {
        'min_samples_split': np.arange(2, 21),
        'max_depth': np.arange(1, 21)
    },
    "KNN": {
        'n_neighbors': np.arange(1, 21),
        'p': np.arange(1, 6)
    },
    "Decision Tree": {
        'max_depth': np.arange(1, 21),
        'min_samples_split': np.arange(2, 21),
    }
}

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
    
    model = models[name]
    param_dist = param_distributions[name]
    
    randomized_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=NUM_TRIALS,
        scoring='f1',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=3
    )
    
    randomized_search.fit(X_train, y_train.values.ravel())
    
    best = randomized_search.best_params_
    model.set_params(**best)
    model.fit(X_train, y_train.values.ravel())
    
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    f1_test = f1_score(y_test, model.predict(X_test))
    
    signature = infer_signature(X_train, model.predict(X_train))
    
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(best)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        model_uri = mlflow.get_artifact_uri("model")
        
        mlflow.evaluate(
            model=model_uri,
            data=evalData,
            targets='label',
            model_type='classifier',
            evaluators=['default'],
        )
        
        if f1_test > best_score:
            best_score = f1_test
            best_model = model
            best_params = best

# Save the best model's parameters in a separate mlflow experiment
mlflow.set_experiment("BestModelParams")
name = best_model.__class__.__name__
with mlflow.start_run(run_name=name) as run:
    mlflow.log_params(best_params)
    mlflow.log_metric("f1", best_score)
