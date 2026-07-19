import os
import gzip
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

def pregunta_01():
    # 1. Crear directorios
    os.makedirs("files/models", exist_ok=True)
    os.makedirs("files/output", exist_ok=True)

    # 2. Cargar datos
    train_data = pd.read_csv("files/input/train_data.csv.zip")
    test_data = pd.read_csv("files/input/test_data.csv.zip")

    # Limpiar datos
    def clean_data(df):
        df = df.copy()
        df = df.rename(columns={"default payment next month": "default"})
        df = df.drop(columns=["ID"])
        df = df.dropna()
        df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
        return df
    
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)

    X_train = train_data.drop(columns=["default"])
    y_train = train_data["default"]
    X_test = test_data.drop(columns=["default"])
    y_test = test_data["default"]

    # 3. Pipeline
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), ['SEX', 'EDUCATION', 'MARRIAGE'])],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

   # 4. Optimización (El punto dulce matemático)
    param_grid = {
        'classifier__n_estimators': [600],        # Un bosque gigante para máxima estabilidad
        'classifier__max_depth': [None], 
        'classifier__min_samples_split': [4],     # El equilibrio exacto entre el 2 y el 5
        'classifier__min_samples_leaf': [1]
    }

    # Optimizamos balanced_accuracy internamente
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)
    
    # 5. Guardar modelo
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(grid, f)

    # 6. Calcular y guardar métricas
    def calculate_metrics(model, X, y, dataset_name):
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        metrics = {
            "type": "metrics",
            "dataset": dataset_name,
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y, y_pred, zero_division=0))
        }
        
        cm_dict = {
            "type": "cm_matrix",
            "dataset": dataset_name,
            "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
            "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
        }
        return metrics, cm_dict

    metrics_train, cm_train = calculate_metrics(grid, X_train, y_train, "train")
    metrics_test, cm_test = calculate_metrics(grid, X_test, y_test, "test")

    with open("files/output/metrics.json", "w") as f:
        f.write(json.dumps(metrics_train) + "\n")
        f.write(json.dumps(metrics_test) + "\n")
        f.write(json.dumps(cm_train) + "\n")
        f.write(json.dumps(cm_test) + "\n")

if __name__ == "__main__":
    pregunta_01()