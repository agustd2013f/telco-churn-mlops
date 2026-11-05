import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def load_data(filepath):
    return pd.read_csv(filepath)

def train_and_evaluate(df):
    X = df.drop(columns=['churn'])
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42)
    }

    results = {}
    for name, model in models.items():
        if name == "Random Forest":
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]

        results[name] = {
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "F1 Score": round(f1_score(y_test, preds), 4),
            "ROC AUC": round(roc_auc_score(y_test, probs), 4)
        }

    return results

if __name__ == "__main__":
    data_path = "data/telco_demo.csv"
    print(f"Cargando datos desde {data_path}...")
    df = load_data(data_path)
    metrics = train_and_evaluate(df)
    print("\n--- Resultados de Evaluacion ---")
    for model_name, res in metrics.items():
        print(f"\n{model_name}:")
        for k, v in res.items():
            print(f"  {k}: {v}")