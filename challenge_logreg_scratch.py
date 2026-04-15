import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred, weights, lambda_reg):
    m = len(y_true)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    loss = -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

    reg_term = (lambda_reg / (2 * m)) * np.sum(weights ** 2)

    return loss + reg_term

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iters=1000, lambda_reg=0.0):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)

            dw = (1 / m) * np.dot(X.T, (y_pred - y)) + (self.lambda_reg / m) * self.weights
            db = (1 / m) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = compute_loss(y, y_pred, self.weights, self.lambda_reg)
            self.loss_history.append(loss)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


def load_data(filepath="data/telecom_churn.csv"):
    return pd.read_csv(filepath)


def prepare_data():
    df = load_data()

    numeric_features = [
        "tenure", "monthly_charges", "total_charges",
        "num_support_calls", "senior_citizen",
        "has_partner", "has_dependents"
    ]

    df_cls = df[numeric_features + ["churned"]].dropna()

    X = df_cls[numeric_features]
    y = df_cls["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy(), X.columns


def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))


def compare_with_sklearn(X_train, X_test, y_train, y_test, feature_names):
    scratch_model = LogisticRegressionScratch(
        learning_rate=0.01,
        n_iters=2000,
        lambda_reg=0.1
    )
    scratch_model.fit(X_train, y_train)
    y_pred_scratch = scratch_model.predict(X_test, threshold=0.3)

    sklearn_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced"
    )
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)

    evaluate_model("Scratch Logistic Regression", y_test, y_pred_scratch)
    evaluate_model("Scikit-learn Logistic Regression", y_test, y_pred_sklearn)

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "scratch_coef": scratch_model.weights,
        "sklearn_coef": sklearn_model.coef_[0]
    })

    print("\nCoefficient Comparison:")
    print(coef_df)

    print("\nScratch bias:", scratch_model.bias)
    print("Scikit-learn bias:", sklearn_model.intercept_[0])

    return scratch_model, sklearn_model, coef_df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    scratch_model, sklearn_model, coef_df = compare_with_sklearn(
        X_train, X_test, y_train, y_test, feature_names
    )