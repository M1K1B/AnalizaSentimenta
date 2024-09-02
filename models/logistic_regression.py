from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class LogisticRegressionModel:
    def __init__(self):
        self.model = Pipeline([
            ('lr', LogisticRegression(max_iter=1000))
        ])

    def train(self, X_train, y_train):
        """Train the Logistic Regression model."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions using the trained model."""
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """Predict class probabilities using the trained model."""
        return self.model.predict_proba(X_test)
