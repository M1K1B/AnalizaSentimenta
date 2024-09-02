from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class NaiveBayesModel:
    def __init__(self):
        self.model = Pipeline([
            ('nb', MultinomialNB())
        ])

    def train(self, X_train, y_train):
        """Train the Naive Bayes model."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions using the trained model."""
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """Predict class probabilities using the trained model."""
        return self.model.predict_proba(X_test)