from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

class SVMModel:
    def __init__(self):
        self.model = Pipeline([
            ('svm', SVC(kernel='linear', probability=True))
        ])

    def train(self, X_train, y_train):
        """Train the SVM model."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions using the trained model."""
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """Predict class probabilities using the trained model."""
        return self.model.predict_proba(X_test)
