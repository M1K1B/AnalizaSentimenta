import argparse
import os
import itertools
import threading
import time
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from utils.preprocess import preprocess_text
from utils.feature_extraction import extract_features
from utils.evaluation import evaluate_model

from models.naive_bayes import NaiveBayesModel
from models.logistic_regression import LogisticRegressionModel
from models.svm import SVMModel
from models.random_forest import RandomForestModel

instanceCount = 0

def load_data(data_dir):
    global instanceCount
    texts, labels = [], []

    for split in ['train', 'test']:
        for label_dir in ['pos', 'neg']:
            folder_path = os.path.join(data_dir, split, label_dir)
            label = 1 if label_dir == 'pos' else 0

            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    instanceCount += 1
                    texts.append(file.read())
                    labels.append(label)

    return pd.Series(texts), pd.Series(labels)

def train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test):
    if model_name == 'naive_bayes':
        model = NaiveBayesModel()
    elif model_name == 'logistic_regression':
        model = LogisticRegressionModel()
    elif model_name == 'svm':
        model = SVMModel()
    elif model_name == 'random_forest':
        model = RandomForestModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.train(X_train, y_train)
    accuracy, precision, recall, f1 =  evaluate_model(model, X_test, y_test)

    return accuracy, precision, recall, f1

def animate(message):
    for c in itertools.cycle(["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]):
        if done:
            break
        sys.stdout.write('\r\033[93m['+c+'] ' + message + '...\033[0m')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r\033[92m[✓] '+ message +' successfuly done!\033[0m\n')

def main(args):
    global done
    done = False
    t = threading.Thread(target=animate, args=("Loading and processing data",))
    t.start()

    # Load and preprocess data
    X, y = load_data(args.data_path)
    X_processed = X.apply(preprocess_text)

    done = True
    t.join()
    done = False
    t = threading.Thread(target=animate, args=("Extracting features",))
    t.start()

    # Extract features
    X_features = extract_features(X_processed, method=args.feature_method)

    done = True
    t.join()
    done = False
    t = threading.Thread(target=animate, args=("Training and evaluating model",))
    t.start()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    # Train and evaluate the model
    accuracy, precision, recall, f1 = train_and_evaluate_model(args.model, X_train, y_train, X_test, y_test)
    done = True
    t.join()

    print(f"Evaluation results ----------------")
    print(f"Model: {args.model}")
    print(f"Number of train instances: {int(instanceCount*0.8)}")
    print(f"Number of test instances: {int(instanceCount*0.2)}")
    print(f"-----------------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis using various classifiers.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the root data directory containing 'train' and 'test' folders.")
    parser.add_argument("--model", type=str, required=True, choices=['naive_bayes', 'logistic_regression', 'svm', 'random_forest'], help="Model to use for classification.")
    parser.add_argument("--feature_method", type=str, default="tfidf", choices=['bow', 'tfidf', 'bert'], help="Feature extraction method.")
    
    args = parser.parse_args()
    main(args)