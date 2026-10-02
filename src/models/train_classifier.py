import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


INPUT_PATH = "data/amazon_sample_labeled.csv"

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


INPUT_PATH = "data/amazon_sample_labeled.csv"


class IntentClassifier:

    def __init__(self, data_path=INPUT_PATH):

        df = pd.read_csv(data_path)
        df = df.dropna(subset=["intent"])

        X = df["text_customer"].fillna("")
        y = df["intent"]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
        )

        X_tfidf = self.vectorizer.fit_transform(X)

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )

        self.model.fit(X_tfidf, y)

    def predict(self, message):
        vector = self.vectorizer.transform([message])

        probabilities = self.model.predict_proba(vector)[0]
        predicted_index = probabilities.argmax()

        return {
            "intent": self.model.classes_[predicted_index],
            "confidence": float(probabilities[predicted_index]),
        }


def main():
    df = pd.read_csv(INPUT_PATH)

    # Remove rows where our provisional labeling failed.
    df = df.dropna(subset=["intent"])

    X = df["text_customer"].fillna("")
    y = df["intent"]

    print("Total labeled examples:", len(df))
    print("\nClass distribution:")
    print(y.value_counts())

    # ---------------------------------------------------------
    # Train / validation split
    # ---------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTraining examples:", len(X_train))
    print("Validation examples:", len(X_val))

    # ---------------------------------------------------------
    # Baseline 1: Majority class
    # ---------------------------------------------------------
    majority_class = y_train.value_counts().idxmax()

    majority_predictions = [majority_class] * len(y_val)

    majority_accuracy = accuracy_score(
        y_val,
        majority_predictions
    )

    print("\n==============================")
    print("BASELINE 1: MAJORITY CLASS")
    print("==============================")
    print("Predicted class:", majority_class)
    print("Accuracy:", round(majority_accuracy, 4))

    # ---------------------------------------------------------
    # Baseline 2: TF-IDF + Logistic Regression
    # ---------------------------------------------------------
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    classifier.fit(
        X_train_tfidf,
        y_train
    )

    predictions = classifier.predict(X_val_tfidf)

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    print("\n==============================")
    print("BASELINE 2: TF-IDF + LOGISTIC REGRESSION")
    print("==============================")
    print("Accuracy:", round(accuracy, 4))

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()