import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from shutil import make_archive


def compute_score(hazards_true, products_true, hazards_pred, products_pred):
    # compute f1 for hazards:
    f1_hazards = f1_score(hazards_true, hazards_pred, average="macro")

    # compute f1 for products:
    f1_products = f1_score(
        products_true[hazards_pred == hazards_true],
        products_pred[hazards_pred == hazards_true],
        average="macro",
    )

    return (f1_hazards + f1_products) / 2.0


if __name__ == "__main__":
    # load training data:
    data = pd.read_csv("../data/incidents_train.csv", index_col=0)
    train_set, dev_set = train_test_split(data, test_size=0.2, random_state=2024)

    print(train_set.sample())

    # load test data:
    test_set = pd.read_csv("../data/incidents_unlabeled_test.csv", index_col=0)

    print(test_set.sample())

    text_clf_lr = Pipeline(
        [
            (
                "vect",
                TfidfVectorizer(
                    strip_accents="unicode",
                    analyzer="char",
                    ngram_range=(2, 5),
                    max_df=0.5,
                    min_df=5,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    for label in ("hazard-category", "product-category", "hazard", "product"):
        print(label.upper())
        text_clf_lr.fit(train_set.title, train_set[label])

        # get development scores:
        dev_set["predictions-" + label] = text_clf_lr.predict(dev_set.title)
        print(
            f'  macro: {f1_score(dev_set[label], dev_set["predictions-" + label], zero_division=0, average="macro"):.2f}'
        )
        print(
            f'  micro: {f1_score(dev_set[label], dev_set["predictions-" + label], zero_division=0, average="micro"):.2f}'
        )

        # predict test set:
        test_set[label] = text_clf_lr.predict(test_set.title)

    print(
        f"Score Sub-Task 1: {compute_score(dev_set['hazard-category'], dev_set['product-category'], dev_set['predictions-hazard-category'], dev_set['predictions-product-category']):.3f}"
    )
    print(
        f"Score Sub-Task 2: {compute_score(dev_set['hazard'], dev_set['product'], dev_set['predictions-hazard'], dev_set['predictions-product']):.3f}"
    )

    # save predictions to a new folder:
    os.makedirs("./submission/", exist_ok=True)
    test_set[["hazard-category", "product-category", "hazard", "product"]].to_csv(
        "./submission/submission.csv"
    )

    # zip the folder (zipfile can be directly uploaded to codalab):
    make_archive("./submission", "zip", "./submission")
