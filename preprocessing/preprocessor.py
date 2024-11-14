import pandas as pd
import re
import numpy as np
from bs4 import BeautifulSoup
import nltk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')

class DataPreprocessor:
    def __init__(self, file_path, mappings_file):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        print("✔ Loaded dataset successfully.")

        # Load mappings from the provided JSON file
        with open(mappings_file, 'r') as f:
            mappings = json.load(f)
            self.product_category_indices = mappings['product_category_indices']
            self.hazard_category_indices = mappings['hazard_category_indices']
            self.product_indices = mappings['product_indices']
            self.hazard_indices = mappings['hazard_indices']

        # Initialize reverse mappings
        self.product_category_labels = {v: k for k, v in self.product_category_indices.items()}
        self.hazard_category_labels = {v: k for k, v in self.hazard_category_indices.items()}
        self.product_labels = {v: k for k, v in self.product_indices.items()}
        self.hazard_labels = {v: k for k, v in self.hazard_indices.items()}

        # Initialize the Lemmatizer and Stop Words
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Save original country names and mapping
        if 'country' in self.data.columns:
            self.original_country_names = self.data['country'].copy()

    def remove_html_tags(self, text):
        return BeautifulSoup(text, "html.parser").get_text() if text else ""

    def preprocess_text(self, text):
        if text is None:
            return ""
        text = self.remove_html_tags(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join(
            [self.lemmatizer.lemmatize(word) for word in text.split() if
             word not in self.stop_words]
        )
        return re.sub(r'\s+', ' ', text).strip()

    def combine_date_columns(self):
        if {'year', 'month', 'day'}.issubset(self.data.columns):
            self.data['date'] = pd.to_datetime(
                self.data[['year', 'month', 'day']])
            self.data['date'] = self.data['date'].astype('int64') // 1e9
            self.data.drop(['year', 'month', 'day'], axis=1, inplace=True)
        print("✔ Combined date columns successfully.")

    def drop_index_column(self):
        if 'Unnamed: 0' in self.data.columns:
            self.data.drop('Unnamed: 0', axis=1, inplace=True)
        print("✔ Dropped index column successfully.")

    def encode_country_column(self):
        if 'country' in self.data.columns:
            # Save the mapping of the numerical index to the country name
            self.country_mapping = dict(
                enumerate(self.data['country'].factorize()[1]))
            self.data['country'] = pd.factorize(self.data['country'])[0]
        print("✔ Encoded country column successfully.")

    def balance_data(self):
        # Convert date and country columns to strings and combine them with text and title
        self.data['date_str'] = self.data['date'].astype(
            str) if 'date' in self.data.columns else ''
        self.data['country_str'] = self.data['country'].astype(
            str) if 'country' in self.data.columns else ''

        # Combine text, title, date, and country columns into a single feature
        self.data['combined_text'] = (
            self.data['text'].fillna('') + ' ' +
            self.data['title'].fillna('') + ' ' +
            self.data['date_str'] + ' ' +
            self.data['country_str']
        )

        # Vectorize the combined text using TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        text_features = vectorizer.fit_transform(
            self.data['combined_text']).toarray()

        # Include date and country columns as numerical features
        date_column = self.data[
            ['date']].values if 'date' in self.data.columns else np.zeros(
            (self.data.shape[0], 1))
        country_column = self.data[
            ['country']].values if 'country' in self.data.columns else np.zeros(
            (self.data.shape[0], 1))

        # Combine all features
        features = np.hstack((text_features, date_column, country_column))

        # Standardize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Apply PCA to reduce the dimensionality (optional, can be skipped if not needed)
        pca = PCA(n_components=200)
        features_pca = pca.fit_transform(features_scaled)

        # Initialize SMOTEENN with k_neighbors=2
        smote_enn = SMOTEENN(smote=SMOTE(k_neighbors=2), random_state=42)

        # Function to balance a single label while preserving rare classes
        def balance_label(X, y):
            class_counts = y.value_counts()
            rare_classes = class_counts[
                class_counts < 3].index  # Identify classes with fewer than 3 samples
            common_classes = class_counts[class_counts >= 3].index

            # Separate rare and common classes
            X_common = X[y.isin(common_classes)]
            y_common = y[y.isin(common_classes)]
            X_rare = X[y.isin(rare_classes)]
            y_rare = y[y.isin(rare_classes)]

            # Apply SMOTEENN only to common classes
            if len(y_common) > 1:  # Ensure there are enough samples for SMOTE
                X_balanced, y_balanced = smote_enn.fit_resample(X_common,
                                                                y_common)
            else:
                X_balanced, y_balanced = X_common, y_common

            # Combine the balanced common classes with the rare classes
            X_final = np.vstack((X_balanced, X_rare))
            y_final = pd.concat([pd.Series(y_balanced), pd.Series(y_rare)],
                                ignore_index=True)

            return X_final, y_final

        # Balancing each label
        y_product_category = self.data['product-category']
        X_balanced, y_balanced_product_category = balance_label(features_pca,
                                                                y_product_category)

        y_hazard_category = self.data['hazard-category']
        _, y_balanced_hazard_category = balance_label(features_pca,
                                                      y_hazard_category)

        y_product = self.data['product']
        _, y_balanced_product = balance_label(features_pca, y_product)

        y_hazard = self.data['hazard']
        _, y_balanced_hazard = balance_label(features_pca, y_hazard)

        # Combine the PCA features into a DataFrame
        balanced_data = pd.DataFrame(X_balanced,
                                     columns=[f'PC{i + 1}' for i in range(200)])

        # Add the original/preprocessed columns and label categories back to the DataFrame
        balanced_data['text'] = self.data['text']
        balanced_data['title'] = self.data['title']
        balanced_data['date'] = self.data['date']

        # Add the 'country' column from the original data
        balanced_data['country'] = self.data['country']

        # Replace the numerical country index with the actual country name
        balanced_data['country'] = balanced_data['country'].map(
            self.country_mapping)

        balanced_data['product-category'] = y_balanced_product_category
        balanced_data['hazard-category'] = y_balanced_hazard_category
        balanced_data['product'] = y_balanced_product
        balanced_data['hazard'] = y_balanced_hazard

        self.data = balanced_data
        print("✔ Balanced data with original columns and labels added back.")

    def preprocess(self):
        self.drop_index_column()
        self.combine_date_columns()
        self.encode_country_column()
        # Removed self.clean_number_format()
        self.data.dropna(subset=['text'], inplace=True)
        self.data['text'] = self.data['text'].apply(self.preprocess_text)
        if 'title' in self.data.columns:
            self.data['title'] = self.data['title'].fillna('').apply(self.preprocess_text)
        print("✔ Preprocessed text and title columns successfully.")
        self.balance_data()
        print("✔ Preprocessing completed successfully.")
        return self.data

    def save_preprocessed_data(self):
        print("dimension - ", self.data.shape)
        self.data.to_csv('../data/final_preprocessed_data.csv', index=False)
        print("✔ Final preprocessed data saved to 'final_preprocessed_data.csv'.")


if __name__ == '__main__':
    file_path = '../data/incidents_train.csv'
    mappings_file = 'label_mappings.json'
    preprocessor = DataPreprocessor(file_path, mappings_file)
    preprocessed_data = preprocessor.preprocess()
    preprocessor.save_preprocessed_data()
