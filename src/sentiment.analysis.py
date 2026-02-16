"""
Comparative Analysis of Machine Learning Algorithms 
for Sentiment Classification of IMDB Movie Reviews

Bachelor's Thesis Implementation
"""

import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# NLTK for text processing
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
from nltk.corpus import stopwords

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"


class SentimentAnalyzer:
    """Complete sentiment analysis system with multiple algorithms"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.models = {}
        self.results = []
        self.seeded_results = []
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        words = text.split()
        text = ' '.join([word for word in words if word not in self.stop_words])
        return text
    
    def prepare_data(self, df, text_column, label_column):
        """Prepare data for training"""
        print("Preprocessing text data...")
        df['clean_text'] = df[text_column].apply(self.preprocess_text)
        
        if df[label_column].dtype == 'object':
            normalized = df[label_column].astype(str).str.strip().str.lower()
            label_map = {
                'negative': 0,
                'neg': 0,
                '0': 0,
                'false': 0,
                'positive': 1,
                'pos': 1,
                '1': 1,
                'true': 1,
            }
            df['label'] = normalized.map(label_map)

            unknown_labels = sorted(normalized[df['label'].isna()].unique().tolist())
            if unknown_labels:
                raise ValueError(
                    "Unexpected sentiment labels found: "
                    f"{unknown_labels}. Expected positive/negative (or 1/0)."
                )
        else:
            numeric_labels = pd.to_numeric(df[label_column], errors='coerce')
            if numeric_labels.isna().any():
                raise ValueError("Numeric label column contains non-numeric values.")

            if not set(numeric_labels.unique().tolist()).issubset({0, 1, 0.0, 1.0}):
                raise ValueError("Numeric label column must contain only 0 and 1 values.")
            df['label'] = numeric_labels.astype(int)
        
        return df
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, model_name, model):
        """Train a model and collect performance metrics"""
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        # Vectorize data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train and measure time
        start_time = time.time()
        model.fit(X_train_vec, y_train)
        training_time = time.time() - start_time
        
        # Predict and measure time
        start_time = time.time()
        y_pred = model.predict(X_test_vec)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store model and results
        self.models[model_name] = model
        
        result = {
            'Algorithm': model_name,
            'Accuracy': accuracy * 100,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1-Score': f1 * 100,
            'Training Time (s)': training_time,
            'Prediction Time (s)': prediction_time,
            'Predictions': y_pred
        }
        self.results.append(result)
        
        # Print results
        print(f"Accuracy:         {accuracy * 100:.2f}%")
        print(f"Precision:        {precision * 100:.2f}%")
        print(f"Recall:           {recall * 100:.2f}%")
        print(f"F1-Score:         {f1 * 100:.2f}%")
        print(f"Training Time:    {training_time:.2f} seconds")
        print(f"Prediction Time:  {prediction_time:.4f} seconds")
        
        return result
    
    def compare_all_algorithms(self, X_train, X_test, y_train, y_test):
        """Train and compare all algorithms"""
        
        algorithms = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': MultinomialNB(),
            'Support Vector Machine': LinearSVC(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        print("\n" + "="*70)
        print("TRAINING ALL ALGORITHMS")
        print("="*70)
        
        for name, model in algorithms.items():
            self.train_and_evaluate(X_train, X_test, y_train, y_test, name, model)
        
        return self.results

    def evaluate_algorithms_with_seeds(self, X, y, seeds, test_size=0.2):
        """Evaluate algorithms across multiple random seeds using split per seed"""
        algorithms = {
            'Logistic Regression': lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
            'Naive Bayes': lambda seed: MultinomialNB(),
            'Support Vector Machine': lambda seed: LinearSVC(random_state=seed, max_iter=1000),
            'Random Forest': lambda seed: RandomForestClassifier(
                n_estimators=100, random_state=seed, n_jobs=-1
            ),
            'Decision Tree': lambda seed: DecisionTreeClassifier(random_state=seed)
        }

        self.seeded_results = []

        print("\n" + "="*70)
        print("SEED-BASED EVALUATION")
        print("="*70)

        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            for name, model_factory in algorithms.items():
                model = model_factory(seed)

                model.fit(X_train_vec, y_train)
                y_pred = model.predict(X_test_vec)

                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                self.seeded_results.append({
                    'Algorithm': name,
                    'Seed': seed,
                    'Accuracy': accuracy,
                    'F1': f1
                })

                print(f"{name} | Seed {seed} | Accuracy: {accuracy:.4f} | F1: {f1:.4f}")

        return self.seeded_results
    
    def generate_comparison_table(self):
        """Generate comparison table of all algorithms"""
        df_results = pd.DataFrame(self.results)
        df_results = df_results.drop('Predictions', axis=1)
        
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON TABLE")
        print("="*70)
        print(df_results.to_string(index=False))
        
        return df_results
    
    def plot_accuracy_comparison(self):
        """Create bar chart comparing accuracies"""
        df_results = pd.DataFrame(self.results)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df_results['Algorithm'], df_results['Accuracy'], 
                       color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
        plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        plt.title('Accuracy Comparison of Machine Learning Algorithms', 
                  fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "accuracy_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] {output_path}")
        plt.show()
    
    def plot_metrics_comparison(self):
        """Create grouped bar chart for all metrics"""
        df_results = pd.DataFrame(self.results)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(df_results['Algorithm']))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, df_results[metric], width, 
                   label=metric, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df_results['Algorithm'], rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "metrics_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_path}")
        plt.show()
    
    def plot_time_comparison(self):
        """Create bar chart comparing training times"""
        df_results = pd.DataFrame(self.results)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Training time
        ax1.bar(df_results['Algorithm'], df_results['Training Time (s)'], 
                color='#2E86AB', alpha=0.8)
        ax1.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax1.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Prediction time
        ax2.bar(df_results['Algorithm'], df_results['Prediction Time (s)'], 
                color='#F18F01', alpha=0.8)
        ax2.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax2.set_title('Prediction Time Comparison', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "time_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_path}")
        plt.show()
    
    def plot_confusion_matrices(self, y_test):
        """Create confusion matrices for all algorithms"""
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, result in enumerate(self.results):
            cm = confusion_matrix(y_test, result['Predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=axes[idx], cbar=True)
            
            axes[idx].set_title(f"{result['Algorithm']}\nAccuracy: {result['Accuracy']:.2f}%", 
                               fontweight='bold')
            axes[idx].set_ylabel('Actual', fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontweight='bold')
        
        # Hide extra subplot
        if n_models < 6:
            axes[5].axis('off')
        
        plt.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "confusion_matrices.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_path}")
        plt.show()
    
    def save_results_to_csv(self):
        """Save results to CSV for thesis"""
        df_results = pd.DataFrame(self.results)
        df_results = df_results.drop('Predictions', axis=1)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / "algorithm_comparison_results.csv"
        df_results.to_csv(output_path, index=False)
        print(f"\n[SAVED] {output_path}")

    def save_seeded_results_to_csv(self, df_results, filename="seeded_algorithm_results.csv"):
        """Save seeded evaluation results to CSV"""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / filename
        df_results.to_csv(output_path, index=False)
        print(f"\n[SAVED] {output_path}")

    def export_error_examples_lr(self, df_raw, seeds=[0], test_size=0.2, n_fp=10, n_fn=10):
        """Export top FP/FN error examples for Logistic Regression"""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            train_idx, test_idx = train_test_split(
                df_raw.index, test_size=test_size, random_state=seed, stratify=df_raw['label']
            )

            X_train = df_raw.loc[train_idx, 'clean_text']
            y_train = df_raw.loc[train_idx, 'label']
            X_test = df_raw.loc[test_idx, 'clean_text']
            y_test = df_raw.loc[test_idx, 'label']

            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            prob_pos = model.predict_proba(X_test_vec)[:, 1]

            df_test = pd.DataFrame({
                'Algorithm': 'Logistic Regression',
                'Seed': seed,
                'review_raw': df_raw.loc[test_idx, 'review'].values,
                'review_clean': X_test.values,
                'true_label': y_test.values,
                'pred_label': y_pred,
                'prob_pos': prob_pos
            })

            df_test['error_type'] = 'Correct'
            df_test.loc[(df_test['true_label'] == 0) & (df_test['pred_label'] == 1), 'error_type'] = 'FP'
            df_test.loc[(df_test['true_label'] == 1) & (df_test['pred_label'] == 0), 'error_type'] = 'FN'

            df_errors = df_test[df_test['error_type'].isin(['FP', 'FN'])]
            df_fp = df_errors[df_errors['error_type'] == 'FP'].sort_values(
                'prob_pos', ascending=False
            ).head(n_fp)
            df_fn = df_errors[df_errors['error_type'] == 'FN'].sort_values(
                'prob_pos', ascending=True
            ).head(n_fn)

            df_export = pd.concat([df_fp, df_fn], ignore_index=True)
            output_path = RESULTS_DIR / f"error_analysis_lr_seed{seed}.csv"
            df_export.to_csv(output_path, index=False)

            total_fp = (df_errors['error_type'] == 'FP').sum()
            total_fn = (df_errors['error_type'] == 'FN').sum()
            print(
                f"Seed {seed} | test_size={test_size} | total FP={total_fp} | "
                f"total FN={total_fn} | saved: {output_path}"
            )


def run_complete_comparison(filepath=None):
    """Run complete comparative analysis"""
    
    print("="*70)
    print("COMPARATIVE ANALYSIS OF ML ALGORITHMS")
    print("Sentiment Classification of IMDB Movie Reviews")
    print("="*70)
    
    # Load dataset
    print("\n1. Loading IMDB dataset...")
    if filepath is None:
        filepath = DATA_DIR / "IMBD Dataset.csv"
    else:
        filepath = Path(filepath)

    try:
        df = pd.read_csv(filepath)
        print(f"[SUCCESS] Dataset loaded!")
        print(f"  Total reviews: {len(df):,}")
        print(f"\nSentiment distribution:")
        print(df['sentiment'].value_counts())
    except FileNotFoundError:
        print(f"[ERROR] Could not find '{filepath}'")
        print("\nMake sure the CSV file is in the same folder!")
        return
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Prepare data
    print("\n2. Preprocessing data...")
    df = analyzer.prepare_data(df, 'review', 'sentiment')
    print("[SUCCESS] Data preprocessed!")
    
    # Seed-based evaluation
    print("\n3. Evaluating algorithms across multiple seeds...")
    print("   (This may take 3-5 minutes...)")
    seeded_results = analyzer.evaluate_algorithms_with_seeds(
        df['clean_text'], df['label'], seeds=[0, 1, 2, 3, 4]
    )

    # Build results table
    print("\n4. Building results table...")
    df_seeded = pd.DataFrame(seeded_results)
    df_seeded = df_seeded.sort_values(['Algorithm', 'Seed']).reset_index(drop=True)
    print(df_seeded.to_string(index=False))

    # Save results
    analyzer.save_seeded_results_to_csv(df_seeded)

    # Step 4: Error analysis export
    analyzer.export_error_examples_lr(df, seeds=[0], n_fp=10, n_fn=10)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print("\nGenerated files for your thesis:")
    print(f"  1. {RESULTS_DIR / 'seeded_algorithm_results.csv'}")
    print("\nAll files are saved in the results folder.")

    return analyzer, df_seeded


# Run the complete comparison
if __name__ == "__main__":
    analyzer, results = run_complete_comparison()
