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

# NLTK for text processing
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
from nltk.corpus import stopwords


class SentimentAnalyzer:
    """Complete sentiment analysis system with multiple algorithms"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.models = {}
        self.results = []
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
            unique_labels = df[label_column].unique()
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            df['label'] = df[label_column].map(label_map)
        else:
            df['label'] = df[label_column]
        
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
        plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print("\n[SAVED] accuracy_comparison.png")
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
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("[SAVED] metrics_comparison.png")
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
        plt.savefig('time_comparison.png', dpi=300, bbox_inches='tight')
        print("[SAVED] time_comparison.png")
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
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("[SAVED] confusion_matrices.png")
        plt.show()
    
    def save_results_to_csv(self):
        """Save results to CSV for thesis"""
        df_results = pd.DataFrame(self.results)
        df_results = df_results.drop('Predictions', axis=1)
        df_results.to_csv('algorithm_comparison_results.csv', index=False)
        print("\n[SAVED] algorithm_comparison_results.csv")


def run_complete_comparison(filepath='IMBD Dataset.csv'):
    """Run complete comparative analysis"""
    
    print("="*70)
    print("COMPARATIVE ANALYSIS OF ML ALGORITHMS")
    print("Sentiment Classification of IMDB Movie Reviews")
    print("="*70)
    
    # Load dataset
    print("\n1. Loading IMDB dataset...")
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], 
        test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"\n[SUCCESS] Training samples: {len(X_train):,}")
    print(f"[SUCCESS] Testing samples: {len(X_test):,}")
    
    # Compare all algorithms
    print("\n3. Training and comparing all algorithms...")
    print("   (This may take 3-5 minutes...)")
    results = analyzer.compare_all_algorithms(X_train, X_test, y_train, y_test)
    
    # Generate comparison table
    print("\n4. Generating comparison results...")
    df_comparison = analyzer.generate_comparison_table()
    
    # Find best algorithm
    best_algo = df_comparison.loc[df_comparison['Accuracy'].idxmax()]
    print(f"\n{'='*70}")
    print("BEST PERFORMING ALGORITHM")
    print(f"{'='*70}")
    print(f"Algorithm: {best_algo['Algorithm']}")
    print(f"Accuracy:  {best_algo['Accuracy']:.2f}%")
    print(f"Precision: {best_algo['Precision']:.2f}%")
    print(f"Recall:    {best_algo['Recall']:.2f}%")
    print(f"F1-Score:  {best_algo['F1-Score']:.2f}%")
    
    # Generate all visualizations
    print(f"\n{'='*70}")
    print("5. Generating visualizations for thesis...")
    print(f"{'='*70}")
    
    analyzer.plot_accuracy_comparison()
    analyzer.plot_metrics_comparison()
    analyzer.plot_time_comparison()
    analyzer.plot_confusion_matrices(y_test)
    
    # Save results
    analyzer.save_results_to_csv()
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print("\nGenerated files for your thesis:")
    print("  1. accuracy_comparison.png")
    print("  2. metrics_comparison.png")
    print("  3. time_comparison.png")
    print("  4. confusion_matrices.png")
    print("  5. algorithm_comparison_results.csv")
    print("\nAll files are saved in the current directory.")
    print("You can use these directly in your thesis!")
    
    return analyzer, df_comparison


# Run the complete comparison
if __name__ == "__main__":
    analyzer, results = run_complete_comparison('IMBD Dataset.csv')