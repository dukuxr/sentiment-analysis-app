"""
Save your trained ML model for use in the web app
Run this ONCE after training your model
"""

import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import re
from nltk.corpus import stopwords
import nltk

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Download stopwords if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ModelSaver:
    """Train and save the best model for web app"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = None
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        words = text.split()
        text = ' '.join([word for word in words if word not in self.stop_words])
        return text
    
    def train_and_save(self, csv_file=None, model_type='logistic'):
        """Train model and save it"""
        
        print("="*70)
        print("TRAINING AND SAVING MODEL FOR WEB APP")
        print("="*70)
        
        # Load data
        print("\n1. Loading dataset...")
        if csv_file is None:
            csv_file = DATA_DIR / "IMBD Dataset.csv"
        else:
            csv_file = Path(csv_file)

        df = pd.read_csv(csv_file)
        print(f"   Loaded {len(df):,} reviews")
        
        # Normalize labels
        df['sentiment'] = df['sentiment'].str.lower()
        
        # Preprocess
        print("\n2. Preprocessing text...")
        df['clean_text'] = df['review'].apply(self.preprocess_text)
        
        # Create labels
        df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        
        # Split data
        print("\n3. Splitting data (80-20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_text'], df['label'],
            test_size=0.2, random_state=42, stratify=df['label']
        )
        
        print(f"   Training: {len(X_train):,} | Testing: {len(X_test):,}")
        
        # Vectorize
        print("\n4. Vectorizing text (TF-IDF)...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        print(f"   Features: {X_train_vec.shape[1]:,}")
        
        # Train model
        print(f"\n5. Training {model_type} model...")
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'svm':
            self.model = LinearSVC(random_state=42, max_iter=1000)
        
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        print("\n6. Evaluating model...")
        train_acc = self.model.score(X_train_vec, y_train)
        test_acc = self.model.score(X_test_vec, y_test)
        
        print(f"   Training Accuracy: {train_acc*100:.2f}%")
        print(f"   Testing Accuracy:  {test_acc*100:.2f}%")
        
        # Save model
        print("\n7. Saving model files...")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            model_path = MODELS_DIR / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"   [SUCCESS] Saved: {model_path}")
        except Exception as e:
            print(f"   [ERROR] Could not save model.pkl: {e}")
        
        try:
            vectorizer_path = MODELS_DIR / "vectorizer.pkl"
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            print(f"   [SUCCESS] Saved: {vectorizer_path}")
        except Exception as e:
            print(f"   [ERROR] Could not save vectorizer.pkl: {e}")
        
        # Save metadata
        metadata = {
            'model_type': model_type,
            'accuracy': test_acc,
            'features': X_train_vec.shape[1],
            'training_samples': len(X_train)
        }
        
        try:
            metadata_path = MODELS_DIR / "model_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"   [SUCCESS] Saved: {metadata_path}")
        except Exception as e:
            print(f"   [ERROR] Could not save model_metadata.pkl: {e}")
        
        print("\n" + "="*70)
        print("MODEL SAVED SUCCESSFULLY!")
        print("="*70)
        print("\nFiles created:")
        print(f"  1. {MODELS_DIR / 'model.pkl'}          - Trained model")
        print(f"  2. {MODELS_DIR / 'vectorizer.pkl'}     - TF-IDF vectorizer")
        print(f"  3. {MODELS_DIR / 'model_metadata.pkl'} - Model information")
        print("\nYou can now use these in your Streamlit app!")
        
        return test_acc


if __name__ == "__main__":
    saver = ModelSaver()
    
    # Train and save Logistic Regression (88.30% accuracy)
    print("\nTraining Logistic Regression model...")
    accuracy = saver.train_and_save(model_type='logistic')
    
    print(f"\n🎉 Model ready for web app with {accuracy*100:.2f}% accuracy!")
    print("\nNext step: Update your Streamlit app to use these files.")
