"""
Sentiment Analysis Web Application - Connected to Real ML Model
Using your actual 88.30% accurate Logistic Regression model

To run:
1. First run save_model.py to create model files
2. Then: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import re
from nltk.corpus import stopwords
import nltk
import os

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Page config
st.set_page_config(
    page_title="Sentiment Analysis - ML Model",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load ML Model
@st.cache_resource
def load_ml_model():
    """Load the trained ML model and vectorizer"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load metadata if available
        try:
            with open('model_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
        except:
            metadata = {'accuracy': 88.30, 'model_type': 'logistic'}
        
        return model, vectorizer, metadata
    except FileNotFoundError:
        st.error("""
        ⚠️ Model files not found! 
        
        Please run `save_model.py` first to train and save the model.
        
        This will create:
        - model.pkl
        - vectorizer.pkl
        - model_metadata.pkl
        """)
        return None, None, None

# Load model at startup
model, vectorizer, metadata = load_ml_model()

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def preprocess_text(text):
    """Clean and preprocess text (same as training)"""
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])
    return text

def analyze_with_ml(text):
    """Analyze sentiment using the trained ML model"""
    if model is None or vectorizer is None:
        return None
    
    # Preprocess
    clean_text = preprocess_text(text)
    
    # Vectorize
    text_vectorized = vectorizer.transform([clean_text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    
    # Get probability (confidence)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(text_vectorized)[0]
        confidence = probabilities[prediction] * 100
    else:
        # For SVM without probability
        decision = model.decision_function(text_vectorized)[0]
        confidence = min(95, 50 + abs(decision) * 10)
    
    # Determine sentiment
    sentiment = "Positive" if prediction == 1 else "Negative"
    emoji = "😊" if prediction == 1 else "😞"
    color = "#10b981" if prediction == 1 else "#ef4444"
    
    # Additional metrics
    word_count = len(text.split())
    exclamations = text.count('!')
    questions = text.count('?')
    
    return {
        'sentiment': sentiment,
        'prediction': prediction,
        'confidence': confidence,
        'emoji': emoji,
        'color': color,
        'word_count': word_count,
        'exclamations': exclamations,
        'questions': questions
    }

# Header
st.markdown('<div class="main-header">🤖 Sentiment Analysis with ML</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Using Real Machine Learning Model (88.30% Accuracy)</div>', unsafe_allow_html=True)

# Check if model is loaded
if model is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    
    if metadata:
        st.metric("Model Type", metadata.get('model_type', 'Logistic Regression').title())
        st.metric("Accuracy", f"{metadata.get('accuracy', 88.30)*100:.2f}%")
        st.metric("Features", f"{metadata.get('features', 5000):,}")
        st.metric("Training Samples", f"{metadata.get('training_samples', 20000):,}")
    
    st.markdown("---")
    
    st.markdown("""
    **🎓 Thesis Project**
    
    This web app uses the actual machine learning model from the thesis:
    
    - **Dataset:** 25,000 IMDB Reviews
    - **Algorithm:** Logistic Regression
    - **Preprocessing:** TF-IDF Vectorization
    - **Validation:** 80-20 Train-Test Split
    
    **Comparison with Other Models:**
    - Logistic Regression: 88.30%
    - SVM: 89.50%
    - Naive Bayes: 85.20%
    - Random Forest: 86.70%
    - Decision Tree: 76.30%
    """)
    
    st.markdown("---")
    
    if st.session_state.analysis_history:
        st.markdown("**📈 Session Stats:**")
        total = len(st.session_state.analysis_history)
        positive = sum(1 for r in st.session_state.analysis_history if r['sentiment'] == 'Positive')
        st.metric("Analyzed", total)
        st.metric("Positive", f"{positive/total*100:.0f}%")

# Main content - Add tabs for single vs batch analysis
tab1, tab2 = st.tabs(["📝 Single Review Analysis", "📊 Batch Analysis"])

# TAB 1: Single Review Analysis (existing functionality)
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 Analyze Movie Review")
    
    # Example reviews
    st.markdown("**Quick Examples:**")
    examples = {
        "": "",  # Empty option
        "Very Positive": "This movie was absolutely fantastic! The acting was superb and I loved every minute of it. Best film I've seen this year!",
        "Very Negative": "Terrible waste of time. Boring and predictable throughout. Would not recommend to anyone. Worst movie ever.",
        "Positive": "One of the best films I've seen this year. Highly recommend to everyone!",
        "Negative": "Awful experience. Poor storyline and bad acting. Complete disappointment.",
        "Mixed": "Good acting but the plot was somewhat disappointing. Some parts were enjoyable though."
    }
    
    # Callback function to load example
    def load_example():
        if st.session_state.example_selector != "":
            st.session_state.review_input = examples[st.session_state.example_selector]
    
    selected_example = st.selectbox(
        "Choose an example:", 
        list(examples.keys()), 
        index=0, 
        key="example_selector",
        on_change=load_example
    )
    
    # Text input (comes AFTER selectbox so callback works)
    review_text = st.text_area(
        "Enter a movie review:",
        height=200,
        placeholder="Type or paste a movie review here...\n\nExample: 'This movie was absolutely fantastic! The acting was superb and I loved every minute of it.'",
        key="review_input"
    )
    
    # Buttons
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        analyze_btn = st.button("🔍 Analyze with ML Model", type="primary")
    with col_btn2:
        clear_btn = st.button("🗑️ Clear")
    
    if clear_btn:
        st.rerun()

    with col2:
        st.subheader("📈 ML Model Results")
        
        if analyze_btn and review_text:
            with st.spinner("🤖 ML model analyzing..."):
                result = analyze_with_ml(review_text)
                
                if result:
                    # Save to history
                    st.session_state.analysis_history.append(result)
                    
                    # Main result card
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {result['color']}22 0%, {result['color']}44 100%); 
                                padding: 2rem; border-radius: 1rem; border: 2px solid {result['color']}88; text-align: center;'>
                        <div style='font-size: 4rem;'>{result['emoji']}</div>
                        <div style='font-size: 2.5rem; font-weight: bold; color: {result['color']};'>
                            {result['sentiment']}
                        </div>
                        <div style='font-size: 1.2rem; color: #6b7280; margin-top: 0.5rem;'>
                            ML Confidence: <strong>{result['confidence']:.2f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Metrics
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.metric("📝 Word Count", result['word_count'])
                    with col_m2:
                        st.metric("❗ Exclamations", result['exclamations'])
                    with col_m3:
                        st.metric("❓ Questions", result['questions'])
                    
                    # Model info
                    st.success(f"""
                    **✅ Analysis Complete!**
                    
                    This prediction was made using your trained **{metadata.get('model_type', 'Logistic Regression').title()}** model 
                    with **{metadata.get('accuracy', 88.30)*100:.2f}% accuracy** on 25,000 IMDB reviews.
                    
                    The model analyzed **{metadata.get('features', 5000):,} TF-IDF features** from your review.
                    """)
                    
                    # Show preprocessing info
                    with st.expander("🔬 Technical Details"):
                        st.markdown(f"""
                        **Preprocessing Applied:**
                        - Converted to lowercase
                        - Removed HTML tags and URLs
                        - Removed special characters
                        - Removed stopwords
                        
                        **Model Pipeline:**
                        1. Text → Preprocessing
                        2. Clean Text → TF-IDF Vectorization ({metadata.get('features', 5000):,} features)
                        3. Vector → ML Model (Logistic Regression)
                        4. Prediction → Confidence Score
                        
                        **Raw Prediction Value:** {result['prediction']} (0=Negative, 1=Positive)
                        """)
        else:
            st.info("👈 Enter a review and click 'Analyze with ML Model' to see results")

# TAB 2: Batch Analysis (NEW FEATURE!)
with tab2:
    st.subheader("📊 Batch Analysis - Analyze Multiple Reviews")
    
    st.markdown("""
    Upload a CSV file containing movie reviews to analyze hundreds or thousands of reviews at once!
    
    **CSV Format Required:**
    - Must have a column named `review` (or `text`, `comment`, `feedback`)
    - Optional: `title` or `movie_name` column
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file with reviews",
        type=['csv'],
        help="CSV must contain a 'review' column with text to analyze"
    )
    
    # Sample CSV download
    col_sample1, col_sample2 = st.columns([1, 3])
    with col_sample1:
        # Create sample CSV
        sample_data = pd.DataFrame({
            'review': [
                'This movie was absolutely fantastic! Loved every minute.',
                'Terrible waste of time. Very disappointing.',
                'Amazing acting and great storyline. Highly recommend!',
                'Boring and predictable. Would not watch again.',
                'One of the best films I have seen this year!'
            ],
            'movie_name': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E']
        })
        
        csv_sample = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv_sample,
            file_name="sample_reviews.csv",
            mime="text/csv",
            help="Download a sample CSV file to see the required format"
        )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df_batch = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df_batch):,} reviews")
            
            # Show preview
            with st.expander("📋 Preview uploaded data"):
                st.dataframe(df_batch.head(10))
            
            # Find review column
            review_col = None
            for col in ['review', 'text', 'comment', 'feedback', 'Review', 'Text']:
                if col in df_batch.columns:
                    review_col = col
                    break
            
            if review_col is None:
                st.error("❌ Could not find review column. Please ensure your CSV has a column named 'review', 'text', 'comment', or 'feedback'")
                st.stop()
            
            st.info(f"Using column: **{review_col}** for analysis")
            
            # Analyze button
            if st.button("🚀 Analyze All Reviews", type="primary", key="batch_analyze"):
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_list = []
                
                # Analyze each review
                for idx, row in df_batch.iterrows():
                    review_text = str(row[review_col])
                    
                    # Skip empty reviews
                    if not review_text or review_text.strip() == '' or review_text == 'nan':
                        continue
                    
                    # Analyze
                    result = analyze_with_ml(review_text)
                    
                    if result:
                        result['review_text'] = review_text[:100] + '...'  # Truncate for display
                        if 'movie_name' in df_batch.columns:
                            result['movie_name'] = row['movie_name']
                        results_list.append(result)
                    
                    # Update progress
                    progress = (idx + 1) / len(df_batch)
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing... {idx + 1}/{len(df_batch)} reviews")
                
                progress_bar.empty()
                status_text.empty()
                
                # Create results dataframe
                df_results = pd.DataFrame(results_list)
                
                st.success(f"✅ Analysis complete! Processed {len(df_results):,} reviews")
                
                # Summary Statistics
                st.markdown("---")
                st.subheader("📊 Summary Statistics")
                
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    total_reviews = len(df_results)
                    st.metric("Total Reviews", f"{total_reviews:,}")
                
                with col_stat2:
                    positive_count = (df_results['sentiment'] == 'Positive').sum()
                    positive_pct = (positive_count / total_reviews) * 100
                    st.metric("Positive Reviews", f"{positive_count:,}", f"{positive_pct:.1f}%")
                
                with col_stat3:
                    negative_count = (df_results['sentiment'] == 'Negative').sum()
                    negative_pct = (negative_count / total_reviews) * 100
                    st.metric("Negative Reviews", f"{negative_count:,}", f"{negative_pct:.1f}%")
                
                with col_stat4:
                    avg_confidence = df_results['confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                
                # Visualizations
                st.markdown("---")
                st.subheader("📈 Visual Analysis")
                
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    # Sentiment Distribution Pie Chart
                    sentiment_counts = df_results['sentiment'].value_counts()
                    fig_pie = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_viz2:
                    # Confidence Distribution
                    fig_hist = px.histogram(
                        df_results,
                        x='confidence',
                        color='sentiment',
                        title="Confidence Score Distribution",
                        color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444'},
                        nbins=20
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Detailed Results Table
                st.markdown("---")
                st.subheader("📋 Detailed Results")
                
                # Prepare display dataframe
                display_cols = ['sentiment', 'confidence', 'review_text']
                if 'movie_name' in df_results.columns:
                    display_cols.insert(0, 'movie_name')
                
                display_df = df_results[display_cols].copy()
                display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # Download Results
                st.markdown("---")
                st.subheader("💾 Download Results")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    # Download full results as CSV
                    csv_output = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results (CSV)",
                        data=csv_output,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
                
                with col_dl2:
                    # Download summary statistics
                    summary_data = {
                        'Metric': ['Total Reviews', 'Positive Reviews', 'Negative Reviews', 
                                  'Positive %', 'Negative %', 'Average Confidence'],
                        'Value': [
                            total_reviews,
                            positive_count,
                            negative_count,
                            f"{positive_pct:.2f}%",
                            f"{negative_pct:.2f}%",
                            f"{avg_confidence:.2f}%"
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    csv_summary = summary_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Summary (CSV)",
                        data=csv_summary,
                        file_name="sentiment_analysis_summary.csv",
                        mime="text/csv"
                    )
                
                # Top Reviews
                st.markdown("---")
                st.subheader("🏆 Notable Reviews")
                
                col_top1, col_top2 = st.columns(2)
                
                with col_top1:
                    st.markdown("**Most Positive (Highest Confidence)**")
                    top_positive = df_results[df_results['sentiment'] == 'Positive'].nlargest(3, 'confidence')
                    for idx, row in top_positive.iterrows():
                        st.success(f"**{row.get('movie_name', 'Review')}** ({row['confidence']:.1f}%)\n\n{row['review_text']}")
                
                with col_top2:
                    st.markdown("**Most Negative (Highest Confidence)**")
                    top_negative = df_results[df_results['sentiment'] == 'Negative'].nlargest(3, 'confidence')
                    for idx, row in top_negative.iterrows():
                        st.error(f"**{row.get('movie_name', 'Review')}** ({row['confidence']:.1f}%)\n\n{row['review_text']}")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted with a 'review' column.")

# Analysis History Section
if st.session_state.analysis_history:
    st.markdown("---")
    st.subheader("📊 Analysis History")
    
    history_df = pd.DataFrame(st.session_state.analysis_history)
    
    col_h1, col_h2, col_h3 = st.columns(3)
    
    with col_h1:
        total = len(history_df)
        st.metric("Total Analyzed", total)
    
    with col_h2:
        positive_pct = (history_df['sentiment'] == 'Positive').sum() / total * 100
        st.metric("Positive Rate", f"{positive_pct:.1f}%")
    
    with col_h3:
        avg_conf = history_df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
    
    # Sentiment distribution
    sentiment_counts = history_df['sentiment'].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Algorithm comparison section
st.markdown("---")
st.subheader("📊 Algorithm Performance Comparison")

algo_data = pd.DataFrame({
    'Algorithm': ['Logistic Regression', 'Naive Bayes', 'SVM', 'Random Forest', 'Decision Tree'],
    'Accuracy': [88.30, 85.20, 89.50, 86.70, 76.30],
    'Color': ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444']
})

fig = go.Figure(data=[
    go.Bar(
        x=algo_data['Algorithm'],
        y=algo_data['Accuracy'],
        marker=dict(
            color=algo_data['Color'],
            line=dict(color='white', width=2)
        ),
        text=algo_data['Accuracy'].apply(lambda x: f'{x}%'),
        textposition='outside',
    )
])

fig.update_layout(
    title='Machine Learning Algorithm Accuracy Comparison (From Thesis)',
    xaxis_title='Algorithm',
    yaxis_title='Accuracy (%)',
    yaxis=dict(range=[70, 95]),
    height=400,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p><strong>🎓 Bachelor's Thesis Project</strong></p>
    <p><strong>Comparative Analysis of Machine Learning Algorithms for Sentiment Classification</strong></p>
    <p style='margin-top: 1rem;'>
        This application uses the actual trained model from the thesis research.
        The model was trained on 25,000 IMDB movie reviews and achieves 88.30% accuracy.
    </p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
        Demonstrates automation of sentiment analysis that would otherwise require 486 hours of manual classification.
    </p>
</div>
""", unsafe_allow_html=True)