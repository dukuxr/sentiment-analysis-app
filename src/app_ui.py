"""
UI rendering helpers for the Streamlit app.
Includes manual/help and testing dashboards.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import streamlit as st

from app_logic import (
    accuracy_to_percentage,
    analyze_with_ml,
    normalize_sentiment_label,
    sentiment_to_binary,
)
from app_store import generate_run_id, get_predictions_for_run, get_recent_runs, save_run


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def _find_column_case_insensitive(df: pd.DataFrame, candidates: list[str]):
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def _load_algorithm_comparison_data():
    results_file = RESULTS_DIR / "algorithm_comparison_results.csv"
    if results_file.exists():
        try:
            df = pd.read_csv(results_file)
            if {"Algorithm", "Accuracy"}.issubset(df.columns):
                df = df[["Algorithm", "Accuracy"]].copy()
                df["Accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")
                df = df.dropna(subset=["Accuracy"])
                if not df.empty:
                    # Normalize values in case stored as 0-1 instead of 0-100.
                    if float(df["Accuracy"].max()) <= 1.0:
                        df["Accuracy"] = df["Accuracy"] * 100.0

                    display_name_map = {"Support Vector Machine": "SVM"}
                    df["DisplayAlgorithm"] = df["Algorithm"].replace(display_name_map)
                    return df, "Measured results loaded from results/algorithm_comparison_results.csv"
        except Exception:
            pass

    fallback_df = pd.DataFrame(
        {
            "Algorithm": [
                "Logistic Regression",
                "Naive Bayes",
                "SVM",
                "Random Forest",
                "Decision Tree",
            ],
            "Accuracy": [88.30, 85.20, 87.30, 86.70, 76.30],
        }
    )
    fallback_df["DisplayAlgorithm"] = fallback_df["Algorithm"]
    return fallback_df, "Fallback values shown (results file not found)."


def _log_run(
    mode: str,
    source: str,
    metadata: dict | None,
    total_records: int,
    df_results: pd.DataFrame,
    duration_seconds: float,
    notes: str = "",
    metrics: dict | None = None,
):
    metrics = metrics or {}
    run_id = generate_run_id(mode[:6].upper())

    positive_count = int((df_results["sentiment"] == "Positive").sum()) if not df_results.empty else 0
    negative_count = int((df_results["sentiment"] == "Negative").sum()) if not df_results.empty else 0
    avg_confidence = float(df_results["confidence"].mean()) if not df_results.empty else None
    throughput = (len(df_results) / duration_seconds) if duration_seconds > 0 else None

    model_type = metadata.get("model_type") if metadata else None
    model_accuracy_pct = (
        accuracy_to_percentage(metadata.get("accuracy", 0.883)) if metadata else None
    )

    run_data = {
        "run_id": run_id,
        "mode": mode,
        "source": source,
        "model_type": model_type,
        "model_accuracy_pct": model_accuracy_pct,
        "total_records": total_records,
        "processed_records": len(df_results),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "avg_confidence": avg_confidence,
        "duration_seconds": duration_seconds,
        "throughput_rows_per_sec": throughput,
        "notes": notes,
    }
    run_data.update(metrics)

    prediction_rows = []
    for idx, row in df_results.reset_index(drop=True).iterrows():
        predicted = row.get("sentiment")
        actual = row.get("actual_sentiment")
        is_correct = None
        error_type = None

        if actual in {"Positive", "Negative"} and predicted in {"Positive", "Negative"}:
            is_correct = int(actual == predicted)
            if is_correct == 0:
                error_type = "FN" if actual == "Positive" else "FP"

        prediction_rows.append(
            {
                "row_index": int(row.get("row_index", idx + 1)),
                "input_text": str(row.get("full_text", row.get("review_text", ""))),
                "predicted_sentiment": predicted,
                "confidence": row.get("confidence"),
                "actual_sentiment": actual,
                "is_correct": is_correct,
                "error_type": error_type,
            }
        )

    save_run(run_data, prediction_rows)
    return run_id


def render_sidebar(metadata, analysis_history):
    with st.sidebar:
        st.header("Model Information")

        if metadata:
            metadata_accuracy_pct = accuracy_to_percentage(metadata.get("accuracy", 0.883))
            st.metric("Model Type", metadata.get("model_type", "Logistic Regression").title())
            st.metric("Accuracy", f"{metadata_accuracy_pct:.2f}%")
            st.metric("Features", f"{metadata.get('features', 5000):,}")
            st.metric("Training Samples", f"{metadata.get('training_samples', 20000):,}")

        st.markdown("---")
        st.markdown(
            """
    **Application Scope**

    - Dataset: 25,000 IMDb reviews
    - Pipeline: preprocessing, vectorization, inference
    - Modes: single analysis, batch analysis, testing and QA
    - Evidence: run-level and prediction-level logs in SQLite
    """
        )

        if analysis_history:
            st.markdown("---")
            st.markdown("**Session Stats**")
            total = len(analysis_history)
            positive = sum(1 for r in analysis_history if r["sentiment"] == "Positive")
            st.metric("Analyzed", total)
            st.metric("Positive", f"{positive / total * 100:.0f}%")

        st.markdown("---")
        try:
            recent = get_recent_runs(limit=1)
            st.metric("Logged Runs", "Available")
            if not recent.empty:
                st.caption(f"Latest run: {recent.iloc[0]['run_id']}")
        except Exception:
            st.metric("Logged Runs", "Unavailable")


def render_single_review(model, vectorizer, metadata, analysis_history):
    st.subheader("Single Review Analysis")

    with st.expander("Context Help: Single Review"):
        st.markdown(
            """
            Use this mode for case-based demonstration.
            - Input: one review text.
            - Output: sentiment, confidence, and technical details.
            - Evidence: each run is logged to the SQLite database.
            """
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Quick Examples**")
        examples = {
            "": "",
            "Very Positive": "This movie was absolutely fantastic! The acting was superb and I loved every minute of it. Best film I've seen this year!",
            "Very Negative": "Terrible waste of time. Boring and predictable throughout. Would not recommend to anyone. Worst movie ever.",
            "Positive": "One of the best films I've seen this year. Highly recommend to everyone!",
            "Negative": "Awful experience. Poor storyline and bad acting. Complete disappointment.",
            "Mixed": "Good acting but the plot was somewhat disappointing. Some parts were enjoyable though.",
        }

        def load_example():
            if st.session_state.example_selector != "":
                st.session_state.review_input = examples[st.session_state.example_selector]

        st.selectbox(
            "Choose an example",
            list(examples.keys()),
            index=0,
            key="example_selector",
            on_change=load_example,
            help="Pick a sample review to pre-fill the input area.",
        )

        review_text = st.text_area(
            "Enter a movie review",
            height=220,
            placeholder="Type or paste a movie review here.",
            key="review_input",
            help="This text is analyzed with the loaded model and logged as a test run.",
        )

        def clear_review():
            st.session_state.review_input = ""
            st.session_state.example_selector = ""

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            analyze_btn = st.button("Analyze with ML Model", type="primary")
        with col_btn2:
            st.button("Clear", on_click=clear_review)

    with col2:
        st.markdown("**Model Results**")

        if analyze_btn and review_text:
            start = time.perf_counter()
            with st.spinner("Analyzing..."):
                result = analyze_with_ml(review_text, model, vectorizer)
            duration = time.perf_counter() - start

            if result:
                analysis_history.append(result)

                st.markdown(
                    f"""
                <div style='background: linear-gradient(135deg, {result['color']}22 0%, {result['color']}44 100%);
                            padding: 1.5rem; border-radius: 0.75rem; border: 1px solid {result['color']}88;'>
                    <div style='font-size: 2rem; font-weight: 700; color: {result['color']};'>
                        {result['sentiment']}
                    </div>
                    <div style='font-size: 1.05rem; color: #374151; margin-top: 0.35rem;'>
                        Confidence: <strong>{result['confidence']:.2f}%</strong>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Word Count", result["word_count"])
                with col_m2:
                    st.metric("Exclamations", result["exclamations"])
                with col_m3:
                    st.metric("Questions", result["questions"])

                single_df = pd.DataFrame(
                    [
                        {
                            "row_index": 1,
                            "full_text": review_text,
                            "review_text": review_text[:100] + ("..." if len(review_text) > 100 else ""),
                            "sentiment": result["sentiment"],
                            "confidence": result["confidence"],
                        }
                    ]
                )
                run_id = _log_run(
                    mode="single",
                    source="manual_input",
                    metadata=metadata,
                    total_records=1,
                    df_results=single_df,
                    duration_seconds=duration,
                    notes="Single review analysis",
                )
                st.caption(f"Run ID logged: {run_id}")

                st.success(
                    f"""
                    Analysis completed with **{metadata.get('model_type', 'Logistic Regression').title()}**
                    ({accuracy_to_percentage(metadata.get('accuracy', 0.883)):.2f}% reference accuracy).
                    """
                )

                with st.expander("Technical Details"):
                    st.markdown(
                        f"""
                        Preprocessing steps:
                        1. Lowercase normalization
                        2. HTML and URL removal
                        3. Special character cleanup
                        4. Stopword removal

                        Pipeline:
                        1. Input text -> preprocessing
                        2. Preprocessed text -> TF-IDF ({metadata.get('features', 5000):,} features)
                        3. Vector -> classification model

                        Runtime:
                        - Inference duration: {duration:.4f} seconds
                        """
                    )
        else:
            st.info("Enter a review and click Analyze with ML Model.")


def render_batch_analysis(model, vectorizer, metadata):
    st.subheader("Batch Analysis")

    with st.expander("Context Help: Batch Analysis"):
        st.markdown(
            """
            Use this mode for scalability evidence.
            - Input: CSV with a review text column.
            - Output: aggregate metrics, distributions, and downloadable reports.
            - Evidence: each batch run is logged in SQLite with throughput.
            """
        )

    st.markdown(
        """
    Upload a CSV file containing movie reviews to analyze many records at once.

    Required:
    - A text column named one of: `review`, `text`, `comment`, `feedback`

    Optional:
    - `movie_name`
    """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file with reviews",
        type=["csv"],
        key="batch_upload",
        help="The app auto-detects the text column from supported names.",
    )

    col_sample1, _ = st.columns([1, 3])
    with col_sample1:
        sample_data = pd.DataFrame(
            {
                "review": [
                    "This movie was absolutely fantastic! Loved every minute.",
                    "Terrible waste of time. Very disappointing.",
                    "Amazing acting and great storyline. Highly recommend!",
                    "Boring and predictable. Would not watch again.",
                    "One of the best films I have seen this year!",
                ],
                "movie_name": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"],
            }
        )
        st.download_button(
            label="Download Sample CSV",
            data=sample_data.to_csv(index=False),
            file_name="sample_reviews.csv",
            mime="text/csv",
        )

    if uploaded_file is None:
        return

    try:
        df_batch = pd.read_csv(uploaded_file)
        st.success(f"File uploaded successfully. Found {len(df_batch):,} rows.")
        with st.expander("Preview uploaded data"):
            st.dataframe(df_batch.head(10), use_container_width=True)

        review_col = _find_column_case_insensitive(
            df_batch, ["review", "text", "comment", "feedback"]
        )
        if review_col is None:
            st.error(
                "Could not find a review text column. Use one of: review, text, comment, feedback."
            )
            return

        st.info(f"Using text column: **{review_col}**")

        if st.button("Analyze All Reviews", type="primary", key="batch_analyze"):
            start = time.perf_counter()
            progress_bar = st.progress(0)
            status_text = st.empty()

            results_list = []
            total_records = len(df_batch)

            for idx, row in df_batch.iterrows():
                raw_review = row.get(review_col)
                if pd.isna(raw_review):
                    progress_bar.progress((idx + 1) / total_records)
                    continue

                review_text = str(raw_review).strip()
                if not review_text:
                    progress_bar.progress((idx + 1) / total_records)
                    continue

                result = analyze_with_ml(review_text, model, vectorizer)
                if result:
                    result_row = {
                        "row_index": idx + 1,
                        "full_text": review_text,
                        "review_text": review_text[:100] + ("..." if len(review_text) > 100 else ""),
                        "sentiment": result["sentiment"],
                        "confidence": result["confidence"],
                    }
                    if "movie_name" in df_batch.columns and pd.notna(row.get("movie_name")):
                        result_row["movie_name"] = row.get("movie_name")
                    results_list.append(result_row)

                progress = (idx + 1) / total_records if total_records else 1.0
                progress_bar.progress(progress)
                status_text.text(f"Analyzing {idx + 1}/{total_records} rows")

            duration = time.perf_counter() - start
            progress_bar.empty()
            status_text.empty()

            df_results = pd.DataFrame(results_list)
            if df_results.empty:
                st.warning("No valid review text was found in the selected column.")
                return

            run_id = _log_run(
                mode="batch",
                source=str(uploaded_file.name),
                metadata=metadata,
                total_records=total_records,
                df_results=df_results,
                duration_seconds=duration,
                notes=f"Batch inference from column {review_col}",
            )

            st.success(
                f"Analysis complete. Processed {len(df_results):,} rows in {duration:.2f} s. Run ID: {run_id}"
            )

            st.markdown("---")
            st.subheader("Summary Statistics")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Total Processed", f"{len(df_results):,}")
            with col_stat2:
                positive_count = int((df_results["sentiment"] == "Positive").sum())
                positive_pct = (positive_count / len(df_results)) * 100
                st.metric("Positive", f"{positive_count:,}", f"{positive_pct:.1f}%")
            with col_stat3:
                negative_count = int((df_results["sentiment"] == "Negative").sum())
                negative_pct = (negative_count / len(df_results)) * 100
                st.metric("Negative", f"{negative_count:,}", f"{negative_pct:.1f}%")
            with col_stat4:
                throughput = len(df_results) / duration if duration > 0 else 0.0
                st.metric("Throughput", f"{throughput:.1f} rows/s")

            st.markdown("---")
            st.subheader("Visual Analysis")
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                sentiment_counts = df_results["sentiment"].value_counts()
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_map={"Positive": "#10b981", "Negative": "#ef4444"},
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_viz2:
                fig_hist = px.histogram(
                    df_results,
                    x="confidence",
                    color="sentiment",
                    title="Confidence Distribution",
                    color_discrete_map={"Positive": "#10b981", "Negative": "#ef4444"},
                    nbins=20,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("---")
            st.subheader("Detailed Results")
            display_cols = ["sentiment", "confidence", "review_text"]
            if "movie_name" in df_results.columns:
                display_cols.insert(0, "movie_name")
            display_df = df_results[display_cols].copy()
            display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display_df, use_container_width=True, height=400)

            st.markdown("---")
            st.subheader("Download Results")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="Download Full Results (CSV)",
                    data=df_results.to_csv(index=False),
                    file_name=f"sentiment_analysis_results_{run_id}.csv",
                    mime="text/csv",
                )
            with col_dl2:
                summary_df = pd.DataFrame(
                    {
                        "metric": [
                            "run_id",
                            "source",
                            "total_records",
                            "processed_records",
                            "duration_seconds",
                            "throughput_rows_per_sec",
                        ],
                        "value": [
                            run_id,
                            uploaded_file.name,
                            total_records,
                            len(df_results),
                            round(duration, 4),
                            round(len(df_results) / duration, 4) if duration > 0 else 0.0,
                        ],
                    }
                )
                st.download_button(
                    label="Download Run Metadata (CSV)",
                    data=summary_df.to_csv(index=False),
                    file_name=f"run_metadata_{run_id}.csv",
                    mime="text/csv",
                )
    except Exception as exc:
        st.error(f"Error processing file: {str(exc)}")


def render_testing_and_qa(model, vectorizer, metadata):
    st.subheader("Testing and Quality Assurance")

    st.markdown(
        """
        This section provides reproducible testing evidence:
        - labeled dataset evaluation (accuracy, precision, recall, F1),
        - confusion matrix and misclassification records,
        - persistent run logs for audit and reporting.
        """
    )

    with st.expander("Context Help: Testing Protocol"):
        st.markdown(
            """
            Recommended protocol:
            1. Upload a labeled CSV with text and ground-truth sentiment columns.
            2. Run evaluation and record run ID.
            3. Export run logs and attach them in reports.
            4. Repeat on multiple datasets to demonstrate scalability and stability.
            """
        )

    st.markdown("---")
    st.markdown("**A) Labeled Evaluation**")

    col_eval_sample, _ = st.columns([1, 3])
    with col_eval_sample:
        sample_labeled_df = pd.DataFrame(
            {
                "review": [
                    "This film was excellent and emotionally engaging.",
                    "A terrible movie with poor writing and weak acting.",
                    "Great visuals and soundtrack, I really enjoyed it.",
                    "The plot was boring and I almost fell asleep.",
                    "Amazing performances and strong direction.",
                    "Not worth watching. Completely disappointing.",
                ],
                "sentiment": [
                    "positive",
                    "negative",
                    "positive",
                    "negative",
                    "positive",
                    "negative",
                ],
                "movie_name": [
                    "Eval Movie A",
                    "Eval Movie B",
                    "Eval Movie C",
                    "Eval Movie D",
                    "Eval Movie E",
                    "Eval Movie F",
                ],
            }
        )
        st.download_button(
            label="Download Labeled Sample CSV",
            data=sample_labeled_df.to_csv(index=False),
            file_name="sample_labeled_reviews.csv",
            mime="text/csv",
            help="Use this file in labeled evaluation to test Accuracy/Precision/Recall/F1.",
        )

    eval_file = st.file_uploader(
        "Upload labeled CSV",
        type=["csv"],
        key="qa_eval_upload",
        help="Must include one text column and one label column.",
    )

    if eval_file is not None:
        try:
            df_eval = pd.read_csv(eval_file)
            st.success(f"Loaded labeled dataset with {len(df_eval):,} rows.")

            text_col = _find_column_case_insensitive(
                df_eval, ["review", "text", "comment", "feedback"]
            )
            label_col = _find_column_case_insensitive(
                df_eval,
                [
                    "sentiment",
                    "label",
                    "target",
                    "class",
                    "polarity",
                    "actual_sentiment",
                    "y_true",
                ],
            )

            if text_col is None or label_col is None:
                found_columns = [str(c) for c in df_eval.columns]
                found_columns_lower = {str(c).strip().lower() for c in df_eval.columns}

                st.error("Could not detect required columns for labeled evaluation.")
                st.info(
                    "Required columns:\n"
                    "- Text: review / text / comment / feedback\n"
                    "- Label: sentiment / label / target / class / polarity / actual_sentiment / y_true"
                )
                st.caption(f"Found columns: {', '.join(found_columns)}")

                if {"metric", "value"}.issubset(found_columns_lower):
                    st.warning(
                        "This file looks like run metadata, not a labeled dataset. "
                        "Please upload a CSV that has both review text and true sentiment labels."
                    )
            else:
                st.info(f"Text column: **{text_col}**  |  Label column: **{label_col}**")

                if st.button("Run Labeled Evaluation", type="primary", key="run_eval"):
                    start = time.perf_counter()
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    rows = []
                    total_records = len(df_eval)
                    for idx, row in df_eval.iterrows():
                        raw_text = row.get(text_col)
                        if pd.isna(raw_text):
                            progress_bar.progress((idx + 1) / total_records)
                            continue

                        review_text = str(raw_text).strip()
                        if not review_text:
                            progress_bar.progress((idx + 1) / total_records)
                            continue

                        actual_sentiment = normalize_sentiment_label(row.get(label_col))
                        if actual_sentiment is None:
                            progress_bar.progress((idx + 1) / total_records)
                            continue

                        result = analyze_with_ml(review_text, model, vectorizer)
                        if result:
                            predicted = result["sentiment"]
                            is_correct = int(predicted == actual_sentiment)
                            error_type = None
                            if is_correct == 0:
                                error_type = "FN" if actual_sentiment == "Positive" else "FP"
                            rows.append(
                                {
                                    "row_index": idx + 1,
                                    "full_text": review_text,
                                    "review_text": review_text[:120]
                                    + ("..." if len(review_text) > 120 else ""),
                                    "actual_sentiment": actual_sentiment,
                                    "sentiment": predicted,
                                    "confidence": result["confidence"],
                                    "is_correct": is_correct,
                                    "error_type": error_type,
                                }
                            )

                        progress_bar.progress((idx + 1) / total_records)
                        status_text.text(f"Evaluating {idx + 1}/{total_records} rows")

                    duration = time.perf_counter() - start
                    progress_bar.empty()
                    status_text.empty()

                    eval_results = pd.DataFrame(rows)
                    if eval_results.empty:
                        st.warning("No valid labeled rows were found for evaluation.")
                    else:
                        y_true = eval_results["actual_sentiment"].apply(sentiment_to_binary).tolist()
                        y_pred = eval_results["sentiment"].apply(sentiment_to_binary).tolist()

                        accuracy = accuracy_score(y_true, y_pred) * 100
                        precision = precision_score(y_true, y_pred, zero_division=0) * 100
                        recall = recall_score(y_true, y_pred, zero_division=0) * 100
                        f1 = f1_score(y_true, y_pred, zero_division=0) * 100

                        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
                        tp = int(cm[0, 0])
                        fn = int(cm[0, 1])
                        fp = int(cm[1, 0])
                        tn = int(cm[1, 1])

                        metrics_data = {
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "tp": tp,
                            "tn": tn,
                            "fp": fp,
                            "fn": fn,
                        }

                        run_id = _log_run(
                            mode="testing",
                            source=str(eval_file.name),
                            metadata=metadata,
                            total_records=len(df_eval),
                            df_results=eval_results,
                            duration_seconds=duration,
                            notes=f"Labeled evaluation using {label_col}",
                            metrics=metrics_data,
                        )

                        st.success(
                            f"Evaluation completed on {len(eval_results):,} rows. Run ID: {run_id}"
                        )

                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1:
                            st.metric("Accuracy", f"{accuracy:.2f}%")
                        with col_m2:
                            st.metric("Precision", f"{precision:.2f}%")
                        with col_m3:
                            st.metric("Recall", f"{recall:.2f}%")
                        with col_m4:
                            st.metric("F1-Score", f"{f1:.2f}%")

                        col_runtime1, col_runtime2 = st.columns(2)
                        with col_runtime1:
                            st.metric("Duration", f"{duration:.2f} s")
                        with col_runtime2:
                            throughput = len(eval_results) / duration if duration > 0 else 0.0
                            st.metric("Throughput", f"{throughput:.1f} rows/s")

                        conf_fig = go.Figure(
                            data=go.Heatmap(
                                z=[[tp, fn], [fp, tn]],
                                x=["Predicted Positive", "Predicted Negative"],
                                y=["Actual Positive", "Actual Negative"],
                                colorscale="Blues",
                                text=[[tp, fn], [fp, tn]],
                                texttemplate="%{text}",
                            )
                        )
                        conf_fig.update_layout(
                            title="Confusion Matrix",
                            xaxis_title="Prediction",
                            yaxis_title="Actual",
                        )
                        st.plotly_chart(conf_fig, use_container_width=True)

                        st.markdown("**Evaluation Records**")
                        st.dataframe(
                            eval_results[
                                [
                                    "row_index",
                                    "actual_sentiment",
                                    "sentiment",
                                    "confidence",
                                    "is_correct",
                                    "error_type",
                                    "review_text",
                                ]
                            ],
                            use_container_width=True,
                            height=340,
                        )

                        st.download_button(
                            label="Download Evaluation Records (CSV)",
                            data=eval_results.to_csv(index=False),
                            file_name=f"testing_records_{run_id}.csv",
                            mime="text/csv",
                        )
        except Exception as exc:
            st.error(f"Could not evaluate file: {exc}")

    st.markdown("---")
    st.markdown("**B) Run Database and Audit Trail**")

    col_limit, _ = st.columns([1, 2])
    with col_limit:
        limit = st.number_input("Recent run limit", min_value=10, max_value=500, value=50, step=10)

    try:
        runs_df = get_recent_runs(limit=int(limit))
    except Exception as exc:
        st.error(f"Could not read run database: {exc}")
        return

    if runs_df.empty:
        st.info("No runs logged yet.")
        return

    st.caption(
        "Quality metrics are populated only for runs with mode='testing'. "
        "Single/batch inference runs do not include ground-truth labels."
    )

    view_mode = st.selectbox(
        "Run view",
        ["Testing runs only", "All runs"],
        index=0,
        help="Use 'Testing runs only' to view rows with accuracy/precision/recall/F1 metrics.",
    )

    display_runs_df = runs_df.copy()
    if view_mode == "Testing runs only":
        display_runs_df = display_runs_df[display_runs_df["mode"] == "testing"].copy()
        if display_runs_df.empty:
            st.info("No testing runs found yet. Run a labeled evaluation to populate metrics.")
            return

    for metric_col in ["accuracy", "precision", "recall", "f1"]:
        if metric_col in display_runs_df.columns:
            display_runs_df[metric_col] = display_runs_df[metric_col].round(2)
            display_runs_df[metric_col] = display_runs_df[metric_col].where(
                display_runs_df[metric_col].notna(), "N/A"
            )

    st.dataframe(
        display_runs_df[
            [
                "run_id",
                "created_at_utc",
                "mode",
                "source",
                "processed_records",
                "duration_seconds",
                "throughput_rows_per_sec",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        ],
        use_container_width=True,
        height=300,
    )
    st.download_button(
        label="Download Runs Table (CSV)",
        data=runs_df.to_csv(index=False),
        file_name="run_log_export.csv",
        mime="text/csv",
    )

    mode_counts = runs_df["mode"].value_counts().reset_index()
    mode_counts.columns = ["mode", "count"]
    fig_modes = px.bar(mode_counts, x="mode", y="count", title="Run Count by Mode")
    st.plotly_chart(fig_modes, use_container_width=True)

    selected_run = st.selectbox("Inspect run predictions", runs_df["run_id"].tolist())
    if selected_run:
        pred_df = get_predictions_for_run(selected_run)
        if pred_df.empty:
            st.info("No prediction rows found for this run.")
        else:
            st.dataframe(
                pred_df[
                    [
                        "row_index",
                        "predicted_sentiment",
                        "confidence",
                        "actual_sentiment",
                        "is_correct",
                        "error_type",
                        "input_text_preview",
                    ]
                ],
                use_container_width=True,
                height=280,
            )
            st.download_button(
                label="Download Selected Run Predictions (CSV)",
                data=pred_df.to_csv(index=False),
                file_name=f"predictions_{selected_run}.csv",
                mime="text/csv",
            )


def render_help_section():
    st.subheader("Help and User Manual")

    st.markdown(
        """
        This section is intended for operational documentation and day-to-day use.
        It provides user manual content and contextual guidance for each app module.
        """
    )

    with st.expander("Manual: Quick Start", expanded=True):
        st.markdown(
            """
            1. Run `python src/save_model.py` once to create model files.
            2. Run `streamlit run src/app.py`.
            3. Use `Single Review Analysis` for case-based checks.
            4. Use `Batch Analysis` for large-scale inference.
            5. Use `Testing and Quality Assurance` for labeled validation and audit logs.
            """
        )

    with st.expander("Manual: Input Data Format"):
        st.markdown(
            """
            Supported text column names:
            - `review`
            - `text`
            - `comment`
            - `feedback`

            Supported label column names (testing mode):
            - `sentiment`
            - `label`
            - `target`
            - `class`
            - `polarity`
            - `actual_sentiment`
            """
        )

    with st.expander("Context Guidance by Section"):
        st.markdown(
            """
            - Single Review:
            Use for demonstrative case studies and step-by-step verification.

            - Batch Analysis:
            Use to demonstrate scalability and throughput on larger datasets.

            - Testing and QA:
            Use labeled datasets to report measurable quality metrics and confusion matrix.

            - Run Database:
            Use run IDs and exports as reproducibility evidence in reports.
            """
        )

    with st.expander("Troubleshooting"):
        st.markdown(
            """
            - Model files not found:
            Run `python src/save_model.py`.

            - CSV upload error:
            Verify delimiter/encoding and required column names.

            - No valid rows processed:
            Check for empty text cells and label format in testing mode.
            """
        )


def render_process_flow():
    st.subheader("Process Flow")

    st.markdown("Workflow and architecture overview.")

    st.markdown("**End-to-End Workflow**")
    workflow_dot = """
digraph Workflow {
  rankdir=TB;
  node [shape=box, style="rounded,filled", color="#1f2937", fillcolor="#f9fafb", fontname="Arial"];
  edge [color="#374151"];

  start [label="Start: streamlit run src/app.py"];
  init [label="Initialize app config, stopwords, SQLite database"];
  load [label="Load model, vectorizer, metadata"];
  mode [label="Select mode", shape=diamond, fillcolor="#eef2ff"];

  single [label="Single Review Analysis"];
  batch [label="Batch Analysis"];
  qa [label="Testing and QA"];
  help [label="Help and User Manual"];

  pre [label="Preprocess text"];
  vec [label="TF-IDF vectorization"];
  infer [label="Model inference: sentiment + confidence"];

  single_out [label="Display single result + runtime"];
  batch_out [label="Aggregate stats, charts, downloads"];
  qa_metrics [label="Compute Accuracy, Precision, Recall, F1, Confusion Matrix"];

  log_single [label="Log run (mode=single)"];
  log_batch [label="Log run (mode=batch)"];
  log_qa [label="Log run (mode=testing)"];
  db [label="SQLite evidence store: runs + predictions", shape=cylinder, fillcolor="#ecfeff"];
  annex [label="Export CSV reports"];

  start -> init -> load -> mode;
  mode -> single;
  mode -> batch;
  mode -> qa;
  mode -> help;

  single -> pre -> vec -> infer -> single_out -> log_single -> db;
  batch -> pre -> vec -> infer -> batch_out -> log_batch -> db;
  qa -> pre -> vec -> infer -> qa_metrics -> log_qa -> db;
  help -> annex;
  db -> annex;
}
"""
    st.graphviz_chart(workflow_dot, use_container_width=True)

    st.markdown("**Module Architecture**")
    architecture_dot = """
digraph Architecture {
  rankdir=LR;
  node [shape=box, style="rounded,filled", color="#1f2937", fillcolor="#f9fafb", fontname="Arial"];
  edge [color="#374151"];

  app [label="src/app.py\\nEntry point and tab routing"];
  ui [label="src/app_ui.py\\nUI rendering + dashboards"];
  logic [label="src/app_logic.py\\nPreprocessing + inference"];
  store [label="src/app_store.py\\nSQLite persistence"];
  model [label="models/model.pkl + vectorizer.pkl", shape=cylinder, fillcolor="#fef3c7"];
  db [label="results/sentiment_app.db", shape=cylinder, fillcolor="#ecfeff"];

  app -> ui;
  app -> logic;
  app -> store;
  ui -> logic;
  ui -> store;
  logic -> model;
  store -> db;
}
"""
    st.graphviz_chart(architecture_dot, use_container_width=True)

    st.markdown("**Stage Summary**")
    stage_df = pd.DataFrame(
        [
            ["1", "Initialization", "Load environment, model assets, and logging database."],
            ["2", "Input", "Collect single text, batch CSV, or labeled test CSV."],
            ["3", "Processing", "Preprocess text and transform to TF-IDF features."],
            ["4", "Inference", "Predict sentiment and confidence."],
            ["5", "Evaluation", "For labeled tests, calculate quality metrics and confusion matrix."],
            ["6", "Persistence", "Store run-level and row-level evidence in SQLite."],
            ["7", "Reporting", "Display dashboards and export CSV reports."],
        ],
        columns=["Stage", "Name", "What Happens"],
    )
    st.dataframe(stage_df, use_container_width=True, hide_index=True)


def render_history(analysis_history):
    if not analysis_history:
        return

    st.markdown("---")
    st.subheader("Session History")

    history_df = pd.DataFrame(analysis_history)
    col_h1, col_h2, col_h3 = st.columns(3)

    with col_h1:
        total = len(history_df)
        st.metric("Total Analyzed", total)

    with col_h2:
        positive_pct = (history_df["sentiment"] == "Positive").sum() / total * 100
        st.metric("Positive Rate", f"{positive_pct:.1f}%")

    with col_h3:
        avg_conf = history_df["confidence"].mean()
        st.metric("Average Confidence", f"{avg_conf:.1f}%")

    sentiment_counts = history_df["sentiment"].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Session Sentiment Distribution",
        color_discrete_map={"Positive": "#10b981", "Negative": "#ef4444"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_algorithm_comparison():
    st.markdown("---")
    st.subheader("Algorithm Performance Comparison")

    algo_data, source_note = _load_algorithm_comparison_data()
    st.caption(source_note)

    color_map = {
        "Logistic Regression": "#3b82f6",
        "Naive Bayes": "#8b5cf6",
        "SVM": "#10b981",
        "Support Vector Machine": "#10b981",
        "Random Forest": "#f59e0b",
        "Decision Tree": "#ef4444",
    }
    algo_data["Color"] = algo_data["DisplayAlgorithm"].map(color_map).fillna("#6b7280")

    fig = go.Figure(
        data=[
            go.Bar(
                x=algo_data["DisplayAlgorithm"],
                y=algo_data["Accuracy"],
                marker=dict(color=algo_data["Color"], line=dict(color="white", width=2)),
                text=algo_data["Accuracy"].apply(lambda x: f"{x}%"),
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="ML Algorithm Accuracy Comparison",
        xaxis_title="Algorithm",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[70, 95]),
        height=400,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_footer():
    st.markdown("---")
    st.markdown(
        """
<div style='text-align: center; color: #6b7280; padding: 1.5rem;'>
    <p><strong>Sentiment Analysis Application</strong></p>
    <p><strong>Comparative Analysis of Machine Learning Algorithms for Sentiment Classification</strong></p>
    <p style='margin-top: 0.6rem;'>
        This application supports automation, testing, and reproducibility.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )
