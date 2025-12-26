"""
Streamlit Application for Digit Recognition.

A modular, optimized application using Custom KMeans, PCA, and Logistic Regression
to recognize handwritten digits.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

# Import custom modules
from preprocessing import get_data, processed_data_exists
from model_manager import get_or_train_model, model_exists, save_model
from training import build_pipeline, train_model, evaluate_model, get_cluster_distribution
from config import N_CLUSTERS, PCA_VARIANCE_RATIO, MAX_ITER_LOGREG

# Page configuration
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data_cached():
    """Load and cache training data."""
    return get_data(use_cache=True)


@st.cache_resource
def get_model(force_retrain=False):
    """Load or train model with caching."""
    X_train, y_train, X_test = load_data_cached()
    pipeline, is_new = get_or_train_model(X_train, y_train, force_retrain=force_retrain)
    return pipeline, X_train, y_train, X_test, is_new


def display_digit_grid(X, y, predictions=None, n_samples=10, title="Sample Digits"):
    """Display a grid of digit images."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(indices):
            i = indices[idx]
            img = X[i].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            
            if predictions is not None:
                color = 'green' if predictions[i] == y[i] else 'red'
                ax.set_title(f"True: {y[i]}, Pred: {predictions[i]}", color=color, fontsize=10)
            else:
                ax.set_title(f"Label: {y[i]}", fontsize=10)
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def display_confusion_matrix(conf_matrix):
    """Display confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=range(10), yticklabels=range(10))
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    return fig


def display_class_performance(report):
    """Display per-class performance metrics."""
    classes = [str(i) for i in range(10)]
    precisions = [report[c]['precision'] for c in classes]
    recalls = [report[c]['recall'] for c in classes]
    f1_scores = [report[c]['f1-score'] for c in classes]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(10)
    width = 0.25
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#667eea')
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#764ba2')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#f093fb')
    
    ax.set_xlabel('Digit Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(range(10))
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    return fig


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🔢 Digit Recognition System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model status
        st.subheader("Model Status")
        if model_exists():
            st.success("✅ Trained model found")
        else:
            st.warning("⚠️ No trained model found")
        
        # Data status
        st.subheader("Data Status")
        if processed_data_exists():
            st.success("✅ Cached data available")
        else:
            st.info("📥 Data will be processed on first run")
        
        st.divider()
        
        # Hyperparameters display
        st.subheader("📊 Hyperparameters")
        st.write(f"**Clusters:** {N_CLUSTERS}")
        st.write(f"**PCA Variance:** {PCA_VARIANCE_RATIO*100:.0f}%")
        st.write(f"**Max Iterations:** {MAX_ITER_LOGREG}")
        
        st.divider()
        
        # Retrain option
        st.subheader("🔄 Actions")
        retrain = st.button("🔄 Retrain Model", width='stretch')
    
    # Main content area
    if retrain:
        st.cache_resource.clear()
    
    # Load model and data
    with st.spinner("🔄 Loading model and data..."):
        try:
            pipeline, X_train, y_train, X_test, is_new = get_model(force_retrain=retrain)
            
            if is_new:
                st.success("✅ Model trained and saved successfully!")
            else:
                st.info("📂 Loaded existing model from cache")
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    
    # Evaluate model
    with st.spinner("📊 Evaluating model..."):
        results = evaluate_model(pipeline, X_train, y_train)
    
    # Display metrics
    st.header("📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Accuracy", f"{results['accuracy']:.2%}")
    
    with col2:
        st.metric("Training Samples", f"{len(X_train):,}")
    
    with col3:
        st.metric("Test Samples", f"{len(X_test):,}" if X_test is not None else "N/A")
    
    with col4:
        st.metric("Clusters", N_CLUSTERS)
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📷 Sample Predictions", 
        "📊 Confusion Matrix", 
        "📈 Class Performance",
        "🎯 Make Predictions"
    ])
    
    with tab1:
        st.subheader("Sample Predictions from Training Data")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            n_samples = st.slider("Number of samples", 5, 20, 10)
            if st.button("🔄 Refresh Samples"):
                pass  # Forces redraw with new random samples
        
        with col2:
            fig = display_digit_grid(X_train, y_train, results['predictions'], 
                                     n_samples=n_samples, title="Random Training Samples")
            st.pyplot(fig)
            plt.close()
    
    with tab2:
        st.subheader("Confusion Matrix Analysis")
        fig = display_confusion_matrix(results['confusion_matrix'])
        st.pyplot(fig)
        plt.close()
        
        st.info("💡 The diagonal shows correct predictions. Off-diagonal elements show misclassifications.")
    
    with tab3:
        st.subheader("Per-Class Performance Metrics")
        fig = display_class_performance(results['classification_report'])
        st.pyplot(fig)
        plt.close()
        
        # Detailed table
        st.subheader("Detailed Metrics")
        report_df = pd.DataFrame(results['classification_report']).transpose()
        report_df = report_df.iloc[:10]  # Only digit classes
        st.dataframe(report_df.style.format("{:.3f}"), width='stretch')
    
    with tab4:
        st.subheader("Predict Specific Digits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_idx = st.number_input(
                "Enter sample index", 
                min_value=0, 
                max_value=len(X_train)-1, 
                value=0
            )
            
            if st.button("🎯 Predict"):
                sample = X_train[sample_idx:sample_idx+1]
                prediction = pipeline.predict(sample)[0]
                actual = y_train[sample_idx]
                
                # Display the digit
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(X_train[sample_idx].reshape(28, 28), cmap='gray')
                ax.axis('off')
                
                is_correct = prediction == actual
                ax.set_title(
                    f"Predicted: {prediction} | Actual: {actual}", 
                    fontsize=14,
                    color='green' if is_correct else 'red'
                )
                st.pyplot(fig)
                plt.close()
                
                if is_correct:
                    st.success(f"✅ Correct! The digit is {actual}")
                else:
                    st.error(f"❌ Incorrect. Predicted {prediction}, but actual is {actual}")
        
        with col2:
            st.info("""
            **How to use:**
            1. Enter a sample index from the training data
            2. Click 'Predict' to see the model's prediction
            3. Green title = correct, Red title = incorrect
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 1rem;'>
            Built with ❤️ using Streamlit | 
            Custom KMeans + PCA + Logistic Regression Pipeline
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
