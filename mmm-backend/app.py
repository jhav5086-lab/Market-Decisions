import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
import io

# App title
st.set_page_config(page_title="Market Mix Modeling", page_icon="📊", layout="wide")
st.title("📊 Market Mix Modeling (MMM) Analysis")
st.markdown("Analyze the impact of your marketing channels on sales")

# Sidebar for user inputs
st.sidebar.header("Configuration")

# Sample data or file upload
st.sidebar.subheader("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])

# If no file uploaded, use sample data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
else:
    # Create sample data
    st.sidebar.info("Using sample data for demonstration")
    np.random.seed(42)
    n_samples = 100
    
    sample_data = {
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='W'),
        'sales': np.random.normal(1000, 200, n_samples).cumsum() + 10000,
        'tv_spend': np.random.uniform(5000, 20000, n_samples),
        'digital_spend': np.random.uniform(3000, 15000, n_samples),
        'print_spend': np.random.uniform(1000, 8000, n_samples),
        'radio_spend': np.random.uniform(500, 5000, n_samples),
        'competitor_spend': np.random.uniform(10000, 50000, n_samples)
    }
    df = pd.DataFrame(sample_data)

# Display data
st.subheader("Data Overview")
st.dataframe(df.head(), use_container_width=True)

# Data summary
st.subheader("Data Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Records", len(df))
with col2:
    st.metric("Total Sales", f"${df['sales'].sum():,.0f}")
with col3:
    st.metric("Average Sales", f"${df['sales'].mean():,.0f}")

# Model configuration
st.sidebar.subheader("Model Settings")
alpha = st.sidebar.slider("Ridge Alpha", 0.1, 10.0, 1.0, 0.1)
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20, 5)

# Prepare features and target
st.subheader("Model Training")

# Select features (exclude date and target)
feature_columns = [col for col in df.columns if col not in ['date', 'sales']]
selected_features = st.multiselect(
    "Select Marketing Channels for Model",
    feature_columns,
    default=feature_columns
)

if len(selected_features) >= 1:
    X = df[selected_features]
    y = df['sales']
    
    # Split data
    split_index = int(len(X) * (1 - test_size/100))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Train model
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        st.metric("Mean Absolute Percentage Error", f"{mape:.2%}")
        st.metric("Root Mean Square Error", f"${rmse:,.0f}")
    
    with col2:
        st.subheader("Channel Coefficients")
        coefficients = pd.DataFrame({
            'Channel': selected_features,
            'Coefficient': model.coef_
        })
        st.dataframe(coefficients, use_container_width=True)
    
    # Visualization
    st.subheader("Actual vs Predicted Sales")
    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Date': df['date'].iloc[split_index:].values
    })
    
    st.line_chart(results_df.set_index('Date')[['Actual', 'Predicted']])
    
else:
    st.warning("Please select at least one marketing channel for analysis")

# Feature importance
if len(selected_features) >= 1:
    st.subheader("Channel Contribution Analysis")
    
    # Calculate contribution percentages
    total_impact = np.sum(np.abs(model.coef_))
    contributions = (np.abs(model.coef_) / total_impact) * 100
    
    contribution_df = pd.DataFrame({
        'Channel': selected_features,
        'Contribution %': contributions
    }).sort_values('Contribution %', ascending=False)
    
    st.bar_chart(contribution_df.set_index('Channel')['Contribution %'])

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Market Mix Modeling Demo")
