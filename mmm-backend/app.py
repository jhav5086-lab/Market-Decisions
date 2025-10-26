import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Model libraries
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import scipy.optimize as opt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set page config first
st.set_page_config(page_title="Media Mix Model Analyzer", layout="wide", page_icon="📊")

class MediaMixModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def hill_saturation(self, spend, alpha, gamma):
        return (spend**alpha) / (spend**alpha + gamma**alpha)
    
    def weibull_saturation(self, spend, alpha, gamma):
        return 1 - np.exp(-(spend/gamma)**alpha)
    
    def geometric_saturation(self, spend, alpha):
        return spend**alpha
    
    def fit_ridge(self, X, y, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.model.fit(X, y)
        return self.model
    
    def calculate_vif(self, X):
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        return vif_data

def enhanced_data_upload():
    st.sidebar.header("📁 Data Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Marketing Data", 
        type=['csv', 'xlsx'],
        help="Upload your CSV or Excel file with marketing data"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            if data.empty:
                st.sidebar.error("Uploaded file is empty!")
                return None
                
            if 'date' not in data.columns:
                st.sidebar.warning("No 'date' column found. Please ensure your data has a date column.")
            else:
                st.sidebar.success(f"✅ Data loaded: {len(data)} rows, {len(data.columns)} columns")
                
                with st.sidebar.expander("Data Preview"):
                    st.dataframe(data.head(3), use_container_width=True)
                    st.write(f"Date range: {data['date'].iloc[0]} to {data['date'].iloc[-1]}")
            
            return data
            
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            return None
    
    return None

def generate_sample_data():
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    
    sample_data = {
        'date': dates,
        'paid_search_net_spend': np.random.uniform(1000, 5000, len(dates)),
        'paid_search_revenue': np.random.uniform(5000, 20000, len(dates)),
        'paid_social_net_spend': np.random.uniform(500, 3000, len(dates)),
        'paid_social_revenue': np.random.uniform(2000, 10000, len(dates)),
        'paid_display_net_spend': np.random.uniform(300, 2000, len(dates)),
        'paid_display_revenue': np.random.uniform(1000, 6000, len(dates)),
        'total_revenue': np.random.uniform(15000, 50000, len(dates)),
        'Promo': np.random.choice(['Avg day', 'Promo day', 'Holiday'], len(dates)),
        'holiday': np.random.choice(['NH', 'New Year Day', 'Valentine Day'], len(dates))
    }
    
    return pd.DataFrame(sample_data)

def preprocess_data(data, date_col, dependent_var, independent_vars):
    df = data.copy()
    
    # Convert date column
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Create time-based features
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
    
    # Handle missing values
    numeric_cols = [dependent_var] + independent_vars
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def train_model(processed_data, dependent_var, independent_vars, model_type, test_size, alpha):
    X = processed_data[independent_vars]
    y = processed_data[dependent_var]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Initialize model
    mmm = MediaMixModel()
    
    # Apply transformations based on model type
    if model_type == "Ridge Regression":
        X_train_scaled = mmm.scaler.fit_transform(X_train)
        X_test_scaled = mmm.scaler.transform(X_test)
        
        model = mmm.fit_ridge(X_train_scaled, y_train, alpha)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
    elif model_type == "Weibull":
        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy()
        
        for col in independent_vars:
            X_train_transformed[col] = 1 - np.exp(-X_train[col]/X_train[col].mean())
            X_test_transformed[col] = 1 - np.exp(-X_test[col]/X_test[col].mean())
        
        X_train_scaled = mmm.scaler.fit_transform(X_train_transformed)
        X_test_scaled = mmm.scaler.transform(X_test_transformed)
        
        model = mmm.fit_ridge(X_train_scaled, y_train, alpha)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    
    elif model_type == "Hill Saturation":
        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy()
        
        for col in independent_vars:
            X_train_transformed[col] = X_train[col] / (X_train[col] + X_train[col].mean())
            X_test_transformed[col] = X_test[col] / (X_test[col] + X_test[col].mean())
        
        X_train_scaled = mmm.scaler.fit_transform(X_train_transformed)
        X_test_scaled = mmm.scaler.transform(X_test_transformed)
        
        model = mmm.fit_ridge(X_train_scaled, y_train, alpha)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    
    else:  # Geometric
        X_train_transformed = np.log(X_train + 1)
        X_test_transformed = np.log(X_test + 1)
        
        X_train_scaled = mmm.scaler.fit_transform(X_train_transformed)
        X_test_scaled = mmm.scaler.transform(X_test_transformed)
        
        model = mmm.fit_ridge(X_train_scaled, y_train, alpha)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_mape = mean_absolute_percentage_error(y_train, y_pred_train) * 100
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    r2_score = model.score(X_test_scaled, y_test)
    
    # Calculate channel contributions
    channel_contributions = {}
    channel_roi = {}
    
    for i, col in enumerate(independent_vars):
        contribution = model.coef_[i] * processed_data[col].mean()
        channel_contributions[col] = max(contribution, 0)  # Avoid negative contributions
        
        if processed_data[col].mean() > 0:
            roi = contribution / processed_data[col].mean()
            channel_roi[col] = max(roi, 0)
        else:
            channel_roi[col] = 0
    
    # VIF analysis
    vif_data = mmm.calculate_vif(X_train)
    
    return {
        'model': model,
        'scaler': mmm.scaler,
        'train_mape': train_mape,
        'test_mape': test_mape,
        'r2_score': r2_score,
        'channel_contributions': channel_contributions,
        'channel_roi': channel_roi,
        'vif': vif_data,
        'model_type': model_type,
        'feature_names': independent_vars
    }

def main():
    st.title("📊 Media Mix Model Analyzer")
    st.markdown("---")
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Data upload
        data = enhanced_data_upload()
        
        # Sample data option
        st.markdown("---")
        if st.button("🎲 Load Sample Data for Testing"):
            sample_data = generate_sample_data()
            st.session_state.raw_data = sample_data
            st.success("Sample data loaded!")
        
        if data is not None:
            st.session_state.raw_data = data
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 Data Setup", "⚙️ Model Config", "📈 Analysis", 
        "💰 Budget Sim", "🔮 Forecast", "📋 Report"
    ])
    
    with tab1:
        st.header("Data Setup & Preprocessing")
        
        if 'raw_data' in st.session_state:
            data = st.session_state.raw_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Preview")
                st.dataframe(data.head(), use_container_width=True)
                
                st.subheader("Data Summary")
                st.json({
                    "Total Records": len(data),
                    "Date Range": f"{data['date'].iloc[0]} to {data['date'].iloc[-1]}" if 'date' in data.columns else "N/A",
                    "Columns": len(data.columns)
                })
            
            with col2:
                st.subheader("Variable Selection")
                
                # KPI Selection
                kpi_options = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
                if kpi_options:
                    dependent_var = st.selectbox("Select Target KPI", options=kpi_options)
                    
                    # Independent variables
                    independent_options = [col for col in kpi_options if col != dependent_var]
                    independent_vars = st.multiselect("Select Independent Variables", options=independent_options)
                    
                    if st.button("Process Data", type="primary") and independent_vars:
                        processed_data = preprocess_data(data, 'date', dependent_var, independent_vars)
                        st.session_state.processed_data = processed_data
                        st.session_state.dependent_var = dependent_var
                        st.session_state.independent_vars = independent_vars
                        st.success("Data processed successfully!")
                else:
                    st.warning("No numeric columns found in the data")
        else:
            st.info("Please upload data or load sample data to begin")
    
    with tab2:
        st.header("Model Configuration")
        
        if st.session_state.processed_data is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Model Settings")
                model_type = st.selectbox("Model Type", ["Ridge Regression", "Geometric", "Weibull", "Hill Saturation"])
                test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
                
                if model_type == "Ridge Regression":
                    alpha = st.slider("Regularization (Alpha)", 0.1, 10.0, 1.0)
                else:
                    alpha = 1.0
            
            with col2:
                st.subheader("Training")
                if st.button("Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        results = train_model(
                            st.session_state.processed_data, 
                            st.session_state.dependent_var,
                            st.session_state.independent_vars,
                            model_type, test_size, alpha
                        )
                        st.session_state.results = results
                        
                        st.metric("Train MAPE", f"{results['train_mape']:.2f}%")
                        st.metric("Test MAPE", f"{results['test_mape']:.2f}%")
                        st.metric("R² Score", f"{results['r2_score']:.3f}")
        else:
            st.warning("Please process data first in Data Setup tab")
    
    with tab3:
        st.header("Performance Analysis")
        
        if st.session_state.results:
            results = st.session_state.results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Channel Contributions")
                
                # Volume decomposition
                fig, ax = plt.subplots(figsize=(10, 6))
                channels = list(results['channel_contributions'].keys())
                contributions = list(results['channel_contributions'].values())
                
                colors = ['green' if x > 0 else 'red' for x in contributions]
                bars = ax.bar(channels, contributions, color=colors, alpha=0.7)
                ax.set_ylabel("Revenue Contribution ($)")
                ax.set_title("Channel Revenue Contributions")
                plt.xticks(rotation=45)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Base vs Incremental
                st.subheader("Base vs Incremental Volume")
                total_volume = sum(results['channel_contributions'].values())
                base_volume = total_volume * 0.6
                incremental_volume = total_volume * 0.4
                
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.pie([base_volume, incremental_volume], labels=['Base Volume', 'Incremental Volume'], 
                       autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
                st.pyplot(fig2)
            
            with col2:
                st.subheader("ROI Analysis")
                
                # ROI bubble chart
                fig3, ax3 = plt.subplots(figsize=(10, 8))
                channels = list(results['channel_roi'].keys())
                roi_values = list(results['channel_roi'].values())
                contributions = list(results['channel_contributions'].values())
                
                bubble_sizes = [abs(c) / max(abs(np.array(contributions))) * 1000 for c in contributions]
                
                scatter = ax3.scatter(range(len(channels)), roi_values, s=bubble_sizes, alpha=0.6, 
                                    c=roi_values, cmap='RdYlGn')
                ax3.set_xticks(range(len(channels)))
                ax3.set_xticklabels(channels, rotation=45)
                ax3.set_ylabel("ROI")
                ax3.set_title("Channel ROI vs Contribution")
                plt.colorbar(scatter, ax=ax3, label='ROI')
                st.pyplot(fig3)
                
                # VIF Analysis
                st.subheader("VIF Analysis")
                st.dataframe(results['vif'], use_container_width=True)
        else:
            st.warning("Please train a model first")
    
    with tab4:
        st.header("Budget Simulator")
        
        if st.session_state.results:
            results = st.session_state.results
            
            st.subheader("Budget Allocation")
            budget_allocation = {}
            channels = st.session_state.independent_vars
            
            cols = st.columns(3)
            for i, channel in enumerate(channels):
                with cols[i % 3]:
                    budget_allocation[channel] = st.number_input(f"{channel} Budget", value=1000, step=100, key=f"budget_{channel}")
            
            total_budget = sum(budget_allocation.values())
            st.metric("Total Budget", f"${total_budget:,.2f}")
            
            if st.button("Simulate Budget Impact"):
                # Simple simulation
                total_revenue = 0
                for channel in channels:
                    if channel in budget_allocation:
                        spend = budget_allocation[channel]
                        # Simple contribution calculation
                        idx = results['feature_names'].index(channel)
                        contribution = results['model'].coef_[idx] * spend
                        total_revenue += contribution
                
                total_revenue += results['model'].intercept_
                roi = (total_revenue - total_budget) / total_budget
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Revenue", f"${total_revenue:,.2f}")
                col2.metric("Total Budget", f"${total_budget:,.2f}")
                col3.metric("ROI", f"{roi:.1%}")
        else:
            st.warning("Please train a model first")
    
    with tab5:
        st.header("Demand Forecasting")
        
        if st.session_state.results:
            results = st.session_state.results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Forecast Settings")
                forecast_period = st.selectbox("Forecast Period", ["Next 3 Months", "Next 6 Months", "Next 12 Months"])
                budget_growth = st.slider("Budget Growth Rate (%)", -20, 50, 5) / 100
            
            with col2:
                if st.button("Generate Forecast"):
                    # Create future dates
                    last_date = st.session_state.processed_data['date'].max()
                    periods = 12 if "12" in forecast_period else (6 if "6" in forecast_period else 3)
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='M')
                    
                    # Simple forecast
                    base_revenue = st.session_state.processed_data[st.session_state.dependent_var].mean()
                    forecast_data = []
                    
                    for i, date in enumerate(future_dates):
                        growth_factor = (1 + budget_growth) ** (i + 1)
                        forecast_revenue = base_revenue * growth_factor
                        forecast_data.append({'Date': date, 'Forecasted_Revenue': forecast_revenue})
                    
                    forecast_df = pd.DataFrame(forecast_data)
                    
                    st.subheader("Forecast Results")
                    st.dataframe(forecast_df.style.format({"Forecasted_Revenue": "${:,.2f}"}), use_container_width=True)
                    
                    # Plot forecast
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(forecast_df['Date'], forecast_df['Forecasted_Revenue'], marker='o', linewidth=2)
                    ax.set_title(f"Revenue Forecast - {forecast_period}")
                    ax.set_ylabel("Revenue ($)")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
        else:
            st.warning("Please train a model first")
    
    with tab6:
        st.header("Summary Report")
        
        if st.session_state.results:
            results = st.session_state.results
            
            st.subheader("Executive Summary")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Model Accuracy", f"{results['test_mape']:.2f}% MAPE")
            col2.metric("R² Score", f"{results['r2_score']:.3f}")
            col3.metric("Channels Analyzed", len(results['feature_names']))
            
            total_revenue = sum(results['channel_contributions'].values())
            col4.metric("Total Revenue Modeled", f"${total_revenue:,.0f}")
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Channel Performance")
                performance_df = pd.DataFrame({
                    'Channel': list(results['channel_contributions'].keys()),
                    'Contribution': list(results['channel_contributions'].values()),
                    'ROI': list(results['channel_roi'].values())
                })
                st.dataframe(performance_df.style.format({
                    'Contribution': '${:,.2f}',
                    'ROI': '{:.2%}'
                }), use_container_width=True)
            
            with col2:
                st.subheader("Strategic Recommendations")
                
                # Generate recommendations based on ROI
                high_roi = [chan for chan, roi in results['channel_roi'].items() if roi > 0.2]
                low_roi = [chan for chan, roi in results['channel_roi'].items() if roi < 0.1]
                
                if high_roi:
                    st.success(f"**Increase Investment**: {', '.join(high_roi)}")
                
                if low_roi:
                    st.warning(f"**Optimize Spending**: {', '.join(low_roi)}")
                
                st.info("**Next Steps**: Use the Budget Simulator to test different allocation scenarios")
        else:
            st.warning("Please train a model first to see the summary")

if __name__ == "__main__":
    main()
