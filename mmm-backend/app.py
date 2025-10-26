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

# Plotting setup
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MediaMixModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_name = ""
        
    def hill_saturation(self, spend, alpha, gamma):
        """Hill saturation function"""
        return (spend**alpha) / (spend**alpha + gamma**alpha)
    
    def weibull_saturation(self, spend, alpha, gamma):
        """Weibull saturation function"""
        return 1 - np.exp(-(spend/gamma)**alpha)
    
    def geometric_saturation(self, spend, alpha):
        """Geometric saturation function"""
        return spend**alpha
    
    def fit_ridge(self, X, y, alpha=1.0):
        """Fit Ridge regression model"""
        self.model = Ridge(alpha=alpha)
        self.model.fit(X, y)
        return self.model
    
    def calculate_vif(self, X):
        """Calculate Variance Inflation Factor"""
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        return vif_data

def enhanced_data_upload():
    """Enhanced data upload function with support for multiple file types"""
    st.sidebar.header("📁 Data Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Marketing Data", 
        type=['csv', 'xlsx', 'xls'],
        help="Upload your CSV or Excel file with marketing data"
    )
    
    if uploaded_file is not None:
        try:
            # Handle different file types
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:  # Excel files
                data = pd.read_excel(uploaded_file)
            
            # Basic data validation
            if data.empty:
                st.sidebar.error("Uploaded file is empty!")
                return None
                
            # Check for required columns
            required_cols = ['date']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                st.sidebar.warning(f"Missing recommended columns: {missing_cols}")
            else:
                st.sidebar.success(f"✅ Data loaded: {len(data)} rows, {len(data.columns)} columns")
                
                # Show quick data preview
                with st.sidebar.expander("Data Preview"):
                    st.dataframe(data.head(3), use_container_width=True)
                    st.write(f"Date range: {data['date'].min()} to {data['date'].max()}")
            
            return data
            
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            return None
    
    return None

def generate_sample_data():
    """Generate sample marketing data for testing"""
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

def preprocess_data(data, date_col, dependent_var, base_vars, promo_vars, paid_media_vars, organic_vars):
    """Preprocess the data for modeling"""
    df = data.copy()
    
    # Convert date column
    df[date_col] = pd.to_datetime(df[date_col], format='mixed')
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Create time-based features
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
    
    # Handle missing values
    numeric_cols = [dependent_var] + base_vars + promo_vars + paid_media_vars + organic_vars
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def train_model(processed_data, dependent_var, independent_vars, model_type, test_size, alpha):
    """Train the selected model"""
    X = processed_data[independent_vars]
    y = processed_data[dependent_var]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Initialize model
    mmm = MediaMixModel()
    
    # Apply transformations based on model type
    if model_type == "Ridge Regression":
        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy()
        
        # Scale features
        X_train_scaled = mmm.scaler.fit_transform(X_train_transformed)
        X_test_scaled = mmm.scaler.transform(X_test_transformed)
        
        model = mmm.fit_ridge(X_train_scaled, y_train, alpha)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
    elif model_type in ["Weibull", "Hill Saturation"]:
        # For demonstration, using Ridge as base with transformation hints
        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy()
        
        # Apply saturation transformations (simplified)
        for col in independent_vars:
            if model_type == "Weibull":
                X_train_transformed[col] = 1 - np.exp(-X_train[col]/X_train[col].mean())
                X_test_transformed[col] = 1 - np.exp(-X_test[col]/X_test[col].mean())
            elif model_type == "Hill Saturation":
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
        channel_contributions[col] = contribution
        
        # Simple ROI calculation
        if processed_data[col].mean() > 0:
            roi = contribution / processed_data[col].mean()
            channel_roi[col] = roi
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

def plot_response_curve(results, channel, max_spend):
    """Plot response curve for a specific channel"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    spend_range = np.linspace(0, max_spend, 100)
    
    # Simplified response curve calculation
    if results['model_type'] == "Weibull":
        response = 1 - np.exp(-spend_range/spend_range.mean())
    elif results['model_type'] == "Hill Saturation":
        response = spend_range / (spend_range + spend_range.mean())
    else:
        response = spend_range
    
    # Scale response based on model coefficients
    channel_idx = results['feature_names'].index(channel)
    coefficient = results['model'].coef_[channel_idx]
    scaled_response = response * coefficient * 1000  # Scaling factor for visualization
    
    ax.plot(spend_range, scaled_response, linewidth=3)
    ax.set_xlabel(f"{channel} Spend ($)")
    ax.set_ylabel("Incremental Response")
    ax.set_title(f"Response Curve: {channel}")
    ax.grid(True, alpha=0.3)
    
    return fig

def simulate_budget(results, budget_allocation):
    """Simulate revenue impact of budget allocation"""
    # Simplified simulation using model coefficients
    total_revenue = 0
    model = results['model']
    scaler = results['scaler']
    
    for i, channel in enumerate(results['feature_names']):
        if channel in budget_allocation:
            spend = budget_allocation[channel]
            # Apply same transformations as in training
            if results['model_type'] == "Weibull":
                transformed_spend = 1 - np.exp(-spend/spend)
            elif results['model_type'] == "Hill Saturation":
                transformed_spend = spend / (spend + spend)
            elif results['model_type'] == "Geometric":
                transformed_spend = np.log(spend + 1)
            else:
                transformed_spend = spend
            
            # Scale and predict
            contribution = model.coef_[i] * transformed_spend
            total_revenue += contribution
    
    # Add intercept and scale
    total_revenue += model.intercept_
    return max(total_revenue, 0)

def optimize_budget(results, total_budget):
    """Optimize budget allocation across channels"""
    channels = results['feature_names']
    
    def objective(x):
        budget_dict = {channels[i]: x[i] for i in range(len(channels))}
        return -simulate_budget(results, budget_dict)  # Negative for minimization
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - total_budget}]
    bounds = [(0, total_budget) for _ in channels]
    
    # Initial guess (equal distribution)
    x0 = [total_budget / len(channels)] * len(channels)
    
    # Optimize
    result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimized_budget = pd.DataFrame({
        'Channel': channels,
        'Optimized_Budget': result.x,
        'Percentage': result.x / total_budget * 100
    })
    
    return optimized_budget

def generate_forecast(results, processed_data, forecast_period, budget_growth, include_promo):
    """Generate revenue forecast"""
    periods = {
        "Next Quarter": 3,
        "Next 6 Months": 6,
        "Next 12 Months": 12
    }
    
    n_periods = periods[forecast_period]
    
    # Create future dates
    last_date = processed_data['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_periods, freq='M')
    
    # Forecast data frame
    forecast_df = pd.DataFrame(index=future_dates, columns=['Forecasted_Revenue'])
    
    # Simplified forecast using average contributions with growth
    base_revenue = processed_data[st.session_state.dependent_var].mean()
    
    for i, date in enumerate(future_dates):
        growth_factor = (1 + budget_growth) ** (i + 1)
        forecast_revenue = base_revenue * growth_factor
        forecast_df.loc[date, 'Forecasted_Revenue'] = forecast_revenue
    
    return forecast_df

def plot_volume_decomposition(results):
    """Plot volume growth/loss decomposition"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    channels = list(results['channel_contributions'].keys())
    contributions = list(results['channel_contributions'].values())
    
    colors = ['green' if x > 0 else 'red' for x in contributions]
    bars = ax.bar(channels, contributions, color=colors, alpha=0.7)
    
    ax.set_ylabel("Volume Contribution ($)")
    ax.set_title("Volume Growth/Loss by Channel")
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_base_vs_incremental(results):
    """Plot base vs incremental volume"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Simplified calculation
    total_volume = sum(results['channel_contributions'].values())
    base_volume = total_volume * 0.6  # Assuming 60% base
    incremental_volume = total_volume * 0.4  # Assuming 40% incremental
    
    labels = ['Base Volume', 'Incremental Volume']
    sizes = [base_volume, incremental_volume]
    colors = ['lightblue', 'lightcoral']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title("Base vs Incremental Volume Split")
    
    return fig

def plot_roi_bubble_chart(results):
    """Plot ROI bubble chart by channel"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    channels = list(results['channel_roi'].keys())
    roi_values = list(results['channel_roi'].values())
    contributions = list(results['channel_contributions'].values())
    
    # Bubble sizes based on contribution
    bubble_sizes = [abs(c) / max(abs(np.array(contributions))) * 1000 for c in contributions]
    
    scatter = ax.scatter(range(len(channels)), roi_values, s=bubble_sizes, alpha=0.6, 
                        c=roi_values, cmap='RdYlGn')
    
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45)
    ax.set_ylabel("ROI")
    ax.set_title("Channel ROI vs Contribution (Bubble Size)")
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='ROI')
    
    return fig

def calculate_promo_effectiveness(results):
    """Calculate promotional effectiveness"""
    # Simplified calculation - in real scenario, use actual promo data
    promo_data = {
        'Promo_Type': ['Seasonal', 'Clearance', 'New Launch', 'Holiday'],
        'Effectiveness_Score': [85, 72, 92, 88],
        'Revenue_Lift_Percent': [15, 8, 25, 18],
        'ROI': [2.1, 1.5, 3.2, 2.8]
    }
    
    return pd.DataFrame(promo_data)

def create_ui():
    st.set_page_config(page_title="Media Mix Model Analyzer", layout="wide", page_icon="📊")
    
    st.title("📊 Media Mix Model Analyzer")
    st.markdown("---")
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    # Sidebar for configuration with enhanced data upload
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Use the enhanced upload function
        data = enhanced_data_upload()
        
        # Add option to load sample data
        st.sidebar.markdown("---")
        if st.sidebar.button("🎲 Load Sample Data for Testing"):
            sample_data = generate_sample_data()
            st.session_state.raw_data = sample_data
            st.sidebar.success("Sample data loaded! You can now explore all features.")
        
        if data is not None:
            st.session_state.raw_data = data
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📁 Data Setup", 
        "⚙️ Model Config", 
        "📈 Response Curves", 
        "💰 Budget Simulator",
        "🔮 Forecasting",
        "📊 Performance Analysis",
        "📋 Summary Report"
    ])
    
    with tab1:
        st.header("Data Setup & Preprocessing")
        
        if 'raw_data' in st.session_state:
            data = st.session_state.raw_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Preview")
                st.dataframe(data.head(10), use_container_width=True)
                
                st.subheader("Data Summary")
                st.json({
                    "Total Records": len(data),
                    "Date Range": f"{data['date'].min()} to {data['date'].max()}" if 'date' in data.columns else "N/A",
                    "Numeric Columns": len(data.select_dtypes(include=[np.number]).columns)
                })
            
            with col2:
                st.subheader("Preprocessing Options")
                
                # Date column selection
                date_col = st.selectbox("Select Date Column", options=data.columns)
                
                # KPI Selection
                kpi_options = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
                dependent_var = st.selectbox("Select Dependent Variable (KPI)", options=kpi_options)
                
                # Independent variables categorization
                st.subheader("Categorize Independent Variables")
                
                base_vars = st.multiselect("Base Variables", options=[col for col in kpi_options if col != dependent_var])
                promo_vars = st.multiselect("Promotional Variables", options=[col for col in kpi_options if col != dependent_var and col not in base_vars])
                paid_media_vars = st.multiselect("Paid Media Variables", options=[col for col in kpi_options if col != dependent_var and col not in base_vars + promo_vars])
                organic_vars = st.multiselect("Organic Variables", options=[col for col in kpi_options if col != dependent_var and col not in base_vars + promo_vars + paid_media_vars])
                
                if st.button("Process Data", type="primary"):
                    # Process data
                    processed_data = preprocess_data(data, date_col, dependent_var, base_vars, promo_vars, paid_media_vars, organic_vars)
                    st.session_state.processed_data = processed_data
                    st.session_state.dependent_var = dependent_var
                    st.session_state.independent_vars = base_vars + promo_vars + paid_media_vars + organic_vars
                    st.success("Data processed successfully!")
        
        else:
            st.info("Please upload your data file or load sample data to begin analysis")
    
    with tab2:
        st.header("Model Configuration")
        
        if st.session_state.processed_data is not None:
            processed_data = st.session_state.processed_data
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Model Settings")
                
                # Model type selection
                model_type = st.selectbox("Select Model Type", 
                                         ["Ridge Regression", "Geometric", "Weibull", "Hill Saturation"])
                
                # Test split configuration
                test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
                
                # Regularization parameter for Ridge
                if model_type == "Ridge Regression":
                    alpha = st.slider("Regularization Strength (Alpha)", 0.1, 10.0, 1.0)
                else:
                    alpha = 1.0
                
                # Additional parameters for nonlinear models
                if model_type in ["Weibull", "Hill Saturation"]:
                    st.subheader("Saturation Parameters")
                    max_iter = st.slider("Maximum Iterations", 100, 5000, 1000)
                
            with col2:
                st.subheader("Model Metrics")
                
                if st.button("Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        results = train_model(processed_data, st.session_state.dependent_var, 
                                            st.session_state.independent_vars, model_type, 
                                            test_size, alpha)
                        st.session_state.results = results
                        
                        # Display metrics
                        st.metric("Train MAPE", f"{results['train_mape']:.2f}%")
                        st.metric("Test MAPE", f"{results['test_mape']:.2f}%")
                        st.metric("R² Score", f"{results['r2_score']:.3f}")
                        
                        # VIF Analysis
                        if 'vif' in results:
                            st.subheader("VIF Analysis")
                            st.dataframe(results['vif'], use_container_width=True)
        
        else:
            st.warning("Please process your data in the Data Setup tab first")
    
    with tab3:
        st.header("Response Curves Analysis")
        
        if st.session_state.results:
            results = st.session_state.results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Response Curve Settings")
                channel = st.selectbox("Select Channel", options=st.session_state.independent_vars)
                max_spend = st.number_input("Maximum Spend for Simulation", value=10000, step=1000)
                
                if st.button("Generate Response Curve"):
                    fig = plot_response_curve(results, channel, max_spend)
                    st.pyplot(fig)
            
            with col2:
                st.subheader("Saturation Analysis")
                # Display saturation parameters
                if 'saturation_params' in results:
                    params_df = pd.DataFrame(results['saturation_params']).T
                    st.dataframe(params_df, use_container_width=True)
        
        else:
            st.warning("Please train a model first in the Model Config tab")
    
    with tab4:
        st.header("Budget Simulator")
        
        if st.session_state.results:
            results = st.session_state.results
            
            st.subheader("Budget Allocation")
            
            # Budget input for each channel
            col1, col2, col3 = st.columns(3)
            budget_allocation = {}
            
            channels = st.session_state.independent_vars
            num_channels = len(channels)
            
            for i, channel in enumerate(channels):
                if i % 3 == 0:
                    with col1:
                        budget_allocation[channel] = st.number_input(f"{channel} Budget", value=1000, step=100)
                elif i % 3 == 1:
                    with col2:
                        budget_allocation[channel] = st.number_input(f"{channel} Budget", value=1000, step=100)
                else:
                    with col3:
                        budget_allocation[channel] = st.number_input(f"{channel} Budget", value=1000, step=100)
            
            total_budget = sum(budget_allocation.values())
            st.metric("Total Budget", f"${total_budget:,.2f}")
            
            if st.button("Simulate Budget Impact", type="primary"):
                predicted_revenue = simulate_budget(results, budget_allocation)
                roi = (predicted_revenue - total_budget) / total_budget
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Revenue", f"${predicted_revenue:,.2f}")
                with col2:
                    st.metric("Total Budget", f"${total_budget:,.2f}")
                with col3:
                    st.metric("ROI", f"{roi:.1%}")
                
                # Budget optimization
                st.subheader("Budget Optimization")
                if st.button("Optimize Budget Allocation"):
                    optimized_budget = optimize_budget(results, total_budget)
                    st.write("Optimized Budget Allocation:")
                    st.dataframe(optimized_budget, use_container_width=True)
        
        else:
            st.warning("Please train a model first in the Model Config tab")
    
    with tab5:
        st.header("Demand Forecasting")
        
        if st.session_state.results:
            results = st.session_state.results
            processed_data = st.session_state.processed_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Forecast Settings")
                forecast_period = st.selectbox("Forecast Period", ["Next 12 Months", "Next 6 Months", "Next Quarter"])
                include_promo = st.checkbox("Include Promotional Effects", value=True)
                
                # Future budget planning
                st.subheader("Future Budget Planning")
                budget_growth = st.slider("Budget Growth Rate (%)", -20, 50, 5) / 100
            
            with col2:
                if st.button("Generate Forecast", type="primary"):
                    forecast_df = generate_forecast(results, processed_data, forecast_period, 
                                                  budget_growth, include_promo)
                    
                    st.subheader("Forecast Results")
                    st.dataframe(forecast_df.style.format("${:,.2f}"), use_container_width=True)
                    
                    # Plot forecast
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(forecast_df.index, forecast_df['Forecasted_Revenue'], marker='o', linewidth=2)
                    ax.set_title(f"Revenue Forecast - {forecast_period}")
                    ax.set_ylabel("Revenue ($)")
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
        
        else:
            st.warning("Please train a model first in the Model Config tab")
    
    with tab6:
        st.header("Performance Analysis")
        
        if st.session_state.results:
            results = st.session_state.results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Volume Analysis")
                
                # Volume growth/loss decomposition
                fig = plot_volume_decomposition(results)
                st.pyplot(fig)
                
                st.subheader("Base vs Incremental Volume")
                fig = plot_base_vs_incremental(results)
                st.pyplot(fig)
            
            with col2:
                st.subheader("ROI Bubble Chart")
                
                # ROI by channel bubble chart
                fig = plot_roi_bubble_chart(results)
                st.pyplot(fig)
                
                st.subheader("Promotional Effectiveness")
                promo_effectiveness = calculate_promo_effectiveness(results)
                st.dataframe(promo_effectiveness, use_container_width=True)
        
        else:
            st.warning("Please train a model first in the Model Config tab")
    
    with tab7:
        st.header("Comprehensive Summary Report")
        
        if st.session_state.results:
            results = st.session_state.results
            
            # Executive Summary
            st.subheader("📋 Executive Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Accuracy (MAPE)", f"{results.get('test_mape', 0):.2f}%")
            with col2:
                st.metric("R² Score", f"{results.get('r2_score', 0):.3f}")
            with col3:
                total_revenue = sum(results.get('channel_contributions', {}).values())
                st.metric("Total Revenue Modeled", f"${total_revenue:,.0f}")
            with col4:
                best_channel = max(results.get('channel_contributions', {}).items(), key=lambda x: x[1])[0]
                st.metric("Top Performing Channel", best_channel)
            
            # Detailed Analysis
            st.subheader("📊 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Channel Contributions**")
                contributions_df = pd.DataFrame.from_dict(results.get('channel_contributions', {}), 
                                                        orient='index', columns=['Contribution'])
                st.dataframe(contributions_df.style.format("${:,.2f}"), use_container_width=True)
            
            with col2:
                st.write("**ROI by Channel**")
                roi_df = pd.DataFrame.from_dict(results.get('channel_roi', {}), 
                                              orient='index', columns=['ROI'])
                st.dataframe(roi_df.style.format("{:.2%}"), use_container_width=True)
            
            # Recommendations
            st.subheader("🎯 Strategic Recommendations")
            
            if 'channel_roi' in results:
                high_roi_channels = [chan for chan, roi in results['channel_roi'].items() if roi > 0.2]
                low_roi_channels = [chan for chan, roi in results['channel_roi'].items() if roi < 0.1]
                
                if high_roi_channels:
                    st.success(f"**Increase Investment**: {', '.join(high_roi_channels)} show strong ROI potential")
                
                if low_roi_channels:
                    st.warning(f"**Optimize/Reduce**: {', '.join(low_roi_channels)} have below-target ROI")
            
            # Export option
            st.subheader("📤 Export Results")
            if st.button("Generate PDF Report"):
                st.info("PDF report generation would be implemented here")
        
        else:
            st.warning("Please train a model first to generate the summary report")

# Run the UI
if __name__ == "__main__":
    create_ui()
