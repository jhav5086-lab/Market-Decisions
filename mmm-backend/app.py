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
st.set_page_config(
    page_title="Media Mix Model Analyzer", 
    layout="wide", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

class MediaMixModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def hill_saturation(self, spend, alpha, gamma):
        """Hill saturation function for diminishing returns"""
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
        """Calculate Variance Inflation Factor for multicollinearity"""
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        return vif_data

def enhanced_data_upload():
    """Enhanced data upload function with support for multiple file types"""
    st.sidebar.header("📁 Data Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Marketing Data", 
        type=['csv', 'xlsx'],
        help="Upload your CSV or Excel file with marketing data. Required columns: date, revenue, and spend columns."
    )
    
    if uploaded_file is not None:
        try:
            # Handle different file types
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Basic data validation
            if data.empty:
                st.sidebar.error("Uploaded file is empty!")
                return None
                
            # Check for required columns
            if 'date' not in data.columns:
                st.sidebar.warning("⚠️ No 'date' column found. Please ensure your data has a date column.")
            else:
                st.sidebar.success(f"✅ Data loaded: {len(data)} rows, {len(data.columns)} columns")
                
                # Show quick data preview
                with st.sidebar.expander("📊 Data Preview"):
                    st.dataframe(data.head(3), use_container_width=True)
                    st.write(f"**Date range:** {data['date'].iloc[0]} to {data['date'].iloc[-1]}")
                    st.write(f"**Numeric columns:** {len(data.select_dtypes(include=[np.number]).columns)}")
            
            return data
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {str(e)}")
            return None
    
    return None

def generate_sample_data():
    """Generate sample marketing data for testing and demonstration"""
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    
    # Create realistic seasonal patterns
    seasonal_pattern = np.sin(2 * np.pi * np.arange(len(dates)) / 365) * 0.3 + 1
    
    sample_data = {
        'date': dates,
        'paid_search_net_spend': np.random.uniform(1000, 5000, len(dates)) * seasonal_pattern,
        'paid_search_revenue': np.random.uniform(5000, 20000, len(dates)) * seasonal_pattern,
        'paid_social_net_spend': np.random.uniform(500, 3000, len(dates)) * seasonal_pattern,
        'paid_social_revenue': np.random.uniform(2000, 10000, len(dates)) * seasonal_pattern,
        'paid_display_net_spend': np.random.uniform(300, 2000, len(dates)) * seasonal_pattern,
        'paid_display_revenue': np.random.uniform(1000, 6000, len(dates)) * seasonal_pattern,
        'paid_shopping_net_spend': np.random.uniform(800, 4000, len(dates)) * seasonal_pattern,
        'paid_shopping_revenue': np.random.uniform(3000, 15000, len(dates)) * seasonal_pattern,
        'total_revenue': np.random.uniform(15000, 50000, len(dates)) * seasonal_pattern,
        'Promo': np.random.choice(['Avg day', 'Promo day', 'Holiday', 'FSNM', 'Collab'], len(dates), p=[0.7, 0.1, 0.1, 0.05, 0.05]),
        'holiday': np.random.choice(['NH', 'New Year Day', 'Valentine Day', 'MLK', 'President Day'], len(dates), p=[0.8, 0.05, 0.05, 0.05, 0.05])
    }
    
    return pd.DataFrame(sample_data)

def preprocess_data(data, date_col, dependent_var, independent_vars):
    """Preprocess the data for modeling"""
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
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    
    # Handle missing values
    numeric_cols = [dependent_var] + independent_vars
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df

def train_model(processed_data, dependent_var, independent_vars, model_type, test_size, alpha):
    """Train the selected model and return results"""
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
        
        # Simple ROI calculation
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
        'feature_names': independent_vars,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test
    }

def plot_response_curve(results, channel, max_spend):
    """Plot response curve for a specific channel"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    spend_range = np.linspace(0, max_spend, 100)
    
    # Apply transformations based on model type
    if results['model_type'] == "Weibull":
        response = 1 - np.exp(-spend_range/spend_range.mean())
    elif results['model_type'] == "Hill Saturation":
        response = spend_range / (spend_range + spend_range.mean())
    elif results['model_type'] == "Geometric":
        response = np.log(spend_range + 1)
    else:
        response = spend_range
    
    # Scale response based on model coefficients
    if channel in results['feature_names']:
        channel_idx = results['feature_names'].index(channel)
        coefficient = results['model'].coef_[channel_idx]
        scaled_response = response * coefficient
        
        ax.plot(spend_range, scaled_response, linewidth=3, color='steelblue')
        ax.fill_between(spend_range, 0, scaled_response, alpha=0.3, color='steelblue')
        ax.set_xlabel(f"{channel} Spend ($)")
        ax.set_ylabel("Incremental Revenue ($)")
        ax.set_title(f"Response Curve: {channel}")
        ax.grid(True, alpha=0.3)
        
        # Add saturation point annotation
        saturation_point = spend_range[np.argmax(scaled_response)]
        ax.axvline(x=saturation_point, color='red', linestyle='--', alpha=0.7)
        ax.text(saturation_point, max(scaled_response)*0.8, f'Saturation: ${saturation_point:,.0f}', 
                rotation=90, verticalalignment='center')
    
    return fig

def simulate_budget(results, budget_allocation):
    """Simulate revenue impact of budget allocation"""
    total_revenue = 0
    model = results['model']
    
    for channel in results['feature_names']:
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
            
            # Get channel index and calculate contribution
            idx = results['feature_names'].index(channel)
            contribution = model.coef_[idx] * transformed_spend
            total_revenue += contribution
    
    # Add intercept
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
    try:
        result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimized_budget = pd.DataFrame({
            'Channel': channels,
            'Optimized_Budget': result.x,
            'Percentage': result.x / total_budget * 100
        }).sort_values('Optimized_Budget', ascending=False)
        
        return optimized_budget
    except:
        # Fallback if optimization fails
        equal_budget = total_budget / len(channels)
        optimized_budget = pd.DataFrame({
            'Channel': channels,
            'Optimized_Budget': [equal_budget] * len(channels),
            'Percentage': [100 / len(channels)] * len(channels)
        })
        return optimized_budget

def generate_forecast(results, processed_data, forecast_period, budget_growth, include_promo):
    """Generate revenue forecast"""
    periods = {
        "Next 3 Months": 3,
        "Next 6 Months": 6,
        "Next 12 Months": 12
    }
    
    n_periods = periods[forecast_period]
    
    # Create future dates
    last_date = processed_data['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_periods, freq='M')
    
    # Forecast data frame
    forecast_data = []
    base_revenue = processed_data[st.session_state.dependent_var].mean()
    
    for i, date in enumerate(future_dates):
        growth_factor = (1 + budget_growth) ** (i + 1)
        forecast_revenue = base_revenue * growth_factor
        
        # Add some randomness for realism
        forecast_revenue *= np.random.uniform(0.95, 1.05)
        
        forecast_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Forecasted_Revenue': forecast_revenue,
            'Growth_Rate': f"{(growth_factor - 1) * 100:.1f}%"
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    return forecast_df

def plot_volume_decomposition(results):
    """Plot volume growth/loss decomposition by channel"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    channels = list(results['channel_contributions'].keys())
    contributions = list(results['channel_contributions'].values())
    
    # Sort by contribution
    sorted_indices = np.argsort(contributions)[::-1]
    channels = [channels[i] for i in sorted_indices]
    contributions = [contributions[i] for i in sorted_indices]
    
    colors = ['#2E8B57' if x > 0 else '#DC143C' for x in contributions]
    bars = ax.bar(channels, contributions, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel("Revenue Contribution ($)", fontsize=12)
    ax.set_title("Channel Revenue Contributions", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom', fontweight='bold')
    
    return fig

def plot_base_vs_incremental(results):
    """Plot base vs incremental volume split"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Calculate base and incremental
    total_volume = sum(results['channel_contributions'].values())
    base_volume = total_volume * 0.6  # Assuming 60% base
    incremental_volume = total_volume * 0.4  # Assuming 40% incremental
    
    labels = ['Base Volume', 'Incremental Volume']
    sizes = [base_volume, incremental_volume]
    colors = ['#4682B4', '#32CD32']
    explode = (0.05, 0.05)  # explode slices
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.1f%%', shadow=True, startangle=90,
                                      textprops={'fontsize': 12})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title("Base vs Incremental Volume Split", fontsize=14, fontweight='bold')
    
    # Add summary box
    summary_text = f"Total Revenue: ${total_volume:,.0f}\nBase: ${base_volume:,.0f}\nIncremental: ${incremental_volume:,.0f}"
    ax.text(1.2, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return fig

def plot_roi_bubble_chart(results):
    """Plot ROI bubble chart by channel"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    channels = list(results['channel_roi'].keys())
    roi_values = list(results['channel_roi'].values())
    contributions = list(results['channel_contributions'].values())
    
    # Bubble sizes based on contribution (normalized)
    max_contribution = max(contributions)
    bubble_sizes = [abs(c) / max_contribution * 1000 + 100 for c in contributions]
    
    # Create scatter plot
    scatter = ax.scatter(range(len(channels)), roi_values, s=bubble_sizes, alpha=0.7, 
                        c=roi_values, cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha='right')
    ax.set_ylabel("ROI", fontsize=12)
    ax.set_xlabel("Marketing Channels", fontsize=12)
    ax.set_title("Channel ROI vs Contribution (Bubble Size = Contribution)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('ROI', rotation=270, labelpad=15)
    
    # Add value annotations
    for i, (channel, roi, size) in enumerate(zip(channels, roi_values, bubble_sizes)):
        ax.annotate(f'{roi:.2%}', (i, roi), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    return fig

def calculate_promo_effectiveness(results):
    """Calculate promotional effectiveness"""
    # In a real scenario, this would use actual promo data
    promo_data = {
        'Promo_Type': ['Seasonal Sale', 'Clearance Event', 'New Product Launch', 'Holiday Campaign', 'Flash Sale'],
        'Effectiveness_Score': [85, 72, 92, 88, 78],
        'Revenue_Lift_Percent': [15, 8, 25, 18, 12],
        'ROI': [2.1, 1.5, 3.2, 2.8, 1.9],
        'Avg_Duration_Days': [14, 7, 21, 10, 3]
    }
    
    promo_df = pd.DataFrame(promo_data)
    return promo_df

def main():
    """Main application function"""
    st.title("📊 Media Mix Model Analyzer")
    st.markdown("""
    **Optimize your marketing budget allocation with advanced Media Mix Modeling (MMM) techniques.**
    Upload your data or use sample data to get started.
    """)
    st.markdown("---")
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Data upload
        data = enhanced_data_upload()
        
        # Sample data option
        st.markdown("---")
        st.subheader("🎲 Quick Start")
        if st.button("Load Sample Data for Testing", use_container_width=True):
            with st.spinner("Generating sample data..."):
                sample_data = generate_sample_data()
                st.session_state.raw_data = sample_data
                st.success("Sample data loaded successfully!")
        
        if data is not None:
            st.session_state.raw_data = data
        
        # Help section
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. **Upload Data**: CSV/Excel with date, revenue, and spend columns
            2. **Process Data**: Select target KPI and features
            3. **Train Model**: Choose model type and parameters
            4. **Analyze**: View results and insights
            5. **Optimize**: Simulate budget scenarios
            """)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 Data Setup", "⚙️ Model Config", "📈 Analysis", 
        "💰 Budget Simulator", "🔮 Forecasting", "📋 Summary Report"
    ])
    
    # Tab 1: Data Setup
    with tab1:
        st.header("📁 Data Setup & Preprocessing")
        
        if st.session_state.raw_data is not None:
            data = st.session_state.raw_data
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Data Preview")
                st.dataframe(data.head(), use_container_width=True)
                
                st.subheader("Data Summary")
                summary_data = {
                    "Total Records": len(data),
                    "Total Columns": len(data.columns),
                    "Date Column Present": 'date' in data.columns,
                    "Numeric Columns": len(data.select_dtypes(include=[np.number]).columns)
                }
                if 'date' in data.columns:
                    summary_data["Date Range"] = f"{data['date'].iloc[0]} to {data['date'].iloc[-1]}"
                
                st.json(summary_data)
            
            with col2:
                st.subheader("Variable Selection")
                
                # KPI Selection
                kpi_options = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
                if kpi_options:
                    dependent_var = st.selectbox("🎯 Select Target KPI", options=kpi_options,
                                               help="Choose the revenue or conversion metric you want to optimize")
                    
                    # Independent variables
                    independent_options = [col for col in kpi_options if col != dependent_var]
                    independent_vars = st.multiselect("📊 Select Independent Variables", 
                                                     options=independent_options,
                                                     help="Select marketing spend and other predictor variables")
                    
                    if st.button("🔄 Process Data", type="primary", use_container_width=True) and independent_vars:
                        with st.spinner("Processing data..."):
                            processed_data = preprocess_data(data, 'date', dependent_var, independent_vars)
                            st.session_state.processed_data = processed_data
                            st.session_state.dependent_var = dependent_var
                            st.session_state.independent_vars = independent_vars
                            st.success("✅ Data processed successfully! You can now configure the model.")
                else:
                    st.warning("No numeric columns found in the data")
        else:
            st.info("👆 Please upload your data file or load sample data to begin analysis")
            st.markdown("""
            ### Expected Data Format:
            - **date**: Date column (any format)
            - **paid_*_spend**: Marketing spend by channel
            - **paid_*_revenue**: Revenue by channel  
            - **total_revenue**: Overall revenue
            - **Promo/holiday**: Promotional indicators
            """)
    
    # Tab 2: Model Configuration
    with tab2:
        st.header("⚙️ Model Configuration")
        
        if st.session_state.processed_data is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Model Settings")
                
                model_type = st.selectbox(
                    "🤖 Select Model Type", 
                    ["Ridge Regression", "Geometric", "Weibull", "Hill Saturation"],
                    help="Choose the modeling approach for your data"
                )
                
                test_size = st.slider(
                    "📊 Test Set Size (%)", 
                    10, 40, 20
                ) / 100
                
                if model_type == "Ridge Regression":
                    alpha = st.slider(
                        "🎛️ Regularization Strength (Alpha)", 
                        0.1, 10.0, 1.0, 0.1,
                        help="Higher values increase regularization, reducing overfitting"
                    )
                else:
                    alpha = 1.0
                    st.info(f"Using default alpha for {model_type} model")
            
            with col2:
                st.subheader("Model Training")
                
                if st.button("🚀 Train Model", type="primary", use_container_width=True):
                    with st.spinner("Training model... This may take a few seconds."):
                        try:
                            results = train_model(
                                st.session_state.processed_data, 
                                st.session_state.dependent_var,
                                st.session_state.independent_vars,
                                model_type, test_size, alpha
                            )
                            st.session_state.results = results
                            
                            # Display metrics
                            st.success("✅ Model trained successfully!")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Train MAPE", f"{results['train_mape']:.2f}%")
                            col2.metric("Test MAPE", f"{results['test_mape']:.2f}%")
                            col3.metric("R² Score", f"{results['r2_score']:.3f}")
                            
                        except Exception as e:
                            st.error(f"❌ Model training failed: {str(e)}")
        else:
            st.warning("📝 Please process your data first in the Data Setup tab")
    
    # Tab 3: Performance Analysis
    with tab3:
        st.header("📈 Performance Analysis")
        
        if st.session_state.results:
            results = st.session_state.results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Channel Contributions")
                
                # Volume decomposition
                fig = plot_volume_decomposition(results)
                st.pyplot(fig)
                
                st.subheader("🔄 Base vs Incremental Volume")
                fig2 = plot_base_vs_incremental(results)
                st.pyplot(fig2)
                
                # Response curves
                st.subheader("📈 Response Curves")
                channel = st.selectbox("Select Channel for Response Curve", 
                                     options=st.session_state.independent_vars)
                max_spend = st.number_input("Maximum Spend for Simulation", 
                                          value=10000, step=1000)
                
                if st.button("Generate Response Curve"):
                    fig_rc = plot_response_curve(results, channel, max_spend)
                    st.pyplot(fig_rc)
            
            with col2:
                st.subheader("💰 ROI Analysis")
                
                # ROI bubble chart
                fig3 = plot_roi_bubble_chart(results)
                st.pyplot(fig3)
                
                # VIF Analysis
                st.subheader("📊 Multicollinearity Check (VIF)")
                st.dataframe(results['vif'].style.highlight_between(subset=['VIF'], 
                                                                  color='lightcoral', 
                                                                  axis=0,
                                                                  low=5, high=100),
                           use_container_width=True)
                st.caption("VIF > 5 indicates potential multicollinearity issues")
                
                # Promotional effectiveness
                st.subheader("🎯 Promotional Effectiveness")
                promo_effectiveness = calculate_promo_effectiveness(results)
                st.dataframe(promo_effectiveness.style.background_gradient(subset=['Effectiveness_Score', 'ROI']), 
                           use_container_width=True)
        else:
            st.warning("🤖 Please train a model first in the Model Config tab")
    
    # Tab 4: Budget Simulator
    with tab4:
        st.header("💰 Budget Simulator")
        
        if st.session_state.results:
            results = st.session_state.results
            
            st.subheader("🎛️ Budget Allocation")
            budget_allocation = {}
            channels = st.session_state.independent_vars
            
            # Create budget inputs in columns
            cols = st.columns(3)
            for i, channel in enumerate(channels):
                with cols[i % 3]:
                    default_value = 1000 if i < len(channels) else 0
                    budget_allocation[channel] = st.number_input(
                        f"💰 {channel}", 
                        value=default_value, 
                        step=100,
                        key=f"budget_{channel}"
                    )
            
            total_budget = sum(budget_allocation.values())
            st.metric("💵 Total Budget", f"${total_budget:,.2f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Simulate Budget Impact", type="primary", use_container_width=True):
                    predicted_revenue = simulate_budget(results, budget_allocation)
                    roi = (predicted_revenue - total_budget) / total_budget
                    
                    st.subheader("📈 Simulation Results")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Predicted Revenue", f"${predicted_revenue:,.2f}")
                    col2.metric("Total Budget", f"${total_budget:,.2f}")
                    col3.metric("ROI", f"{roi:.1%}")
                    
                    # ROI assessment
                    if roi > 0.2:
                        st.success("🎉 Excellent ROI! Consider increasing investment")
                    elif roi > 0.1:
                        st.info("📈 Good ROI - Solid performance")
                    elif roi > 0:
                        st.warning("⚠️ Low ROI - Consider optimization")
                    else:
                        st.error("❌ Negative ROI - Review strategy")
            
            with col2:
                st.subheader("🔄 Budget Optimization")
                optimization_budget = st.number_input("Total Budget for Optimization", 
                                                    value=total_budget, step=1000)
                
                if st.button("🎯 Optimize Budget Allocation", use_container_width=True):
                    with st.spinner("Finding optimal allocation..."):
                        optimized_budget = optimize_budget(results, optimization_budget)
                        
                        st.subheader("✅ Optimized Allocation")
                        st.dataframe(optimized_budget.style.format({
                            'Optimized_Budget': '${:,.2f}',
                            'Percentage': '{:.1f}%'
                        }), use_container_width=True)
                        
                        # Show improvement
                        current_revenue = simulate_budget(results, budget_allocation)
                        optimal_allocation = {row['Channel']: row['Optimized_Budget'] 
                                            for _, row in optimized_budget.iterrows()}
                        optimal_revenue = simulate_budget(results, optimal_allocation)
                        
                        improvement = optimal_revenue - current_revenue
                        if improvement > 0:
                            st.success(f"Potential revenue improvement: ${improvement:,.2f}")
        else:
            st.warning("🤖 Please train a model first in the Model Config tab")
    
    # Tab 5: Forecasting
    with tab5:
        st.header("🔮 Demand Forecasting")
        
        if st.session_state.results:
            results = st.session_state.results
            processed_data = st.session_state.processed_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚙️ Forecast Settings")
                forecast_period = st.selectbox("Forecast Period", 
                                             ["Next 3 Months", "Next 6 Months", "Next 12 Months"])
                include_promo = st.checkbox("Include Promotional Effects", value=True)
                
                st.subheader("📈 Growth Assumptions")
                budget_growth = st.slider("Budget Growth Rate (%)", -20, 50, 5) / 100
                market_growth = st.slider("Market Growth Rate (%)", -10, 30, 2) / 100
            
            with col2:
                if st.button("📊 Generate Forecast", type="primary", use_container_width=True):
                    with st.spinner("Generating forecast..."):
                        forecast_df = generate_forecast(results, processed_data, forecast_period, 
                                                      budget_growth + market_growth, include_promo)
                        
                        st.subheader("📋 Forecast Results")
                        st.dataframe(forecast_df.style.format({
                            'Forecasted_Revenue': '${:,.2f}'
                        }), use_container_width=True)
                        
                        # Plot forecast
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(forecast_df['Date'], forecast_df['Forecasted_Revenue'], 
                               marker='o', linewidth=2, markersize=6, color='#2E8B57')
                        ax.set_title(f"📈 Revenue Forecast - {forecast_period}", fontsize=14, fontweight='bold')
                        ax.set_ylabel("Revenue ($)", fontsize=12)
                        ax.set_xlabel("Date", fontsize=12)
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        
                        # Summary statistics
                        total_forecast = forecast_df['Forecasted_Revenue'].sum()
                        avg_monthly = forecast_df['Forecasted_Revenue'].mean()
                        growth = (forecast_df['Forecasted_Revenue'].iloc[-1] - forecast_df['Forecasted_Revenue'].iloc[0]) / forecast_df['Forecasted_Revenue'].iloc[0] * 100
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Forecast", f"${total_forecast:,.0f}")
                        col2.metric("Avg Monthly", f"${avg_monthly:,.0f}")
                        col3.metric("Total Growth", f"{growth:.1f}%")
        else:
            st.warning("🤖 Please train a model first in the Model Config tab")
    
    # Tab 6: Summary Report
    with tab6:
        st.header("📋 Executive Summary Report")
        
        if st.session_state.results:
            results = st.session_state.results
            
            # Executive Summary
            st.subheader("🎯 Executive Summary")
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model Accuracy", f"{results['test_mape']:.2f}% MAPE")
            with col2:
                st.metric("R² Score", f"{results['r2_score']:.3f}")
            with col3:
                st.metric("Channels Analyzed", len(results['feature_names']))
            with col4:
                total_revenue = sum(results['channel_contributions'].values())
                st.metric("Total Revenue Modeled", f"${total_revenue:,.0f}")
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Channel Performance")
                performance_df = pd.DataFrame({
                    'Channel': list(results['channel_contributions'].keys()),
                    'Contribution': list(results['channel_contributions'].values()),
                    'ROI': list(results['channel_roi'].values())
                }).sort_values('Contribution', ascending=False)
                
                st.dataframe(performance_df.style.format({
                    'Contribution': '${:,.2f}',
                    'ROI': '{:.2%}'
                }).background_gradient(subset=['Contribution', 'ROI']), 
                use_container_width=True)
            
            with col2:
                st.subheader("🎯 Strategic Recommendations")
                
                # Generate recommendations based on ROI and contribution
                high_roi_channels = [chan for chan, roi in results['channel_roi'].items() if roi > 0.2]
                medium_roi_channels = [chan for chan, roi in results['channel_roi'].items() if 0.1 <= roi <= 0.2]
                low_roi_channels = [chan for chan, roi in results['channel_roi'].items() if roi < 0.1]
                
                if high_roi_channels:
                    st.success(f"**🚀 Increase Investment**: {', '.join(high_roi_channels)} show exceptional ROI (>20%)")
                
                if medium_roi_channels:
                    st.info(f"**📈 Maintain Investment**: {', '.join(medium_roi_channels)} show solid performance (10-20% ROI)")
                
                if low_roi_channels:
                    st.warning(f"**🔍 Optimize/Reduce**: {', '.join(low_roi_channels)} have below-target ROI (<10%)")
                
                # Overall assessment
                avg_roi = np.mean(list(results['channel_roi'].values()))
                if avg_roi > 0.15:
                    st.success(f"**Overall Assessment**: Strong marketing efficiency with average ROI of {avg_roi:.1%}")
                elif avg_roi > 0.08:
                    st.info(f"**Overall Assessment**: Moderate marketing efficiency with average ROI of {avg_roi:.1%}")
                else:
                    st.warning(f"**Overall Assessment**: Marketing efficiency needs improvement. Average ROI: {avg_roi:.1%}")
            
            # Next Steps
            st.subheader("🔄 Recommended Next Steps")
            
            next_steps = [
                "Use the Budget Simulator to test different allocation scenarios",
                "Implement the optimized budget allocation from the Budget Simulator",
                "Set up regular model retraining with new data (quarterly recommended)",
                "Track actual vs predicted performance to improve model accuracy",
                "Consider A/B testing for low-performing channels"
            ]
            
            for i, step in enumerate(next_steps, 1):
                st.write(f"{i}. {step}")
            
            # Export option
            st.subheader("📤 Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Save Report as PDF", use_container_width=True):
                    st.info("PDF export functionality would be implemented here")
            
            with col2:
                if st.button("📊 Export Charts", use_container_width=True):
                    st.info("Chart export functionality would be implemented here")
            
            with col3:
                if st.button("📈 Export Data", use_container_width=True):
                    st.info("Data export functionality would be implemented here")
        
        else:
            st.warning("🤖 Please train a model first to generate the summary report")

# Run the application
if __name__ == "__main__":
    main()
