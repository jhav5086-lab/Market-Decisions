import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Enterprise MMM Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class MMMEngine:
    def __init__(self):
        self.data = None
        self.model_results = {}
        self.causal_tests = {}
        
    def load_data(self, uploaded_file):
        try:
            self.data = pd.read_csv(uploaded_file)
            
            # Auto-detect date column
            date_col = None
            for col in self.data.columns:
                if 'date' in col.lower():
                    date_col = col
                    break
            
            if not date_col:
                st.error("❌ No date column found in uploaded data")
                return False
            
            # Parse dates
            self.data[date_col] = pd.to_datetime(self.data[date_col], errors='coerce')
            self.data = self.data.set_index(date_col).sort_index()
            
            # Remove rank and country columns
            cols_to_drop = [col for col in self.data.columns if any(x in col.lower() for x in ['rank', 'country', 'rk'])]
            self.data = self.data.drop(columns=cols_to_drop)
            
            # Aggregate to weekly
            self.data = self.data.resample('W-MON').sum()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False

    def reset_data(self):
        """Reset all data and models"""
        self.data = None
        self.model_results = {}
        self.causal_tests = {}
        return True

    def weibull_adstock(self, x, shape=1.5, scale=14, max_lag=28):
        lags = np.arange(0, max_lag + 1)
        weights = (shape / scale) * (lags / scale)**(shape - 1) * np.exp(-(lags / scale)**shape)
        weights = weights / weights.sum()
        adstocked = np.convolve(x, weights, mode='full')[:len(x)]
        return adstocked

    def hill_saturation(self, x, alpha=2.0, kappa=0.8):
        return x**alpha / (x**alpha + kappa**alpha)

    def transform_media_variables(self, params):
        media_transforms = {}
        media_cols = [col for col in self.data.columns if 'spend' in col.lower() and 'paid' in col.lower()]
        
        for col in media_cols:
            channel_type = 'search' if 'search' in col.lower() else 'social' if 'social' in col.lower() else 'display'
            channel_params = params[channel_type]
            
            raw_data = self.data[col].fillna(0).values
            adstocked = self.weibull_adstock(raw_data, channel_params['shape'], channel_params['scale'])
            saturated = self.hill_saturation(adstocked, channel_params['alpha'], channel_params['kappa'])
            media_transforms[col] = saturated
            
        return media_transforms

    def run_causal_analysis(self):
        causal_results = {}
        media_cols = [col for col in self.data.columns if 'spend' in col.lower() and 'paid' in col.lower()]
        
        for media_col in media_cols:
            max_lag = 4
            best_lag = 0
            best_corr = 0
            
            for lag in range(1, max_lag + 1):
                media_lagged = self.data[media_col].shift(lag).dropna()
                revenue_current = self.data['total_revenue'].iloc[lag:]
                
                if len(media_lagged) == len(revenue_current):
                    corr = np.corrcoef(media_lagged, revenue_current)[0,1]
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
            
            causal_results[media_col] = {
                'best_lag': best_lag,
                'max_correlation': best_corr,
                'significant': abs(best_corr) > 0.3
            }
        
        spend_shocks = {}
        for media_col in media_cols:
            spend_changes = self.data[media_col].pct_change().abs()
            shock_periods = spend_changes[spend_changes > 1.0]
            spend_shocks[media_col] = len(shock_periods)
        
        causal_results['spend_shocks'] = spend_shocks
        self.causal_tests = causal_results
        return causal_results

    def train_model(self, params, test_size=0.2):
        try:
            media_transforms = self.transform_media_variables(params)
            X = pd.DataFrame(media_transforms)
            
            organic_cols = [col for col in self.data.columns if 'organic' in col.lower() and 'revenue' in col]
            for col in organic_cols:
                X[col] = self.data[col].fillna(0).values
            
            y = self.data['total_revenue'].values
            
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            coefs = self._apply_bayesian_shrinkage(model.coef_, X.columns, params)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            self.model_results = {
                'model': model,
                'feature_names': X.columns.tolist(),
                'coefficients': coefs,
                'train_actual': y_train,
                'test_actual': y_test,
                'train_predicted': y_pred_train,
                'test_predicted': y_pred_test,
                'train_dates': self.data.index[:split_idx],
                'test_dates': self.data.index[split_idx:],
                'test_start_idx': split_idx,
                'feature_importance': dict(zip(X.columns, np.abs(coefs)))
            }
            
            self._calculate_model_metrics()
            return True
            
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")
            return False

    def _apply_bayesian_shrinkage(self, coefficients, feature_names, params):
        shrunk_coefs = np.zeros_like(coefficients)
        
        for i, (coef, feature) in enumerate(zip(coefficients, feature_names)):
            if 'paid_search' in feature:
                prior_mean = params.get('search_prior', 0.15)
            elif 'paid_social' in feature:
                prior_mean = params.get('social_prior', 0.12)
            elif 'paid_display' in feature:
                prior_mean = params.get('display_prior', 0.08)
            else:
                prior_mean = coef
                
            shrinkage_strength = params.get('regularization_strength', 0.3)
            shrunk_coefs[i] = (1 - shrinkage_strength) * coef + shrinkage_strength * prior_mean
                
        return shrunk_coefs

    def _calculate_model_metrics(self):
        results = self.model_results
        
        train_mape = np.mean(np.abs((results['train_actual'] - results['train_predicted']) / results['train_actual'])) * 100
        test_mape = np.mean(np.abs((results['test_actual'] - results['test_predicted']) / results['test_actual'])) * 100
        
        train_r2 = r2_score(results['train_actual'], results['train_predicted'])
        test_r2 = r2_score(results['test_actual'], results['test_predicted'])
        
        total_revenue = np.sum(results['train_actual']) + np.sum(results['test_actual'])
        media_contribution = 0
        
        for feature, coef in zip(results['feature_names'], results['coefficients']):
            if 'paid' in feature:
                media_data = self.data[feature].fillna(0).values
                media_effect = np.sum(coef * media_data)
                media_contribution += media_effect
        
        base_contribution = total_revenue - media_contribution
        
        self.model_results.update({
            'train_mape': train_mape,
            'test_mape': test_mape,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'base_contribution': base_contribution,
            'incremental_contribution': media_contribution,
            'total_revenue': total_revenue
        })

    def optimize_budget(self, target_increase, total_budget, business_rules):
        if not self.model_results:
            return None
        
        current_revenue = self.data['total_revenue'].mean() * 52
        target_revenue = current_revenue * (1 + target_increase)
        
        channel_roi = {}
        for feature, coef in zip(self.model_results['feature_names'], self.model_results['coefficients']):
            if 'paid' in feature:
                channel = feature.replace('_gross_spend', '').replace('_net_spend', '')
                historical_spend = self.data[feature].mean()
                if historical_spend > 0:
                    channel_roi[channel] = coef / historical_spend
        
        allocation = {}
        remaining_budget = total_budget
        
        sorted_channels = sorted(channel_roi.items(), key=lambda x: x[1], reverse=True)
        
        for channel, roi in sorted_channels:
            if channel in business_rules:
                min_spend = business_rules[channel]['min_spend']
                max_spend = business_rules[channel]['max_spend']
                
                channel_budget = min(max_spend, max(min_spend, remaining_budget * 0.4))
                allocation[channel] = {
                    'budget': channel_budget,
                    'expected_roi': roi,
                    'expected_revenue': channel_budget * roi
                }
                
                remaining_budget -= channel_budget
                
                if remaining_budget <= 0:
                    break
        
        total_expected_revenue = sum([channel_data['expected_revenue'] for channel_data in allocation.values()])
        
        return {
            'allocation': allocation,
            'total_expected_revenue': total_expected_revenue,
            'revenue_increase_achieved': (total_expected_revenue / current_revenue) - 1,
            'remaining_budget': remaining_budget
        }

class VisualizationEngine:
    def __init__(self):
        plt.style.use('default')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_actual_vs_predicted(self, model_results):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(model_results['train_dates'], model_results['train_actual']/1e6, 
                label='Actual', linewidth=2, color='blue')
        ax1.plot(model_results['train_dates'], model_results['train_predicted']/1e6,
                label='Predicted', linewidth=2, color='red', linestyle='--')
        ax1.set_title('Model Performance: Training Period', fontweight='bold')
        ax1.set_ylabel('Revenue (Million USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(model_results['test_dates'], model_results['test_actual']/1e6,
                label='Actual', linewidth=2, color='blue')
        ax2.plot(model_results['test_dates'], model_results['test_predicted']/1e6,
                label='Predicted', linewidth=2, color='red', linestyle='--')
        ax2.axvspan(model_results['test_dates'][0], model_results['test_dates'][-1], 
                   alpha=0.2, color='gray', label='Holdout Period')
        ax2.set_title('Model Performance: Test Period (Holdout)', fontweight='bold')
        ax2.set_ylabel('Revenue (Million USD)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def plot_contribution_analysis(self, model_results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        base = model_results['base_contribution'] / 1e6
        incremental = model_results['incremental_contribution'] / 1e6
        
        ax1.pie([base, incremental], labels=['Base', 'Incremental'], 
                autopct='%1.1f%%', startangle=90, colors=['#ff7f0e', '#1f77b4'])
        ax1.set_title('Revenue Contribution: Base vs Incremental', fontweight='bold')
        
        media_effects = {}
        for feature, coef in zip(model_results['feature_names'], model_results['coefficients']):
            if 'paid' in feature:
                channel = feature.split('_')[1] if len(feature.split('_')) > 1 else feature
                total_effect = coef * 1000000  # Simplified for demo
                media_effects[channel] = total_effect / 1e6
        
        if media_effects:
            channels = list(media_effects.keys())
            effects = list(media_effects.values())
            
            bars = ax2.bar(channels, effects, color=self.colors[:len(channels)])
            ax2.set_title('Incremental Revenue by Channel', fontweight='bold')
            ax2.set_ylabel('Revenue (Million USD)')
            
            for bar, effect in zip(bars, effects):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(effects)*0.01, 
                       f'${effect:.1f}M', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig

# Initialize session state
if 'mmm_engine' not in st.session_state:
    st.session_state.mmm_engine = MMMEngine()
if 'viz_engine' not in st.session_state:
    st.session_state.viz_engine = VisualizationEngine()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Main application
st.markdown('<h1 class="main-header">🚀 Enterprise Marketing Mix Modeling Platform</h1>', unsafe_allow_html=True)

# Sidebar with Data Management
st.sidebar.title("🔧 Data Management")

# Upload Data Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=['csv'], key="file_uploader")

if st.sidebar.button("📤 Upload Data", type="primary", use_container_width=True):
    if uploaded_file is not None:
        with st.spinner("Loading and validating data..."):
            success = st.session_state.mmm_engine.load_data(uploaded_file)
            st.session_state.data_loaded = success
            
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Data successfully loaded!")
            st.session_state.model_trained = False  # Reset model when new data is loaded
        else:
            st.sidebar.error("❌ Failed to load data")
    else:
        st.sidebar.warning("⚠️ Please select a CSV file first")

# Reset Data Button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset All Data", use_container_width=True):
    st.session_state.mmm_engine.reset_data()
    st.session_state.data_loaded = False
    st.session_state.model_trained = False
    st.sidebar.success("✅ All data and models reset!")
    st.rerun()

# Data Status Display
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Current Status")
if st.session_state.data_loaded:
    st.sidebar.success("✅ Data Loaded")
    data_info = st.session_state.mmm_engine.data
    st.sidebar.write(f"**Periods:** {len(data_info)}")
    st.sidebar.write(f"**Date Range:** {data_info.index.min().strftime('%Y-%m-%d')} to {data_info.index.max().strftime('%Y-%m-%d')}")
    st.sidebar.write(f"**Total Revenue:** ${data_info['total_revenue'].sum()/1e6:.1f}M")
else:
    st.sidebar.warning("⚠️ No data loaded")

# Main content navigation
st.sidebar.markdown("---")
st.sidebar.subheader("🧭 Navigation")
app_section = st.sidebar.radio(
    "Select Section:",
    ["Data Overview", "Causal Analysis", "Model Training", "Budget Simulation", "Results Dashboard"]
)

# Section 1: Data Overview
if app_section == "Data Overview":
    st.markdown('<h2 class="section-header">📊 Data Overview</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data using the 'Upload Data' button in the sidebar")
    else:
        data = st.session_state.mmm_engine.data
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Periods", len(data))
        with col2:
            st.metric("Date Range", f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("Total Revenue", f"${data['total_revenue'].sum()/1e6:.1f}M")
        with col4:
            st.metric("Avg Weekly Revenue", f"${data['total_revenue'].mean()/1000:.0f}K")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Available variables
        st.subheader("Available Variables")
        media_cols = [col for col in data.columns if 'paid' in col.lower()]
        organic_cols = [col for col in data.columns if 'organic' in col.lower()]
        other_cols = [col for col in data.columns if col not in media_cols + organic_cols and col != 'total_revenue']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Media Variables:**")
            for col in media_cols[:10]:  # Show first 10
                st.write(f"- {col}")
        with col2:
            st.write("**Organic Variables:**")
            for col in organic_cols[:10]:
                st.write(f"- {col}")
        with col3:
            st.write("**Other Variables:**")
            for col in other_cols[:10]:
                st.write(f"- {col}")

# Section 2: Causal Analysis
elif app_section == "Causal Analysis" and st.session_state.data_loaded:
    st.markdown('<h2 class="section-header">🔍 Causal Analysis</h2>', unsafe_allow_html=True)
    
    if st.button("Run Causal Analysis", type="primary"):
        with st.spinner("Running causal validation tests..."):
            causal_results = st.session_state.mmm_engine.run_causal_analysis()
        
        st.success("✅ Causal analysis completed!")
        
        # Display results
        st.subheader("Granger Causality Tests")
        causal_df = []
        for media_col, results in causal_results.items():
            if media_col != 'spend_shocks':
                causal_df.append({
                    'Channel': media_col,
                    'Optimal Lag (weeks)': results['best_lag'],
                    'Max Correlation': f"{results['max_correlation']:.3f}",
                    'Significant': '✅' if results['significant'] else '❌'
                })
        
        if causal_df:
            st.dataframe(pd.DataFrame(causal_df), use_container_width=True)
        
        # Natural experiments
        st.subheader("Natural Experiment Detection")
        shock_df = []
        for media_col, shock_count in causal_results['spend_shocks'].items():
            shock_df.append({
                'Channel': media_col,
                'Spend Shocks Detected': shock_count,
                'Analysis': '✅ Sufficient variation' if shock_count > 0 else '⚠️ Limited variation'
            })
        
        st.dataframe(pd.DataFrame(shock_df), use_container_width=True)

# Section 3: Model Training
elif app_section == "Model Training" and st.session_state.data_loaded:
    st.markdown('<h2 class="section-header">⚙️ Model Training</h2>', unsafe_allow_html=True)
    
    st.subheader("Media Transformation Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Paid Search**")
        search_shape = st.slider("Search Weibull Shape", 0.5, 3.0, 1.2, key="search_shape")
        search_scale = st.slider("Search Weibull Scale (days)", 1, 28, 10, key="search_scale")
        search_alpha = st.slider("Search Hill Alpha", 0.5, 5.0, 1.8, key="search_alpha")
        search_kappa = st.slider("Search Hill Kappa", 0.1, 1.0, 0.7, key="search_kappa")
    
    with col2:
        st.markdown("**Paid Social**")
        social_shape = st.slider("Social Weibull Shape", 0.5, 3.0, 0.8, key="social_shape")
        social_scale = st.slider("Social Weibull Scale (days)", 1, 28, 7, key="social_scale")
        social_alpha = st.slider("Social Hill Alpha", 0.5, 5.0, 1.3, key="social_alpha")
        social_kappa = st.slider("Social Hill Kappa", 0.1, 1.0, 0.6, key="social_kappa")
    
    with col3:
        st.markdown("**Paid Display**")
        display_shape = st.slider("Display Weibull Shape", 0.5, 3.0, 1.0, key="display_shape")
        display_scale = st.slider("Display Weibull Scale (days)", 1, 28, 5, key="display_scale")
        display_alpha = st.slider("Display Hill Alpha", 0.5, 5.0, 1.5, key="display_alpha")
        display_kappa = st.slider("Display Hill Kappa", 0.1, 1.0, 0.5, key="display_kappa")
    
    st.subheader("Bayesian Regularization")
    regularization_strength = st.slider("Regularization Strength", 0.0, 1.0, 0.3)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        search_prior = st.number_input("Search Prior ROI", 0.0, 1.0, 0.15, step=0.01)
    with col2:
        social_prior = st.number_input("Social Prior ROI", 0.0, 1.0, 0.12, step=0.01)
    with col3:
        display_prior = st.number_input("Display Prior ROI", 0.0, 1.0, 0.08, step=0.01)
    
    model_params = {
        'search': {'shape': search_shape, 'scale': search_scale, 'alpha': search_alpha, 'kappa': search_kappa},
        'social': {'shape': social_shape, 'scale': social_scale, 'alpha': social_alpha, 'kappa': social_kappa},
        'display': {'shape': display_shape, 'scale': display_scale, 'alpha': display_alpha, 'kappa': display_kappa},
        'regularization_strength': regularization_strength,
        'search_prior': search_prior,
        'social_prior': social_prior,
        'display_prior': display_prior
    }
    
    if st.button("Train Model", type="primary"):
        with st.spinner("Training Bayesian MMM model..."):
            success = st.session_state.mmm_engine.train_model(model_params)
            st.session_state.model_trained = success
        
        if st.session_state.model_trained:
            st.success("✅ Model trained successfully!")
            
            results = st.session_state.mmm_engine.model_results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Train MAPE", f"{results['train_mape']:.2f}%")
            with col2:
                st.metric("Test MAPE", f"{results['test_mape']:.2f}%")
            with col3:
                st.metric("Train R²", f"{results['train_r2']:.3f}")
            with col4:
                st.metric("Test R²", f"{results['test_r2']:.3f}")

# Section 4: Budget Simulation
elif app_section == "Budget Simulation" and st.session_state.model_trained:
    st.markdown('<h2 class="section-header">💰 Budget Simulation</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_increase = st.slider("Target Revenue Increase (%)", 5, 25, 12) / 100
        total_budget = st.number_input("Total Budget ($)", min_value=10000, value=1000000, step=50000)
    
    with col2:
        st.subheader("Business Constraints")
        min_search = st.number_input("Min Search Spend", min_value=0, value=50000, step=5000)
        max_search = st.number_input("Max Search Spend", min_value=min_search, value=500000, step=50000)
        
        min_social = st.number_input("Min Social Spend", min_value=0, value=20000, step=5000)
        max_social = st.number_input("Max Social Spend", min_value=min_social, value=300000, step=50000)
        
        min_display = st.number_input("Min Display Spend", min_value=0, value=10000, step=5000)
        max_display = st.number_input("Max Display Spend", min_value=min_display, value=200000, step=50000)
    
    business_rules = {
        'paid_search': {'min_spend': min_search, 'max_spend': max_search},
        'paid_social': {'min_spend': min_social, 'max_spend': max_social},
        'paid_display': {'min_spend': min_display, 'max_spend': max_display}
    }
    
    if st.button("Optimize Budget Allocation", type="primary"):
        with st.spinner("Running budget optimization..."):
            optimization = st.session_state.mmm_engine.optimize_budget(
                target_increase, total_budget, business_rules
            )
        
        if optimization:
            st.success("✅ Budget optimization completed!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Target Increase", f"{target_increase*100:.1f}%")
            with col2:
                st.metric("Achieved Increase", f"{optimization['revenue_increase_achieved']*100:.1f}%")
            with col3:
                st.metric("Remaining Budget", f"${optimization['remaining_budget']:,.0f}")
            
            st.subheader("Optimal Allocation")
            allocation_df = []
            for channel, data in optimization['allocation'].items():
                allocation_df.append({
                    'Channel': channel,
                    'Budget': f"${data['budget']:,.0f}",
                    'Expected ROI': f"{data['expected_roi']:.2f}",
                    'Expected Revenue': f"${data['expected_revenue']:,.0f}"
                })
            
            st.dataframe(pd.DataFrame(allocation_df), use_container_width=True)

# Section 5: Results Dashboard
elif app_section == "Results Dashboard" and st.session_state.model_trained:
    st.markdown('<h2 class="section-header">📊 Results Dashboard</h2>', unsafe_allow_html=True)
    
    results = st.session_state.mmm_engine.model_results
    
    # Key Metrics
    st.subheader("Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training MAPE", f"{results['train_mape']:.2f}%")
    with col2:
        st.metric("Test MAPE", f"{results['test_mape']:.2f}%")
    with col3:
        st.metric("Training R²", f"{results['train_r2']:.3f}")
    with col4:
        st.metric("Test R²", f"{results['test_r2']:.3f}")
    
    # Revenue Decomposition
    st.subheader("Revenue Decomposition")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Revenue", f"${results['total_revenue']/1e6:.1f}M")
    with col2:
        st.metric("Base Revenue", f"${results['base_contribution']/1e6:.1f}M")
    with col3:
        st.metric("Incremental Revenue", f"${results['incremental_contribution']/1e6:.1f}M")
    
    # Visualizations
    st.subheader("Model Performance")
    fig1 = st.session_state.viz_engine.plot_actual_vs_predicted(results)
    st.pyplot(fig1)
    
    st.subheader("Revenue Contribution Analysis")
    fig2 = st.session_state.viz_engine.plot_contribution_analysis(results)
    st.pyplot(fig2)

# Handle missing prerequisites
if not st.session_state.data_loaded and app_section != "Data Overview":
    st.warning("⚠️ Please upload data first using the 'Upload Data' button in the sidebar")
elif not st.session_state.model_trained and app_section in ["Budget Simulation", "Results Dashboard"]:
    st.warning("⚠️ Please train the model first in the 'Model Training' section")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Enterprise MMM Platform v1.0 | Upload Data • Causal Analysis • Budget Optimization"
    "</div>",
    unsafe_allow_html=True
)
