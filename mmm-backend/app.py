import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Marketing Mix Modeling Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MarketingMixModel:
    def __init__(self):
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def load_sample_data(self):
        """Create sample dataset based on the provided structure"""
        # Generate sample data for demonstration
        dates = pd.date_range('2018-01-01', '2020-12-31', freq='D')
        
        data = {
            'date': dates,
            'report_date_3': dates,
            'day_type': np.random.choice(['Avg day', 'Promo day- low', 'Promo day- mid', 
                                        'Promo day- high', 'Sale', 'Early access'], len(dates)),
            'descriptor': 'Sample campaign',
            'paid_search_gross_spend': np.random.normal(1500, 500, len(dates)),
            'paid_search_net_spend': np.random.normal(1350, 450, len(dates)),
            'paid_search_revenue': np.random.normal(10000, 3000, len(dates)),
            'paid_shopping_gross_spend': np.random.normal(2700, 800, len(dates)),
            'paid_shopping_net_spend': np.random.normal(2450, 700, len(dates)),
            'paid_shopping_revenue': np.random.normal(1500, 500, len(dates)),
            'paid_display_gross_spend': np.random.normal(3300, 1000, len(dates)),
            'paid_display_net_spend': np.random.normal(2800, 800, len(dates)),
            'paid_display_revenue': np.random.normal(120, 50, len(dates)),
            'paid_social_gross_spend': np.random.normal(1800, 600, len(dates)),
            'paid_social_net_spend': np.random.normal(1500, 500, len(dates)),
            'paid_social_revenue': np.random.normal(1600, 500, len(dates)),
            'organic_direct_revenue': np.random.normal(11500, 3000, len(dates)),
            'organic_search_revenue': np.random.normal(8300, 2000, len(dates)),
            'holiday': np.random.choice(['None', 'New Year Day', 'Christmas', 'Black Friday'], 
                                      len(dates), p=[0.85, 0.05, 0.05, 0.05]),
            'year': [d.year for d in dates],
            'total_revenue': 0  # Will be calculated
        }
        
        self.df = pd.DataFrame(data)
        
        # Calculate total revenue
        revenue_columns = [col for col in self.df.columns if 'revenue' in col.lower()]
        self.df['total_revenue'] = self.df[revenue_columns].sum(axis=1)
        
        # Add seasonality and trends
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        
        # Add promotional impact
        promo_impact = {
            'Avg day': 1.0,
            'Promo day- low': 1.2,
            'Promo day- mid': 1.5,
            'Promo day- high': 2.0,
            'Sale': 2.5,
            'Early access': 1.3
        }
        
        self.df['promo_multiplier'] = self.df['day_type'].map(promo_impact)
        self.df['total_revenue'] = self.df['total_revenue'] * self.df['promo_multiplier']
        
        return self.df
    
    def prepare_features(self):
        """Prepare features for modeling"""
        # Select relevant features
        spend_features = [
            'paid_search_net_spend', 'paid_shopping_net_spend', 
            'paid_display_net_spend', 'paid_social_net_spend'
        ]
        
        # Create lagged variables
        for feature in spend_features:
            for lag in [1, 7, 30]:  # 1-day, 1-week, 1-month lags
                self.df[f'{feature}_lag_{lag}'] = self.df[feature].shift(lag)
        
        # Create rolling averages
        for feature in spend_features:
            self.df[f'{feature}_rolling_7'] = self.df[feature].rolling(7).mean()
            self.df[f'{feature}_rolling_30'] = self.df[feature].rolling(30).mean()
        
        # Create seasonal features
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month']/12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month']/12)
        
        # Create holiday dummy
        self.df['is_holiday'] = (self.df['holiday'] != 'None').astype(int)
        
        # Create promo type dummies
        promo_dummies = pd.get_dummies(self.df['day_type'], prefix='promo')
        self.df = pd.concat([self.df, promo_dummies], axis=1)
        
        # Drop rows with NaN values from lag features
        self.df = self.df.dropna()
        
        return self.df
    
    def train_model(self, model_type='linear'):
        """Train the marketing mix model"""
        # Define features and target
        feature_columns = [
            'paid_search_net_spend', 'paid_shopping_net_spend', 
            'paid_display_net_spend', 'paid_social_net_spend',
            'paid_search_net_spend_rolling_7', 'paid_shopping_net_spend_rolling_7',
            'paid_display_net_spend_rolling_7', 'paid_social_net_spend_rolling_7',
            'month_sin', 'month_cos', 'is_holiday'
        ]
        
        # Add promo type features
        promo_features = [col for col in self.df.columns if col.startswith('promo_')]
        feature_columns.extend(promo_features)
        
        X = self.df[feature_columns]
        y = self.df['total_revenue']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=1.0)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate feature importance
        if hasattr(self.model, 'coef_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': abs(self.model.coef_)
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'metrics': {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            },
            'predictions': {
                'actual': y_test,
                'predicted': y_pred
            },
            'features_used': feature_columns
        }

def main():
    st.markdown('<h1 class="main-header">📊 Marketing Mix Modeling Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'mmm' not in st.session_state:
        st.session_state.mmm = MarketingMixModel()
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_section = st.sidebar.radio(
        "Select Section:",
        ["Data Overview", "Exploratory Analysis", "MMM Modeling", "Results & Insights", "Budget Optimization"]
    )
    
    # Load data
    if st.session_state.mmm.df is None:
        with st.spinner("Loading sample data..."):
            st.session_state.mmm.load_sample_data()
            st.session_state.mmm.prepare_features()
    
    # Data Overview Section
    if app_section == "Data Overview":
        st.markdown('<h2 class="section-header">📈 Data Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_revenue = st.session_state.mmm.df['total_revenue'].sum()
            st.metric("Total Revenue", f"${total_revenue:,.0f}")
        
        with col2:
            avg_daily_revenue = st.session_state.mmm.df['total_revenue'].mean()
            st.metric("Average Daily Revenue", f"${avg_daily_revenue:,.0f}")
        
        with col3:
            total_spend = st.session_state.mmm.df[[
                'paid_search_net_spend', 'paid_shopping_net_spend', 
                'paid_display_net_spend', 'paid_social_net_spend'
            ]].sum().sum()
            st.metric("Total Marketing Spend", f"${total_spend:,.0f}")
        
        # Data preview
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.mmm.df.head(10), use_container_width=True)
        
        # Key metrics table
        st.subheader("Key Marketing Metrics")
        metrics_df = st.session_state.mmm.df[[
            'paid_search_net_spend', 'paid_search_revenue',
            'paid_shopping_net_spend', 'paid_shopping_revenue',
            'paid_display_net_spend', 'paid_display_revenue',
            'paid_social_net_spend', 'paid_social_revenue',
            'total_revenue'
        ]].describe()
        
        st.dataframe(metrics_df, use_container_width=True)
    
    # Exploratory Analysis Section
    elif app_section == "Exploratory Analysis":
        st.markdown('<h2 class="section-header">🔍 Exploratory Analysis</h2>', unsafe_allow_html=True)
        
        # Time series analysis
        st.subheader("Revenue & Spend Over Time")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           subplot_titles=('Total Revenue Over Time', 'Marketing Spend by Channel'))
        
        # Revenue plot
        fig.add_trace(
            go.Scatter(x=st.session_state.mmm.df['date'], y=st.session_state.mmm.df['total_revenue'],
                      name='Total Revenue', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Spend plot
        spend_columns = ['paid_search_net_spend', 'paid_shopping_net_spend', 
                        'paid_display_net_spend', 'paid_social_net_spend']
        
        colors = ['red', 'green', 'orange', 'purple']
        for col, color in zip(spend_columns, colors):
            fig.add_trace(
                go.Scatter(x=st.session_state.mmm.df['date'], y=st.session_state.mmm.df[col],
                          name=col.replace('_net_spend', '').replace('_', ' ').title(),
                          line=dict(color=color)),
                row=2, col=1
            )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Channel performance
        st.subheader("Channel Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROI by channel
            channel_roi = {}
            for channel in ['search', 'shopping', 'display', 'social']:
                spend_col = f'paid_{channel}_net_spend'
                revenue_col = f'paid_{channel}_revenue'
                total_spend = st.session_state.mmm.df[spend_col].sum()
                total_revenue = st.session_state.mmm.df[revenue_col].sum()
                roi = (total_revenue - total_spend) / total_spend if total_spend > 0 else 0
                channel_roi[channel.title()] = roi * 100  # Convert to percentage
            
            roi_df = pd.DataFrame(list(channel_roi.items()), columns=['Channel', 'ROI (%)'])
            fig_roi = px.bar(roi_df, x='Channel', y='ROI (%)', 
                            title='Return on Investment by Channel',
                            color='ROI (%)', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_roi, use_container_width=True)
        
        with col2:
            # Spend distribution
            spend_data = []
            for channel in ['search', 'shopping', 'display', 'social']:
                spend_col = f'paid_{channel}_net_spend'
                total_spend = st.session_state.mmm.df[spend_col].sum()
                spend_data.append({
                    'Channel': channel.title(),
                    'Spend': total_spend
                })
            
            spend_df = pd.DataFrame(spend_data)
            fig_spend = px.pie(spend_df, values='Spend', names='Channel', 
                              title='Marketing Spend Distribution')
            st.plotly_chart(fig_spend, use_container_width=True)
        
        # Seasonal analysis
        st.subheader("Seasonal Patterns")
        
        # Monthly aggregation
        monthly_data = st.session_state.mmm.df.groupby('month').agg({
            'total_revenue': 'mean',
            'paid_search_net_spend': 'mean',
            'paid_shopping_net_spend': 'mean',
            'paid_display_net_spend': 'mean',
            'paid_social_net_spend': 'mean'
        }).reset_index()
        
        fig_seasonal = px.line(monthly_data, x='month', y='total_revenue',
                              title='Average Monthly Revenue Pattern')
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # MMM Modeling Section
    elif app_section == "MMM Modeling":
        st.markdown('<h2 class="section-header">🤖 Marketing Mix Modeling</h2>', unsafe_allow_html=True)
        
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type:",
                ["linear", "ridge", "lasso", "random_forest"],
                help="Choose the regression algorithm for MMM"
            )
        
        with col2:
            st.write("Model Description:")
            model_descriptions = {
                "linear": "Linear Regression - Basic model assuming linear relationships",
                "ridge": "Ridge Regression - Handles multicollinearity with L2 regularization",
                "lasso": "Lasso Regression - Feature selection with L1 regularization", 
                "random_forest": "Random Forest - Ensemble method capturing non-linear relationships"
            }
            st.info(model_descriptions[model_type])
        
        if st.button("Train Marketing Mix Model", type="primary"):
            with st.spinner("Training model... This may take a few moments."):
                results = st.session_state.mmm.train_model(model_type)
                st.session_state.model_results = results
                st.session_state.model_trained = True
            
            st.success("Model trained successfully!")
            
            # Display model metrics
            st.subheader("Model Performance")
            
            metrics = results['metrics']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R² Score", f"{metrics['R2']:.3f}")
            
            with col2:
                st.metric("Mean Absolute Error", f"${metrics['MAE']:,.0f}")
            
            with col3:
                st.metric("Root Mean Square Error", f"${metrics['RMSE']:,.0f}")
            
            # Actual vs Predicted plot
            st.subheader("Model Predictions vs Actual")
            
            pred_data = results['predictions']
            fig_pred = go.Figure()
            
            fig_pred.add_trace(go.Scatter(
                x=pred_data['actual'].index, y=pred_data['actual'],
                name='Actual Revenue', mode='markers', marker=dict(color='blue')
            ))
            
            fig_pred.add_trace(go.Scatter(
                x=pred_data['predicted'].index, y=pred_data['predicted'],
                name='Predicted Revenue', mode='markers', marker=dict(color='red')
            ))
            
            fig_pred.update_layout(
                title='Actual vs Predicted Revenue',
                xaxis_title='Observation',
                yaxis_title='Revenue ($)',
                showlegend=True
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
        
        if st.session_state.model_trained:
            # Feature importance
            st.subheader("Feature Importance")
            
            importance_df = st.session_state.model_results['feature_importance']
            fig_importance = px.bar(
                importance_df.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title='Top 10 Most Important Features',
                color='importance',
                color_continuous_scale='viridis'
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
    
    # Results & Insights Section
    elif app_section == "Results & Insights":
        st.markdown('<h2 class="section-header">💡 Results & Business Insights</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first in the 'MMM Modeling' section.")
            return
        
        results = st.session_state.model_results
        
        # Key insights
        st.subheader("Key Marketing Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Channel contribution
            st.info("📊 **Channel Contribution Analysis**")
            channel_contributions = results['feature_importance'][
                results['feature_importance']['feature'].str.contains('net_spend')
            ].head(4)
            
            for _, row in channel_contributions.iterrows():
                channel_name = row['feature'].replace('_net_spend', '').replace('_', ' ').title()
                st.write(f"• **{channel_name}**: {row['importance']:.3f}")
        
        with col2:
            # Model performance insights
            st.success("🎯 **Model Performance**")
            st.write(f"• **R² Score**: {results['metrics']['R2']:.3f}")
            st.write(f"• **Prediction Error**: ${results['metrics']['MAE']:,.0f}")
            st.write(f"• **Best Performing Channel**: {channel_contributions.iloc[0]['feature'].split('_')[1].title()}")
        
        # ROI analysis
        st.subheader("Return on Investment Analysis")
        
        roi_data = []
        for channel in ['search', 'shopping', 'display', 'social']:
            spend_col = f'paid_{channel}_net_spend'
            total_spend = st.session_state.mmm.df[spend_col].sum()
            total_revenue = st.session_state.mmm.df[f'paid_{channel}_revenue'].sum()
            roi = (total_revenue - total_spend) / total_spend if total_spend > 0 else 0
            
            roi_data.append({
                'Channel': channel.title(),
                'Total Spend': total_spend,
                'Total Revenue': total_revenue,
                'ROI (%)': roi * 100,
                'Efficiency': 'High' if roi > 1 else 'Medium' if roi > 0.5 else 'Low'
            })
        
        roi_df = pd.DataFrame(roi_data)
        st.dataframe(roi_df, use_container_width=True)
        
        # Budget allocation recommendations
        st.subheader("Budget Allocation Recommendations")
        
        # Simple optimization based on ROI
        total_budget = st.slider("Set Total Marketing Budget ($):", 
                                min_value=100000, max_value=1000000, 
                                value=500000, step=50000)
        
        if total_budget:
            # Allocate based on ROI performance
            roi_df['Weight'] = roi_df['ROI (%)'] / roi_df['ROI (%)'].sum()
            roi_df['Recommended Allocation'] = roi_df['Weight'] * total_budget
            roi_df['Recommended vs Current'] = roi_df['Recommended Allocation'] - roi_df['Total Spend']
            
            fig_allocation = px.bar(
                roi_df,
                x='Channel',
                y=['Total Spend', 'Recommended Allocation'],
                title='Current vs Recommended Budget Allocation',
                barmode='group'
            )
            
            st.plotly_chart(fig_allocation, use_container_width=True)
            
            st.subheader("Allocation Summary")
            for _, row in roi_df.iterrows():
                change = row['Recommended vs Current']
                action = "Increase" if change > 0 else "Decrease"
                st.write(f"• **{row['Channel']}**: {action} by ${abs(change):,.0f}")
    
    # Budget Optimization Section
    elif app_section == "Budget Optimization":
        st.markdown('<h2 class="section-header">💰 Budget Optimization Simulator</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first in the 'MMM Modeling' section.")
            return
        
        st.subheader("Marketing Budget Simulator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Adjust channel budgets to see predicted revenue impact:")
            
            search_budget = st.slider("Paid Search Budget", 0, 500000, 150000, 10000)
            shopping_budget = st.slider("Paid Shopping Budget", 0, 500000, 120000, 10000)
            display_budget = st.slider("Paid Display Budget", 0, 500000, 80000, 10000)
            social_budget = st.slider("Paid Social Budget", 0, 500000, 100000, 10000)
        
        with col2:
            # Calculate predicted revenue
            if hasattr(st.session_state.mmm, 'model'):
                # Prepare input features for prediction
                input_features = np.array([[
                    search_budget, shopping_budget, display_budget, social_budget,
                    0, 0, 0, 0,  # rolling averages (set to 0 for simulation)
                    0, 0, 0  # seasonal and holiday features
                ]])
                
                # Add promo features (assuming average day)
                promo_features = np.zeros((1, 6))  # 6 promo types
                input_features = np.concatenate([input_features, promo_features], axis=1)
                
                # Scale features and predict
                input_scaled = st.session_state.mmm.scaler.transform(input_features)
                predicted_revenue = st.session_state.mmm.model.predict(input_scaled)[0]
                
                total_budget = search_budget + shopping_budget + display_budget + social_budget
                predicted_roi = (predicted_revenue - total_budget) / total_budget if total_budget > 0 else 0
                
                st.metric("Total Budget", f"${total_budget:,.0f}")
                st.metric("Predicted Revenue", f"${predicted_revenue:,.0f}")
                st.metric("Predicted ROI", f"{predicted_roi:.1%}")
        
        # Scenario analysis
        st.subheader("Scenario Analysis")
        
        scenarios = {
            "Current Allocation": [150000, 120000, 80000, 100000],
            "ROI Optimized": [200000, 150000, 50000, 120000],
            "Balanced Mix": [180000, 130000, 60000, 110000],
            "Aggressive Growth": [250000, 180000, 70000, 150000]
        }
        
        scenario_results = []
        for scenario_name, budgets in scenarios.items():
            input_features = np.array([[
                budgets[0], budgets[1], budgets[2], budgets[3],
                0, 0, 0, 0, 0, 0, 0
            ]])
            promo_features = np.zeros((1, 6))
            input_features = np.concatenate([input_features, promo_features], axis=1)
            
            input_scaled = st.session_state.mmm.scaler.transform(input_features)
            predicted_revenue = st.session_state.mmm.model.predict(input_scaled)[0]
            total_budget = sum(budgets)
            roi = (predicted_revenue - total_budget) / total_budget
            
            scenario_results.append({
                'Scenario': scenario_name,
                'Total Budget': total_budget,
                'Predicted Revenue': predicted_revenue,
                'ROI': roi
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        st.dataframe(scenario_df, use_container_width=True)
        
        # Visualization of scenarios
        fig_scenarios = px.bar(
            scenario_df,
            x='Scenario',
            y=['Total Budget', 'Predicted Revenue'],
            title='Budget Scenarios Comparison',
            barmode='group'
        )
        st.plotly_chart(fig_scenarios, use_container_width=True)

if __name__ == "__main__":
    main()
