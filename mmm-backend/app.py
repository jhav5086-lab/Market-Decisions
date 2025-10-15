# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
class MMMConfig:
    REQUIRED_COLUMNS = {
        'date': 'Date column',
        'total_revenue': 'Target revenue'
    }
    OPTIONAL_MEDIA_COLUMNS = {
        'paid_search_net_spend': 'Paid search',
        'paid_social_net_spend': 'Paid social', 
        'paid_display_net_spend': 'Paid display',
        'paid_shopping_net_spend': 'Paid shopping',
        'paid_affiliate_net_spend': 'Paid affiliate',
        'organic_search_sessions': 'Organic search',
        'organic_social_sessions': 'Organic social',
        'organic_direct_sessions': 'Organic direct',
        'organic_email_sessions': 'Organic email',
        'organic_referral_sessions': 'Organic referral'
    }
    OPTIONAL_BASE_COLUMNS = {
        'holiday': 'Holiday indicators',
        'day_type': 'Promotion types',
        'descriptor': 'Promotion details'
    }
    PROMOTION_PATTERNS = {
        'promo_50_pct': ['50% off', '50 off'],
        'promo_30_pct': ['30% off', '30 off', '30% Off Fall Favorites'],
        'promo_25_pct': ['25% off', '25 off', 'Extra 25% Off'],
        'promo_20_pct': ['20% off', '20 off'],
        'promo_gwp': ['GWP', 'gift with purchase', 'free gift', 'Brush Set'],
        'promo_collab': ['Collab', 'Collaboration', 'Relaunch'],
        'promo_influencer': ['Influencer', 'Jeffree Star', 'Jstarr', 'Nabela'],
        'promo_launch': ['Product Launch', 'back in-stock', 'Relaunch', 'Restock'],
        'promo_clearance': ['Clearance', 'Clearence', 'End Of Summer'],
        'promo_bogo': ['Buy 1, Get 1', 'BOGO'],
        'promo_tiered': ['Buy More, Save More', 'Spend $20', 'Spend $30', 'Spend $40'],
        'promo_shipping': ['Free Shipping'],
        'promo_early_access': ['Early access', 'loyalty only'],
        'promo_points': ['Double points', 'loyalty']
    }
    PARAM_RANGES = {
        'half_life': [1, 2, 4, 8, 12],
        'penetration': [30, 50, 70, 90],
        'effective_frequency': [3, 6, 9, 12],
        'hill_alpha': [0.5, 1.0, 1.5, 2.0]
    }

class DataValidator:
    def __init__(self, config): 
        self.config = config
    
    def robust_date_parser(self, date_series):
        """Handle multiple date formats automatically"""
        try:
            parsed_dates = pd.to_datetime(date_series, dayfirst=False, errors='coerce')
            if parsed_dates.isnull().any():
                parsed_dates_eu = pd.to_datetime(date_series, dayfirst=True, errors='coerce')
                if parsed_dates_eu.isnull().sum() < parsed_dates.isnull().sum():
                    parsed_dates = parsed_dates_eu
                    st.info("✅ Used dayfirst=True (European format)")
                else:
                    st.info("✅ Used dayfirst=False (US format)")
            else:
                st.info("✅ Auto date parsing successful")
            return parsed_dates
        except Exception as e:
            st.warning(f"⚠️ Auto parsing failed: {e}, using coerce")
            return pd.to_datetime(date_series, errors='coerce')
    
    def validate_data(self, data):
        missing_required = [col for col in self.config.REQUIRED_COLUMNS.keys() if col not in data.columns]
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")
        
        if data['total_revenue'].isnull().any():
            raise ValueError("Target variable 'total_revenue' has missing values")
        
        available_media = [col for col in self.config.OPTIONAL_MEDIA_COLUMNS.keys() if col in data.columns]
        if not available_media:
            raise ValueError("No media variables found in dataset")
        
        st.success(f"✅ Data validated! Found {len(available_media)} media variables")
        return True
    
    def create_features(self, data, frequency='daily'):
        df = data.copy()
        
        df['date'] = self.robust_date_parser(df['date'])
        
        if df['date'].isnull().any():
            null_count = df['date'].isnull().sum()
            st.warning(f"⚠️ Warning: {null_count} dates could not be parsed and were set to NaT")
            df = df.dropna(subset=['date']).reset_index(drop=True)
        
        df = df.sort_values('date').reset_index(drop=True)
        
        if frequency == 'monthly':
            original_rows = len(df)
            df['year_month'] = df['date'].dt.to_period('M')
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'trend' in numeric_cols:
                numeric_cols.remove('trend')
            
            df_monthly = df.groupby('year_month').agg({
                **{col: 'sum' for col in numeric_cols if 'revenue' in col or 'spend' in col or 'sessions' in col},
                **{col: 'mean' for col in numeric_cols if col not in ['total_revenue', 'paid_media_revenue', 'organic_media_revenue'] and any(x in col for x in ['revenue', 'spend', 'sessions'])},
                'date': 'first',
                'holiday': lambda x: x.mode().iloc[0] if not x.mode().empty else 'NH',
                'day_type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Avg day',
                'descriptor': lambda x: '; '.join(x.unique())
            }).reset_index(drop=True)
            
            df = df_monthly
            st.info(f"✅ Aggregated from {original_rows} days to {len(df)} months")
        
        df['trend'] = range(len(df))
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        
        if 'holiday' in df.columns:
            df['is_holiday'] = (df['holiday'] != 'NH').astype(int)
        else:
            df['is_holiday'] = 0
            st.warning("⚠️ 'holiday' column not found, using default")
        
        df['seasonal_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['seasonal_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        if 'day_type' in df.columns:
            day_type_dummies = pd.get_dummies(df['day_type'], prefix='day_type')
            st.info(f"✅ Created {len(day_type_dummies.columns)} day_type features")
        else:
            day_type_dummies = pd.DataFrame(index=df.index)
            st.warning("⚠️ 'day_type' column not found")
        
        promo_dummies = pd.DataFrame(index=df.index)
        if 'descriptor' in df.columns:
            for dummy_name, patterns in self.config.PROMOTION_PATTERNS.items():
                mask = False
                for pattern in patterns:
                    mask |= df['descriptor'].str.contains(pattern, case=False, na=False)
                promo_dummies[dummy_name] = mask.astype(int)
            
            promo_features_created = promo_dummies.sum().sum()
            st.info(f"✅ Created {len(promo_dummies.columns)} promotion features ({promo_features_created} total promotions detected)")
        else:
            st.warning("⚠️ 'descriptor' column not found, skipping promotion features")
        
        result_df = pd.concat([df, day_type_dummies, promo_dummies], axis=1)
        
        st.success(f"✅ Features created: {len(result_df.columns)} total columns")
        return result_df

class MediaTransformer:
    def __init__(self, config): self.config = config
    
    def adstock_transform(self, data, column, half_life):
        if half_life <= 0: 
            return data[column].values
        retention = 0.5 ** (1 / half_life)
        adstock = np.zeros(len(data))
        adstock[0] = data[column].iloc[0]
        for i in range(1, len(data)):
            adstock[i] = data[column].iloc[i] + retention * adstock[i-1]
        return adstock
    
    def hill_saturation(self, adstock, penetration, ef, alpha):
        reach_adstock = adstock * (penetration / 100)
        K = ef
        numerator = reach_adstock ** alpha
        denominator = K ** alpha + numerator
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    
    def transform_media_variables(self, data, media_vars, params_dict):
        transformed_data = pd.DataFrame(index=data.index)
        for media_var in media_vars:
            if media_var in params_dict and media_var in data.columns:
                params = params_dict[media_var]
                adstock = self.adstock_transform(data, media_var, params['half_life'])
                saturation = self.hill_saturation(adstock, params['penetration'], params['effective_frequency'], params['hill_alpha'])
                transformed_data[media_var] = saturation
        return transformed_data

class ParameterEstimator:
    def __init__(self, config): 
        self.config = config
        self.transformer = MediaTransformer(config)
    
    def estimate_parameters(self, data, media_vars, target):
        best_params = {}
        available_media_vars = [var for var in media_vars if var in data.columns]
        
        st.info(f"🔍 Estimating parameters for {len(available_media_vars)} media variables...")
        
        for media_var in available_media_vars:
            st.write(f"  Estimating {media_var}...")
            best_score = float('inf')
            best_channel_params = {}
            
            for hl in self.config.PARAM_RANGES['half_life']:
                for pen in self.config.PARAM_RANGES['penetration']:
                    for ef in self.config.PARAM_RANGES['effective_frequency']:
                        for alpha in self.config.PARAM_RANGES['hill_alpha']:
                            try:
                                adstock = self.transformer.adstock_transform(data, media_var, hl)
                                saturation = self.transformer.hill_saturation(adstock, pen, ef, alpha)
                                
                                if len(np.unique(saturation)) > 1 and np.std(saturation) > 0:
                                    correlation = np.corrcoef(saturation, target)[0, 1]
                                    if not np.isnan(correlation):
                                        score = -abs(correlation)
                                    else:
                                        score = float('inf')
                                else:
                                    score = float('inf')
                                
                                if score < best_score:
                                    best_score = score
                                    best_channel_params = {
                                        'half_life': hl,
                                        'penetration': pen,
                                        'effective_frequency': ef,
                                        'hill_alpha': alpha,
                                        'score': best_score
                                    }
                            except Exception:
                                continue
            
            if best_channel_params:
                best_params[media_var] = best_channel_params
                st.success(f"  ✅ {media_var}: HL={best_channel_params['half_life']}, PEN={best_channel_params['penetration']}, EF={best_channel_params['effective_frequency']}")
            else:
                st.warning(f"  ⚠️ Could not estimate parameters for {media_var}")
        
        return best_params

class TwoStageMMM:
    def __init__(self, config):
        self.config = config
        self.validator = DataValidator(config)
        self.transformer = MediaTransformer(config)
    
    def build_baseline_model(self, data, frequency='daily'):
        df = data
        
        baseline_features = ['trend', 'day_of_week', 'month', 'quarter', 'year', 'is_holiday', 'seasonal_sin', 'seasonal_cos']
        baseline_features += [col for col in df.columns if col.startswith(('day_type_', 'promo_'))]
        
        baseline_features = [f for f in baseline_features if f in df.columns and 'paid_' not in f and 'organic_' not in f]
        
        X_baseline = df[baseline_features].fillna(0)
        y = data['total_revenue']
        
        baseline_model = Ridge(alpha=1.0)
        baseline_model.fit(X_baseline, y)
        baseline_predictions = baseline_model.predict(X_baseline)
        
        return {
            'model': baseline_model,
            'predictions': baseline_predictions,
            'features': baseline_features,
            'r2': r2_score(y, baseline_predictions),
            'mae': mean_absolute_error(y, baseline_predictions)
        }
    
    def build_incremental_model(self, data, baseline_predictions, best_params):
        incremental_revenue = data['total_revenue'] - baseline_predictions
        
        paid_media_vars = [col for col in data.columns if 'paid_' in col and 'net_spend' in col]
        organic_media_vars = [col for col in data.columns if 'organic_' in col and 'sessions' in col]
        media_vars = paid_media_vars + organic_media_vars
        
        media_transformed = self.transformer.transform_media_variables(data, media_vars, best_params).fillna(0)
        
        comp_vars = [col for col in data.columns if col.startswith('comp_')]
        if comp_vars:
            media_transformed = pd.concat([media_transformed, data[comp_vars]], axis=1)
        
        incremental_model = Ridge(alpha=1.0)
        incremental_model.fit(media_transformed, incremental_revenue)
        incremental_predictions = incremental_model.predict(media_transformed)
        
        return {
            'model': incremental_model,
            'predictions': incremental_predictions,
            'media_vars': media_vars,
            'feature_names': list(media_transformed.columns),
            'r2': r2_score(incremental_revenue, incremental_predictions),
            'mae': mean_absolute_error(incremental_revenue, incremental_predictions)
        }

class MMMVisualizer:
    def create_data_quality_dashboard(self, data):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Data Completeness', 'Revenue Distribution', 'Media Spend Trends', 'Channel Correlation'),
            specs=[[{"type": "heatmap"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        completeness = (1 - data.isnull().mean()).to_frame('Completeness')
        fig.add_trace(
            go.Heatmap(
                z=completeness.T.values,
                x=completeness.index,
                y=['Completeness'],
                colorscale=['#FF6B6B', '#4ECDC4'],
                showscale=True,
                hoverinfo='x+z'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=data['total_revenue'], nbinsx=50, marker_color='#1f77b4', opacity=0.7),
            row=1, col=2
        )
        
        paid_cols = [col for col in data.columns if 'paid_' in col and 'net_spend' in col]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, col in enumerate(paid_cols[:3]):
            fig.add_trace(
                go.Scatter(
                    x=data.index if 'date' not in data.columns else data['date'],
                    y=data[col],
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        media_cols = [col for col in data.columns if 'paid_' in col or 'organic_' in col] + ['total_revenue']
        available_media_cols = [col for col in media_cols if col in data.columns]
        
        if len(available_media_cols) > 1:
            corr_data = data[available_media_cols].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_data.values,
                    x=corr_data.columns,
                    y=corr_data.index,
                    colorscale='RdBu',
                    zmid=0
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="📊 Data Quality Dashboard")
        return fig
    
    def plot_model_decomposition(self, data, baseline_results, incremental_results, frequency='daily'):
        if 'date' in data.columns:
            dates = pd.to_datetime(data['date'])
        else:
            dates = data.index
        
        total_predictions = baseline_results['predictions'] + incremental_results['predictions']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=baseline_results['predictions'],
            fill='tozeroy',
            name='Baseline',
            line=dict(color='gray', width=0.5),
            fillcolor='rgba(128, 128, 128, 0.3)'
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=total_predictions,
            fill='tonexty',
            name='Media Contributions',
            line=dict(color='#4ECDC4', width=0.5),
            fillcolor='rgba(78, 205, 196, 0.5)'
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=data['total_revenue'],
            line=dict(color='black', width=2),
            name='Actual Revenue'
        ))
        
        title = f'Revenue Decomposition ({frequency.capitalize()} Model)'
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Revenue',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_media_contributions(self, insights):
        media_contributions = insights.get('media_contributions', {})
        
        if not media_contributions:
            st.warning("No media contributions to plot")
            return go.Figure()
        
        channels = []
        contributions = []
        roas_values = []
        
        for channel, stats in media_contributions.items():
            channels.append(channel)
            contributions.append(stats['contribution'])
            roas_values.append(stats['roas'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=channels,
            y=contributions,
            name='Revenue Contribution',
            marker_color='#4ECDC4',
            text=[f'${c:,.0f}' for c in contributions],
            textposition='auto'
        ))
        
        fig.add_trace(go.Scatter(
            x=channels,
            y=roas_values,
            name='ROAS',
            yaxis='y2',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Media Channel Contributions and ROAS',
            xaxis_title='Media Channels',
            yaxis_title='Revenue Contribution ($)',
            yaxis2=dict(
                title='ROAS',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=500
        )
        
        return fig

class MMMPlatform:
    def __init__(self):
        self.config = MMMConfig()
        self.validator = DataValidator(self.config)
        self.transformer = MediaTransformer(self.config)
        self.estimator = ParameterEstimator(self.config)
        self.model_builder = TwoStageMMM(self.config)
        self.visualizer = MMMVisualizer()
        self.data = None
        self.results = None
    
    def run_complete_analysis(self, data, frequency='daily'):
        try:
            st.header(f"🎯 MMM Analysis ({frequency.upper()})")
            
            st.subheader("🚀 Step 1: Data Loading & Validation")
            self.validator.validate_data(data)
            
            self.data = self.validator.create_features(data, frequency=frequency)
            
            st.subheader("📊 Data Quality Dashboard")
            fig = self.visualizer.create_data_quality_dashboard(self.data)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🔍 Step 2: Media Parameter Estimation")
            media_vars = list(self.config.OPTIONAL_MEDIA_COLUMNS.keys())
            self.best_params = self.estimator.estimate_parameters(self.data, media_vars, self.data['total_revenue'])
            
            st.subheader("🏗️ Step 3: Two-Stage Model Building")
            self.baseline_results = self.model_builder.build_baseline_model(self.data, frequency=frequency)
            self.incremental_results = self.model_builder.build_incremental_model(
                self.data, self.baseline_results['predictions'], self.best_params
            )
            
            st.subheader("📈 Model Decomposition")
            fig2 = self.visualizer.plot_model_decomposition(self.data, self.baseline_results, self.incremental_results, frequency)
            st.plotly_chart(fig2, use_container_width=True)
            
            st.subheader("💡 Step 4: Business Insights")
            total_predictions = self.baseline_results['predictions'] + self.incremental_results['predictions']
            total_r2 = r2_score(self.data['total_revenue'], total_predictions)
            
            insights = {
                'frequency': frequency,
                'total_r2': total_r2,
                'baseline_r2': self.baseline_results['r2'],
                'incremental_r2': self.incremental_results['r2'],
                'media_contributions': {},
                'parameter_summary': self.best_params,
                'model_details': {
                    'baseline_features': self.baseline_results['features'],
                    'media_features': self.incremental_results['feature_names']
                }
            }
            
            for media_var in self.incremental_results['media_vars']:
                if media_var in self.best_params and media_var in self.incremental_results['feature_names']:
                    try:
                        coef_idx = self.incremental_results['feature_names'].index(media_var)
                        coef = self.incremental_results['model'].coef_[coef_idx]
                        avg_spend = self.data[media_var].mean() if media_var in self.data.columns else 0
                        contribution = coef * avg_spend if avg_spend > 0 else 0
                        
                        insights['media_contributions'][media_var] = {
                            'coefficient': coef,
                            'roas': coef,
                            'avg_spend': avg_spend,
                            'contribution': contribution,
                            'half_life': self.best_params[media_var]['half_life'],
                            'saturation_level': self.best_params[media_var]['effective_frequency']
                        }
                    except Exception as e:
                        st.warning(f"⚠️ Could not calculate contribution for {media_var}: {e}")
            
            self._display_results(insights, frequency)
            
            st.subheader("💰 Media Channel Performance")
            fig3 = self.visualizer.plot_media_contributions(insights)
            st.plotly_chart(fig3, use_container_width=True)
            
            self.results = insights
            return insights
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            raise
    
    def _display_results(self, insights, frequency):
        st.header("📊 Final Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Frequency", frequency.upper())
        with col2:
            st.metric("Total R²", f"{insights['total_r2']:.3f}")
        with col3:
            st.metric("Baseline R²", f"{insights['baseline_r2']:.3f}")
        
        st.subheader("💰 Media Channel Performance")
        
        if insights['media_contributions']:
            sorted_media = sorted(
                insights['media_contributions'].items(),
                key=lambda x: abs(x[1]['contribution']),
                reverse=True
            )
            
            for media_var, stats in sorted_media:
                with st.expander(f"📈 {media_var}"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("ROAS", f"{stats['roas']:.2f}")
                    with col2:
                        st.metric("Avg Spend", f"${stats['avg_spend']:,.0f}")
                    with col3:
                        st.metric("Contribution", f"${stats['contribution']:,.0f}")
                    with col4:
                        st.metric("Half-life", f"{stats['half_life']} weeks")
        else:
            st.warning("No media contributions calculated")
        
        st.success("🎉 MMM ANALYSIS COMPLETE!")

def generate_sample_data(days=100):
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    
    data = pd.DataFrame({
        'date': dates,
        'total_revenue': np.random.normal(50000, 15000, days).clip(1000),
        'holiday': 'NH',
        'day_type': 'Avg day',
        'descriptor': 'Avg day'
    })
    
    holiday_indices = np.random.choice(days, size=10, replace=False)
    data.loc[holiday_indices, 'holiday'] = 'Holiday'
    
    promo_data = [
        ('Promo day- mid', '50% off sitewide', 5),
        ('Promo day- high', '50% off everything', 3),
        ('Promo day- low', '30% off select items', 8),
        ('Promo day- mid', '30% off entire collection', 4),
        ('Promo day- low', 'Free GWP with $25 purchase', 6),
        ('Sale', 'Clearance event - extra 25% off', 4),
        ('Promo day- low', 'Buy 1 Get 1 30% off', 5),
        ('Early access', 'Early access for loyalty members', 3),
        ('Influencer', 'Special influencer collaboration', 3),
        ('Promo day- mid', 'Tiered promotion: Spend $20 get $5 off', 4),
        ('Promo day- low', 'New product launch', 4),
        ('Promo day- low', 'Limited edition restock', 3),
        ('Promo day- mid', 'Free shipping on all orders', 5),
        ('Promo day- low', 'Double loyalty points', 4)
    ]
    
    promo_indices_used = set()
    for day_type, descriptor, count in promo_data:
        available_days = [i for i in range(days) if i not in promo_indices_used]
        if len(available_days) >= count:
            chosen_indices = np.random.choice(available_days, size=count, replace=False)
            data.loc[chosen_indices, 'day_type'] = day_type
            data.loc[chosen_indices, 'descriptor'] = descriptor
            promo_indices_used.update(chosen_indices)
    
    paid_media_vars = {
        'paid_search_net_spend': (1000, 300),
        'paid_social_net_spend': (800, 200),
        'paid_display_net_spend': (600, 150),
        'paid_shopping_net_spend': (400, 100),
        'paid_affiliate_net_spend': (200, 50)
    }
    
    for var, (mean, std) in paid_media_vars.items():
        data[var] = np.random.normal(mean, std, days).clip(0)
    
    organic_media_vars = {
        'organic_search_sessions': (5000, 1000),
        'organic_social_sessions': (2000, 500),
        'organic_direct_sessions': (3000, 800),
        'organic_email_sessions': (1000, 300),
        'organic_referral_sessions': (800, 200)
    }
    
    for var, (mean, std) in organic_media_vars.items():
        data[var] = np.random.normal(mean, std, days).clip(0)
    
    return data

# Streamlit App
def main():
    st.set_page_config(
        page_title="MMM Modeling Platform",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 MMM Modeling Platform")
    st.markdown("""
    Complete Marketing Mix Modeling platform for analyzing media effectiveness and revenue contribution.
    Upload your data or use sample data to get started!
    """)
    
    if 'mmm' not in st.session_state:
        st.session_state.mmm = MMMPlatform()
    
    # Sidebar
    st.sidebar.header("📁 Data Input")
    
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV", "Use Sample Data"]
    )
    
    frequency = st.sidebar.selectbox(
        "Analysis Frequency:",
        ["daily", "monthly"],
        help="Choose between daily granularity or monthly aggregated analysis"
    )
    
    data = None
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Upload a CSV file with your marketing data"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ Loaded {len(data)} rows, {len(data.columns)} columns")
                
                with st.sidebar.expander("📋 Data Preview"):
                    st.dataframe(data.head(), use_container_width=True)
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {e}")
    
    else:  # Sample Data
        if st.sidebar.button("Generate Sample Data"):
            data = generate_sample_data(100)
            st.session_state.sample_data = data
            st.sidebar.success("✅ Sample data generated!")
        
        if 'sample_data' in st.session_state:
            data = st.session_state.sample_data
            with st.sidebar.expander("📋 Sample Data Preview"):
                st.dataframe(data.head(), use_container_width=True)
    
    # Main content
    if data is not None:
        st.header("🚀 Ready to Analyze!")
        st.write(f"**Data Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
        st.write(f"**Selected Frequency:** {frequency.upper()}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Run Daily Analysis", type="primary", use_container_width=True):
                with st.spinner("Running daily analysis... This may take a few minutes"):
                    st.session_state.results = st.session_state.mmm.run_complete_analysis(data, frequency='daily')
        
        with col2:
            if st.button("📈 Run Monthly Analysis", type="secondary", use_container_width=True):
                with st.spinner("Running monthly analysis... This may take a few minutes"):
                    st.session_state.results = st.session_state.mmm.run_complete_analysis(data, frequency='monthly')
    
    else:
        st.info("👆 Please upload your data or generate sample data to get started!")
        
        # Data requirements
        with st.expander("📋 Data Requirements & Examples", expanded=True):
            st.markdown("""
            ### 📅 Required Columns:
            - **date**: Date column (any format: 2023-01-15, 1/15/2023, 15/1/2023)
            - **total_revenue**: Daily revenue in dollars ($12,456)
            
            ### 📊 Optional Media Columns (use what you have):
            - **paid_search_net_spend**: Paid search spend ($1,234)
            - **paid_social_net_spend**: Paid social media spend ($987)
            - **paid_display_net_spend**: Display advertising spend ($654)
            - **organic_search_sessions**: Organic search traffic (1,234 sessions)
            - **organic_social_sessions**: Organic social traffic (567 sessions)
            - *...and other paid/organic media variables*
            
            ### 🎯 Optional Promotion Columns (for better modeling):
            - **day_type**: Promotion intensity (Avg day, Promo day- low, Promo day- mid, Promo day- high, Sale, etc.)
            - **descriptor**: Promotion details ("50% off sitewide", "Free GWP with $25 purchase", "New product launch")
            - **holiday**: Holiday indicators (NH, Holiday, etc.)
            
            ### 💡 Example Data Structure:
            ```
            date | total_revenue | paid_search_net_spend | day_type | descriptor
            2023-01-01 | 45234 | 1245 | Avg day | Regular day
            2023-01-02 | 67890 | 1567 | Promo day- mid | 30% off select items
            2023-01-03 | 89234 | 1987 | Promo day- high | 50% off everything
            ```
            
            *Note: Analysis will run with whatever columns are available! More data = better insights.*
            """)

if __name__ == "__main__":
    main()
