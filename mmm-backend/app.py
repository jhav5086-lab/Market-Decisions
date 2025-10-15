# ====================
# COMPLETE MMM MODELING PLATFORM WITH UPLOAD FUNCTIONALITY
# ====================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output
import io
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
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
    # Realistic promotion patterns based on actual data
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
    DAY_TYPES = [
        'Avg day', 'Promo day- low', 'Promo day- mid', 'Promo day- high', 
        'Promo day- super high', 'Sale', 'Early access', 'Influencer', 
        'Collab', 'FSNM', 'Product launch'
    ]
    PARAM_RANGES = {
        'half_life': [1, 2, 4, 8, 12],
        'penetration': [30, 50, 70, 90],
        'effective_frequency': [3, 6, 9, 12],
        'hill_alpha': [0.5, 1.0, 1.5, 2.0]
    }
    HOLDOUT_PERCENT = 0.2

class DataValidator:
    def __init__(self, config): 
        self.config = config
    
    def robust_date_parser(self, date_series):
        """Handle multiple date formats automatically"""
        # Try pandas auto-detection first with dayfirst for European formats
        try:
            parsed_dates = pd.to_datetime(date_series, dayfirst=False, errors='coerce')
            # Check if any dates couldn't be parsed
            if parsed_dates.isnull().any():
                # Try with dayfirst=True for European formats
                parsed_dates_eu = pd.to_datetime(date_series, dayfirst=True, errors='coerce')
                # Use whichever method parsed more dates successfully
                if parsed_dates_eu.isnull().sum() < parsed_dates.isnull().sum():
                    parsed_dates = parsed_dates_eu
                    print("✅ Used dayfirst=True (European format)")
                else:
                    print("✅ Used dayfirst=False (US format)")
            else:
                print("✅ Auto date parsing successful")
            return parsed_dates
        except Exception as e:
            print(f"⚠️  Auto parsing failed: {e}, using coerce")
            return pd.to_datetime(date_series, errors='coerce')
    
    def validate_data(self, data):
        # Check required columns
        missing_required = [col for col in self.config.REQUIRED_COLUMNS.keys() if col not in data.columns]
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")
        
        # Check for target variable issues
        if data['total_revenue'].isnull().any():
            raise ValueError("Target variable 'total_revenue' has missing values")
        
        # Check available media columns
        available_media = [col for col in self.config.OPTIONAL_MEDIA_COLUMNS.keys() if col in data.columns]
        if not available_media:
            raise ValueError("No media variables found in dataset")
        
        print(f"✅ Data validated! Found {len(available_media)} media variables")
        return True
    
    def create_features(self, data, frequency='daily'):
        df = data.copy()
        
        # Robust date parsing
        df['date'] = self.robust_date_parser(df['date'])
        
        # Check for any failed date parses
        if df['date'].isnull().any():
            null_count = df['date'].isnull().sum()
            print(f"⚠️  Warning: {null_count} dates could not be parsed and were set to NaT")
            df = df.dropna(subset=['date']).reset_index(drop=True)
        
        # Sort by date to ensure time series order
        df = df.sort_values('date').reset_index(drop=True)
        
        # Handle frequency aggregation
        if frequency == 'monthly':
            original_rows = len(df)
            # Create month-year identifier
            df['year_month'] = df['date'].dt.to_period('M')
            
            # Aggregate numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'trend' in numeric_cols:
                numeric_cols.remove('trend')
            
            # Group by month and aggregate
            df_monthly = df.groupby('year_month').agg({
                **{col: 'sum' for col in numeric_cols if 'revenue' in col or 'spend' in col or 'sessions' in col},
                **{col: 'mean' for col in numeric_cols if col not in ['total_revenue', 'paid_media_revenue', 'organic_media_revenue'] and any(x in col for x in ['revenue', 'spend', 'sessions'])},
                'date': 'first',
                'holiday': lambda x: x.mode().iloc[0] if not x.mode().empty else 'NH',
                'day_type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Avg day',
                'descriptor': lambda x: '; '.join(x.unique())
            }).reset_index(drop=True)
            
            df = df_monthly
            print(f"✅ Aggregated from {original_rows} days to {len(df)} months")
        
        # Create time-based features
        df['trend'] = range(len(df))
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        
        # Holiday indicator (handle missing holiday column)
        if 'holiday' in df.columns:
            df['is_holiday'] = (df['holiday'] != 'NH').astype(int)
        else:
            df['is_holiday'] = 0
            print("⚠️  'holiday' column not found, using default")
        
        # Seasonal features
        df['seasonal_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['seasonal_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Handle categorical variables - day_type dummies
        if 'day_type' in df.columns:
            # Create dummies for all day types found in data
            day_type_dummies = pd.get_dummies(df['day_type'], prefix='day_type')
            print(f"✅ Created {len(day_type_dummies.columns)} day_type features")
        else:
            day_type_dummies = pd.DataFrame(index=df.index)
            print("⚠️  'day_type' column not found")
        
        # Promotion patterns from descriptor (using realistic patterns)
        promo_dummies = pd.DataFrame(index=df.index)
        if 'descriptor' in df.columns:
            for dummy_name, patterns in self.config.PROMOTION_PATTERNS.items():
                mask = False
                for pattern in patterns:
                    # Handle NaN values in descriptor
                    mask |= df['descriptor'].str.contains(pattern, case=False, na=False)
                promo_dummies[dummy_name] = mask.astype(int)
            
            # Count how many promotion features were created
            promo_features_created = promo_dummies.sum().sum()
            print(f"✅ Created {len(promo_dummies.columns)} promotion features ({promo_features_created} total promotions detected)")
        else:
            print("⚠️  'descriptor' column not found, skipping promotion features")
        
        # Combine all features
        result_df = pd.concat([df, day_type_dummies, promo_dummies], axis=1)
        
        print(f"✅ Features created: {len(result_df.columns)} total columns")
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
        
        print(f"🔍 Estimating parameters for {len(available_media_vars)} media variables...")
        
        for media_var in available_media_vars:
            print(f"  Estimating {media_var}...")
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
                            except Exception as e:
                                continue
            
            if best_channel_params:
                best_params[media_var] = best_channel_params
                print(f"  ✅ {media_var}: HL={best_channel_params['half_life']}, PEN={best_channel_params['penetration']}, EF={best_channel_params['effective_frequency']}")
            else:
                print(f"  ⚠️  Could not estimate parameters for {media_var}")
        
        return best_params

class TwoStageMMM:
    def __init__(self, config):
        self.config = config
        self.validator = DataValidator(config)
        self.transformer = MediaTransformer(config)
    
    def build_baseline_model(self, data, frequency='daily'):
        # Use the data that already has features created
        df = data
        
        # Define baseline features (excluding media variables)
        baseline_features = ['trend', 'day_of_week', 'month', 'quarter', 'year', 'is_holiday', 'seasonal_sin', 'seasonal_cos']
        baseline_features += [col for col in df.columns if col.startswith(('day_type_', 'promo_'))]
        
        # Remove any media-related features
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
        
        # Find available media variables
        paid_media_vars = [col for col in data.columns if 'paid_' in col and 'net_spend' in col]
        organic_media_vars = [col for col in data.columns if 'organic_' in col and 'sessions' in col]
        media_vars = paid_media_vars + organic_media_vars
        
        media_transformed = self.transformer.transform_media_variables(data, media_vars, best_params).fillna(0)
        
        # Add competitor variables if available
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
        
        # Data completeness heatmap
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
        
        # Revenue distribution
        fig.add_trace(
            go.Histogram(x=data['total_revenue'], nbinsx=50, marker_color='#1f77b4', opacity=0.7),
            row=1, col=2
        )
        
        # Media spend trends (top 3 paid media channels)
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
        
        # Channel correlation heatmap
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
        
        # Baseline area
        fig.add_trace(go.Scatter(
            x=dates, y=baseline_results['predictions'],
            fill='tozeroy',
            name='Baseline',
            line=dict(color='gray', width=0.5),
            fillcolor='rgba(128, 128, 128, 0.3)'
        ))
        
        # Media contributions area
        fig.add_trace(go.Scatter(
            x=dates, y=total_predictions,
            fill='tonexty',
            name='Media Contributions',
            line=dict(color='#4ECDC4', width=0.5),
            fillcolor='rgba(78, 205, 196, 0.5)'
        ))
        
        # Actual revenue line
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
            print("No media contributions to plot")
            return go.Figure()
        
        # Prepare data for plotting
        channels = []
        contributions = []
        roas_values = []
        
        for channel, stats in media_contributions.items():
            channels.append(channel)
            contributions.append(stats['contribution'])
            roas_values.append(stats['roas'])
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=channels,
            y=contributions,
            name='Revenue Contribution',
            marker_color='#4ECDC4',
            text=[f'${c:,.0f}' for c in contributions],
            textposition='auto'
        ))
        
        # Add ROAS as line
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
        print("🎯 MMM Platform Initialized!")
        
    def run_complete_analysis(self, data, frequency='daily'):  # FIXED: Added frequency parameter
        try:
            print(f"🎯 STARTING COMPLETE MMM ANALYSIS ({frequency.upper()})")
            print("=" * 50)
            
            # Step 1: Data loading and validation
            print("🚀 STEP 1: Data Loading & Validation")
            self.validator.validate_data(data)
            
            # Create features with specified frequency
            self.data = self.validator.create_features(data, frequency=frequency)
            
            # Show data quality dashboard
            fig = self.visualizer.create_data_quality_dashboard(self.data)
            fig.show()
            
            # Step 2: Parameter estimation
            print("🔍 STEP 2: Media Parameter Estimation")
            media_vars = list(self.config.OPTIONAL_MEDIA_COLUMNS.keys())
            self.best_params = self.estimator.estimate_parameters(self.data, media_vars, self.data['total_revenue'])
            
            # Step 3: Model building
            print("🏗️ STEP 3: Two-Stage Model Building")
            self.baseline_results = self.model_builder.build_baseline_model(self.data, frequency=frequency)
            self.incremental_results = self.model_builder.build_incremental_model(
                self.data, self.baseline_results['predictions'], self.best_params
            )
            
            # Show model decomposition
            fig2 = self.visualizer.plot_model_decomposition(self.data, self.baseline_results, self.incremental_results, frequency)
            fig2.show()
            
            # Step 4: Insights generation
            print("💡 STEP 4: Generating Business Insights")
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
            
            # Calculate media contributions
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
                        print(f"⚠️  Could not calculate contribution for {media_var}: {e}")
            
            # Display results
            self._display_results(insights, frequency)
            
            # Show media contributions chart
            fig3 = self.visualizer.plot_media_contributions(insights)
            fig3.show()
            
            self.results = insights
            return insights
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            raise
    
    def _display_results(self, insights, frequency):
        print("📊 FINAL RESULTS:")
        print("=" * 50)
        print(f"Model Frequency: {frequency.upper()}")
        print(f"Total Model R²: {insights['total_r2']:.3f}")
        print(f"Baseline R²: {insights['baseline_r2']:.3f}")
        print(f"Incremental R²: {insights['incremental_r2']:.3f}")
        
        print("\n💰 MEDIA CHANNEL PERFORMANCE:")
        print("-" * 40)
        
        if insights['media_contributions']:
            sorted_media = sorted(
                insights['media_contributions'].items(),
                key=lambda x: abs(x[1]['contribution']),
                reverse=True
            )
            
            for media_var, stats in sorted_media:
                print(f"  📈 {media_var}:")
                print(f"     ROAS: {stats['roas']:.2f}")
                print(f"     Avg Spend: ${stats['avg_spend']:,.0f}")
                print(f"     Contribution: ${stats['contribution']:,.0f}")
                print(f"     Half-life: {stats['half_life']} weeks")
                print()
        else:
            print("  No media contributions calculated")
        
        print("=" * 50)
        print("🎉 MMM ANALYSIS COMPLETE!")

def generate_sample_data(days=100):
    """Generate realistic sample data with proper promotion patterns"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    
    # Base dataframe with essential columns
    data = pd.DataFrame({
        'date': dates,
        'total_revenue': np.random.normal(50000, 15000, days).clip(1000),
        'holiday': 'NH',
        'day_type': 'Avg day',
        'descriptor': 'Avg day'
    })
    
    # Add realistic holidays
    holiday_indices = np.random.choice(days, size=10, replace=False)
    data.loc[holiday_indices, 'holiday'] = 'Holiday'
    
    # Add realistic promotions based on actual patterns
    promo_data = [
        # (day_type, descriptor, count)
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
    
    # Paid media variables (net spend)
    paid_media_vars = {
        'paid_search_net_spend': (1000, 300),
        'paid_social_net_spend': (800, 200),
        'paid_display_net_spend': (600, 150),
        'paid_shopping_net_spend': (400, 100),
        'paid_affiliate_net_spend': (200, 50)
    }
    
    for var, (mean, std) in paid_media_vars.items():
        data[var] = np.random.normal(mean, std, days).clip(0)
    
    # Organic media variables (sessions)
    organic_media_vars = {
        'organic_search_sessions': (5000, 1000),
        'organic_social_sessions': (2000, 500),
        'organic_direct_sessions': (3000, 800),
        'organic_email_sessions': (1000, 300),
        'organic_referral_sessions': (800, 200)
    }
    
    for var, (mean, std) in organic_media_vars.items():
        data[var] = np.random.normal(mean, std, days).clip(0)
    
    print(f"✅ Sample data generated with {len(data)} days")
    print("📊 Sample variables created:")
    print("   - date, total_revenue, holiday, day_type, descriptor")
    print("   - 5 paid media net spend variables")
    print("   - 5 organic media sessions variables")
    print("   - Realistic promotion patterns including:")
    print("     * 50% off, 30% off, 25% off promotions")
    print("     * GWP (gift with purchase) offers")
    print("     * Influencer collaborations")
    print("     * Product launches and restocks")
    print("     * Tiered promotions and early access")
    
    return data

# ====================
# ENHANCED UI WITH FREQUENCY OPTIONS
# ====================
def create_mmm_ui():
    """Create the complete MMM UI with upload functionality and frequency options"""
    
    # Initialize platform
    mmm = MMMPlatform()
    
    # Create widgets
    upload_btn = widgets.FileUpload(
        description='📁 Upload CSV',
        multiple=False,
        accept='.csv',
        style={'button_color': '#4ECDC4'}
    )
    
    sample_btn = widgets.Button(
        description='🎲 Use Sample Data',
        style={'button_color': '#FF6B6B'}
    )
    
    frequency_dropdown = widgets.Dropdown(
        options=['daily', 'monthly'],
        value='daily',
        description='Frequency:',
        style={'description_width': 'initial'}
    )
    
    run_daily_btn = widgets.Button(
        description='📊 Run Daily Model',
        style={'button_color': '#45B7D1'},
        disabled=True
    )
    
    run_monthly_btn = widgets.Button(
        description='📈 Run Monthly Model', 
        style={'button_color': '#96CEB4'},
        disabled=True
    )
    
    output = widgets.Output()
    
    # Enhanced data requirements display with realistic examples
    requirements_html = """
    <div style="background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0;">
    <h4>📋 Data Requirements & Examples:</h4>
    
    <p><strong>📅 Required Columns:</strong></p>
    <ul>
    <li><b>date</b>: Date column (any format: 2023-01-15, 1/15/2023, 15/1/2023)</li>
    <li><b>total_revenue</b>: Daily revenue in dollars ($12,456)</li>
    </ul>
    
    <p><strong>📊 Optional Media Columns (use what you have):</strong></p>
    <ul>
    <li><b>paid_search_net_spend</b>: Paid search spend ($1,234)</li>
    <li><b>paid_social_net_spend</b>: Paid social media spend ($987)</li>
    <li><b>paid_display_net_spend</b>: Display advertising spend ($654)</li>
    <li><b>organic_search_sessions</b>: Organic search traffic (1,234 sessions)</li>
    <li><b>organic_social_sessions</b>: Organic social traffic (567 sessions)</li>
    <li><em>...and other paid/organic media variables</em></li>
    </ul>
    
    <p><strong>🎯 Optional Promotion Columns (for better modeling):</strong></p>
    <ul>
    <li><b>day_type</b>: Promotion intensity (Avg day, Promo day- low, Promo day- mid, Promo day- high, Sale, etc.)</li>
    <li><b>descriptor</b>: Promotion details ("50% off sitewide", "Free GWP with $25 purchase", "New product launch")</li>
    <li><b>holiday</b>: Holiday indicators (NH, Holiday, etc.)</li>
    </ul>
    
    <div style="background: #e8f4f8; padding: 10px; border-radius: 3px; margin: 10px 0;">
    <strong>💡 Example Data Structure:</strong><br>
    <small>
    date | total_revenue | paid_search_net_spend | day_type | descriptor<br>
    2023-01-01 | 45234 | 1245 | Avg day | Regular day<br>
    2023-01-02 | 67890 | 1567 | Promo day- mid | 30% off select items<br>
    2023-01-03 | 89234 | 1987 | Promo day- high | 50% off everything<br>
    </small>
    </div>
    
    <p><em>Note: Analysis will run with whatever columns are available! More data = better insights.</em></p>
    </div>
    """
    
    requirements = widgets.HTML(requirements_html)
    
    # Status display
    status = widgets.HTML("<div style='padding: 10px; background: #e8f4fd; border-radius: 5px;'>📝 Ready to upload data</div>")
    
    # Event handlers
    def on_upload_change(change):
        if upload_btn.value:
            run_daily_btn.disabled = False
            run_monthly_btn.disabled = False
            status.value = "<div style='padding: 10px; background: #e8f4f8; border-radius: 5px;'>✅ CSV file uploaded - Ready to analyze!</div>"
    
    def on_sample_click(b):
        with output:
            clear_output()
            print("🎲 Generating realistic sample data...")
            sample_data = generate_sample_data(100)
            mmm.data = sample_data
            run_daily_btn.disabled = False
            run_monthly_btn.disabled = False
            status.value = "<div style='padding: 10px; background: #e8f4f8; border-radius: 5px;'>✅ Sample data generated - Ready to analyze!</div>"
    
    def run_analysis(frequency):
        with output:
            clear_output()
            try:
                if upload_btn.value:
                    # Handle uploaded file
                    uploaded_file = list(upload_btn.value.values())[0]
                    content = uploaded_file['content']
                    data = pd.read_csv(io.BytesIO(content))
                    print(f"📁 Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
                else:
                    # Use sample data
                    data = mmm.data
                    print(f"🎲 Using sample data: {data.shape[0]} rows, {data.shape[1]} columns")
                
                # Run analysis
                status.value = f"<div style='padding: 10px; background: #fff3cd; border-radius: 5px;'>🔄 Running {frequency} analysis... This may take a few minutes</div>"
                
                results = mmm.run_complete_analysis(data, frequency=frequency)  # FIXED: Now passes frequency parameter
                
                # Show success status
                status.value = f"""
                <div style='padding: 10px; background: #d4edda; border-radius: 5px;'>
                ✅ {frequency.capitalize()} Analysis Complete! Model R²: {results['total_r2']:.3f}
                </div>
                """
                
            except Exception as e:
                status.value = f"""
                <div style='padding: 10px; background: #f8d7da; border-radius: 5px;'>
                ❌ Error in {frequency} analysis: {str(e)}
                </div>
                """
                print(f"Error details: {str(e)}")
    
    def on_run_daily_click(b):
        run_analysis('daily')
    
    def on_run_monthly_click(b):
        run_analysis('monthly')
    
    # Attach event handlers
    upload_btn.observe(on_upload_change, names='value')
    sample_btn.on_click(on_sample_click)
    run_daily_btn.on_click(on_run_daily_click)
    run_monthly_btn.on_click(on_run_monthly_click)
    
    # Create layout
    header = widgets.HTML("<h2>🎯 MMM Modeling Platform</h2>")
    upload_row = widgets.HBox([upload_btn, sample_btn])
    frequency_row = widgets.HBox([frequency_dropdown])
    run_row = widgets.HBox([run_daily_btn, run_monthly_btn])
    
    # Display UI
    display(header)
    display(requirements)
    display(upload_row)
    display(frequency_row) 
    display(run_row)
    display(status)
    display(output)
    
    return mmm

# ====================
# LAUNCH THE UI
# ====================
print("🚀 Launching Enhanced MMM Platform with Realistic Promotion Patterns...")
mmm_ui = create_mmm_ui()
