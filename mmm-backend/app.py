import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Marketing Mix Model Analyzer",
    page_icon="📊",
    layout="wide"
)

class MMModel:
    def __init__(self):
        self.media_channels = [
            'paid_search_net_spend', 'paid_shopping_net_spend', 
            'paid_display_net_spend', 'paid_social_net_spend', 
            'paid_affiliate_net_spend'
        ]
        
    def prepare_data(self, df):
        """Prepare and validate data for modeling"""
        try:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.sort_values('date').reset_index(drop=True)
            
            # Check for required columns
            missing_columns = [col for col in self.media_channels if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            y = df['total_revenue'].values
            media_data = np.column_stack([df[channel].values for channel in self.media_channels])
            
            # Handle missing values
            media_data = np.nan_to_num(media_data)
            
            return y, media_data, df
            
        except Exception as e:
            logger.error(f"Data preparation error: {str(e)}")
            raise
    
    def calculate_roas(self, model, media_data):
        """Calculate ROAS for each channel"""
        try:
            roas_results = []
            current_allocation = np.mean(media_data, axis=0)
            
            for i, channel in enumerate(self.media_channels):
                roas = model.coef_[i] * current_allocation[i]
                roas_results.append({
                    'channel': channel,
                    'roas_mean': max(roas, 0.1),  # Minimum ROAS of 0.1
                    'roas_std': abs(roas * 0.1)
                })
            
            return roas_results, current_allocation
            
        except Exception as e:
            logger.error(f"ROAS calculation error: {str(e)}")
            raise
    
    def optimize_budget(self, roas_results, current_allocation, total_budget):
        """Optimize budget allocation based on ROAS"""
        try:
            sorted_channels = sorted(roas_results, key=lambda x: x['roas_mean'], reverse=True)
            roas_values = [channel['roas_mean'] for channel in sorted_channels]
            total_roas = sum(roas_values)
            
            # Avoid division by zero
            if total_roas == 0:
                total_roas = 1
                
            optimal_weights = [roas / total_roas for roas in roas_values]
            optimized_allocation = [weight * total_budget for weight in optimal_weights]
            
            budget_recommendations = []
            for i, channel in enumerate(sorted_channels):
                current = current_allocation[i]
                optimized = optimized_allocation[i]
                
                # Handle division by zero for current allocation
                if current > 0:
                    change_pct = ((optimized - current) / current) * 100
                else:
                    change_pct = 100 if optimized > 0 else 0
                
                budget_recommendations.append({
                    'channel': channel['channel'],
                    'current_allocation': round(current, 2),
                    'optimized_allocation': round(optimized, 2),
                    'change_percentage': round(change_pct, 1),
                    'roas': round(channel['roas_mean'], 2)
                })
            
            return budget_recommendations
            
        except Exception as e:
            logger.error(f"Budget optimization error: {str(e)}")
            raise

def main():
    st.title("📊 Marketing Mix Model Analyzer")
    st.markdown("Upload your marketing data to analyze ROAS and get budget optimization recommendations")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read and validate data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded data with {len(df)} rows")
            
            # Show data preview
            with st.expander("Data Preview"):
                st.dataframe(df.head())
                st.write(f"Data shape: {df.shape}")
            
            # Initialize model
            mmm_model = MMModel()
            
            # Check required columns
            required_columns = ['total_revenue'] + mmm_model.media_channels
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
                st.info("Please ensure your CSV contains all required columns")
                return
            
            # Process data
            with st.spinner("Analyzing your marketing data..."):
                y, media_data, processed_df = mmm_model.prepare_data(df)
                
                # Use Ridge regression for stability
                model = Ridge(alpha=1.0)
                X = media_data
                model.fit(X, y)
                
                current_allocation = np.mean(media_data, axis=0)
                total_budget = np.sum(current_allocation)
                
                roas_results, current_allocation = mmm_model.calculate_roas(model, media_data)
                budget_recommendations = mmm_model.optimize_budget(roas_results, current_allocation, total_budget)
                
                top_channel = max(roas_results, key=lambda x: x['roas_mean'])
                worst_channel = min(roas_results, key=lambda x: x['roas_mean'])
            
            # Display results
            st.header("📈 Executive Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Top Performing Channel", 
                    top_channel['channel'].replace('_', ' ').title(),
                    f"ROAS: {top_channel['roas_mean']:.2f}"
                )
            with col2:
                st.metric(
                    "Worst Performing Channel", 
                    worst_channel['channel'].replace('_', ' ').title(),
                    f"ROAS: {worst_channel['roas_mean']:.2f}"
                )
            with col3:
                st.metric("R² Score", f"{model.score(X, y):.3f}")
            with col4:
                mape = mean_absolute_percentage_error(y, model.predict(X))
                st.metric("MAPE", f"{mape:.3f}")
            
            # ROAS Analysis
            st.header("💰 ROAS Analysis")
            roas_df = pd.DataFrame(roas_results)
            roas_df['channel'] = roas_df['channel'].str.replace('_', ' ').str.title()
            roas_df = roas_df.rename(columns={
                'channel': 'Channel',
                'roas_mean': 'ROAS',
                'roas_std': 'Std Dev'
            })
            st.dataframe(roas_df[['Channel', 'ROAS', 'Std Dev']])
            
            # Budget Optimization
            st.header("🎯 Budget Optimization Recommendations")
            budget_df = pd.DataFrame(budget_recommendations)
            budget_df['channel'] = budget_df['channel'].str.replace('_', ' ').str.title()
            budget_df = budget_df.rename(columns={
                'channel': 'Channel',
                'current_allocation': 'Current ($)',
                'optimized_allocation': 'Recommended ($)',
                'change_percentage': 'Change %',
                'roas': 'ROAS'
            })
            
            # Color code the change percentage
            def color_change(val):
                if val > 5:
                    return 'color: green; font-weight: bold'
                elif val < -5:
                    return 'color: red; font-weight: bold'
                else:
                    return 'color: gray'
            
            styled_df = budget_df.style.format({
                'Current ($)': '${:,.2f}',
                'Recommended ($)': '${:,.2f}',
                'Change %': '{:.1f}%',
                'ROAS': '{:.2f}'
            }).applymap(color_change, subset=['Change %'])
            
            st.dataframe(styled_df)
            
            # Visualization
            st.header("📊 Channel Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # ROAS Comparison
                st.subheader("ROAS by Channel")
                roas_chart_df = roas_df[['Channel', 'ROAS']].set_index('Channel')
                st.bar_chart(roas_chart_df)
            
            with col2:
                # Budget Allocation
                st.subheader("Budget Allocation Comparison")
                allocation_df = budget_df[['Channel', 'Current ($)', 'Recommended ($)']].set_index('Channel')
                st.bar_chart(allocation_df)
                
            # Download recommendations
            st.header("📥 Export Results")
            recommendations_csv = budget_df.to_csv(index=False)
            st.download_button(
                label="Download Budget Recommendations as CSV",
                data=recommendations_csv,
                file_name="budget_recommendations.csv",
                mime="text/csv"
            )
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            st.info("Please check your data format and try again.")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Sample data format
        st.subheader("Expected CSV Format")
        sample_data = {
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'total_revenue': [10000, 12000, 11000],
            'paid_search_net_spend': [1000, 1200, 1100],
            'paid_shopping_net_spend': [500, 600, 550],
            'paid_display_net_spend': [300, 350, 325],
            'paid_social_net_spend': [400, 450, 425],
            'paid_affiliate_net_spend': [200, 250, 225]
        }
        st.dataframe(pd.DataFrame(sample_data))
        
        st.info("💡 **Tips for best results:**")
        st.markdown("""
        - Ensure all spend columns are in the same currency
        - Include at least 4-6 weeks of data for reliable analysis
        - Check for missing values in your data
        - Ensure revenue and spend figures are accurate
        """)

if __name__ == "__main__":
    main()
