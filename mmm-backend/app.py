import streamlit as st
import pandas as pd
import numpy as np

def model_configuration_homepage():
    """Model Configuration Homepage - Wireframe Design"""
    
    st.set_page_config(page_title="MMM Model Configuration", layout="wide")
    
    # Header Section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>🎯 Model Configuration Center</h1>
        <p style='color: white; opacity: 0.9; margin: 0.5rem 0 0 0;'>Configure your Marketing Mix Model with automated settings or custom controls</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Choice Section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div style='border: 2px dashed #4CAF50; border-radius: 10px; padding: 2rem; text-align: center; height: 300px; display: flex; flex-direction: column; justify-content: center;'>
            <h2>🚀 Automated Configuration</h2>
            <p><strong>Recommended for most users</strong></p>
            <p>AI-powered parameter estimation based on your data patterns</p>
            <br>
            <ul style='text-align: left;'>
                <li>Auto-detect media channels</li>
                <li>Smart adstock & saturation estimation</li>
                <li>Optimal time series components</li>
                <li>One-click model training</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 Use Automated Configuration", type="primary", use_container_width=True):
            st.session_state.config_mode = "automated"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style='border: 2px dashed #FF9800; border-radius: 10px; padding: 2rem; text-align: center; height: 300px; display: flex; flex-direction: column; justify-content: center;'>
            <h2>⚙️ Custom Configuration</h2>
            <p><strong>For advanced users</strong></p>
            <p>Full control over model parameters and transformations</p>
            <br>
            <ul style='text-align: left;'>
                <li>Manual channel selection</li>
                <li>Custom adstock & saturation parameters</li>
                <li>Advanced time series settings</li>
                <li>Bayesian prior configuration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("⚙️ Use Custom Configuration", use_container_width=True):
            st.session_state.config_mode = "custom"
            st.rerun()
    
    # Data Status Section
    st.markdown("---")
    st.subheader("📊 Current Data Status")
    
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        data = st.session_state.processed_data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Periods", len(data))
            st.metric("Date Range", f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
        
        with col2:
            # Auto-detected media channels
            if 'paid_media_columns' in st.session_state:
                media_channels = len([k for k,v in st.session_state.paid_media_columns.items() if v])
                st.metric("Media Channels Detected", media_channels)
            
            total_revenue = data['total_revenue'].sum() if 'total_revenue' in data.columns else 0
            st.metric("Total Revenue", f"${total_revenue/1e6:.1f}M")
        
        with col3:
            # Data quality indicators
            completeness = (1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
            
            if 'paid_media_columns' in st.session_state:
                total_media_spend = 0
                for media_type, columns in st.session_state.paid_media_columns.items():
                    for col in columns:
                        if any(x in col.lower() for x in ['spend', 'cost']) and col in data.columns:
                            total_media_spend += data[col].sum()
                st.metric("Total Media Spend", f"${total_media_spend/1e6:.1f}M")
        
        with col4:
            st.metric("Recommended Model", "BSTS + Ridge")
            st.metric("Time Series Structure", "Weekly")
    
    else:
        st.warning("📁 No data loaded. Please upload your marketing data first.")
        if st.button("📤 Upload Data", type="secondary"):
            st.switch_page("pages/data_upload.py")
    
    # Quick Start Recommendations
    st.markdown("---")
    st.subheader("🚀 Quick Start Recommendations")
    
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        if 'paid_media_columns' in st.session_state:
            paid_media = st.session_state.paid_media_columns
            
            # Show detected channels and recommendations
            st.write("**Detected Media Channels:**")
            cols = st.columns(4)
            channel_count = 0
            
            for media_type, columns in paid_media.items():
                if columns:
                    with cols[channel_count % 4]:
                        st.info(f"**{media_type.replace('_', ' ').title()}**")
                        st.write(f"{len(columns)} variables")
                    channel_count += 1
            
            # Model recommendations based on data characteristics
            st.write("**Recommended Model Settings:**")
            
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            
            with rec_col1:
                st.markdown("""
                **📈 Time Series Components**
                - Trend: Local Linear
                - Seasonality: Weekly + Yearly
                - Cyclical: Auto-detect
                """)
            
            with rec_col2:
                st.markdown("""
                **🔄 Media Transformations**
                - Adstock: Weibull Distribution
                - Saturation: Hill Function
                - Carryover: 4-12 weeks
                """)
            
            with rec_col3:
                st.markdown("""
                **🎯 Validation Strategy**
                - Holdout: Last 12 weeks
                - Cross-validation: Time-based
                - Metrics: MAPE, R², RMSE
                """)
    
    # Configuration Preview Section
    st.markdown("---")
    st.subheader("⚙️ Configuration Preview")
    
    preview_col1, preview_col2 = st.columns(2)
    
    with preview_col1:
        st.markdown("""
        **Automated Configuration Will:**
        
        ✅ **Auto-detect media channels** from your data
        ✅ **Estimate optimal parameters** using ML
        ✅ **Configure time series components** automatically
        ✅ **Set validation strategy** based on data size
        ✅ **Apply business rules** from industry benchmarks
        
        **Estimated Setup Time:** 2-3 minutes
        """)
    
    with preview_col2:
        st.markdown("""
        **Custom Configuration Allows:**
        
        ⚙️ **Manual channel selection** and grouping
        ⚙️ **Custom adstock parameters** per channel
        ⚙️ **Advanced saturation curves** configuration
        ⚙️ **Bayesian prior settings** and regularization
        ⚙️ **Component-level controls** for trend/seasonality
        
        **Estimated Setup Time:** 5-10 minutes
        """)
    
    # Bottom Action Buttons
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        
        if st.button("📊 View Data Summary First", use_container_width=True):
            st.switch_page("pages/data_explorer.py")
        
        st.markdown("</div>", unsafe_allow_html=True)

def automated_configuration_page():
    """Automated Configuration Page Wireframe"""
    
    st.title("🚀 Automated Model Configuration")
    st.info("Our AI will analyze your data and configure optimal model settings automatically.")
    
    # Progress Tracker
    st.markdown("""
    <div style='background: #f0f2f6; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h4 style='margin: 0;'>Configuration Progress</h4>
        <div style='display: flex; justify-content: space-between; margin-top: 1rem;'>
            <span>📊 Data Analysis</span>
            <span>🔍 Media Detection</span>
            <span>⚙️ Parameter Estimation</span>
            <span>✅ Model Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration Steps
    with st.expander("🔍 Step 1: Media Channel Detection", expanded=True):
        st.write("**Automatically detected channels:**")
        
        # Mock detected channels
        channels = {
            'Paid Search': ['paid_search_gross_spend', 'paid_search_impressions'],
            'Paid Social': ['paid_social_gross_spend', 'paid_social_clicks'],
            'Paid Display': ['paid_display_gross_spend']
        }
        
        for channel, vars in channels.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.checkbox(f"Include {channel}", value=True, key=f"auto_{channel}")
            with col2:
                st.write(f"Variables: {', '.join(vars)}")
    
    with st.expander("⚙️ Step 2: Automated Parameter Estimation"):
        st.write("**Estimated parameters based on data patterns:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Adstock Type", "Weibull")
            st.metric("Avg Carryover", "8.2 weeks")
        
        with col2:
            st.metric("Saturation", "Hill Function")
            st.metric("Diminishing Returns", "Medium")
        
        with col3:
            st.metric("Seasonality", "Weekly + Yearly")
            st.metric("Trend Type", "Local Linear")
    
    with st.expander("🎯 Step 3: Validation Settings"):
        st.slider("Holdout Period (weeks)", 4, 26, 12, help="Last X weeks for model testing")
        st.selectbox("Validation Metric", ["MAPE", "RMSE", "R²", "Combined"])
    
    # Action Buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Automated Configuration", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is configuring your model... This may take 2-3 minutes."):
                # Simulate processing
                import time
                time.sleep(2)
                st.success("✅ Model configured successfully!")
                st.session_state.model_configured = True
        
        if st.button("← Back to Configuration Choice", use_container_width=True):
            st.session_state.config_mode = None
            st.rerun()

def custom_configuration_page():
    """Custom Configuration Page Wireframe"""
    
    st.title("⚙️ Custom Model Configuration")
    st.warning("Advanced settings - recommended for users familiar with MMM modeling")
    
    # Configuration Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Media Channels", 
        "🔄 Transformations", 
        "📈 Time Series", 
        "🎯 Validation",
        "⚡ Advanced"
    ])
    
    with tab1:
        st.subheader("Media Channel Selection")
        st.write("**Select which media channels to include in the model:**")
        
        # Mock channel options
        channels = {
            'Paid Search': {
                'variables': ['paid_search_gross_spend', 'paid_search_net_spend', 'paid_search_impressions'],
                'description': 'Search engine marketing'
            },
            'Paid Social': {
                'variables': ['paid_social_gross_spend', 'paid_social_impressions', 'paid_social_clicks'],
                'description': 'Social media advertising'
            },
            'Paid Display': {
                'variables': ['paid_display_gross_spend', 'paid_display_impressions'],
                'description': 'Display and banner ads'
            }
        }
        
        for channel, info in channels.items():
            col1, col2, col3 = st.columns([1, 3, 2])
            with col1:
                include = st.checkbox(f"Include", value=True, key=f"custom_{channel}")
            with col2:
                st.write(f"**{channel}**")
                st.write(f"*{info['description']}*")
            with col3:
                if include:
                    st.selectbox("Spend Variable", info['variables'], key=f"var_{channel}")
    
    with tab2:
        st.subheader("Media Transformations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Adstock Method", ["Weibull", "Geometric", "None"])
            st.slider("Carryover Weeks", 1, 26, 8)
            st.selectbox("Saturation Curve", ["Hill", "Logistic", "Power Law", "None"])
        
        with col2:
            st.number_input("Shape Parameter (α)", min_value=0.1, max_value=5.0, value=1.5)
            st.number_input("Scale Parameter (λ)", min_value=1, max_value=52, value=10)
            st.number_input("Saturation Point", min_value=0.1, max_value=1.0, value=0.7)
    
    with tab3:
        st.subheader("Time Series Components")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Include Trend Component", value=True)
            st.selectbox("Trend Type", ["Local Linear", "Semi-local", "Static"])
            st.checkbox("Include Seasonality", value=True)
        
        with col2:
            st.multiselect("Seasonal Periods", ["Weekly (52)", "Monthly (12)", "Quarterly (4)"], default=["Weekly (52)"])
            st.checkbox("Include Cyclical Components", value=False)
            st.checkbox("Include Holiday Effects", value=True)
    
    with tab4:
        st.subheader("Model Validation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.slider("Training Period (%)", 50, 90, 80)
            st.selectbox("Primary Metric", ["MAPE", "RMSE", "R²", "MAE"])
        
        with col2:
            st.checkbox("Time Series Cross-Validation", value=True)
            st.number_input("CV Folds", min_value=2, max_value=10, value=5)
            st.checkbox("Statistical Significance Testing", value=True)
    
    with tab5:
        st.subheader("Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Regression Method", ["Bayesian", "Ridge", "LASSO", "Elastic Net"])
            st.number_input("Regularization Strength", min_value=0.0, max_value=1.0, value=0.1)
            st.checkbox("Enable Business Priors", value=True)
        
        with col2:
            st.number_input("MCMC Samples", min_value=100, max_value=5000, value=1000)
            st.number_input("Warmup Period", min_value=100, max_value=2000, value=500)
            st.checkbox("Parallel Sampling", value=True)
    
    # Action Buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💾 Save Configuration", use_container_width=True):
            st.success("Configuration saved!")
    
    with col2:
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.config_mode = None
            st.rerun()
    
    with col3:
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training model with custom configuration..."):
                import time
                time.sleep(3)
                st.success("✅ Model training completed!")
                st.session_state.model_trained = True

# Main App Router
def main():
    # Initialize session state
    if 'config_mode' not in st.session_state:
        st.session_state.config_mode = None
    
    # Route to appropriate page
    if st.session_state.config_mode == "automated":
        automated_configuration_page()
    elif st.session_state.config_mode == "custom":
        custom_configuration_page()
    else:
        model_configuration_homepage()

if __name__ == "__main__":
    main()
