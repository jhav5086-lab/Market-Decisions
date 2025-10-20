! pip install streamlit pandas numpy scikit-learn scipy matplotlib seaborn
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Marketing Mix Modeling (MMM) Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FrequentistHierarchicalMMM:
    def __init__(self, data, media_channels, organic_channels, competitor_channels, 
                 date_column, target_column, group_column, parameters):
        self.data = data.copy()
        self.media_channels = media_channels
        self.organic_channels = organic_channels
        self.competitor_channels = competitor_channels
        self.date_column = date_column
        self.target_column = target_column
        self.group_column = group_column
        self.groups = data[group_column].unique()
        self.parameters = parameters
        
        # Parameter storage
        self.models = {}
        self.contributions = {}
        self.transformed_data = None
        
    def weibull_adstock(self, x, alpha, lam, scale):
        """Weibull Adstock transformation"""
        x = np.array(x)
        t = len(x)
        weights = np.zeros(t)
        
        for i in range(t):
            if alpha <= 0 or lam <= 0:
                weights[i] = 0
            else:
                exponent = (i / max(alpha, 1e-6)) ** max(lam, 1e-6)
                weights[i] = (lam / alpha) * ((i / alpha) ** (lam - 1)) * np.exp(-exponent)
        
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        adstocked = np.convolve(x, weights, mode='full')[:t]
        return adstocked * scale
    
    def hill_saturation(self, x, half_saturation, shape):
        """Hill saturation transformation"""
        x = np.array(x)
        half_saturation = max(half_saturation, 1e-6)
        shape = max(shape, 1e-6)
        
        x_safe = np.clip(x, 0, 1e10)
        numerator = x_safe ** shape
        denominator = half_saturation ** shape + numerator
        
        return numerator / denominator
    
    def media_transformation(self, x, alpha, lam, half_saturation, shape, scale):
        """Combine Weibull adstock and Hill saturation"""
        try:
            adstocked = self.weibull_adstock(x, alpha, lam, scale)
            saturated = self.hill_saturation(adstocked, half_saturation, shape)
            return saturated
        except:
            return np.array(x) * scale
    
    def prepare_data(self, test_size=0.2):
        """Prepare data for modeling with train/test split"""
        self.data = self.data.sort_values(self.date_column).reset_index(drop=True)
        
        n = len(self.data)
        split_idx = int(n * (1 - test_size))
        
        self.train_data = self.data.iloc[:split_idx].copy()
        self.test_data = self.data.iloc[split_idx:].copy()
        
        return self.train_data, self.test_data
    
    def optimize_media_parameters(self, channel_data, target_data):
        """Optimize media transformation parameters"""
        def objective(params):
            try:
                alpha, lam, half_saturation, shape, scale = params
                
                if (alpha < self.parameters['alpha_bounds'][0] or alpha > self.parameters['alpha_bounds'][1] or
                    lam < self.parameters['lam_bounds'][0] or lam > self.parameters['lam_bounds'][1] or
                    half_saturation < self.parameters['half_sat_bounds'][0] or half_saturation > self.parameters['half_sat_bounds'][1] or
                    shape < self.parameters['shape_bounds'][0] or shape > self.parameters['shape_bounds'][1] or
                    scale < self.parameters['scale_bounds'][0] or scale > self.parameters['scale_bounds'][1]):
                    return 1e6
                    
                transformed = self.media_transformation(channel_data, alpha, lam, half_saturation, shape, scale)
                
                if len(np.unique(transformed)) > 1:
                    r2 = np.corrcoef(transformed, target_data)[0, 1] ** 2
                    return -r2
                else:
                    return 1e6
            except:
                return 1e6
        
        # Use parameter bounds from UI
        bounds = [
            self.parameters['alpha_bounds'],
            self.parameters['lam_bounds'],
            self.parameters['half_sat_bounds'],
            self.parameters['shape_bounds'],
            self.parameters['scale_bounds']
        ]
        
        # Grid search within bounds
        best_score = 1e6
        best_params = [2, 1, 1000, 2, 0.1]
        
        alpha_range = np.linspace(bounds[0][0], bounds[0][1], 3)
        lam_range = np.linspace(bounds[1][0], bounds[1][1], 3)
        half_sat_range = np.linspace(bounds[2][0], bounds[2][1], 3)
        shape_range = np.linspace(bounds[3][0], bounds[3][1], 3)
        scale_range = np.linspace(bounds[4][0], bounds[4][1], 3)
        
        for params in product(alpha_range, lam_range, half_sat_range, shape_range, scale_range):
            try:
                score = objective(params)
                if score < best_score:
                    best_score = score
                    best_params = params
            except:
                continue
        
        # Refine with optimization
        try:
            result = minimize(objective, best_params, method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': self.parameters['max_iterations']})
            if result.success:
                return result.x
        except:
            pass
        
        return best_params
    
    def transform_media_variables(self):
        """Transform all media variables using optimized parameters"""
        self.transformed_data = self.data.copy()
        self.media_params = {}
        
        for channel in self.media_channels:
            train_indices = self.transformed_data.index[:len(self.train_data)]
            channel_train = self.transformed_data.loc[train_indices, channel].values
            target_train = self.transformed_data.loc[train_indices, self.target_column].values
            
            params = self.optimize_media_parameters(channel_train, target_train)
            
            self.media_params[channel] = {
                'alpha': params[0],
                'lam': params[1],
                'half_saturation': params[2],
                'shape': params[3],
                'scale': params[4]
            }
            
            transformed = self.media_transformation(
                self.transformed_data[channel].values,
                params[0], params[1], params[2], params[3], params[4]
            )
            
            self.transformed_data[f'{channel}_transformed'] = transformed
        
        return self.transformed_data
    
    def build_features(self):
        """Build features for modeling"""
        if self.transformed_data is None:
            self.transform_media_variables()
        
        features_df = self.transformed_data.copy()
        
        # Create group dummies
        group_dummies = pd.get_dummies(features_df[self.group_column], prefix='group')
        features_df = pd.concat([features_df, group_dummies], axis=1)
        
        # Store feature names
        self.feature_columns = []
        
        # Group fixed effects
        self.group_columns = [col for col in group_dummies.columns if col.startswith('group_')]
        self.feature_columns.extend(self.group_columns)
        
        # Transformed media variables
        self.media_transformed_columns = [f'{channel}_transformed' for channel in self.media_channels]
        self.feature_columns.extend(self.media_transformed_columns)
        
        # Organic variables
        self.feature_columns.extend(self.organic_channels)
        
        # Competitor variables
        self.feature_columns.extend(self.competitor_channels)
        
        # Create group-media interactions
        self.interaction_columns = []
        for channel in self.media_transformed_columns:
            for group_col in self.group_columns:
                interaction_name = f'{channel}_{group_col}'
                features_df[interaction_name] = features_df[channel] * features_df[group_col]
                self.interaction_columns.append(interaction_name)
        
        self.feature_columns.extend(self.interaction_columns)
        
        self.features_df = features_df
        return features_df
    
    def fit_model(self):
        """Fit the hierarchical model using regularized regression"""
        features_df = self.build_features()
        
        # Prepare training data
        train_mask = features_df.index[:len(self.train_data)]
        X_train = features_df.loc[train_mask, self.feature_columns].values
        y_train = features_df.loc[train_mask, self.target_column].values
        
        # Add intercept
        X_train = np.column_stack([np.ones(len(X_train)), X_train])
        
        # Fit Ridge regression with cross-validation
        model = RidgeCV(alphas=self.parameters['ridge_alphas'], cv=5)
        model.fit(X_train, y_train)
        
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        
        # Store coefficients with feature names
        feature_names = ['intercept'] + self.feature_columns
        self.coefficients = dict(zip(feature_names, 
                                   np.concatenate([[model.intercept_], model.coef_])))
        
        return model
    
    def predict(self, data=None):
        """Generate predictions"""
        if data is None:
            features_df = self.features_df
            X = features_df[self.feature_columns].values
            X = np.column_stack([np.ones(len(X)), X])
            return self.model.predict(X)
        else:
            X_new = self._prepare_new_data(data)
            return self.model.predict(X_new)
    
    def _prepare_new_data(self, data):
        """Prepare new data for prediction"""
        data_copy = data.copy()
        
        # Apply media transformations using stored parameters
        for channel in self.media_channels:
            if channel in self.media_params:
                params = self.media_params[channel]
                transformed = self.media_transformation(
                    data_copy[channel].values,
                    params['alpha'], params['lam'], params['half_saturation'],
                    params['shape'], params['scale']
                )
                data_copy[f'{channel}_transformed'] = transformed
        
        # Create group dummies
        group_dummies = pd.get_dummies(data_copy[self.group_column], prefix='group')
        
        # Ensure all group columns are present
        for col in self.group_columns:
            if col not in group_dummies.columns:
                group_dummies[col] = 0
        
        data_copy = pd.concat([data_copy, group_dummies], axis=1)
        
        # Create interactions
        for channel in self.media_transformed_columns:
            for group_col in self.group_columns:
                interaction_name = f'{channel}_{group_col}'
                if channel in data_copy.columns and group_col in data_copy.columns:
                    data_copy[interaction_name] = data_copy[channel] * data_copy[group_col]
        
        # Ensure all feature columns are present
        X_new = []
        for col in self.feature_columns:
            if col in data_copy.columns:
                X_new.append(data_copy[col].values)
            else:
                X_new.append(np.zeros(len(data_copy)))
        
        X_new = np.column_stack(X_new)
        X_new = np.column_stack([np.ones(len(X_new)), X_new])
        
        return X_new
    
    def calculate_metrics(self):
        """Calculate model performance metrics"""
        # Training predictions
        train_predictions = self.predict(self.train_data)
        train_actual = self.y_train
        
        # Test predictions
        test_predictions = self.predict(self.test_data)
        test_actual = self.test_data[self.target_column].values
        
        def calculate_mape(actual, predicted):
            mask = actual != 0
            if np.sum(mask) == 0:
                return np.inf
            return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        
        metrics = {
            'train': {
                'mape': calculate_mape(train_actual, train_predictions),
                'r2': r2_score(train_actual, train_predictions),
                'rmse': np.sqrt(mean_squared_error(train_actual, train_predictions)),
                'mae': mean_absolute_error(train_actual, train_predictions)
            },
            'test': {
                'mape': calculate_mape(test_actual, test_predictions),
                'r2': r2_score(test_actual, test_predictions),
                'rmse': np.sqrt(mean_squared_error(test_actual, test_predictions)),
                'mae': mean_absolute_error(test_actual, test_predictions)
            }
        }
        
        self.metrics = metrics
        return metrics
    
    def calculate_contributions(self):
        """Calculate contribution decomposition"""
        if not hasattr(self, 'coefficients'):
            raise ValueError("Model must be fitted first")
        
        # Get average feature values from training data
        avg_features = {}
        for col in self.feature_columns:
            if col in self.features_df.columns:
                avg_features[col] = self.features_df[col].mean()
            else:
                avg_features[col] = 0
        
        contributions = {
            'base': {'intercept': self.coefficients['intercept']},
            'media': {},
            'organic': {},
            'competitor': {},
            'group_effects': {}
        }
        
        # Calculate contributions
        for feature, coef in self.coefficients.items():
            if feature == 'intercept':
                continue
                
            avg_value = avg_features.get(feature, 0)
            contribution = coef * avg_value
            
            if 'transformed' in feature and any(channel in feature for channel in self.media_channels):
                channel = feature.replace('_transformed', '').split('_group')[0]
                if channel not in contributions['media']:
                    contributions['media'][channel] = 0
                contributions['media'][channel] += contribution
            elif feature in self.organic_channels:
                contributions['organic'][feature] = contribution
            elif feature in self.competitor_channels:
                contributions['competitor'][feature] = contribution
            elif feature.startswith('group_') and 'transformed' not in feature:
                contributions['group_effects'][feature] = contribution
            elif 'interaction' in feature or ('transformed' in feature and 'group' in feature):
                if any(channel in feature for channel in self.media_channels):
                    channel = feature.split('_transformed')[0]
                    if channel not in contributions['media']:
                        contributions['media'][channel] = 0
                    contributions['media'][channel] += contribution
        
        self.contributions = contributions
        return contributions

    def calculate_roi(self, media_spend_data):
        """Calculate ROI for each media channel"""
        roi_results = {}
        
        for channel in self.media_channels:
            media_contribution = self.contributions['media'].get(channel, 0)
            
            # Find corresponding spend column
            spend_column = None
            for col in media_spend_data.columns:
                if channel.replace('_impressions', '') in col and 'spend' in col:
                    spend_column = col
                    break
            
            if spend_column and spend_column in media_spend_data.columns:
                avg_spend = media_spend_data[spend_column].mean()
            else:
                # Fallback: estimate spend from impressions
                avg_spend = self.data[channel].mean() * 0.1  # Assume 0.1 cost per impression
            
            if avg_spend > 0:
                roi = media_contribution / avg_spend
            else:
                roi = 0
                
            roi_results[channel] = {
                'contribution': media_contribution,
                'avg_spend': avg_spend,
                'roi': roi
            }
        
        self.roi_results = roi_results
        return roi_results

    def response_curves(self, channel, spend_range=np.linspace(0, 10000, 50)):
        """Generate response curves for media channels"""
        if channel not in self.media_params:
            raise ValueError(f"No parameters found for channel {channel}")
        
        params = self.media_params[channel]
        transformed_channel = f'{channel}_transformed'
        
        # Find coefficient for this channel
        channel_coef = 0
        for feature, coef in self.coefficients.items():
            if transformed_channel in feature:
                channel_coef += coef
        
        responses = []
        for spend in spend_range:
            impressions = spend  # Simplified conversion
            transformed = self.media_transformation(
                [impressions], 
                params['alpha'], params['lam'], params['half_saturation'],
                params['shape'], params['scale']
            )[0]
            response = channel_coef * transformed
            responses.append(response)
        
        return spend_range, np.array(responses)

# Streamlit UI
def main():
    st.title("📊 Marketing Mix Modeling (MMM) Dashboard")
    st.markdown("Frequentist Hierarchical MMM with Weibull Adstock & Hill Saturation")
    
    # Sidebar for file upload and parameters
    st.sidebar.header("Data Configuration")
    
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")
            
            # Display basic info
            st.sidebar.subheader("Data Overview")
            st.sidebar.write(f"Shape: {data.shape}")
            st.sidebar.write(f"Columns: {len(data.columns)}")
            
            # Column selection
            st.sidebar.subheader("Column Mapping")
            
            date_column = st.sidebar.selectbox("Date Column", options=data.columns, index=0)
            target_column = st.sidebar.selectbox("Target Variable", options=data.columns, index=1)
            group_column = st.sidebar.selectbox("Group Column", options=data.columns, index=2)
            
            # Auto-detect channel types
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            media_candidates = [col for col in numeric_columns if any(x in col.lower() for x in ['media', 'impression', 'ad', 'spend'])]
            organic_candidates = [col for col in numeric_columns if any(x in col.lower() for x in ['organic', 'base', 'direct'])]
            competitor_candidates = [col for col in numeric_columns if any(x in col.lower() for x in ['competitor', 'comp'])]
            
            media_channels = st.sidebar.multiselect("Media Channels (Impressions)", options=numeric_columns, default=media_candidates[:2])
            organic_channels = st.sidebar.multiselect("Organic Variables", options=numeric_columns, default=organic_candidates[:1])
            competitor_channels = st.sidebar.multiselect("Competitor Variables", options=numeric_columns, default=competitor_candidates[:1])
            
            # Model parameters
            st.sidebar.header("Model Parameters")
            
            parameters = {
                'test_size': st.sidebar.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05),
                'max_iterations': st.sidebar.slider("Max Optimization Iterations", 50, 500, 100, 50),
                'ridge_alphas': [0.1, 1.0, 10.0, 100.0, 1000.0]
            }
            
            # Weibull parameters
            st.sidebar.subheader("Weibull Adstock Parameters")
            parameters['alpha_bounds'] = (
                st.sidebar.slider("Alpha Lower Bound", 0.1, 5.0, 0.1, 0.1),
                st.sidebar.slider("Alpha Upper Bound", 1.0, 20.0, 10.0, 1.0)
            )
            parameters['lam_bounds'] = (
                st.sidebar.slider("Lambda Lower Bound", 0.1, 2.0, 0.1, 0.1),
                st.sidebar.slider("Lambda Upper Bound", 0.5, 5.0, 3.0, 0.1)
            )
            
            # Hill parameters
            st.sidebar.subheader("Hill Saturation Parameters")
            parameters['half_sat_bounds'] = (
                st.sidebar.slider("Half-Saturation Lower Bound", 100, 5000, 10, 100),
                st.sidebar.slider("Half-Saturation Upper Bound", 1000, 20000, 10000, 1000)
            )
            parameters['shape_bounds'] = (
                st.sidebar.slider("Shape Lower Bound", 0.1, 3.0, 0.1, 0.1),
                st.sidebar.slider("Shape Upper Bound", 1.0, 10.0, 5.0, 0.5)
            )
            parameters['scale_bounds'] = (
                st.sidebar.slider("Scale Lower Bound", 0.01, 0.5, 0.01, 0.01),
                st.sidebar.slider("Scale Upper Bound", 0.1, 2.0, 1.0, 0.1)
            )
            
            # Run model button
            if st.sidebar.button("🚀 Run MMM Analysis", type="primary"):
                with st.spinner("Running MMM analysis... This may take a few minutes."):
                    try:
                        # Initialize and run model
                        mmm = FrequentistHierarchicalMMM(
                            data=data,
                            media_channels=media_channels,
                            organic_channels=organic_channels,
                            competitor_channels=competitor_channels,
                            date_column=date_column,
                            target_column=target_column,
                            group_column=group_column,
                            parameters=parameters
                        )
                        
                        # Prepare data
                        train_data, test_data = mmm.prepare_data(test_size=parameters['test_size'])
                        
                        # Fit model
                        model = mmm.fit_model()
                        
                        # Calculate metrics
                        metrics = mmm.calculate_metrics()
                        
                        # Display results
                        st.header("📈 Model Results")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Train R²", f"{metrics['train']['r2']:.3f}")
                        with col2:
                            st.metric("Test R²", f"{metrics['test']['r2']:.3f}")
                        with col3:
                            st.metric("Train MAPE", f"{metrics['train']['mape']:.2f}%")
                        with col4:
                            st.metric("Test MAPE", f"{metrics['test']['mape']:.2f}%")
                        
                        # Actual vs Predicted plots
                        st.subheader("Actual vs Predicted")
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Training data
                        train_pred = mmm.predict(mmm.train_data)
                        train_actual = mmm.y_train
                        ax1.scatter(train_actual, train_pred, alpha=0.6)
                        min_val = min(train_actual.min(), train_pred.min())
                        max_val = max(train_actual.max(), train_pred.max())
                        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                        ax1.set_xlabel('Actual')
                        ax1.set_ylabel('Predicted')
                        ax1.set_title(f'Training (R² = {metrics["train"]["r2"]:.3f})')
                        ax1.grid(True, alpha=0.3)
                        
                        # Test data
                        test_pred = mmm.predict(mmm.test_data)
                        test_actual = mmm.test_data[target_column].values
                        ax2.scatter(test_actual, test_pred, alpha=0.6, color='green')
                        min_val = min(test_actual.min(), test_pred.min())
                        max_val = max(test_actual.max(), test_pred.max())
                        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                        ax2.set_xlabel('Actual')
                        ax2.set_ylabel('Predicted')
                        ax2.set_title(f'Test (R² = {metrics["test"]["r2"]:.3f})')
                        ax2.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        
                        # Contributions
                        st.subheader("Contribution Analysis")
                        contributions = mmm.calculate_contributions()
                        
                        # Prepare contribution data
                        contrib_data = []
                        colors = []
                        
                        # Base
                        contrib_data.append(('Base', contributions['base']['intercept']))
                        colors.append('gray')
                        
                        # Media
                        for channel, value in contributions['media'].items():
                            contrib_data.append((f'Media: {channel}', value))
                            colors.append('blue')
                        
                        # Organic
                        for channel, value in contributions['organic'].items():
                            contrib_data.append((f'Organic: {channel}', value))
                            colors.append('green')
                        
                        # Competitor
                        for channel, value in contributions['competitor'].items():
                            contrib_data.append((f'Competitor: {channel}', value))
                            colors.append('red')
                        
                        # Group effects
                        for group, value in contributions['group_effects'].items():
                            contrib_data.append((f'Group: {group}', value))
                            colors.append('orange')
                        
                        labels, values = zip(*contrib_data)
                        
                        fig2, ax = plt.subplots(figsize=(12, 6))
                        bars = ax.bar(labels, values, color=colors, alpha=0.7)
                        plt.xticks(rotation=45, ha='right')
                        plt.ylabel('Contribution')
                        plt.title('Marketing Mix Contribution Decomposition')
                        plt.tight_layout()
                        st.pyplot(fig2)
                        
                        # ROI Analysis
                        st.subheader("ROI Analysis")
                        roi_results = mmm.calculate_roi(data)
                        
                        roi_df = pd.DataFrame.from_dict(roi_results, orient='index')
                        roi_df = roi_df.reset_index().rename(columns={'index': 'Channel'})
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.dataframe(roi_df.style.format({
                                'contribution': '{:.2f}',
                                'avg_spend': '{:.2f}',
                                'roi': '{:.3f}'
                            }))
                        
                        with col2:
                            fig3, ax = plt.subplots(figsize=(8, 6))
                            channels = roi_df['Channel']
                            roi_values = roi_df['roi']
                            
                            bars = ax.bar(channels, roi_values, alpha=0.7, 
                                        color=['green' if x > 0 else 'red' for x in roi_values])
                            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                            ax.set_ylabel('ROI')
                            ax.set_title('ROI by Media Channel')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig3)
                        
                        # Response Curves
                        st.subheader("Media Response Curves")
                        fig4, ax = plt.subplots(figsize=(10, 6))
                        
                        for channel in media_channels:
                            try:
                                spend_range, responses = mmm.response_curves(channel)
                                ax.plot(spend_range, responses, label=channel, linewidth=2)
                            except Exception as e:
                                st.warning(f"Could not generate response curve for {channel}: {e}")
                        
                        ax.set_xlabel('Spend')
                        ax.set_ylabel('Incremental Response')
                        ax.set_title('Media Response Curves')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig4)
                        
                        # Budget Optimization
                        st.subheader("Budget Optimization")
                        total_budget = st.number_input("Total Budget", value=10000, min_value=1000, step=1000)
                        
                        if st.button("Optimize Budget Allocation"):
                            current_budget = {channel: total_budget / len(media_channels) for channel in media_channels}
                            
                            # Simple optimization based on ROI
                            roi_values = [roi_results[channel]['roi'] for channel in media_channels]
                            total_positive_roi = sum(max(0, roi) for roi in roi_values)
                            
                            if total_positive_roi > 0:
                                optimized_budget = [total_budget * max(0, roi) / total_positive_roi for roi in roi_values]
                            else:
                                optimized_budget = [total_budget / len(media_channels)] * len(media_channels)
                            
                            opt_df = pd.DataFrame({
                                'Channel': media_channels,
                                'Current Budget': [total_budget / len(media_channels)] * len(media_channels),
                                'Optimized Budget': optimized_budget,
                                'Change %': [((opt / (total_budget / len(media_channels))) - 1) * 100 for opt in optimized_budget]
                            })
                            
                            st.dataframe(opt_df.style.format({
                                'Current Budget': '{:.0f}',
                                'Optimized Budget': '{:.0f}',
                                'Change %': '{:.1f}%'
                            }))
                        
                        # Model Parameters
                        st.subheader("Media Transformation Parameters")
                        param_df = pd.DataFrame.from_dict(mmm.media_params, orient='index')
                        st.dataframe(param_df.style.format("{:.3f}"))
                        
                    except Exception as e:
                        st.error(f"Error running MMM analysis: {str(e)}")
                        st.info("Please check your data configuration and try again.")
            
            else:
                # Data preview
                st.header("📋 Data Preview")
                st.dataframe(data.head())
                
                st.subheader("Data Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Info:**")
                    st.write(f"- Rows: {data.shape[0]}")
                    st.write(f"- Columns: {data.shape[1]}")
                    st.write(f"- Date range: {data[date_column].min()} to {data[date_column].max()}")
                
                with col2:
                    st.write("**Selected Variables:**")
                    st.write(f"- Target: {target_column}")
                    st.write(f"- Media channels: {len(media_channels)}")
                    st.write(f"- Organic variables: {len(organic_channels)}")
                    st.write(f"- Competitor variables: {len(competitor_channels)}")
                
                # Show sample statistics
                st.subheader("Variable Statistics")
                st.dataframe(data[media_channels + organic_channels + competitor_channels + [target_column]].describe())
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        # Welcome message and instructions
        st.header("Welcome to the MMM Dashboard!")
        
        st.markdown("""
        ### 📁 How to use this dashboard:
        
        1. **Upload your CSV file** using the sidebar
        2. **Configure your data** by selecting the appropriate columns:
           - Date column
           - Target variable (sales, conversions, etc.)
           - Group column (for hierarchical modeling)
           - Media channels (impressions data)
           - Organic variables
           - Competitor variables
        
        3. **Adjust model parameters** in the sidebar:
           - Test/train split ratio
           - Weibull adstock parameters
           - Hill saturation parameters
           - Regularization strength
        
        4. **Run the analysis** and explore results:
           - Model performance metrics
           - Contribution decomposition
           - ROI analysis
           - Response curves
           - Budget optimization
        
        ### 📊 Expected CSV format:
        
        Your CSV should contain:
        - A date column (weekly or monthly data)
        - A target variable (sales, revenue, etc.)
        - Media impression columns (not spend)
        - Optional: organic, competitor, and grouping variables
        
        Example columns:
        - `date`, `sales`, `region`
        - `tv_impressions`, `digital_impressions`, `social_impressions`
        - `organic_traffic`, `competitor_spend`
        """)
        
        # Sample data download
        st.subheader("Need sample data?")
        if st.button("Download Sample CSV Template"):
            # Create sample data
            dates = pd.date_range('2020-01-01', periods=100, freq='W')
            groups = ['Region_A', 'Region_B', 'Region_C']
            
            sample_data = []
            for i, date in enumerate(dates):
                for group in groups:
                    row = {
                        'date': date,
                        'region': group,
                        'sales': np.random.normal(1000, 200) + i * 5,
                        'tv_impressions': np.random.gamma(2, 500),
                        'digital_impressions': np.random.gamma(3, 300),
                        'social_impressions': np.random.gamma(1, 200),
                        'organic_traffic': np.random.normal(5000, 1000),
                        'competitor_impressions': np.random.gamma(2, 400),
                        'tv_spend': np.random.normal(5000, 1000),
                        'digital_spend': np.random.normal(3000, 500),
                        'social_spend': np.random.normal(2000, 300)
                    }
                    sample_data.append(row)
            
            sample_df = pd.DataFrame(sample_data)
            csv = sample_df.to_csv(index=False)
            
            st.download_button(
                label="Download Sample CSV",
                data=csv,
                file_name="mmm_sample_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
