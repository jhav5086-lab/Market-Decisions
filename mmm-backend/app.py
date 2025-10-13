# mmm-backend/app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Marketing Mix Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class MMModel:
    def __init__(self):
        self.media_channels = [
            'paid_search_net_spend', 'paid_shopping_net_spend', 
            'paid_display_net_spend', 'paid_social_net_spend', 
            'paid_affiliate_net_spend'
        ]
        
    def prepare_data(self, df):
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        y = df['total_revenue'].values
        media_data = np.column_stack([df[channel].values for channel in self.media_channels])
        
        # Handle missing values
        media_data = np.nan_to_num(media_data)
        
        return y, media_data, df
    
    def calculate_roas(self, model, media_data):
        roas_results = []
        current_allocation = np.mean(media_data, axis=0)
        
        for i, channel in enumerate(self.media_channels):
            roas = model.coef_[i] * current_allocation[i]
            roas_results.append({
                'channel': channel,
                'roas_mean': max(roas, 0.1),
                'roas_std': abs(roas * 0.1)
            })
        
        return roas_results, current_allocation
    
    def optimize_budget(self, roas_results, current_allocation, total_budget):
        sorted_channels = sorted(roas_results, key=lambda x: x['roas_mean'], reverse=True)
        roas_values = [channel['roas_mean'] for channel in sorted_channels]
        total_roas = sum(roas_values)
        optimal_weights = [roas / total_roas for roas in roas_values]
        optimized_allocation = [weight * total_budget for weight in optimal_weights]
        
        budget_recommendations = []
        for i, channel in enumerate(sorted_channels):
            current = current_allocation[i]
            optimized = optimized_allocation[i]
            change_pct = ((optimized - current) / current) * 100 if current > 0 else 100
            
            budget_recommendations.append({
                'channel': channel['channel'],
                'current_allocation': current,
                'optimized_allocation': optimized,
                'change_percentage': change_pct,
                'roas': channel['roas_mean']
            })
        
        return budget_recommendations

mmm_model = MMModel()

@app.post("/analyze")
async def analyze_mmm(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing file: {file.filename}")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        y, media_data, processed_df = mmm_model.prepare_data(df)
        
        # Use Ridge regression for demo
        model = Ridge(alpha=1.0)
        X = media_data
        model.fit(X, y)
        
        current_allocation = np.mean(media_data, axis=0)
        total_budget = np.sum(current_allocation)
        
        roas_results, current_allocation = mmm_model.calculate_roas(model, media_data)
        budget_recommendations = mmm_model.optimize_budget(roas_results, current_allocation, total_budget)
        
        top_channel = max(roas_results, key=lambda x: x['roas_mean'])
        worst_channel = min(roas_results, key=lambda x: x['roas_mean'])
        total_optimization_impact = sum(abs(rec['change_percentage']) for rec in budget_recommendations)
        
        results = {
            'status': 'success',
            'roas_analysis': roas_results,
            'budget_optimization': budget_recommendations,
            'executive_summary': {
                'top_performing_channel': top_channel['channel'],
                'top_channel_roas': round(top_channel['roas_mean'], 2),
                'worst_performing_channel': worst_channel['channel'],
                'worst_channel_roas': round(worst_channel['roas_mean'], 2),
                'total_optimization_impact': round(total_optimization_impact, 1),
                'recommended_budget_shift': f"Increase investment in {top_channel['channel']}"
            },
            'model_metrics': {
                'r_squared': round(model.score(X, y), 3),
                'mape': round(mean_absolute_percentage_error(y, model.predict(X)), 3)
            }
        }
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
