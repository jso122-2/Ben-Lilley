# ebay_webapp/app.py

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import os

# Plotly imports
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)

# ===============================
# LOAD REPRESENTATIVE DUMMY DATA
# ===============================
data_path = os.path.join(os.getcwd(), "models", "representative_dummy_data.csv")
churn_data = pd.read_csv(data_path)

def predict_churn(tenure, age, avg_purchase):
    """
    Dummy churn prediction using representative dummy data.
    For simplicity, returns the average churn probability from the dataset.
    """
    return churn_data['churn_prob'].mean()

def predict_nps(features):
    """
    Dummy NPS prediction that returns the average NPS from the dataset.
    """
    return churn_data['nps_score'].mean()

def classify_nps(score):
    """
    Simple utility to classify NPS based on the score.
    """
    if score >= 9:
        return 'Promoter'
    elif 7 <= score < 9:
        return 'Passive'
    else:
        return 'Detractor'

# ===============================
# CHURN SCENARIOS FOR DEMO
# ===============================
CHURN_SCENARIOS = [
    {'id': 1, 'tenure_months': 1, 'age': 18, 'avg_purchase': 20,
     'desc': 'New, young, low spend', 'prediction': 'High Risk',
     'funnel_data': [1000, 700, 500, 300]},
    {'id': 2, 'tenure_months': 3, 'age': 25, 'avg_purchase': 50,
     'desc': 'Early stage, moderate spend', 'prediction': 'Moderate Risk',
     'funnel_data': [800, 600, 250, 100]},
    {'id': 3, 'tenure_months': 5, 'age': 30, 'avg_purchase': 80,
     'desc': 'Mid-level tenure, decent spend', 'prediction': 'Moderate Risk',
     'funnel_data': [900, 700, 200, 90]},
    {'id': 4, 'tenure_months': 7, 'age': 35, 'avg_purchase': 100,
     'desc': 'Steady, decent age, good spend', 'prediction': 'Low Risk',
     'funnel_data': [1000, 850, 150, 50]},
    {'id': 5, 'tenure_months': 10, 'age': 40, 'avg_purchase': 120,
     'desc': 'Longer tenure, good spend', 'prediction': 'Low Risk',
     'funnel_data': [1000, 900, 120, 40]},
    {'id': 6, 'tenure_months': 12, 'age': 45, 'avg_purchase': 10,
     'desc': 'Medium tenure, low spend', 'prediction': 'Moderate Risk',
     'funnel_data': [1000, 800, 400, 200]},
    {'id': 7, 'tenure_months': 15, 'age': 50, 'avg_purchase': 200,
     'desc': 'Solid tenure, high spend', 'prediction': 'Very Low Risk',
     'funnel_data': [1000, 920, 80, 20]},
    {'id': 8, 'tenure_months': 20, 'age': 55, 'avg_purchase': 5,
     'desc': 'Older, large tenure, minimal spend', 'prediction': 'High Risk',
     'funnel_data': [1000, 750, 400, 300]},
    {'id': 9, 'tenure_months': 25, 'age': 60, 'avg_purchase': 50,
     'desc': 'Long tenure, moderate spend, older demographic', 'prediction': 'Moderate Risk',
     'funnel_data': [1000, 850, 300, 120]},
    {'id': 10, 'tenure_months': 36, 'age': 65, 'avg_purchase': 150,
     'desc': 'Longest tenure, high spend, older demographic', 'prediction': 'Very Low Risk',
     'funnel_data': [1000, 950, 80, 10]},
]

# ============== HOME ROUTE ==============
@app.route('/')
def index():
    return render_template('index.html')

# ===============================
# NPS FUNCTIONALITY
# ===============================
@app.route('/nps_simulation', methods=['GET', 'POST'])
def nps_simulation():
    nps_prediction = None
    sentiment_html = None

    if request.method == 'POST':
        support_quality = float(request.form.get('support_quality', 5))
        delivery_speed = float(request.form.get('delivery_speed', 5))
        pricing = float(request.form.get('pricing', 5))

        features = np.array([[support_quality, delivery_speed, pricing]])
        nps_prediction = predict_nps(features)

        promoters = max(0, nps_prediction + 20)
        detractors = max(0, 100 - nps_prediction)
        passives = max(0, 100 - (promoters + detractors))

        sentiment_df = pd.DataFrame({
            'Category': ['Promoters', 'Passives', 'Detractors'],
            'Count': [promoters, passives, detractors]
        })

        sentiment_fig = px.bar(
            sentiment_df, 
            x='Category', 
            y='Count',
            title='Simulated Customer Sentiment',
            color='Category',
            color_discrete_map={
                'Promoters': 'green',
                'Passives': 'yellow',
                'Detractors': 'red'
            }
        )
        sentiment_html = sentiment_fig.to_html(full_html=False)

    return render_template('nps_simulation.html',
                           nps_prediction=nps_prediction,
                           sentiment_html=sentiment_html)

@app.route('/nps_live_dashboard')
def nps_live_dashboard():
    overall_nps = np.random.randint(-100, 100)
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_nps,
        title={'text': "Overall NPS Score"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-100, 100]},
            'steps': [
                {'range': [-100, 0], 'color': "red"},
                {'range': [0, 50], 'color': "yellow"},
                {'range': [50, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': overall_nps
            }
        }
    ))
    gauge_fig.update_layout(height=400)

    if overall_nps >= 50:
        mood = "😄"
    elif overall_nps < 0:
        mood = "😞"
    else:
        mood = "😐"

    gauge_fig.add_annotation(
        x=0.5, y=-0.2, 
        text=f"Customer Mood: {mood}", 
        showarrow=False, 
        font=dict(size=24)
    )

    gauge_html = gauge_fig.to_html(full_html=False)
    return render_template('nps_live_dashboard.html', gauge_html=gauge_html)

# ===============================
# CHURN FUNCTIONALITY
# ===============================
@app.route('/churn_interactive', methods=['GET', 'POST'])
def churn_interactive():
    scenario_id = int(request.form.get('scenario_id', 1))
    scenario = next((s for s in CHURN_SCENARIOS if s['id'] == scenario_id), CHURN_SCENARIOS[0])
    
    funnel_df = pd.DataFrame({
        'Stage': ['Onboarding', 'Engaged', 'At Risk', 'Churned'],
        'Count': scenario['funnel_data']
    })
    funnel_fig = go.Figure(
        go.Funnel(
            y=funnel_df['Stage'],
            x=funnel_df['Count'],
            textinfo="value+percent previous"
        )
    )
    funnel_fig.update_layout(title=f'Churn Funnel for Scenario #{scenario_id}')
    funnel_html = funnel_fig.to_html(full_html=False)

    return render_template('churn_interactive.html',
                           scenario=scenario,
                           scenario_id=scenario_id,
                           funnel_html=funnel_html)

# ===============================
# ADVANCED DASHBOARD & SCENARIOS
# ===============================
@app.route('/advanced_dashboard')
def advanced_dashboard():
    region = request.args.get('region', 'All')
    filtered_df = churn_data if region == 'All' else churn_data[churn_data['region'] == region]
    avg_nps = filtered_df['nps_score'].mean()
    avg_churn = filtered_df['churn_prob'].mean()

    bar_fig = px.bar(
        filtered_df, 
        x='customer_id', 
        y='nps_score', 
        color='region', 
        title='NPS Scores by Customer'
    )
    bar_html = bar_fig.to_html(full_html=False)

    hist_fig = px.histogram(
        filtered_df, 
        x='churn_prob', 
        nbins=5, 
        title='Distribution of Churn Probability'
    )
    hist_html = hist_fig.to_html(full_html=False)

    regions_available = sorted(churn_data['region'].unique())

    return render_template('advanced_dashboard.html',
                           bar_html=bar_html,
                           hist_html=hist_html,
                           avg_nps=round(avg_nps, 2),
                           avg_churn=round(avg_churn, 2),
                           region=region,
                           regions_available=regions_available)

@app.route('/scenarios_comparison', methods=['GET', 'POST'])
def scenarios_comparison():
    baseline_nps = 5
    baseline_churn = 0.20

    discount_rate = float(request.form.get('discount_rate', 0))
    support_quality = float(request.form.get('support_quality', 5))

    scenario_nps = baseline_nps + (discount_rate / 5) + (support_quality / 10)
    scenario_churn = baseline_churn - (discount_rate * 0.01) - (support_quality * 0.005)
    scenario_churn = max(scenario_churn, 0)

    comparison_df = pd.DataFrame({
        'Metric': ['NPS', 'Churn Probability'],
        'Baseline': [baseline_nps, baseline_churn],
        'Scenario': [scenario_nps, scenario_churn]
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(x=comparison_df['Metric'], y=comparison_df['Baseline'], name='Baseline'))
    fig.add_trace(go.Bar(x=comparison_df['Metric'], y=comparison_df['Scenario'], name='Scenario'))
    fig.update_layout(barmode='group', title='Scenario Comparison')

    comparison_html = fig.to_html(full_html=False)
    return render_template('scenarios_comparison.html', comparison_html=comparison_html)

@app.route('/customer_journey')
def customer_journey():
    timeline = [
        {'step': 'Onboarding', 'nps': 5,  'churn': 0.2},
        {'step': 'First Purchase', 'nps': 7, 'churn': 0.15},
        {'step': 'Received Reward', 'nps': 9, 'churn': 0.10},
        {'step': 'Had Complaint', 'nps': 3, 'churn': 0.3},
        {'step': 'Resolution', 'nps': 8, 'churn': 0.12},
        {'step': 'Renewal', 'nps': 9, 'churn': 0.05}
    ]
    df = pd.DataFrame(timeline)

    nps_fig = px.line(
        df, x='step', y='nps', 
        markers=True, 
        title='Customer Journey - NPS Over Time'
    )
    nps_html = nps_fig.to_html(full_html=False)

    churn_fig = px.line(
        df, x='step', y='churn', 
        markers=True, 
        title='Customer Journey - Churn Probability Over Time'
    )
    churn_html = churn_fig.to_html(full_html=False)

    return render_template('customer_journey.html', nps_html=nps_html, churn_html=churn_html)

@app.route('/customer_profiles')
def customer_profiles():
    data_copy = churn_data.copy()
    data_copy['nps_class'] = data_copy['nps_score'].apply(classify_nps)
    data_copy['recommendation'] = data_copy.apply(
        lambda row: "Offer discount" if row['churn_prob'] > 0.3 else "Send loyalty email", axis=1
    )
    profiles = data_copy.to_dict(orient='records')
    return render_template('customer_profiles.html', profiles=profiles)

if __name__ == "__main__":
    from waitress import serve
    logging.info("Starting Flask app with Waitress...")
    serve(app, host="0.0.0.0", port=5000)