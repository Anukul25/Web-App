from flask import Blueprint, render_template, flash, redirect, url_for, session
from app import db
from app.routes.predict import ChurnRisk
from datetime import datetime  # Added for creation_date

# Define the Blueprint
dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
def dashboard():
    # Fetch prediction stats from session
    stats = session.get('prediction_stats', None)
    
    if not stats:
        flash('No prediction data available. Please run a prediction first.', 'warning')
        return redirect(url_for('predict.predict'))

    # Fetch all churn prediction data
    churn_risks = ChurnRisk.query.all()
    creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Define creation_date
    if not churn_risks:
        return render_template('dashboard.html', 
                              message="No churn prediction data available. Please run predictions first.",
                              stats=stats,
                              results=None,
                              creation_date=creation_date)  # Pass creation_date

    # Prepare detailed results for display (all records, not limited to 10)
    results = [{
        'customer_id': cr.customer_id,
        'age': cr.age,
        'account_balance': cr.account_balance,
        'account_tenure': cr.account_tenure,
        'account_type': cr.account_type,
        'credit_score': cr.credit_score,
        'transaction_volume': cr.transaction_volume,
        'online_usage': cr.online_usage,
        'complaints': cr.complaints,
        'complaints_per_transaction': cr.complaints_per_transaction,
        'churn_probability': cr.churn_probability,
        'risk_category': cr.risk_category
    } for cr in churn_risks]

    print(f"DEBUG: Dashboard - stats: {stats}, results count: {len(results)}, creation_date: {creation_date}")
    return render_template('dashboard.html', 
                          message=None,
                          stats=stats,
                          results=results,
                          creation_date=creation_date)  # Pass creation_date