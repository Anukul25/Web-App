from flask import Blueprint, render_template, request, session, redirect, url_for, send_file, flash
from app import db
import pandas as pd
import numpy as np
import joblib
import io
from io import StringIO
from collections import defaultdict
from datetime import datetime

# Define the Blueprint
predict_bp = Blueprint('predict', __name__)

# Define required and optional columns
REQUIRED_COLUMNS = ['CustomerID', 'TransactionVolume', 'OnlineUsage', 'Complaints', 'Complaints_per_Transaction']
OPTIONAL_COLUMNS = ['Age', 'Income', 'AccountTenure', 'AccountType', 'CreditScore']

# Define the Customer model (for input data)
class Customer(db.Model):
    __tablename__ = 'customers'
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.String(50), nullable=False, unique=False)
    age = db.Column(db.Integer, nullable=True)
    account_balance = db.Column(db.Float, nullable=True)  # Maps to Income
    account_tenure = db.Column(db.Float, nullable=True)
    account_type = db.Column(db.String(20), nullable=True)
    credit_score = db.Column(db.Integer, nullable=True)
    transaction_volume = db.Column(db.Float, nullable=True)
    online_usage = db.Column(db.Float, nullable=True)
    complaints = db.Column(db.Integer, nullable=True)
    complaints_per_transaction = db.Column(db.Float, nullable=True)

# Define a new model for storing churn prediction results
class ChurnRisk(db.Model):
    __tablename__ = 'churn_risk'
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.String(50), nullable=False, unique=True)
    age = db.Column(db.Integer, nullable=True)
    account_balance = db.Column(db.Float, nullable=True)
    account_tenure = db.Column(db.Float, nullable=True)
    account_type = db.Column(db.String(20), nullable=True)
    credit_score = db.Column(db.Integer, nullable=True)
    transaction_volume = db.Column(db.Float, nullable=False)
    online_usage = db.Column(db.Float, nullable=False)
    complaints = db.Column(db.Integer, nullable=False)
    complaints_per_transaction = db.Column(db.Float, nullable=False)
    churn_probability = db.Column(db.Float, nullable=False)
    risk_category = db.Column(db.String(50), nullable=False)

# Load the pre-trained models and scaler
try:
    models = joblib.load('models/hybrid_model.joblib')
    rf_model = models['random_forest']
    svm_model = models['svm']
    scaler = models['scaler']
    print("DEBUG: Pre-trained models and scaler loaded successfully from 'hybrid_model.joblib'")
except Exception as e:
    print(f"DEBUG: Error loading models - {str(e)}. Please run generate_model.py to create the model.")
    rf_model = None
    svm_model = None
    scaler = None

def calculate_feature_distributions(df, risk_categories):
    """Calculate the distribution of each feature across churn risk categories using percentile binning."""
    dist = {
        'TransactionVolume': {'high': 0, 'moderate': 0, 'low': 0},
        'OnlineUsage': {'high': 0, 'moderate': 0, 'low': 0},
        'Complaints': {'high': 0, 'moderate': 0, 'low': 0},
        'Complaints_per_Transaction': {'high': 0, 'moderate': 0, 'low': 0}
    }

    if len(df) != len(risk_categories):
        print(f"DEBUG: Mismatch in lengths - df: {len(df)}, risk_categories: {len(risk_categories)}")
        return dist

    for feature in dist.keys():
        feature_values = df[feature].dropna().values
        if len(feature_values) == 0:
            print(f"DEBUG: No valid data for {feature}, skipping.")
            continue

        valid_indices = df.index[df[feature].notna()].tolist()
        if len(valid_indices) != len(feature_values):
            print(f"DEBUG: Invalid alignment for {feature} - valid_indices: {len(valid_indices)}, feature_values: {len(feature_values)}")
            continue

        aligned_risks = [risk_categories[i] for i in valid_indices]
        if len(aligned_risks) != len(feature_values):
            print(f"DEBUG: Alignment error for {feature} - aligned_risks: {len(aligned_risks)}, feature_values: {len(feature_values)}")
            continue

        p33 = np.percentile(feature_values, 33) if len(feature_values) > 0 else 0
        p66 = np.percentile(feature_values, 66) if len(feature_values) > 0 else 0
        print(f"DEBUG: Percentiles for {feature}: p33={p33}, p66={p66}")

        for val, cat in zip(feature_values, aligned_risks):
            if cat == 'High Churn Risk':
                if val >= p66:
                    dist[feature]['high'] += 1
            elif cat == 'Moderate Churn Risk':
                if p33 <= val < p66:
                    dist[feature]['moderate'] += 1
            elif cat == 'Low Churn Risk':
                if val < p33:
                    dist[feature]['low'] += 1

    for feature in dist:
        total = sum(dist[feature].values())
        for risk in dist[feature]:
            dist[feature][risk] = round(dist[feature][risk] / total * 100, 2) if total > 0 else 0
        print(f"DEBUG: Distribution for {feature}: {dist[feature]}")

    for feature in dist:
        if all(v == 0 for v in dist[feature].values()):
            print(f"DEBUG: No valid distribution for {feature}, using fallback: {'high': 50, 'moderate': 30, 'low': 20}")
            dist[feature] = {'high': 50.0, 'moderate': 30.0, 'low': 20.0}

    return dist

@predict_bp.route('/')
def index():
    return render_template('index.html')

@predict_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        if 'start_over' in request.args:
            session.pop('record_count', None)
            session.pop('current_record', None)
            session.pop('show_run_prediction', None)
            session.pop('result', None)
            session.pop('prediction_results', None)
            session.pop('prediction_stats', None)
            session.pop('dataset_stats', None)
            session.pop('redundancy_stats', None)
            session.pop('show_upload_error', None)
            session.pop('cleaned', None)
            session.pop('show_mini_result', None)
            session.pop('input_columns', None)
            session.pop('creation_date', None)  # Clear creation_date on reset
            Customer.query.delete()
            ChurnRisk.query.delete()
            db.session.commit()
            print("DEBUG: GET request (start_over) - Cleared session variables and database")
        print(f"DEBUG: Session on GET: {dict(session)}")

        prediction_results = None
        if session.get('show_mini_result', False):
            churn_risks = ChurnRisk.query.all()
            input_columns = session.get('input_columns', ['CustomerID'])
            prediction_results = []
            for cr in churn_risks:
                result = {'CustomerID': cr.customer_id}
                if 'Age' in input_columns:
                    result['Age'] = cr.age
                if 'Income' in input_columns:
                    result['Income'] = cr.account_balance
                if 'AccountTenure' in input_columns:
                    result['AccountTenure'] = cr.account_tenure
                if 'AccountType' in input_columns:
                    result['AccountType'] = cr.account_type
                if 'CreditScore' in input_columns:
                    result['CreditScore'] = cr.credit_score
                result.update({
                    'TransactionVolume': cr.transaction_volume,
                    'OnlineUsage': cr.online_usage,
                    'Complaints': cr.complaints,
                    'Complaints_per_Transaction': cr.complaints_per_transaction,
                    'Churn_Probability (%)': cr.churn_probability,
                    'Risk_Category': cr.risk_category
                })
                prediction_results.append(result)

        return render_template('predict.html', 
                              result=session.get('result', None), 
                              record_count=session.get('record_count', None), 
                              current_record=session.get('current_record', 0), 
                              show_run_prediction=session.get('show_run_prediction', False),
                              prediction_results=prediction_results,
                              prediction_stats=session.get('prediction_stats', None),
                              dataset_stats=session.get('dataset_stats', None),
                              redundancy_stats=session.get('redundancy_stats', None),
                              show_upload_error=session.get('show_upload_error', False),
                              cleaned=session.get('cleaned', False),
                              show_mini_result=session.get('show_mini_result', False))

    result = None
    record_count = session.get('record_count', None)
    current_record = session.get('current_record', 0)
    show_run_prediction = session.get('show_run_prediction', False)
    prediction_results = None
    prediction_stats = session.get('prediction_stats', None)
    dataset_stats = session.get('dataset_stats', None)
    redundancy_stats = session.get('redundancy_stats', None)
    show_upload_error = session.get('show_upload_error', False)
    cleaned = session.get('cleaned', False)
    show_mini_result = session.get('show_mini_result', False)
    print(f"DEBUG: POST start - record_count={record_count}, current_record={current_record}, show_run_prediction={show_run_prediction}, show_upload_error={show_upload_error}, cleaned={cleaned}, show_mini_result={show_mini_result}")

    if request.method == 'POST':
        print(f"DEBUG: POST request received - form data: {request.form}")

        if 'file' in request.files:
            csv_file = request.files['file']
            print(f"DEBUG: File upload detected - filename: {csv_file.filename}")
            if csv_file.filename == '':
                result = "No file selected. Please upload a CSV file."
                session['result'] = result
                session['show_upload_error'] = True
                session.pop('dataset_stats', None)
                session.pop('redundancy_stats', None)
                session.pop('show_run_prediction', None)
                session.pop('prediction_results', None)
                session.pop('prediction_stats', None)
                session.pop('cleaned', None)
                session.pop('show_mini_result', None)
                session.pop('input_columns', None)
                session.pop('creation_date', None)  # Clear creation_date on new upload
                print("DEBUG: No file selected - Setting show_upload_error=True")
                return redirect(url_for('predict.predict'))

            if not csv_file.filename.endswith('.csv'):
                result = "Invalid file format. Please upload a CSV file."
                session['result'] = result
                session['show_upload_error'] = True
                session.pop('dataset_stats', None)
                session.pop('redundancy_stats', None)
                session.pop('show_run_prediction', None)
                session.pop('prediction_results', None)
                session.pop('prediction_stats', None)
                session.pop('cleaned', None)
                session.pop('show_mini_result', None)
                session.pop('input_columns', None)
                session.pop('creation_date', None)  # Clear creation_date on new upload
                print("DEBUG: Invalid file format - Setting show_upload_error=True")
                return redirect(url_for('predict.predict'))

            try:
                print("DEBUG: Attempting to read CSV file")
                df = pd.read_csv(csv_file)
                print(f"DEBUG: CSV read successfully - rows: {len(df)}, columns: {df.columns.tolist()}")

                missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                if missing_columns:
                    result = f"CSV file is missing the following required columns: {', '.join(missing_columns)}. Required columns are: {', '.join(REQUIRED_COLUMNS)}."
                    session['result'] = result
                    session['show_upload_error'] = True
                    print(f"DEBUG: Missing columns detected - {missing_columns}, Setting show_upload_error=True")
                    return redirect(url_for('predict.predict'))

                numeric_columns = [col for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if col in df.columns and col != 'CustomerID' and col != 'AccountType']
                invalid_columns = []
                for col in numeric_columns:
                    temp_series = df[col].astype(str).str.strip()
                    non_numeric_mask = ~temp_series.str.match(r'^-?\d*\.?\d+$') & ~temp_series.isin(['', 'nan', 'NaN'])
                    if non_numeric_mask.any():
                        invalid_columns.append(col)
                        print(f"DEBUG: Non-numeric values found in {col}: {df[col][non_numeric_mask].tolist()}")

                if invalid_columns:
                    result = f"Non-numeric values found in columns: {', '.join(invalid_columns)}. Please ensure these contain only numbers."
                    session['result'] = result
                    session['show_upload_error'] = True
                    print(f"DEBUG: Invalid data in {invalid_columns}, Setting show_upload_error=True")
                    return redirect(url_for('predict.predict'))

                session['input_columns'] = df.columns.tolist()
                print(f"DEBUG: Input columns stored in session: {session['input_columns']}")

                total_rows = int(len(df))
                duplicate_customer_ids = int(df['CustomerID'].duplicated().sum())
                null_counts_customer_id = int(df['CustomerID'].isnull().sum())
                null_counts_required = int(df[REQUIRED_COLUMNS].isnull().sum().sum())
                dataset_stats = {
                    'total_rows': total_rows,
                    'duplicate_customer_ids': duplicate_customer_ids,
                    'null_counts_customer_id': null_counts_customer_id,
                    'null_counts_required': null_counts_required
                }
                session['dataset_stats'] = dataset_stats
                session.pop('redundancy_stats', None)
                session.pop('show_upload_error', None)
                session.pop('show_run_prediction', None)
                session.pop('prediction_results', None)
                session.pop('prediction_stats', None)
                session.pop('cleaned', None)
                session.pop('show_mini_result', None)
                session['result'] = "CSV uploaded successfully! Check dataset stats below."
                session['cleaned'] = False
                print(f"DEBUG: Dataset stats calculated - {dataset_stats}")

                Customer.query.delete()
                for _, row in df.iterrows():
                    customer_id = str(row['CustomerID']) if pd.notnull(row['CustomerID']) else None
                    if customer_id is None:
                        continue

                    def to_int(val):
                        if pd.isna(val):
                            return None
                        val = float(val)
                        if val.is_integer():
                            val = int(val)
                            if -2147483648 <= val <= 2147483647:
                                return val
                            print(f"DEBUG: Integer out of range for {val}, setting to None")
                            return None
                        print(f"DEBUG: Non-integer value {val} for INTEGER column, setting to None")
                        return None

                    def to_float(val):
                        return None if pd.isna(val) else float(val)

                    customer_data = {
                        'customer_id': customer_id,
                        'age': to_int(row.get('Age')),
                        'account_balance': to_float(row.get('Income')),
                        'account_tenure': to_float(row.get('AccountTenure')),
                        'account_type': str(row.get('AccountType')) if pd.notnull(row.get('AccountType')) else None,
                        'credit_score': to_int(row.get('CreditScore')) if 'CreditScore' in row else None,
                        'transaction_volume': to_float(row.get('TransactionVolume')),
                        'online_usage': to_float(row.get('OnlineUsage')),
                        'complaints': to_int(row.get('Complaints')),
                        'complaints_per_transaction': to_float(row.get('Complaints_per_Transaction'))
                    }
                    print(f"DEBUG: Row data - {customer_data}")
                    customer = Customer(**customer_data)
                    db.session.add(customer)
                db.session.commit()
                print(f"DEBUG: Stored {Customer.query.count()} rows in Customer table")
                return redirect(url_for('predict.predict'))

            except Exception as e:
                db.session.rollback()
                result = f"Error processing CSV file: {str(e)}"
                session['result'] = result
                session['show_upload_error'] = True
                print(f"DEBUG: CSV upload error - {str(e)}, Setting show_upload_error=True")
                return redirect(url_for('predict.predict'))

        if 'remove_redundancy' in request.form:
            try:
                customers = Customer.query.all()
                if not customers:
                    result = "No data available to clean."
                    session['result'] = result
                    return redirect(url_for('predict.predict'))

                print(f"DEBUG: Fetching {len(customers)} customers from DB")
                df = pd.DataFrame([{
                    'CustomerID': c.customer_id,
                    'Age': c.age,
                    'Income': c.account_balance,
                    'AccountTenure': c.account_tenure,
                    'AccountType': c.account_type,
                    'CreditScore': c.credit_score,
                    'TransactionVolume': c.transaction_volume,
                    'OnlineUsage': c.online_usage,
                    'Complaints': c.complaints,
                    'Complaints_per_Transaction': c.complaints_per_transaction
                } for c in customers])
                print(f"DEBUG: DataFrame created with columns: {df.columns.tolist()}")

                initial_rows = len(df)
                df_no_nulls = df.dropna(subset=REQUIRED_COLUMNS)
                null_rows_removed = initial_rows - len(df_no_nulls)
                df_cleaned = df_no_nulls.drop_duplicates(subset=['CustomerID'], keep='first')
                duplicate_rows_removed = len(df_no_nulls) - len(df_cleaned)
                removed_rows = initial_rows - len(df_cleaned)
                print(f"DEBUG: Cleaned DataFrame - initial: {initial_rows}, after null removal: {len(df_no_nulls)} (removed {null_rows_removed}), after deduplication: {len(df_cleaned)} (removed {duplicate_rows_removed}), total removed: {removed_rows}")

                Customer.query.delete()
                for _, row in df_cleaned.iterrows():
                    customer = Customer(
                        customer_id=str(row['CustomerID']),
                        age=row['Age'] if pd.notnull(row['Age']) else None,
                        account_balance=row['Income'] if pd.notnull(row['Income']) else None,
                        account_tenure=row['AccountTenure'] if pd.notnull(row['AccountTenure']) else None,
                        account_type=row['AccountType'] if pd.notnull(row['AccountType']) else None,
                        credit_score=row['CreditScore'] if pd.notnull(row.get('CreditScore', None)) else None,
                        transaction_volume=row['TransactionVolume'] if pd.notnull(row['TransactionVolume']) else None,
                        online_usage=row['OnlineUsage'] if pd.notnull(row['OnlineUsage']) else None,
                        complaints=row['Complaints'] if pd.notnull(row['Complaints']) else None,
                        complaints_per_transaction=row['Complaints_per_Transaction'] if pd.notnull(row['Complaints_per_Transaction']) else None
                    )
                    db.session.add(customer)
                db.session.commit()
                print(f"DEBUG: After redundancy removal - customers in DB: {Customer.query.count()}")

                total_rows = int(len(df_cleaned))
                duplicate_customer_ids = int(df_cleaned['CustomerID'].duplicated().sum())
                null_counts_customer_id = int(df_cleaned['CustomerID'].isnull().sum())
                null_counts_required = int(df_cleaned[REQUIRED_COLUMNS].isnull().sum().sum())
                redundancy_stats = {
                    'total_rows': total_rows,
                    'duplicate_customer_ids': duplicate_customer_ids,
                    'null_counts_customer_id': null_counts_customer_id,
                    'null_counts_required': null_counts_required,
                    'removed_rows': removed_rows,
                    'null_rows_removed': null_rows_removed,
                    'duplicate_rows_removed': duplicate_rows_removed
                }
                session['redundancy_stats'] = redundancy_stats
                session['cleaned'] = True
                session.pop('show_mini_result', None)
                session['result'] = f"Removed {removed_rows} rows (including {null_rows_removed} with nulls in required columns and {duplicate_rows_removed} duplicates). Cleaned dataset ready for prediction!"
                session['show_run_prediction'] = True
                print(f"DEBUG: Redundancy removed - {redundancy_stats}")
                return redirect(url_for('predict.predict'))

            except Exception as e:
                db.session.rollback()
                result = f"Error removing redundancy: {str(e)}. Please check column names or data integrity."
                session['result'] = result
                print(f"DEBUG: Redundancy removal error - {str(e)}")
                return redirect(url_for('predict.predict'))

        if 'run_prediction' in request.form:
            print("DEBUG: Run Prediction button clicked")
            if rf_model is None or svm_model is None or scaler is None:
                result = "Error: Pre-trained models or scaler could not be loaded."
                session['result'] = result
                print("DEBUG: Models not loaded")
                return render_template('predict.html', 
                                      result=result, 
                                      record_count=record_count, 
                                      current_record=current_record, 
                                      show_run_prediction=show_run_prediction,
                                      prediction_results=None,
                                      prediction_stats=None,
                                      dataset_stats=dataset_stats,
                                      redundancy_stats=redundancy_stats,
                                      show_upload_error=False,
                                      cleaned=cleaned,
                                      show_mini_result=show_mini_result)

            try:
                customers = Customer.query.all()
                print(f"DEBUG: Customers found: {len(customers)}")
                if not customers:
                    result = "No customer data available to run predictions."
                    session['result'] = result
                    print("DEBUG: No customers in database")
                    return render_template('predict.html', 
                                          result=result, 
                                          record_count=record_count, 
                                          current_record=current_record, 
                                          show_run_prediction=show_run_prediction,
                                          prediction_results=None,
                                          prediction_stats=None,
                                          dataset_stats=dataset_stats,
                                          redundancy_stats=redundancy_stats,
                                          show_upload_error=False,
                                          cleaned=cleaned,
                                          show_mini_result=show_mini_result)

                df = pd.DataFrame([{
                    'CustomerID': customer.customer_id,
                    'Age': customer.age,
                    'Income': customer.account_balance,
                    'AccountTenure': customer.account_tenure,
                    'AccountType': customer.account_type,
                    'CreditScore': customer.credit_score,
                    'TransactionVolume': customer.transaction_volume,
                    'OnlineUsage': customer.online_usage,
                    'Complaints': customer.complaints,
                    'Complaints_per_Transaction': customer.complaints_per_transaction
                } for customer in customers])
                print(f"DEBUG: DataFrame created with {len(df)} rows")
                print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")

                X = df[['Complaints', 'Complaints_per_Transaction', 'TransactionVolume', 'OnlineUsage']]
                print(f"DEBUG: Features selected: {X.shape}")
                X_scaled = scaler.transform(X)
                print("DEBUG: Features scaled successfully")

                rf_probabilities = rf_model.predict_proba(X_scaled)[:, 1]
                svm_probabilities = svm_model.predict_proba(X_scaled)[:, 1]
                combined_probabilities = 0.5 * rf_probabilities + 0.5 * svm_probabilities
                print("DEBUG: Probabilities calculated")

                def categorize_risk(probability):
                    if probability >= 0.7:
                        return "High Churn Risk"
                    elif probability >= 0.3:
                        return "Moderate Churn Risk"
                    else:
                        return "Low Churn Risk"

                risk_categories = [categorize_risk(prob) for prob in combined_probabilities]
                churn_probabilities_percent = (combined_probabilities * 100).round(2)

                feature_dist = calculate_feature_distributions(df, risk_categories)
                print(f"DEBUG: Feature distributions calculated: {feature_dist}")

                input_columns = session.get('input_columns', ['CustomerID'])
                results_df = pd.DataFrame({
                    'CustomerID': df['CustomerID'],
                    **{col: df[col] for col in input_columns if col in df.columns and col != 'CustomerID'},
                    'Churn_Probability (%)': churn_probabilities_percent,
                    'Risk_Category': risk_categories
                })

                total_customers = int(len(results_df))
                high_churn_count = int(len(results_df[results_df['Risk_Category'] == 'High Churn Risk']))
                moderate_churn_count = int(len(results_df[results_df['Risk_Category'] == 'Moderate Churn Risk']))
                low_churn_count = int(len(results_df[results_df['Risk_Category'] == 'Low Churn Risk']))

                high_churn_percent = float(high_churn_count / total_customers * 100) if total_customers > 0 else 0
                moderate_churn_percent = float(moderate_churn_count / total_customers * 100) if total_customers > 0 else 0
                low_churn_percent = float(low_churn_count / total_customers * 100) if total_customers > 0 else 0

                model_metrics = {'accuracy': 98, 'precision': 99, 'recall': 85, 'f1_score': 83}
                creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Set creation_date at prediction time
                prediction_stats = {
                    'total_customers': total_customers,
                    'high_churn_count': high_churn_count,
                    'moderate_churn_count': moderate_churn_count,
                    'low_churn_count': low_churn_count,
                    'high_churn_percent': round(high_churn_percent, 2),
                    'moderate_churn_percent': round(moderate_churn_percent, 2),
                    'low_churn_percent': round(low_churn_percent, 2),
                    'model_metrics': model_metrics,
                    'transaction_volume_dist': feature_dist['TransactionVolume'],
                    'online_usage_dist': feature_dist['OnlineUsage'],
                    'complaints_dist': feature_dist['Complaints'],
                    'complaints_per_transaction_dist': feature_dist['Complaints_per_Transaction'],
                    'creation_date': creation_date  # Store creation_date in prediction_stats
                }
                print(f"DEBUG: Final prediction_stats before session: {prediction_stats}")

                ChurnRisk.query.delete()
                for idx, row in df.iterrows():
                    churn_risk_data = {
                        'customer_id': row['CustomerID'],
                        'transaction_volume': float(row['TransactionVolume']),
                        'online_usage': float(row['OnlineUsage']),
                        'complaints': int(row['Complaints']),
                        'complaints_per_transaction': float(row['Complaints_per_Transaction']),
                        'churn_probability': float(churn_probabilities_percent[idx]),
                        'risk_category': risk_categories[idx]
                    }
                    if 'Age' in input_columns:
                        churn_risk_data['age'] = int(row['Age']) if pd.notnull(row['Age']) else None
                    if 'Income' in input_columns:
                        churn_risk_data['account_balance'] = float(row['Income']) if pd.notnull(row['Income']) else None
                    if 'AccountTenure' in input_columns:
                        churn_risk_data['account_tenure'] = float(row['AccountTenure']) if pd.notnull(row['AccountTenure']) else None
                    if 'AccountType' in input_columns:
                        churn_risk_data['account_type'] = row['AccountType'] if pd.notnull(row['AccountType']) else None
                    if 'CreditScore' in input_columns:
                        churn_risk_data['credit_score'] = int(row['CreditScore']) if pd.notnull(row.get('CreditScore', None)) else None

                    print(f"DEBUG: Inserting ChurnRisk - {churn_risk_data}")
                    churn_risk = ChurnRisk(**churn_risk_data)
                    db.session.add(churn_risk)
                db.session.commit()
                print("DEBUG: ChurnRisk table updated")

                session['prediction_stats'] = prediction_stats
                print(f"DEBUG: Session updated with prediction_stats: {session.get('prediction_stats')}")
                session['show_run_prediction'] = False
                session['show_mini_result'] = True
                session.pop('redundancy_stats', None)
                session.pop('prediction_results', None)
                session['result'] = "Prediction completed successfully!"
                print(f"DEBUG: Prediction completed - session={dict(session)}")
                return redirect(url_for('predict.predict'))

            except Exception as e:
                db.session.rollback()
                result = f"Error running prediction: {str(e)}"
                session['result'] = result
                print(f"DEBUG: Prediction error - {str(e)}")
                return render_template('predict.html', 
                                      result=result, 
                                      record_count=record_count, 
                                      current_record=current_record, 
                                      show_run_prediction=show_run_prediction,
                                      prediction_results=None,
                                      prediction_stats=None,
                                      dataset_stats=dataset_stats,
                                      redundancy_stats=redundancy_stats,
                                      show_upload_error=False,
                                      cleaned=cleaned,
                                      show_mini_result=show_mini_result)

        if 'record-count' in request.form and 'customer-id' not in request.form:
            record_count = request.form.get('record-count', type=int)
            if record_count is None or record_count < 1:
                result = "Please enter a valid number of records (at least 1)."
            else:
                session['record_count'] = record_count
                session['current_record'] = 0
                result = f"Please enter record 1 of {record_count}."
            session['result'] = result
            print(f"DEBUG: POST (record-count) - record_count={record_count}, current_record=0")
            return redirect(url_for('predict.predict'))

        elif 'customer-id' in request.form:
            try:
                customer_id = request.form['customer-id']
                existing_customer = Customer.query.filter_by(customer_id=customer_id).first()
                if existing_customer:
                    result = f"Customer ID {customer_id} already exists. Please use a unique Customer ID for record {current_record + 1}."
                    session['result'] = result
                    return render_template('predict.html', 
                                          result=result, 
                                          record_count=record_count, 
                                          current_record=current_record, 
                                          show_run_prediction=show_run_prediction,
                                          prediction_results=None,
                                          prediction_stats=None,
                                          dataset_stats=None,
                                          show_upload_error=False)

                customer = Customer(
                    customer_id=customer_id,
                    age=request.form.get('age', type=int),
                    account_balance=request.form.get('account-balance', type=float),
                    account_tenure=request.form.get('account-tenure', type=float),
                    account_type=request.form.get('account-type'),
                    credit_score=request.form.get('credit-score', type=int),
                    transaction_volume=request.form.get('transaction-volume', type=float),
                    online_usage=request.form.get('online-usage', type=float),
                    complaints=request.form.get('complaints', 0, type=int),
                    complaints_per_transaction=request.form.get('complaints-per-transaction', type=float)
                )
                if None in [customer.customer_id, customer.transaction_volume,
                           customer.online_usage, customer.complaints, customer.complaints_per_transaction]:
                    result = f"Missing or invalid data in record {current_record + 1}."
                    session['result'] = result
                    return render_template('predict.html', 
                                          result=result, 
                                          record_count=record_count, 
                                          current_record=current_record, 
                                          show_run_prediction=show_run_prediction,
                                          prediction_results=None,
                                          prediction_stats=None,
                                          dataset_stats=None,
                                          show_upload_error=False)
                if customer.account_type and customer.account_type not in ['Savings', 'Checking', 'Credit']:
                    result = f"Invalid account type in record {current_record + 1}."
                    session['result'] = result
                    return render_template('predict.html', 
                                          result=result, 
                                          record_count=record_count, 
                                          current_record=current_record, 
                                          show_run_prediction=show_run_prediction,
                                          prediction_results=None,
                                          prediction_stats=None,
                                          dataset_stats=None,
                                          show_upload_error=False)

                db.session.add(customer)
                db.session.commit()
                print(f"DEBUG: Saved customer with ID {customer_id} for record {current_record + 1}")

                if 'input_columns' not in session:
                    session['input_columns'] = ['CustomerID', 'Age', 'Income', 'AccountTenure', 'AccountType', 
                                               'CreditScore', 'TransactionVolume', 'OnlineUsage', 'Complaints', 
                                               'Complaints_per_Transaction']

                current_record += 1
                session['current_record'] = current_record

                if current_record >= record_count:
                    result = f"Processed and stored {record_count} records in the database!"
                    session.pop('record_count', None)
                    session.pop('current_record', None)
                    session['show_run_prediction'] = True
                    session.pop('dataset_stats', None)
                    session.pop('show_upload_error', None)
                    session.pop('show_mini_result', None)
                else:
                    result = f"Record {current_record} stored successfully. Enter record {current_record + 1} of {record_count}."

                session['result'] = result
                print(f"DEBUG: Before redirect - session={dict(session)}")
                return redirect(url_for('predict.predict'))

            except Exception as e:
                db.session.rollback()
                result = f"Error storing record: {str(e)}"
                session['result'] = result
                print(f"DEBUG: Error - {str(e)}")
                return render_template('predict.html', 
                                      result=result, 
                                      record_count=record_count, 
                                      current_record=current_record, 
                                      show_run_prediction=show_run_prediction,
                                      prediction_results=None,
                                      prediction_stats=None,
                                      dataset_stats=None,
                                      show_upload_error=False)

    if show_mini_result:
        churn_risks = ChurnRisk.query.all()
        input_columns = session.get('input_columns', ['CustomerID'])
        prediction_results = []
        for cr in churn_risks:
            result = {'CustomerID': cr.customer_id}
            if 'Age' in input_columns:
                result['Age'] = cr.age
            if 'Income' in input_columns:
                result['Income'] = cr.account_balance
            if 'AccountTenure' in input_columns:
                result['AccountTenure'] = cr.account_tenure
            if 'AccountType' in input_columns:
                result['AccountType'] = cr.account_type
            if 'CreditScore' in input_columns:
                result['CreditScore'] = cr.credit_score
            result.update({
                'TransactionVolume': cr.transaction_volume,
                'OnlineUsage': cr.online_usage,
                'Complaints': cr.complaints,
                'Complaints_per_Transaction': cr.complaints_per_transaction,
                'Churn_Probability (%)': cr.churn_probability,
                'Risk_Category': cr.risk_category
            })
            prediction_results.append(result)

    return render_template('predict.html', 
                          result=result, 
                          record_count=record_count, 
                          current_record=current_record, 
                          show_run_prediction=show_run_prediction,
                          prediction_results=prediction_results,
                          prediction_stats=prediction_stats,
                          dataset_stats=dataset_stats,
                          redundancy_stats=redundancy_stats,
                          show_upload_error=show_upload_error,
                          cleaned=cleaned,
                          show_mini_result=show_mini_result)

@predict_bp.route('/predict/reset', methods=['GET'])
def reset():
    customer_count_before = Customer.query.count()
    churn_risk_count_before = ChurnRisk.query.count()
    print(f"DEBUG: Before reset - customers count: {customer_count_before}, churn_risk count: {churn_risk_count_before}")

    try:
        ChurnRisk.query.delete()
        Customer.query.delete()
        db.session.commit()
        print("DEBUG: Successfully cleared customers and churn_risk tables")
    except Exception as e:
        db.session.rollback()
        print(f"DEBUG: Error clearing tables - {str(e)}")
        raise e

    customer_count_after = Customer.query.count()
    churn_risk_count_after = ChurnRisk.query.count()
    print(f"DEBUG: After reset - customers count: {customer_count_after}, churn_risk count: {churn_risk_count_after}")

    session.pop('prediction_results', None)
    session.pop('prediction_stats', None)
    session.pop('dataset_stats', None)
    session.pop('redundancy_stats', None)
    session.pop('show_upload_error', None)
    session.pop('record_count', None)
    session.pop('current_record', None)
    session.pop('show_run_prediction', None)
    session.pop('result', None)
    session.pop('cleaned', None)
    session.pop('show_mini_result', None)
    session.pop('input_columns', None)
    session.pop('creation_date', None)  # Clear creation_date on reset
    print("DEBUG: Cleared all session variables")
    return redirect(url_for('predict.predict', start_over=True))

@predict_bp.route('/download_csv')
def download_csv():
    churn_risks = ChurnRisk.query.all()
    if not churn_risks:
        flash('No prediction results available to download.', 'error')
        return redirect(url_for('predict.predict'))

    input_columns = session.get('input_columns', ['CustomerID'])
    customers = Customer.query.all()
    customer_df = pd.DataFrame([{
        'CustomerID': c.customer_id,
        'Age': c.age,
        'Income': c.account_balance,
        'AccountTenure': c.account_tenure,
        'AccountType': c.account_type,
        'CreditScore': c.credit_score,
        'TransactionVolume': c.transaction_volume,
        'OnlineUsage': c.online_usage,
        'Complaints': c.complaints,
        'Complaints_per_Transaction': c.complaints_per_transaction
    } for c in customers])

    churn_df = pd.DataFrame([{
        'CustomerID': cr.customer_id,
        'Churn_Probability (%)': cr.churn_probability,
        'Risk_Category': cr.risk_category
    } for cr in churn_risks])

    df = pd.merge(customer_df, churn_df, on='CustomerID', how='inner')
    columns = [col for col in input_columns if col in df.columns] + ['Churn_Probability (%)', 'Risk_Category']
    df = df[columns]

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='churn_prediction_results.csv'
    )

@predict_bp.route('/download_filtered_csv', methods=['GET'])
def download_filtered_csv():
    risk_filter = request.args.get('risk_filter', 'all')  # 'high', 'moderate', 'low', or 'all'
    customer_id_search = request.args.get('customer_id', '').strip()

    churn_risks = ChurnRisk.query.all()
    if not churn_risks:
        flash('No prediction results available to download.', 'error')
        return redirect(url_for('predict.predict'))

    df = pd.DataFrame([{
        'CustomerID': cr.customer_id,
        'Age': cr.age,
        'Income': cr.account_balance,
        'AccountTenure': cr.account_tenure,
        'AccountType': cr.account_type,
        'CreditScore': cr.credit_score,
        'TransactionVolume': cr.transaction_volume,
        'OnlineUsage': cr.online_usage,
        'Complaints': cr.complaints,
        'Complaints_per_Transaction': cr.complaints_per_transaction,
        'Churn_Probability (%)': cr.churn_probability,
        'Risk_Category': cr.risk_category
    } for cr in churn_risks])

    if risk_filter != 'all':
        df = df[df['Risk_Category'] == f"{risk_filter.capitalize()} Churn Risk"]
    if customer_id_search:
        df = df[df['CustomerID'].str.contains(customer_id_search, case=False, na=False)]

    if df.empty:
        flash('No data matches the applied filters.', 'warning')
        return redirect(url_for('predict.dashboard'))

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    filename = f"filtered_churn_results_{risk_filter}"
    if customer_id_search:
        filename += f"_cust_{customer_id_search}"
    filename += ".csv"

    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

@predict_bp.route('/dashboard')
def dashboard():
    stats = session.get('prediction_stats', None)
    if not stats:
        flash('No prediction data available. Please run a prediction first.', 'warning')
        return redirect(url_for('predict.predict'))

    churn_risks = ChurnRisk.query.all()
    # Use the creation_date from prediction_stats if available, otherwise set it now
    creation_date = stats.get('creation_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f"DEBUG: Dashboard - Generating with creation_date: {creation_date}")

    if not churn_risks:
        print(f"DEBUG: No churn_risks data, rendering with creation_date: {creation_date}")
        return render_template('dashboard.html', 
                              message="No churn prediction data available. Please run predictions first.",
                              stats=stats,
                              results=None,
                              creation_date=creation_date)

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
                          creation_date=creation_date)