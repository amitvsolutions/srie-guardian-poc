# SRIE 10-Shield Guardian System - Complete POC
# System Reliability Intelligence Engineering
# Save this as: srie_guardian_system.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Required libraries for SRIE shields
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import joblib
    
    # SRIE Shield Libraries
    import shap  # Shield 3: Explainability
    # Note: Some libraries will be simulated for demo purposes
    # In production: alibi-detect, fairlearn, chaos-engineering tools
    
except ImportError as e:
    st.error(f"Missing library: {e}")
    st.info("Run: pip install streamlit pandas numpy plotly scikit-learn shap")

# ============================================================================
# SRIE CONFIGURATION & STATE MANAGEMENT
# ============================================================================

class SRIEState:
    """Global SRIE system state management"""
    def __init__(self):
        if 'srie_initialized' not in st.session_state:
            self.initialize_system()
    
    def initialize_system(self):
        """Initialize SRIE system state"""
        st.session_state.srie_initialized = True
        st.session_state.model_deployed = False
        st.session_state.shields_active = {f"shield_{i+1}": False for i in range(10)}
        st.session_state.system_health = 100
        st.session_state.incidents = []
        st.session_state.model_accuracy = 0.95
        st.session_state.trust_score = 8.5
        st.session_state.security_status = "SECURE"
        
        # Guardian status
        st.session_state.guardians = {
            "Star-Lord": {"status": "ACTIVE", "role": "Orchestrator", "shield": "Central Dashboard"},
            "Gamora": {"status": "ACTIVE", "role": "Precision Guardian", "shield": "Bias/Fairness Audit"},
            "Rocket": {"status": "ACTIVE", "role": "Engineering Guardian", "shield": "Chaos Testing"},
            "Groot": {"status": "ACTIVE", "role": "Reliability Guardian", "shield": "Data Drift Detection"},
            "Drax": {"status": "ACTIVE", "role": "Defense Guardian", "shield": "Security & Adversarial Defense"}
        }

# ============================================================================
# DATA GENERATION & MODEL TRAINING
# ============================================================================

@st.cache_data
def generate_loan_dataset(n_samples=10000):
    """Generate realistic loan approval dataset for SRIE demonstration"""
    np.random.seed(42)
    
    # Generate features
    data = {
        'age': np.random.normal(40, 12, n_samples).clip(18, 80),
        'income': np.random.exponential(50000, n_samples).clip(20000, 200000),
        'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
        'loan_amount': np.random.exponential(150000, n_samples).clip(10000, 500000),
        'employment_years': np.random.exponential(5, n_samples).clip(0, 40),
        'debt_to_income': np.random.beta(2, 5, n_samples) * 0.8,
    }
    
    # Add demographic features (for bias testing)
    gender = np.random.choice(['M', 'F'], n_samples)
    race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
                           n_samples, p=[0.6, 0.13, 0.18, 0.06, 0.03])
    
    data['gender'] = gender
    data['race'] = race
    
    # Create loan approval target (with subtle bias)
    risk_score = (
        (data['credit_score'] - 300) / 550 * 0.4 +
        np.log(data['income']) / np.log(200000) * 0.3 +
        (1 - data['debt_to_income']) * 0.2 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Introduce subtle bias (for demonstration)
    bias_factor = np.where(
        (gender == 'F') | (race.isin(['Black', 'Hispanic'])), 
        -0.1, 0
    )
    risk_score += bias_factor
    
    data['loan_approved'] = (risk_score > 0.5).astype(int)
    
    df = pd.DataFrame(data)
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df['race_encoded'] = le_race.fit_transform(df['race'])
    
    return df, le_gender, le_race

@st.cache_resource
def train_srie_model():
    """Train the main ML model for SRIE demonstration"""
    df, le_gender, le_race = generate_loan_dataset()
    
    # Prepare features
    feature_cols = ['age', 'income', 'credit_score', 'loan_amount', 
                   'employment_years', 'debt_to_income', 'gender_encoded', 'race_encoded']
    X = df[feature_cols]
    y = df['loan_approved']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'feature_names': feature_cols,
        'df': df,
        'encoders': {'gender': le_gender, 'race': le_race}
    }

# ============================================================================
# SHIELD 1: DATA DRIFT DETECTOR
# ============================================================================

def shield_1_data_drift_detector():
    """Shield 1: Data Drift Detection using Alibi Detect simulation"""
    st.subheader("🛡️ Shield 1: Data Drift Detector (Groot's Domain)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Guardian**: Groot 🌱\n**Protection**: Continuous Monitoring & Adaptability")
        
        if st.button("🔍 Run Drift Detection", key="drift_detect"):
            with st.spinner("Groot analyzing feature distributions..."):
                time.sleep(2)
                
                # Simulate drift detection
                model_data = train_srie_model()
                df = model_data['df']
                
                # Generate drifted data
                drift_data = df.copy()
                drift_data['income'] *= np.random.normal(1.2, 0.1, len(drift_data))  # Income inflation
                drift_data['credit_score'] += np.random.normal(20, 10, len(drift_data))  # Credit improvement
                
                # Calculate drift scores
                drift_scores = {}
                for col in ['income', 'credit_score', 'age']:
                    original_mean = df[col].mean()
                    drifted_mean = drift_data[col].mean()
                    drift_score = abs(drifted_mean - original_mean) / original_mean
                    drift_scores[col] = drift_score
                
                st.session_state.drift_detected = True
                st.session_state.drift_scores = drift_scores
    
    with col2:
        # Drift visualization
        if hasattr(st.session_state, 'drift_detected') and st.session_state.drift_detected:
            fig = go.Figure()
            
            features = list(st.session_state.drift_scores.keys())
            scores = [st.session_state.drift_scores[f] * 100 for f in features]
            colors = ['red' if s > 10 else 'orange' if s > 5 else 'green' for s in scores]
            
            fig.add_trace(go.Bar(
                x=features,
                y=scores,
                marker_color=colors,
                text=[f"{s:.1f}%" for s in scores],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Feature Drift Detection Results",
                yaxis_title="Drift Score (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Status
            max_drift = max(st.session_state.drift_scores.values()) * 100
            if max_drift > 10:
                st.error(f"🚨 HIGH DRIFT DETECTED: {max_drift:.1f}% - Retraining recommended")
                st.session_state.guardians["Groot"]["status"] = "ALERT"
            elif max_drift > 5:
                st.warning(f"⚠️ MODERATE DRIFT: {max_drift:.1f}% - Monitor closely")
                st.session_state.guardians["Groot"]["status"] = "WARNING"
            else:
                st.success(f"✅ LOW DRIFT: {max_drift:.1f}% - System stable")
                st.session_state.guardians["Groot"]["status"] = "ACTIVE"

# ============================================================================
# SHIELD 2: BIAS/FAIRNESS AUDIT
# ============================================================================

def shield_2_bias_fairness_audit():
    """Shield 2: Bias/Fairness Audit using Fairlearn simulation"""
    st.subheader("🛡️ Shield 2: Bias/Fairness Audit (Gamora's Domain)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Guardian**: Gamora ⚔️\n**Protection**: Trustworthiness & Ethical AI")
        
        if st.button("🔍 Run Bias Audit", key="bias_audit"):
            with st.spinner("Gamora analyzing model fairness..."):
                time.sleep(2)
                
                model_data = train_srie_model()
                df = model_data['df']
                y_pred = model_data['model'].predict(model_data['X_test'])
                
                # Calculate bias metrics
                test_df = df.iloc[-len(y_pred):].copy()
                test_df['predicted'] = y_pred
                
                # Gender bias analysis
                gender_bias = {}
                for gender in ['M', 'F']:
                    mask = test_df['gender'] == gender
                    approval_rate = test_df[mask]['predicted'].mean()
                    gender_bias[gender] = approval_rate
                
                # Race bias analysis
                race_bias = {}
                for race in test_df['race'].unique():
                    mask = test_df['race'] == race
                    approval_rate = test_df[mask]['predicted'].mean()
                    race_bias[race] = approval_rate
                
                st.session_state.bias_results = {
                    'gender': gender_bias,
                    'race': race_bias
                }
    
    with col2:
        if hasattr(st.session_state, 'bias_results'):
            # Gender bias chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Gender Bias Analysis', 'Race Bias Analysis'),
                vertical_spacing=0.1
            )
            
            # Gender bias
            genders = list(st.session_state.bias_results['gender'].keys())
            gender_rates = [st.session_state.bias_results['gender'][g] * 100 for g in genders]
            
            fig.add_trace(go.Bar(
                x=genders, y=gender_rates,
                name="Approval Rate %", showlegend=False,
                marker_color=['lightblue', 'lightpink']
            ), row=1, col=1)
            
            # Race bias
            races = list(st.session_state.bias_results['race'].keys())
            race_rates = [st.session_state.bias_results['race'][r] * 100 for r in races]
            
            fig.add_trace(go.Bar(
                x=races, y=race_rates,
                name="Approval Rate %", showlegend=False,
                marker_color=['lightgreen', 'lightcoral', 'lightyellow', 'lightgray', 'lightsteelblue']
            ), row=2, col=1)
            
            fig.update_layout(height=500, title="Bias Analysis Results")
            st.plotly_chart(fig, use_container_width=True)
            
            # Bias assessment
            gender_diff = abs(gender_rates[0] - gender_rates[1])
            race_max_diff = max(race_rates) - min(race_rates)
            
            if gender_diff > 10 or race_max_diff > 15:
                st.error("🚨 SIGNIFICANT BIAS DETECTED - Immediate intervention required")
                st.session_state.guardians["Gamora"]["status"] = "ALERT"
                st.session_state.trust_score = 4.2
            elif gender_diff > 5 or race_max_diff > 10:
                st.warning("⚠️ MODERATE BIAS - Monitor and mitigate")
                st.session_state.guardians["Gamora"]["status"] = "WARNING"
                st.session_state.trust_score = 7.1
            else:
                st.success("✅ BIAS WITHIN ACCEPTABLE LIMITS")
                st.session_state.guardians["Gamora"]["status"] = "ACTIVE"
                st.session_state.trust_score = 8.9

# ============================================================================
# SHIELD 3: EXPLAINABILITY
# ============================================================================

def shield_3_explainability():
    """Shield 3: Model Explainability using SHAP/LIME"""
    st.subheader("🛡️ Shield 3: Model Explainability (Transparency Shield)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Guardian**: Star-Lord 🌟\n**Protection**: Transparency & Understanding")
        
        if st.button("🔍 Generate Explanations", key="explainability"):
            with st.spinner("Generating model explanations..."):
                time.sleep(2)
                
                model_data = train_srie_model()
                model = model_data['model']
                X_test = model_data['X_test']
                feature_names = model_data['feature_names']
                
                # Feature importance (simplified SHAP simulation)
                feature_importance = model.feature_importances_
                importance_dict = dict(zip(feature_names, feature_importance))
                
                st.session_state.feature_importance = importance_dict
                
                # Sample prediction explanation
                sample_idx = 0
                sample_prediction = model.predict_proba(X_test[sample_idx:sample_idx+1])
                st.session_state.sample_explanation = {
                    'prediction_proba': sample_prediction[0],
                    'features': dict(zip(feature_names, X_test[sample_idx]))
                }
    
    with col2:
        if hasattr(st.session_state, 'feature_importance'):
            # Feature importance chart
            features = list(st.session_state.feature_importance.keys())
            importances = list(st.session_state.feature_importance.values())
            
            fig = go.Figure(go.Bar(
                y=features,
                x=importances,
                orientation='h',
                marker_color='skyblue'
            ))
            
            fig.update_layout(
                title="Global Feature Importance",
                xaxis_title="Importance Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation quality
            max_importance = max(importances)
            if max_importance > 0.3:
                st.success("✅ HIGH EXPLAINABILITY - Clear feature dominance")
                explainability_score = 92
            elif max_importance > 0.2:
                st.info("ℹ️ MODERATE EXPLAINABILITY - Multiple important features")
                explainability_score = 78
            else:
                st.warning("⚠️ LOW EXPLAINABILITY - Complex feature interactions")
                explainability_score = 65
            
            st.metric("Explainability Score", f"{explainability_score}%")

# ============================================================================
# SHIELD 4: CHAOS TESTING
# ============================================================================

def shield_4_chaos_testing():
    """Shield 4: Chaos Engineering for AI Systems"""
    st.subheader("🛡️ Shield 4: Chaos Testing (Rocket's Domain)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Guardian**: Rocket 🚀\n**Protection**: Resilience & Reliability")
        
        chaos_type = st.selectbox("Select Chaos Test", [
            "Data Corruption",
            "Feature Missing", 
            "Latency Spike",
            "Memory Pressure",
            "Network Partition"
        ])
        
        if st.button("🧨 Launch Chaos Test", key="chaos_test"):
            with st.spinner(f"Rocket launching {chaos_type} test..."):
                time.sleep(3)
                
                # Simulate chaos test results
                model_data = train_srie_model()
                
                if chaos_type == "Data Corruption":
                    # Simulate corrupted data
                    X_corrupted = model_data['X_test'].copy()
                    corruption_mask = np.random.random(X_corrupted.shape) < 0.1
                    X_corrupted[corruption_mask] = np.nan
                    
                    # Test model resilience
                    try:
                        # Fill NaN values (model resilience)
                        X_filled = pd.DataFrame(X_corrupted).fillna(0).values
                        predictions = model_data['model'].predict(X_filled)
                        degradation = 1 - (len(predictions) / len(model_data['X_test']))
                        resilience_score = max(0, 100 - degradation * 100)
                    except:
                        resilience_score = 0
                        
                elif chaos_type == "Feature Missing":
                    # Simulate missing features
                    X_missing = model_data['X_test'].copy()
                    X_missing[:, 0] = 0  # Remove first feature
                    predictions = model_data['model'].predict(X_missing)
                    
                    original_acc = accuracy_score(model_data['y_test'], model_data['y_pred'])
                    degraded_acc = accuracy_score(model_data['y_test'], predictions)
                    resilience_score = (degraded_acc / original_acc) * 100
                    
                else:
                    # Simulate other chaos scenarios
                    resilience_score = random.uniform(70, 95)
                
                st.session_state.chaos_results = {
                    'test_type': chaos_type,
                    'resilience_score': resilience_score,
                    'timestamp': datetime.now()
                }
    
    with col2:
        if hasattr(st.session_state, 'chaos_results'):
            results = st.session_state.chaos_results
            
            # Resilience gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = results['resilience_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Resilience Score<br>{results['test_type']}"},
                delta = {'reference': 90},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Assessment
            if results['resilience_score'] > 90:
                st.success("✅ EXCELLENT RESILIENCE - System handles chaos well")
                st.session_state.guardians["Rocket"]["status"] = "ACTIVE"
            elif results['resilience_score'] > 70:
                st.warning("⚠️ MODERATE RESILIENCE - Some vulnerabilities detected")
                st.session_state.guardians["Rocket"]["status"] = "WARNING"
            else:
                st.error("🚨 POOR RESILIENCE - Critical improvements needed")
                st.session_state.guardians["Rocket"]["status"] = "ALERT"

# ============================================================================
# SHIELD 5: FAILOVER & REDUNDANCY
# ============================================================================

def shield_5_failover_redundancy():
    """Shield 5: Failover and Redundancy Systems"""
    st.subheader("🛡️ Shield 5: Failover & Redundancy (Infrastructure Shield)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Guardian**: Groot 🌳\n**Protection**: Robust Infrastructure")
        
        if st.button("🔄 Test Failover System", key="failover_test"):
            with st.spinner("Testing failover mechanisms..."):
                time.sleep(2)
                
                # Simulate failover test
                systems = {
                    "Primary Model": random.choice([True, False]),
                    "Backup Model": True,  # Always available
                    "Database": random.choice([True, False]),
                    "Backup Database": True,
                    "API Gateway": random.choice([True, False]),
                    "Load Balancer": True
                }
                
                # Calculate system availability
                primary_systems = ["Primary Model", "Database", "API Gateway"]
                backup_systems = ["Backup Model", "Backup Database", "Load Balancer"]
                
                availability = 0
                for i, primary in enumerate(primary_systems):
                    if systems[primary]:
                        availability += 33.33
                    elif systems[backup_systems[i]]:
                        availability += 25  # Reduced performance with backup
                
                st.session_state.failover_results = {
                    'systems': systems,
                    'availability': availability
                }
    
    with col2:
        if hasattr(st.session_state, 'failover_results'):
            results = st.session_state.failover_results
            
            # System status visualization
            fig = go.Figure()
            
            systems = list(results['systems'].keys())
            statuses = list(results['systems'].values())
            colors = ['green' if status else 'red' for status in statuses]
            
            fig.add_trace(go.Bar(
                y=systems,
                x=[100 if status else 0 for status in statuses],
                orientation='h',
                marker_color=colors,
                text=['ONLINE' if status else 'OFFLINE' for status in statuses],
                textposition='inside'
            ))
            
            fig.update_layout(
                title="System Component Status",
                xaxis_title="Status",
                height=400,
                xaxis=dict(range=[0, 100], tickmode='array', tickvals=[0, 100], ticktext=['OFFLINE', 'ONLINE'])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Availability assessment
            availability = results['availability']
            st.metric("System Availability", f"{availability:.1f}%")
            
            if availability >= 95:
                st.success("✅ HIGH AVAILABILITY - Failover systems working correctly")
            elif availability >= 80:
                st.warning("⚠️ REDUCED AVAILABILITY - Some failover issues")
            else:
                st.error("🚨 LOW AVAILABILITY - Critical failover problems")
