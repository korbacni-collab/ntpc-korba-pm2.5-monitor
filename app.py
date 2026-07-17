
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import time
import os


import requests
from bs4 import BeautifulSoup

@st.cache_data(ttl=3600) # CRITICAL: Only run this once per hour
def get_weather_network_temp():
    try:
        # User-Agent is required, or the site will reject the request
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = "https://www.theweathernetwork.com/en/city/in/chhattisgarh/korba/current"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # This specific selector path is an educated guess based on your text
            # Note: The Weather Network renders data via React/JS; this may return nothing
            temp_element = soup.find('div', class_='temp') 
            if temp_element:
                return float(temp_element.text.replace('°', ''))
    except Exception:
        pass
    return 30.0 # Fallback default
st.set_page_config(page_title="NTPC Korba - AI Decision Support System", layout="wide")

# ==============================================================================
# HELPER: TIMEZONE (Indian Standard Time - IST)
# ==============================================================================
def get_ist_now():
    """Returns the current time in Indian Standard Time (UTC + 5:30)"""
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

# ==============================================================================
# 1. DATABASE COMPONENT 
# ==============================================================================
class SafetyDatabaseManager:
    def __init__(self, db_path="ntpc_korba_dss.db"):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Events (
                    event_id TEXT PRIMARY KEY, start_time TEXT, end_time TEXT, alarm_duration REAL,
                    current_pm25 REAL, max_predicted_pm25 REAL, alarm_severity TEXT, alarm_status TEXT,
                    current_shift TEXT, operator TEXT, plant_load REAL, coal_feed REAL, wind_speed REAL,
                    wind_direction REAL, temperature REAL, humidity REAL, esp_efficiency REAL, stack_pm REAL,
                    so2 REAL, nox REAL, flue_gas REAL, prediction_confidence REAL, root_cause TEXT,
                    recommended_action TEXT, operator_action TEXT, remarks TEXT, resolution_status TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS OperatorActions (
                    action_id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, timestamp TEXT,
                    action_taken TEXT, remarks TEXT, operator_name TEXT, FOREIGN KEY(event_id) REFERENCES Events(event_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS MaintenanceWarnings (
                    warning_id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, subsystem TEXT,
                    parameter_tracked TEXT, current_value REAL, drift_rate REAL, predicted_failure_window_hours REAL,
                    alert_severity TEXT, status TEXT
                )
            """)
            conn.commit()

    def get_active_event(self):
        with self.get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM Events WHERE alarm_status = 'Active' LIMIT 1", conn)
            return df.iloc[0].to_dict() if not df.empty else None

    def log_or_update_event(self, metrics, predictions, ai_analysis):
        active = self.get_active_event()
        now_str = get_ist_now().strftime("%Y-%m-%d %H:%M:%S")
        
        max_pred = float(max(predictions))
        threshold = 60.0
        is_breached = metrics['PM2.5'] > threshold or max_pred > threshold
        
        hour = get_ist_now().hour
        if 6 <= hour < 14: current_shift = "A-Shift"
        elif 14 <= hour < 22: current_shift = "B-Shift"
        else: current_shift = "C-Shift"

        if active:
            if not is_breached:
                start = datetime.strptime(active['start_time'], "%Y-%m-%d %H:%M:%S")
                duration_mins = round((get_ist_now() - start).total_seconds() / 60.0, 2)
                with self.get_connection() as conn:
                    conn.execute("UPDATE Events SET end_time = ?, alarm_duration = ?, alarm_status = 'Closed', resolution_status = 'Resolved' WHERE event_id = ?", (now_str, duration_mins, active['event_id']))
                return None
            else:
                with self.get_connection() as conn:
                    conn.execute("UPDATE Events SET current_pm25 = ?, max_predicted_pm25 = ? WHERE event_id = ?", (float(metrics['PM2.5']), max_pred, active['event_id']))
                return active['event_id']
        else:
            if is_breached:
                date_stamp = get_ist_now().strftime("%Y%m%d")
                with self.get_connection() as conn:
                    count = conn.execute("SELECT COUNT(*) FROM Events WHERE event_id LIKE ?", (f"EVT-{date_stamp}-%",)).fetchone()[0] + 1
                evt_id = f"EVT-{date_stamp}-{count:03d}"
                severity = "Critical" if metrics['PM2.5'] > 120 or max_pred > 120 else "Warning"
                
                with self.get_connection() as conn:
                    conn.execute("""
                        INSERT INTO Events (
                            event_id, start_time, alarm_duration, current_pm25, max_predicted_pm25, alarm_severity, alarm_status,
                            current_shift, operator, plant_load, coal_feed, wind_speed, wind_direction, temperature, humidity,
                            esp_efficiency, stack_pm, so2, nox, flue_gas, prediction_confidence, root_cause, recommended_action,
                            operator_action, remarks, resolution_status
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        evt_id, now_str, 0.0, float(metrics['PM2.5']), max_pred, severity, "Active",
                        current_shift, "NTPC_Desk_Eng", float(metrics['Unit_Load_MW']), float(metrics['Coal_Consumption_TPH']),
                        float(metrics['Wind_Speed']), float(metrics['Wind_Direction']), float(metrics['Temperature']),
                        float(metrics['Relative_Humidity']), float(metrics['ESP_Efficiency']), float(metrics['Stack_PM_Emission']),
                        float(metrics['SO2_Emission']), float(metrics['NOx_Emission']), float(metrics['Flue_Gas_Flow_Rate']),
                        88.5, ai_analysis['top_cause'], json.dumps(ai_analysis['recommendations']), "None", "", "Unresolved"
                    ))
                return evt_id
        return None

    def submit_operator_response(self, event_id, action, remarks):
        now_str = get_ist_now().strftime("%Y-%m-%d %H:%M:%S")
        with self.get_connection() as conn:
            conn.execute("UPDATE Events SET operator_action = ?, remarks = ?, resolution_status = 'Acknowledged' WHERE event_id = ?", (action, remarks, event_id))
            conn.execute("INSERT INTO OperatorActions (event_id, timestamp, action_taken, remarks, operator_name) VALUES (?, ?, ?, ?, ?)", (event_id, now_str, action, remarks, "NTPC_Desk_Eng"))

    def log_maintenance_warning(self, subsystem, parameter, val, drift, window, severity):
        now_str = get_ist_now().strftime("%Y-%m-%d %H:%M:%S")
        with self.get_connection() as conn:
            conn.execute("INSERT INTO MaintenanceWarnings (timestamp, subsystem, parameter_tracked, current_value, drift_rate, predicted_failure_window_hours, alert_severity, status) VALUES (?, ?, ?, ?, ?, ?, ?, 'Active')", (now_str, subsystem, parameter, float(val), float(drift), float(window), severity))

# ==============================================================================
# 2. SCADA TELEMETRY INGESTION INTERFACE
# ==============================================================================
class SCADAIngestionHub:
    @staticmethod
    def fetch_telemetry_frame(buffer_df, force_esp_failure=False):
        last_row = buffer_df.iloc[-1].copy()
        new_time = last_row.name + timedelta(minutes=15)
        new_row = pd.Series(index=buffer_df.columns, dtype=float)
        new_row.name = new_time
        
        units = [np.clip(np.random.normal(185, 4), 100, 210) for _ in range(3)] + [np.clip(np.random.normal(470, 8), 250, 525) for _ in range(4)]
        new_row['Unit_Load_MW'] = sum(units)
        new_row['Coal_Consumption_TPH'] = new_row['Unit_Load_MW'] * 0.64 + np.random.normal(0, 4)
        
        if not force_esp_failure:
            decay = 0.002 * (np.sin(time.time() / 10000)) 
            new_row['ESP_Efficiency'] = np.clip(last_row['ESP_Efficiency'] - decay + np.random.normal(0, 0.01), 94.0, 99.9)
        else:
            new_row['ESP_Efficiency'] = np.random.uniform(85.5, 88.0)
            
        new_row['Stack_PM_Emission'] = (100 - new_row['ESP_Efficiency']) * 12.5 + np.random.normal(0, 0.4)
        new_row['SO2_Emission'] = new_row['Coal_Consumption_TPH'] * 1.38 + np.random.normal(0, 1)
        new_row['NOx_Emission'] = new_row['Unit_Load_MW'] * 0.82 + np.random.normal(0, 2)
        new_row['Flue_Gas_Flow_Rate'] = new_row['Unit_Load_MW'] * 1580 + np.random.normal(0, 80)
        
        new_row['Wind_Speed'] = np.clip(last_row['Wind_Speed'] + np.random.normal(0, 0.15), 0.4, 15)
        new_row['Wind_Direction'] = (last_row['Wind_Direction'] + np.random.normal(0, 4)) % 360
        new_row['Temperature'] = get_weather_network_temp() # Or your API function
        new_row['Relative_Humidity'] = np.clip(last_row['Relative_Humidity'] + np.random.normal(0, 0.8), 12, 98)
        new_row['Rainfall'] = 0
        new_row['Hour'] = new_time.hour
        new_row['Season_Month'] = new_time.month
        
        base_pm = (new_row['Stack_PM_Emission'] * 0.65) - (new_row['Wind_Speed'] * 1.6) + np.random.normal(36, 1.5)
        new_row['PM2.5'] = np.clip(base_pm, 12, 600)
        
        new_row['PM2.5_lag_1h'] = buffer_df.iloc[-4]['PM2.5']
        new_row['PM2.5_lag_3h'] = buffer_df.iloc[-12]['PM2.5']
        new_row['PM2.5_lag_6h'] = buffer_df.iloc[-24]['PM2.5']
        new_row['PM2.5_lag_24h'] = buffer_df.iloc[-96]['PM2.5']
        
        return pd.DataFrame([new_row])

# ==============================================================================
# 3. HYBRID PREDICTION COMPONENT
# ==============================================================================
class HybridPredictionModel:
    def __init__(self):
        self.model_rf = None
        self.model_xgb = None
        self.features = ['PM2.5_lag_1h', 'PM2.5_lag_3h', 'PM2.5_lag_6h', 'PM2.5_lag_24h', 'Wind_Speed', 'Wind_Direction', 'Temperature', 'Relative_Humidity', 'Rainfall', 'Unit_Load_MW', 'Coal_Consumption_TPH', 'Stack_PM_Emission', 'SO2_Emission', 'NOx_Emission', 'ESP_Efficiency', 'Flue_Gas_Flow_Rate', 'Hour', 'Season_Month']
        self.baselines = {}
        self.stds = {}

    def train_baseline_data(self):
        np.random.seed(42)
        periods = 30 * 24 * 4 
        dates = pd.date_range(end=get_ist_now(), periods=periods, freq="15min")
        df = pd.DataFrame({'Datetime': dates})
        
        df['Unit_Load_MW'] = np.random.normal(2150, 120, periods).clip(1200, 2600)
        df['Coal_Consumption_TPH'] = df['Unit_Load_MW'] * 0.65 + np.random.normal(0, 8)
        
        esp_eff = np.random.normal(99.5, 0.1, periods).clip(98.0, 99.9)
        i = 0
        while i < periods - 16:
            if np.random.rand() > 0.96: 
                duration = np.random.randint(8, 20)
                esp_eff[i:i+duration] = np.random.uniform(84.0, 90.0)
                i += duration
            else:
                i += 1
                
        df['ESP_Efficiency'] = esp_eff
        df['Flue_Gas_Flow_Rate'] = df['Unit_Load_MW'] * 1590 + np.random.normal(0, 150)
        df['Stack_PM_Emission'] = (100 - df['ESP_Efficiency']) * 12.5 + np.random.normal(0, 0.8)
        df['SO2_Emission'] = df['Coal_Consumption_TPH'] * 1.4 + np.random.normal(0, 4)
        df['NOx_Emission'] = df['Unit_Load_MW'] * 0.84 + np.random.normal(0, 4)
        df['Wind_Speed'] = np.random.lognormal(mean=1.1, sigma=0.35, size=periods)
        df['Wind_Direction'] = np.random.uniform(0, 360, periods)
        df['Temperature'] = 28 + 8 * np.sin(np.linspace(0, 4 * np.pi, periods)) + np.random.normal(0, 1.5)
        df['Relative_Humidity'] = np.random.normal(58, 10, periods).clip(15, 100)
        df['Rainfall'] = np.where(np.random.rand(periods) > 0.99, np.random.exponential(5, periods), 0)
        
        df['PM2.5'] = (df['Stack_PM_Emission'] * 0.65) - (df['Wind_Speed'] * 1.5) + np.random.normal(36, 4, periods)
        df['PM2.5'] = df['PM2.5'].clip(10, 600)
        df = df.set_index('Datetime')

        df['Hour'] = df.index.hour
        df['Season_Month'] = df.index.month
        df['PM2.5_lag_1h'] = df['PM2.5'].shift(4)
        df['PM2.5_lag_3h'] = df['PM2.5'].shift(12)
        df['PM2.5_lag_6h'] = df['PM2.5'].shift(24)
        df['PM2.5_lag_24h'] = df['PM2.5'].shift(96)
        
        rf_targets, xgb_targets = [], []
        for s in range(1, 3): rf_targets.append(f'T_{s}'); df[f'T_{s}'] = df['PM2.5'].shift(-s)
        for s in range(3, 9): xgb_targets.append(f'T_{s}'); df[f'T_{s}'] = df['PM2.5'].shift(-s)

        df.dropna(inplace=True)
        X = df[self.features]
        self.baselines = X.mean().to_dict()
        self.stds = X.std().to_dict()
        
        self.model_rf = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
        self.model_rf.fit(X, df[rf_targets])
        bx = xgb.XGBRegressor(n_estimators=30, max_depth=5, learning_rate=0.15, random_state=42)
        self.model_xgb = MultiOutputRegressor(bx)
        self.model_xgb.fit(X, df[xgb_targets])
        return df

    def execute_hybrid_inference(self, current_frame):
        X = current_frame[self.features]
        return np.concatenate([self.model_rf.predict(X)[0], self.model_xgb.predict(X)[0]])

# ==============================================================================
# 4. DIAGNOSTICS ENGINE (DEEP POWER PLANT ROOT CAUSE ANALYSIS)
# ==============================================================================
class RootCauseAIAnalyzer:
    @staticmethod
    def evaluate_root_causes(metrics, baselines, stds):
        scores = {}
        
        # Calculate standard deviations from normal
        esp_dev = (baselines['ESP_Efficiency'] - metrics['ESP_Efficiency']) / stds['ESP_Efficiency']
        coal_dev = (metrics['Coal_Consumption_TPH'] - baselines['Coal_Consumption_TPH']) / stds['Coal_Consumption_TPH']
        fg_dev = (metrics['Flue_Gas_Flow_Rate'] - baselines['Flue_Gas_Flow_Rate']) / stds['Flue_Gas_Flow_Rate']
        wind_dev = (baselines['Wind_Speed'] - metrics['Wind_Speed']) / stds['Wind_Speed']
        hum_dev = (metrics['Relative_Humidity'] - baselines['Relative_Humidity']) / stds['Relative_Humidity']
        
        # Detailed Power Plant Diagnoses
        scores['ESP Corona Quenching / TR Set Failure'] = max(0.0, min(100.0, esp_dev * 25.0 if esp_dev > 0 else 0))
        scores['High Inlet Dust Burden (Poor Coal Quality)'] = max(0.0, min(100.0, coal_dev * 20.0 if coal_dev > 0 else 0))
        scores['Boiler Overload / High Gas Velocity'] = max(0.0, min(100.0, fg_dev * 18.0 if fg_dev > 0 else 0))
        scores['Thermal Inversion / Poor Dispersion'] = max(0.0, min(100.0, wind_dev * 20.0 if wind_dev > 0 else 0))
        scores['High Moisture / Acid Dew Point Issue'] = max(0.0, min(100.0, hum_dev * 15.0 if hum_dev > 0 else 0))
        
        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        explanations, recommendations = [], []
        
        for cause, score in top_3:
            if cause == 'ESP Corona Quenching / TR Set Failure':
                explanations.append(f"ESP telemetry dropped to {metrics['ESP_Efficiency']:.2f}%. High probability of field tripping or hopper ash grounding the discharge electrodes.")
                recommendations.append({"rec": "Inspect Transformer-Rectifier (TR) sets, check rapper sequencing, and verify ash hopper levels.", "prio": "CRITICAL", "impact": "High PM Reduction", "time": "20-30 Mins"})
            elif cause == 'High Inlet Dust Burden (Poor Coal Quality)':
                explanations.append(f"Coal feed abnormally high at {metrics['Coal_Consumption_TPH']:.1f} TPH relative to current generation load. Indicates unburned carbon or high-ash coal.")
                recommendations.append({"rec": "Optimize pulverizer mills, adjust secondary air damper, and blend with washed coal.", "prio": "HIGH", "impact": "Reduces inlet burden", "time": "45 Mins"})
            elif cause == 'Boiler Overload / High Gas Velocity':
                explanations.append(f"Flue Gas Flow excessive at {metrics['Flue_Gas_Flow_Rate']:.0f} Nm³/h. High velocity reduces ESP collection time causing ash carryover.")
                recommendations.append({"rec": "Throttle ID Fan loading and slightly reduce generation load to increase gas residence time in ESP.", "prio": "HIGH", "impact": "Direct Load Deflection", "time": "15 Mins"})
            elif cause == 'Thermal Inversion / Poor Dispersion':
                explanations.append(f"Wind dropped to {metrics['Wind_Speed']:.2f} m/s. Stagnant atmospheric conditions are trapping the plume near ground level.")
                recommendations.append({"rec": "Temporarily reduce generation load until atmospheric conditions lift.", "prio": "MEDIUM", "impact": "Prevents Build-up", "time": "Ongoing"})
            else:
                explanations.append(f"Humidity high at {metrics['Relative_Humidity']:.1f}%. Risk of sticky ash blinding ESP collector plates.")
                recommendations.append({"rec": "Check air pre-heater (APH) temperatures and adjust flue gas conditioning.", "prio": "MEDIUM", "impact": "Maintains Plate Cleanliness", "time": "60 Mins"})

        return {
            "top_cause": top_3[0][0] if top_3[0][1] > 0 else "Normal Operations",
            "scores": top_3, "explanations": explanations, "recommendations": recommendations
        }

class PredictiveMaintenanceCore:
    @staticmethod
    def process_degradation_checks(buffer_df, db_manager):
        if len(buffer_df) < 20: return []
        warnings = []
        x = np.arange(len(buffer_df))
        slope, _ = np.polyfit(x, buffer_df['ESP_Efficiency'].values, 1)
        if slope < -0.0005: 
            current_val = buffer_df['ESP_Efficiency'].values[-1]
            hours_to_fail = round((((current_val - 90.0) / (abs(slope) + 1e-9)) * 15.0) / 60.0, 1)
            if hours_to_fail < 24:
                warnings.append({"subsystem": "ESP Unit Array", "param": "ESP_Efficiency", "val": current_val, "drift": slope * 4, "hours": hours_to_fail, "severity": "Critical"})
        return warnings

# ==============================================================================
# 5. INITIALIZATION & STATE PROCESSING
# ==============================================================================
db = SafetyDatabaseManager()
predictive_engine = HybridPredictionModel()

if 'models_loaded' not in st.session_state:
    with st.spinner("Training Hybrid Predictive Matrix Models (IST Timezone)..."):
        hist_df = predictive_engine.train_baseline_data()
        st.session_state.stream_buffer = hist_df.tail(100).copy()
        st.session_state.models_loaded = True
        st.session_state.engine_ref = predictive_engine

predictive_engine = st.session_state.engine_ref

st.sidebar.markdown("<h2 style='text-align: center; color: #4da6ff;'>⚡ Korba Control Desk</h2>", unsafe_allow_html=True)

# ---> NAVIGATION SIDEBAR (Outside fragment)
page = st.sidebar.radio("Navigation Matrix", [
    "Live Monitoring", 
    "Active Diagnostics & Response", 
    "Alarm History", 
    "Analytics Dashboard", 
    "Predictive Maintenance", 
    "Shift Reports"
])

st.sidebar.markdown("---")
st.sidebar.header("🕹️ Ingestion Controls")
is_streaming = st.sidebar.toggle("Connect Live Plant Feed", value=False)
force_failure = st.sidebar.checkbox("💥 Inject ESP Field Trip Failure")


def process_current_state(advance_stream, force_fail):
    """Handles background data ingestion and calculations."""
    if advance_stream:
        new_frame = SCADAIngestionHub.fetch_telemetry_frame(st.session_state.stream_buffer, force_fail)
        st.session_state.stream_buffer = pd.concat([st.session_state.stream_buffer, new_frame]).tail(100)

    current_df = st.session_state.stream_buffer.iloc[-1:]
    current_metrics = current_df.iloc[0]
    predictions = predictive_engine.execute_hybrid_inference(current_df)
    ai_analysis = RootCauseAIAnalyzer.evaluate_root_causes(current_metrics, predictive_engine.baselines, predictive_engine.stds)
    active_evt_id = db.log_or_update_event(current_metrics, predictions, ai_analysis)

    threshold = 60.0
    
    # Check if we have an active alarm or forecast breach
    is_alarm_active = current_metrics['PM2.5'] > threshold or max(predictions) > threshold
    
    return current_df, current_metrics, predictions, ai_analysis, active_evt_id, is_alarm_active


# ==============================================================================
# 6. DASHBOARD DRAWING COMPONENT
# ==============================================================================
def draw_dashboard(current_page, current_df, current_metrics, predictions, ai_analysis, active_evt_id, is_alarm_active, is_live_streaming):
    threshold = 60.0

    # DYNAMIC SCREEN RED WARNING STYLE (Make entire screen red if live value is above safety abnormal)
    is_live_abnormal = current_metrics['PM2.5'] > threshold
    if is_live_abnormal:
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(135deg, #2b0000 0%, #610000 50%, #2b0000 100%) !important;
                color: #ffffff !important;
            }
            div[data-testid="stMetricValue"] {
                color: #ffffff !important;
            }
            .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, label {
                color: #ffffff !important;
            }
            div[style*="background"] {
                background-color: rgba(0, 0, 0, 0.5) !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    # Trigger Sound alert only on Live Monitoring pages
    if is_alarm_active and is_live_streaming:
        # Plays a dynamic alarm tone
        components.html("""
            <script>
                if (!window.audioCtx) { window.audioCtx = new (window.AudioContext || window.webkitAudioContext)(); }
                function playAlarm() {
                    let osc = window.audioCtx.createOscillator(); let gain = window.audioCtx.createGain();
                    osc.type = 'sawtooth'; osc.frequency.setValueAtTime(880, window.audioCtx.currentTime);
                    gain.gain.setValueAtTime(0.12, window.audioCtx.currentTime);
                    osc.connect(gain); gain.connect(window.audioCtx.destination);
                    osc.start(); setTimeout(() => { osc.stop(); }, 600);
                } playAlarm();
            </script>
        """, height=0, width=0)

    # PAGE 1: LIVE MONITORING DESK
    if current_page == "Live Monitoring":
        st.markdown("### 🖥️ Live PM2.5 & Emission Tracking Desk | NTPC KORBA")

        # STATIC AI ALARM PANEL
        if is_alarm_active:
            alarm_color = "#4a0000"
            border = "#ff0000" 
            title = "🚨 ACTIVE PM2.5 ALARM"
            body = f"Root Cause : <b>{ai_analysis['top_cause']}</b><br>Recommendation : {ai_analysis['recommendations'][0]['rec']}"
        else:
            alarm_color = "#112211"
            border = "#00ff66" 
            title = "✅ SYSTEM NORMAL"
            body = "All monitored parameters are within safe operating limits."

        st.markdown(f"""<div style="
    height:110px; background:{alarm_color}; border-left:10px solid {border}; border-radius:10px; padding:15px; margin-bottom:15px; display:flex; flex-direction:column; justify-content:center;
    ">
    <div style="font-size:22px;font-weight:bold;color:white;">{title}</div>
    <div style="font-size:16px;color:white;margin-top:6px;">{body}</div>
    </div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### 1. Live Ambient PM2.5")
            bg = "#4a0000" if current_metrics['PM2.5'] > threshold else "#112211"
            border = "#ff0000" if current_metrics['PM2.5'] > threshold else "#00ff66"
            st.markdown(f"""<div style="background:{bg}; border:8px solid {border}; border-radius:10px; padding:20px; text-align:center;">
    <div style="font-size:54px; font-weight:bold; color:white;">{current_metrics['PM2.5']:.2f}</div>
    <div style="font-size:18px; color:white;">µg/m³</div>
    </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("##### 2. Max Forecast (Next 30m)")
            mx_30 = max(predictions[:2])
            bg = "#4a0000" if mx_30 > threshold else "#112211"
            border = "#ff0000" if mx_30 > threshold else "#00ff66"
            st.markdown(f"""<div style="background:{bg}; border:8px solid {border}; border-radius:10px; padding:20px; text-align:center;">
    <div style="font-size:54px; font-weight:bold; color:white;">{mx_30:.2f}</div>
    <div style="font-size:18px; color:white;">Next 30 Mins (RF)</div>
    </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown("##### 3. Max Forecast (Next 2h)")
            mx_120 = max(predictions)
            bg = "#4a0000" if mx_120 > threshold else "#112211"
            border = "#ff0000" if mx_120 > threshold else "#00ff66"
            st.markdown(f"""<div style="background:{bg}; border:8px solid {border}; border-radius:10px; padding:20px; text-align:center;">
    <div style="font-size:54px; font-weight:bold; color:white;">{mx_120:.2f}</div>
    <div style="font-size:18px; color:white;">Next 2 Hrs (RF+XGB)</div>
    </div>""", unsafe_allow_html=True)

        st.write("")
        ch1, ch2 = st.columns(2)
        future_times = [current_df.index[0] + timedelta(minutes=15*i) for i in range(1, 9)]
        time_labels = ["+15m (RF)", "+30m (RF)", "+45m (XGB)", "+60m (XGB)", "+75m (XGB)", "+90m (XGB)", "+105m (XGB)", "+120m (XGB)"]

        with ch1:
            st.markdown("##### 4. Plant Trend & Forward Trajectory")
            fig1 = go.Figure()
            hist_view = st.session_state.stream_buffer.tail(24)
            fig1.add_trace(go.Scatter(x=hist_view.index, y=hist_view['PM2.5'], mode='lines', name='Actual history', line=dict(color='#00FFFF', width=2)))
            fig1.add_trace(go.Scatter(x=[current_df.index[0]] + future_times, y=[current_metrics['PM2.5']] + list(predictions),
                                      mode='lines+markers', name='Hybrid Path', line=dict(color='#ff0000' if is_alarm_active else '#00ff66', width=3, dash='dash')))
            fig1.add_hline(y=threshold, line_dash="dot", line_color="#ff0000")
            fig1.update_layout(plot_bgcolor='#111', paper_bgcolor='#111', font=dict(color='white'), margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig1, use_container_width=True)

        with ch2:
            st.markdown("##### 5. Direct-Read Interval Value Matrix")
            colors = []
            for idx, val in enumerate(predictions):
                if val <= threshold:
                    colors.append("#00ff66")
                else:
                    if idx in [0, 1]:
                        colors.append("#ff0000") # Red alerts for +15m, +30m
                    else:
                        colors.append("#ffd000") # Orange/Yellow alerts for +45m and above
                        
            fig2 = go.Figure(data=[go.Bar(
                x=time_labels, y=predictions, text=[f"{v:.1f}" for v in predictions], textposition='outside', marker_color=colors
            )])
            fig2.add_hline(y=threshold, line_dash="solid", line_color="#ff0000")
            fig2.update_layout(plot_bgcolor='#111', paper_bgcolor='#111', font=dict(color='white'), margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(range=[0, max(140, mx_120 + 40)]))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("##### 6. Active Plant Telemetry Registers")
        def generate_telemetry_html(label, val, unit, feat):
            z = (val - predictive_engine.baselines[feat]) / predictive_engine.stds[feat]
            is_bad = (feat in ['ESP_Efficiency', 'Wind_Speed'] and z < -1.5) or (feat not in ['ESP_Efficiency', 'Wind_Speed'] and z > 1.5)
            color = "#ff0000" if is_bad else "#00ff66"
            icon = "🔴" if is_bad else "🟢"
            return f"""<div style="background:#1a1a1a; padding:12px; border-radius:6px; border-left: 4px solid {color};">
    <div style="font-size:12px; color:#aaa; margin-bottom:4px;">{label}</div>
    <div style="font-size:18px; color:white; font-weight:bold;">{icon} {val:.2f} <span style="font-size:12px; font-weight:normal; color:#888;">{unit}</span></div></div>"""

        telemetry_html = f"""<div style="background:#111; padding:20px; border-radius:10px; border: 1px solid #333;">
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
    {generate_telemetry_html("Plant Generation Load", current_metrics['Unit_Load_MW'], "MW", "Unit_Load_MW")}
    {generate_telemetry_html("Stack Opacity (PM)", current_metrics['Stack_PM_Emission'], "mg/Nm³", "Stack_PM_Emission")}
    {generate_telemetry_html("Anemometer Wind Speed", current_metrics['Wind_Speed'], "m/s", "Wind_Speed")}
    {generate_telemetry_html("Coal Feed Rate", current_metrics['Coal_Consumption_TPH'], "TPH", "Coal_Consumption_TPH")}
    {generate_telemetry_html("Continuous SO2 Emission", current_metrics['SO2_Emission'], "ppm", "SO2_Emission")}
    {generate_telemetry_html("Ambient Air Temperature", current_metrics['Temperature'], "°C", "Temperature")}
    {generate_telemetry_html("Flue Gas Flow Volume", current_metrics['Flue_Gas_Flow_Rate'], "Nm³/h", "Flue_Gas_Flow_Rate")}
    {generate_telemetry_html("Continuous NOx Emission", current_metrics['NOx_Emission'], "ppm", "NOx_Emission")}
    {generate_telemetry_html("Relative Air Humidity", current_metrics['Relative_Humidity'], "%", "Relative_Humidity")}
    </div></div>"""
        st.markdown(telemetry_html, unsafe_allow_html=True)

    # PAGE 1.5: ACTIVE DIAGNOSTICS & RESPONSE
    elif current_page == "Active Diagnostics & Response":
        st.markdown("### 🧠 Active Diagnostics & Operations Response")
        if not is_alarm_active:
            st.success("✅ **STATUS NORMAL:** Plant operations are completely nominal. No active diagnostics required.")
        else:
            st.error(f"### 🚨 PRIMARY DIAGNOSIS: {ai_analysis['top_cause']}")
            st.markdown(f"**Detailed AI Trace:** {ai_analysis['explanations'][0]}")
            st.divider()

            diag_col, act_col = st.columns([1, 1.2])
            with diag_col:
                st.markdown("#### 📊 Root Cause Confidence")
                for cause, score in ai_analysis['scores']:
                    st.markdown(f"**{cause}** ({score:.1f}%)")
                    st.progress(int(score)/100.0)

            with act_col:
                st.markdown("#### 🛠️ Action Strategy Plan")
                for idx, rec in enumerate(ai_analysis['recommendations'][:2]): 
                    st.info(f"**Action {idx+1}:** {rec['rec']}\n\n**Priority:** `{rec['prio']}` | **Impact:** `{rec['impact']}` | **Est. Time:** `{rec['time']}`")

            st.divider()
            st.markdown("#### 📥 Operator Response & Logging")
            current_active = db.get_active_event()
            if current_active:
                with st.form("operator_response_form"):
                    st.warning(f"Logging actions against open event reference: **{current_active['event_id']}**")
                    op_action = st.selectbox("Action Executed", ["Reduce Load", "Increase ESP Voltage", "Inspect Hopper", "Call Maintenance", "Other"])
                    op_remarks = st.text_area("Engineering Logging Remarks", placeholder="Enter specific actions taken...")
                    if st.form_submit_button("Submit Logs to Database"):
                        db.submit_operator_response(current_active['event_id'], op_action, op_remarks)
                        st.success("Action parameters appended to active transaction log.")
            else:
                st.info("No active events found for response logging.")

            st.divider()
            st.markdown("##### ⏱️ Active Event Lifecycle")
            steps = st.columns(4)
            with steps[0]: st.error("🚨 **1. Alarm Raised**\n\nThreshold Breached")
            with steps[1]: st.info("🧠 **2. AI Root Cause**\n\nConfidence Scored")
            with steps[2]: st.warning("⏳ **3. Operator Verification**\n\nPending Input")
            with steps[3]: st.success("✅ **4. Recovery Frame**\n\nMonitoring Normalization")

    # PAGE 2: ALARM HISTORY AUDIT PAGE
    elif current_page == "Alarm History":
        st.markdown("### 🗃️ Historical Event Storage Registry")
        with db.get_connection() as conn:
            history_df = pd.read_sql_query("SELECT * FROM Events ORDER BY start_time DESC", conn)
        
        if history_df.empty: 
            st.info("No recorded historical compliance events stored yet.")
        else:
            fl1, fl2, fl3 = st.columns(3)
            with fl1: severity_filter = st.multiselect("Severity Tier", options=history_df['alarm_severity'].unique(), default=history_df['alarm_severity'].unique() if 'alarm_severity' in history_df.columns else [])
            with fl2: status_filter = st.multiselect("Resolution Status", options=history_df['resolution_status'].unique(), default=history_df['resolution_status'].unique() if 'resolution_status' in history_df.columns else [])
            with fl3: cause_filter = st.multiselect("Primary Flagged Cause", options=history_df['root_cause'].unique(), default=history_df['root_cause'].unique() if 'root_cause' in history_df.columns else [])
            
            # Prevent failures if historical columns are blank
            filtered_df = history_df
            if not filtered_df.empty and 'alarm_severity' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['alarm_severity'].isin(severity_filter)]
            if not filtered_df.empty and 'resolution_status' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['resolution_status'].isin(status_filter)]
            if not filtered_df.empty and 'root_cause' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['root_cause'].isin(cause_filter)]
                
            st.dataframe(filtered_df, use_container_width=True)
            st.download_button("Export Dataset as CSV", data=filtered_df.to_csv(index=False).encode('utf-8'), file_name="NTPC_Korba_Alarms.csv", mime="text/csv")

    # PAGE 3: ANALYTICS DASHBOARD
    elif current_page == "Analytics Dashboard":
        st.markdown("### 📊 Enterprise Analytics Dashboard")
        with db.get_connection() as conn: all_df = pd.read_sql_query("SELECT * FROM Events", conn)
        if all_df.empty: st.info("Insufficient transactional samples.")
        else:
            al1, al2 = st.columns(2)
            with al1: st.plotly_chart(px.pie(all_df, names='root_cause', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu, title="Incident Frequencies by Root Cause"), use_container_width=True)
            with al2: st.plotly_chart(px.bar(all_df, x='current_shift', y='alarm_duration', color='alarm_severity', barmode='group', title="Loss Duration across Shifts"), use_container_width=True)

    # PAGE 4: PREDICTIVE MAINTENANCE MONITOR
    elif current_page == "Predictive Maintenance":
        st.markdown("### 🔧 Predictive Maintenance Monitor")
        st.info("💡 **How this works:** Detects slow degradation over time (e.g. losing 0.1% ESP efficiency daily) and warns you days before an alarm triggers.")
        
        warnings = PredictiveMaintenanceCore.process_degradation_checks(st.session_state.stream_buffer, db)
        if not warnings: st.success("✅ Operational Parameter Slopes Nominal.")
        else:
            for w in warnings: st.error(f"#### ⚠️ PREDICTIVE ALERT: {w['subsystem'].upper()}\n* **Parameter:** `{w['param']}` | **Drift:** {w['drift']:.4f}\n* **Breach Horizon:** **{w['hours']} Hours**")
        st.markdown("##### Recorded Log History")
        with db.get_connection() as conn: st.dataframe(pd.read_sql_query("SELECT * FROM MaintenanceWarnings ORDER BY timestamp DESC LIMIT 20", conn), use_container_width=True)

    # PAGE 5: SHIFT REPORTS
    elif current_page == "Shift Reports":
        st.markdown("### 📋 Executive Shift Summary")
        with db.get_connection() as conn: rep_df = pd.read_sql_query("SELECT * FROM Events", conn)
        report_type = st.selectbox("Select Report Interval", ["Current Operative Shift", "Full Daily Report Summary", "Weekly Compliance Statement"])
        if not rep_df.empty:
            total_alarms = len(rep_df)
            st.markdown(f"""
            ```text
            Total Emission Excursions: {total_alarms}
            Mean Alarm Hold Window: {rep_df['alarm_duration'].mean():.2f} Mins
            Peak Discharged Emission: {rep_df['current_pm25'].max():.2f} µg/m³
            ```
            """)
            st.download_button("Download Report", data=rep_df.to_csv(index=False).encode('utf-8'), file_name="Report.txt", mime="text/plain")


# ==============================================================================
# 7. ROUTING AND FRAGMENT EXECUTION
# ==============================================================================

# Defining isolated fragment loops to avoid page switching conflicts
@st.fragment(run_every=1.2)
def live_stream_fragment(page_name, force_fail):
    current_df, current_metrics, predictions, ai_analysis, active_evt_id, is_alarm_active = process_current_state(True, force_fail)
    draw_dashboard(page_name, current_df, current_metrics, predictions, ai_analysis, active_evt_id, is_alarm_active, is_live_streaming=True)

@st.fragment
def static_ui_fragment(page_name, force_fail):
    current_df, current_metrics, predictions, ai_analysis, active_evt_id, is_alarm_active = process_current_state(False, force_fail)
    draw_dashboard(page_name, current_df, current_metrics, predictions, ai_analysis, active_evt_id, is_alarm_active, is_live_streaming=False)


# Main router routing logic
if page in ["Live Monitoring", "Active Diagnostics & Response"]:
    if is_streaming:
        live_stream_fragment(page, force_failure)
    else:
        static_ui_fragment(page, force_failure)
else:
    # Render interactive filters page completely outside background loops
    c_df, c_metrics, preds, ai_analysis, a_evt_id, is_alarm = process_current_state(False, force_failure)
    draw_dashboard(page, c_df, c_metrics, preds, ai_analysis, a_evt_id, is_alarm, is_live_streaming=False)
