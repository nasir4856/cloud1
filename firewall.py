import re
import pandas as pd
import numpy as np
import joblib
import os
import subprocess
from collections import deque

# Load the trained ML model and label encoder
MODEL_PATH = 'ddos_detection_model.pkl'
ENCODER_PATH = 'label_encoder.pkl'
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Columns expected by the model
MODEL_FEATURES = [
    'mean_header_bytes', 
    'fwd_packets_IAT_mode', 
    'fwd_packets_IAT_median', 
    'packets_IAT_mean', 
    'mode_header_bytes', 
    'fwd_mean_header_bytes', 
    'fwd_min_header_bytes', 
    'fwd_median_header_bytes', 
    'fwd_mode_header_bytes', 
    'fwd_max_header_bytes'
]

# Data structure to compute features
ip_data = {}

# Helper functions for feature calculations
def compute_features(ip):
    """Compute features for the given IP based on logged data."""
    records = ip_data[ip]
    df = pd.DataFrame(records)
    return {
        'mean_header_bytes': df['header_bytes'].mean(),
        'fwd_packets_IAT_mode': df['IAT'].mode()[0] if not df['IAT'].mode().empty else 0,
        'fwd_packets_IAT_median': df['IAT'].median(),
        'packets_IAT_mean': df['IAT'].mean(),
        'mode_header_bytes': df['header_bytes'].mode()[0] if not df['header_bytes'].mode().empty else 0,
        'fwd_mean_header_bytes': df['header_bytes'].mean(),
        'fwd_min_header_bytes': df['header_bytes'].min(),
        'fwd_median_header_bytes': df['header_bytes'].median(),
        'fwd_mode_header_bytes': df['header_bytes'].mode()[0] if not df['header_bytes'].mode().empty else 0,
        'fwd_max_header_bytes': df['header_bytes'].max()
    }

# Parse Apache log entries
def parse_log_line(line):
    """Parse a single log line and extract required fields."""
    pattern = r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<datetime>[^\]]+)\] "(?P<method>\w+) (?P<url>[^\s]+) HTTP/\d+\.\d+" (?P<status>\d+) (?P<header_bytes>\d+)'
    match = re.match(pattern, line)
    if match:
        log_data = match.groupdict()
        log_data['header_bytes'] = int(log_data['header_bytes'])
        log_data['timestamp'] = pd.to_datetime(log_data['datetime'], format='%d/%b/%Y:%H:%M:%S %z')
        return log_data
    return None

# Block IP address using iptables
def block_ip(ip):
    """Block an IP address using iptables."""
    try:
        subprocess.run(["sudo", "iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"], check=True)
        print(f"Blocked IP: {ip}")
    except Exception as e:
        print(f"Error blocking IP {ip}: {e}")

# Main log processing function
def process_logs():
    """Continuously read the Apache logs and process each entry."""
    blocked_ips = set()
    log_file = "/var/log/httpd/access_log"

    with open(log_file, 'r') as f:
        f.seek(0, os.SEEK_END)  # Start at the end of the file for real-time processing
        while True:
            line = f.readline()
            if not line.strip():
                continue

            log_data = parse_log_line(line)
            if log_data:
                ip = log_data['ip']
                timestamp = log_data['timestamp']

                # Initialize or update IP data
                if ip not in ip_data:
                    ip_data[ip] = deque(maxlen=100)  # Keep the last 100 requests
                ip_data[ip].append({
                    'header_bytes': log_data['header_bytes'],
                    'IAT': (timestamp - ip_data[ip][-1]['timestamp']).total_seconds() if ip_data[ip] else 0
                })

                # Compute features
                features = compute_features(ip)
                features_df = pd.DataFrame([features])[MODEL_FEATURES]

                # Predict using the trained model
                prediction = model.predict(features_df)
                predicted_label = label_encoder.inverse_transform(prediction)[0]

                # If the request is flagged as a DDoS, block the IP
                if predicted_label == 'ddos':  # Adjust 'ddos' to match your label
                    if ip not in blocked_ips:
                        block_ip(ip)
                        blocked_ips.add(ip)

if __name__ == "__main__":
    process_logs()
