import time
import requests
import json
import os
from datetime import datetime

# URLs of your services
DATA_API_BASE_URL = "https://lotto-api-production-a6f3.up.railway.app"
MODEL_API_BASE_URL = "https://web-production-09cd3.up.railway.app"

# Local data file (using Railway subscription ended)
LOCAL_DATA_FILE = "local_lottery_data.json"

# How often to check for new data (in minutes)
CHECK_INTERVAL_MINUTES = 1440  # Every 24 hours (once per day)

# File to track the last processed draw date
LAST_DRAW_FILE = "last_processed_draw.json"

def load_last_processed_date():
    """Load the last processed draw date from file"""
    if os.path.exists(LAST_DRAW_FILE):
        try:
            with open(LAST_DRAW_FILE, "r") as f:
                data = json.load(f)
                return data.get("last_date")
        except Exception as e:
            print(f"Error loading last processed date: {e}")
    return None

def save_last_processed_date(date_str):
    """Save the last processed draw date to file"""
    try:
        with open(LAST_DRAW_FILE, "w") as f:
            json.dump({"last_date": date_str, "updated_at": datetime.now().isoformat()}, f)
        print(f"✓ Saved last processed date: {date_str}")
    except Exception as e:
        print(f"Error saving last processed date: {e}")

def fetch_new_draws():
    """
    Load recent draws from local data file.
    Returns list of new draws (newer than last processed).
    """
    print(f"[{datetime.now().isoformat()}] Loading draws from local data file...")
    
    try:
        if not os.path.exists(LOCAL_DATA_FILE):
            print(f"Local data file {LOCAL_DATA_FILE} not found!")
            return []
        
        try:
            with open(LOCAL_DATA_FILE, "rb") as f:
                raw = f.read()
            if b"\x00" in raw:
                raw = raw.replace(b"\x00", b"")
            text = raw.decode("utf-8", errors="ignore")
            data = json.loads(text)
        except Exception as e:
            print(f"Error parsing local data file {LOCAL_DATA_FILE}: {e}")
            return []

        draws = data.get("draws", [])
        
        if not draws:
            print("No draws found in local data file")
            return []
        
        # Get last processed date
        last_date = load_last_processed_date()
        
        if last_date is None:
            # First run - just save the latest date, don't trigger retrain
            latest_date = draws[0]["fecha"]
            save_last_processed_date(latest_date)
            print(f"First run - initialized with latest draw: {latest_date}")
            return []
        
        # Filter for new draws (dates after last_date)
        new_draws = [d for d in draws if d["fecha"] > last_date]
        
        if new_draws:
            # Update last processed date to the newest draw
            newest_date = max(d["fecha"] for d in new_draws)
            save_last_processed_date(newest_date)
            print(f"Found {len(new_draws)} new draw(s) since {last_date}")
        
        return new_draws
        
    except Exception as e:
        print(f"Error loading local data: {e}")
        return []

def trigger_retrain():
    """
    Call the /admin/retrain endpoint on the Model API.
    """
    print(f"[{datetime.now().isoformat()}] Triggering retrain on Model API...")
    url = f"{MODEL_API_BASE_URL}/admin/retrain"
    resp = requests.post(url, timeout=30)
    print(f"Model API response: {resp.status_code} {resp.text}")

def main_loop():
    while True:
        try:
            new_draws = fetch_new_draws()
            if new_draws:
                print(f"Found {len(new_draws)} new draws - updating data and retraining...")
                # TODO: save new_draws to file / DB here
                trigger_retrain()
            else:
                print("No new draws found.")
        except Exception as e:
            print(f"Orchestrator error: {e}")

        print(f"Sleeping for {CHECK_INTERVAL_MINUTES} minutes...\n")
        time.sleep(CHECK_INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    print("🧠 Starting orchestrator (middle agent)...")
    main_loop()
