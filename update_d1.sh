#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "🔄 NASDAQ-50 DAILY DATA PIPELINE & CLOUDFLARE SYNC"
echo "=================================================="

# 1. Download latest daily prices using yfinance
echo "STEP 1: Fetching latest daily prices from Yahoo Finance..."
python3 simulator/db_prep.py

# 2. Precompute all indicators (SMA, RSI, ATR, Minervini)
echo "STEP 2: Precalculating technical indicators..."
python3 simulator/precompute_metrics.py

# 3. Export daily_metrics table to SQL file
echo "STEP 3: Dumping metrics table from SQLite..."
sqlite3 simulator/data/nasdaq_simulator.db ".dump daily_metrics" > simulator/data/daily_metrics.sql

# 4. Clean up transaction statements from SQL file (unsupported by D1 upload)
echo "STEP 4: Cleaning SQL dump file..."
python3 -c '
with open("simulator/data/daily_metrics.sql", "r") as f:
    lines = f.readlines()
with open("simulator/data/daily_metrics_clean.sql", "w") as f:
    for line in lines:
        if line.strip() in ["BEGIN TRANSACTION;", "COMMIT;", "PRAGMA foreign_keys=OFF;"]:
            continue
        f.write(line)
'

# 5. Sync with Cloudflare D1
echo "STEP 5: Deploying updated records to Cloudflare D1..."
npx wrangler d1 execute nasdaq-simulator-db --remote --file=simulator/data/daily_metrics_clean.sql

echo "=================================================="
echo "✅ SYNC COMPLETED SUCCESSFULLY!"
echo "=================================================="
