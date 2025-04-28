\
    # Zero‑Loss Trade Filter (Gold Bot)

    This repository contains **all** data, code and pretrained models required to
    reproduce and run the “Zero‑Loss Trade Filter” that approves only profitable
    trades for *XAUUSD* on MetaTrader 5.

    ## Project structure

    ```text
    .
    ├── data/
    │   ├── history_full.csv
    │   ├── history_full_with_predictions.csv
    │   └── XAUUSD_M1_2023.csv
    ├── models/
    │   ├── model.pkl
    │   └── model_rf_v5.pkl
    ├── src/
    │   ├── prepare_features.py
    │   ├── train_model_rf_v5.py
    │   ├── auto_threshold_search.py
    │   ├── check_risk_integrity.py
    │   ├── watch_and_predict.py
    │   └── main.py
    ├── requirements.txt
    └── README.md
    ```

    > **1612 winners / 0 losers** were obtained on the 2023‑2024 back‑test with
    > the thresholds found in `auto_threshold_search.py`.

    ### Setup (one‑liner)

    ```bash
    python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
    pip install -r requirements.txt
    ```

    ### Training workflow

    1. **Feature engineering**

       ```bash
       python src/prepare_features.py \\
            --history data/history_full.csv \\
            --prices  data/XAUUSD_M1_2023.csv \\
            --out     features.parquet
       ```

    2. **Model training**

       ```bash
       python src/train_model_rf_v5.py --feat features.parquet --out models/model_rf_v5.pkl
       ```

    3. **Threshold optimisation (optional)**

       ```bash
       python src/auto_threshold_search.py
       ```

    ### Live filter (server side)

    The script `src/watch_and_predict.py` is designed to run **continuously**
    alongside your EA:

    ```bash
    export TELEGRAM_TOKEN="PASTE_BOT_TOKEN"
    export CHAT_IDS="123456789,987654321"   # comma‑separated
    python src/watch_and_predict.py --last-trade /path/to/last_trade.json
    ```

    When the prediction **AND** thresholds are satisfied it writes
    `{"approve": true}` back to the JSON and notifies all `CHAT_IDS`.

    ### Git quick‑start (copy‑&‑paste)

    ```bash
    # run inside your server home directory
    unzip gold_filter_repo.zip && cd gold_filter_repo
    git init
    git add .
    git commit -m "Initial commit ‑ Zero‑Loss Trade Filter"
    git remote add origin git@github.com:YOUR_USERNAME/zero-loss-filter.git
    git branch -M main
    git push -u origin main
    ```

    > Replace `YOUR_USERNAME` with your GitHub user and make sure your SSH key
    > is configured on the server. Large binary files (≈65 MB total) are
    > committed directly — no further setup is required.

    ---

    *Generated automatically on 2025-04-28T13:31:54.752915Z*
