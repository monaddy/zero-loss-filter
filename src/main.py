# main.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests
import json
import traceback
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from model_utils import compute_indicators, calculate_risk_score, format_telegram_message_proportional

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/trade", methods=["POST"])
def handle_trade():
    try:
        raw_data = request.data
        print("📦 Raw request body:", raw_data)

        try:
            text_data = raw_data.decode("utf-8")
            print("📄 Decoded text:", text_data)
            data = json.loads(text_data)
        except Exception as json_err:
            print("❌ JSON decode failed:", json_err)
            return jsonify({"status": "error", "message": str(json_err)})

        df = pd.DataFrame([data])
        df = compute_indicators(df)

        X = df[[
            "RSI_14", "CCI_20", "Momentum_10", "BB_Width", "Williams_%R",
            "ROC_10", "MA_20", "MA_50", "EMA_20", "EMA_50"
        ]]

        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X)[0][1] * 100
        risk_score = calculate_risk_score(df.iloc[0])

        df.iloc[0]["Prediction"] = prediction
        df.iloc[0]["Confidence"] = confidence
        df.iloc[0]["Risk_Score"] = risk_score

        message = format_telegram_message_proportional(df.iloc[0])
        send_telegram_message(message)

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "confidence": confidence,
            "risk_score": risk_score
        })

    except Exception as e:
        print("❌ EXCEPTION:", traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, data=payload)
    print("🔁 Telegram response:", response.status_code, response.text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
