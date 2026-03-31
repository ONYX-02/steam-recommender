import os
from flask import Flask, jsonify
from dotenv import load_dotenv

load_dotenv()

app=Flask(__name__)

@app.route("/")
def health_check():
    key_status = "loaded" if os.getenv("STEAM_API_KEY") else "missing"
    return jsonify({
        "status": "online",
        "project": "steam recommender",
        "api_key": key_status

    })

if __name__=="__main__":
    app.run(debug=True)