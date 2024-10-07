import logging
logging.basicConfig(level=logging.INFO)
import json
from flask import Flask, Response
from model import download_data, train_model, get_inference
from config import model_file_path, TOKEN

app = Flask(__name__)

def update_data():
    """Download price data, format data and train model."""
    for i in TOKEN:
        logging.info(f"Downloading and training data for token {i}")
        try:
            files = download_data(i)
            train_model(i)
            logging.info(f"Training completed for token {i}")
        except Exception as e:
            logging.error(f"Error during update_data for token {i}: {str(e)}")
            return False
    return True

@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    logging.info(f"Received inference request for token: {token}")
    
    # Validasi token
    if not token or token.upper() not in TOKEN:
        error_msg = "Token is required" if not token else "Token not supported"
        logging.error(error_msg)
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        # Perform inference using the model for a given token
        inference = get_inference(token.upper())
        logging.info(f"Inference result for {token}: {inference}")
        return Response(str(inference), status=200)
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')


@app.route("/update")
def update():
    """Update data and return status."""
    try:
        success = update_data()
        if success:
            return "0"  # Indicator success
        else:
            return "1"  # Indicator failed
    except Exception as e:
        logging.error(f"Error during update: {str(e)}")
        return "1"


if __name__ == "__main__":
    # First run update_data before the server starts
    logging.info("Starting data update process.")
    update_data()
    logging.info("Starting Flask server.")
    app.run(host="0.0.0.0", port=8000)