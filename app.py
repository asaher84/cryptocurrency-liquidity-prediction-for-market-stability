from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and label encoder
model = pickle.load(open('model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        price = float(request.form['price'])
        volume = float(request.form['volume'])
        mkt_cap = float(request.form['mkt_cap'])
        change_1h = abs(float(request.form['change_1h']))
        change_24h = abs(float(request.form['change_24h']))
        change_7d = abs(float(request.form['change_7d']))

        # Feature engineering
        price_volatility = change_24h
        liquidity_score = volume / (price_volatility + 1e-6)
        log_volume = np.log1p(volume)
        log_mkt_cap = np.log1p(mkt_cap)

        # Final features (order must match training)
        features = [[

            price,
            price_volatility,
            change_1h,
            change_7d,
            log_mkt_cap,
            volume,
            liquidity_score

        ]]

        # Predict
        prediction = model.predict(features)
        label = label_encoder.inverse_transform(prediction)[0]

        return render_template('index.html', prediction_text=f"Predicted Liquidity Level: {label}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
