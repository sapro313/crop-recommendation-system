from flask import Flask, request, render_template
import numpy as np
import pickle

# Load model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=[ 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form values and convert to float
            N = float(request.form['Nitrogen'])
            P = float(request.form['Phosporus'])
            K = float(request.form['Potassium'])
            temp = float(request.form['Temperature'])
            humidity = float(request.form['Humidity'])
            ph = float(request.form['Ph'])
            rainfall = float(request.form['Rainfall'])

            # Prepare feature array
            features = np.array([N, P, K, temp, humidity, ph, rainfall]).reshape(1, -1)

            # Apply MinMaxScaler then StandardScaler
            scaled = ms.transform(features)
            final = sc.transform(scaled)

            # Make prediction
            prediction = model.predict(final)[0]

            # Crop dictionary
            crop_dict = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
                6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
                11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
                16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
                20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }

            # Result
            result = f"{crop_dict[prediction]} is the best crop to be cultivated 🌾" \
                     if prediction in crop_dict else \
                     "Sorry, we could not determine the best crop to be cultivated with the provided data."

        except Exception as e:
            # Catch errors (like missing form values)
            result = f"Error: {str(e)}"

        return render_template("index.html", result=result)

    # GET request returns empty form
    return render_template("index.html", result=None)


# Run Flask
if __name__ == "__main__":
    app.run(debug=True)
