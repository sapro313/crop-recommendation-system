from flask import Flask, request, render_template
import numpy as np
import pickle
import os

# Get base directory

BASE_DIR = os.path.dirname(os.path.abspath(**file**))

# Load model and scalers

model_path = os.path.join(BASE_DIR, 'model.pkl')
standscaler_path = os.path.join(BASE_DIR, 'standscaler.pkl')
minmaxscaler_path = os.path.join(BASE_DIR, 'minmaxscaler.pkl')

model = pickle.load(open(model_path, 'rb'))
sc = pickle.load(open(standscaler_path, 'rb'))
ms = pickle.load(open(minmaxscaler_path, 'rb'))

# Initialize Flask app

app = Flask(**name**)

# Home route

@app.route('/')
def index():
return render_template("index.html")

# Prediction route

@app.route("/predict", methods=['POST'])
def predict():
try:
# Get form values and convert to float
N = float(request.form['Nitrogen'])
P = float(request.form['Phosporus'])
K = float(request.form['Potassium'])
temp = float(request.form['Temperature'])
humidity = float(request.form['Humidity'])
ph = float(request.form['Ph'])
rainfall = float(request.form['Rainfall'])

```
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

    # Result message
    result = f"{crop_dict[prediction]} is the best crop to be cultivated 🌾" \
        if prediction in crop_dict else \
        "Sorry, we could not determine the best crop with the provided data."

except Exception as e:
    result = f"Error: {str(e)}"

return render_template("index.html", result=result)
```

# Run app

if **name** == "**main**":
port = int(os.environ.get("PORT", 5000))  # Render sets PORT automatically
app.run(host="0.0.0.0", port=port)
