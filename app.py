from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
h = pd.read_csv("data.csv")


h.dropna(inplace=True)


X = h.drop("label", axis=1)
y = h["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_clf = DecisionTreeClassifier(max_depth=7, random_state=42)
tree_clf.fit(X_train, y_train)

@app.route('/',methods=['GET','POST'])
def signup():
    return render_template('signup.html')

@app.route('/index',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET','POST'])
def result():
    if request.method == "POST":
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        
        user_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        predicted_crop_label = tree_clf.predict(user_input)
    
        return render_template('result.html', predicted_crop_label=predicted_crop_label[0])

if __name__ == '__main__':
    app.run(debug=True)
