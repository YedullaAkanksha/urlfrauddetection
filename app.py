from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Define the makeTokens function

def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')  # make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')  # make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0, len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')  # make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))  # remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')  # removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens

app = Flask(__name__)

# Load the model and vectorizer
model = LogisticRegression()
vectorizer = TfidfVectorizer(tokenizer=makeTokens)

# Load the dataset
urls_data = pd.read_csv("C:/Users/Akanksha/Documents/urldet1/urldata.csv")

# Train the model (you can add more preprocessing here if needed)
X = vectorizer.fit_transform(urls_data["url"])
y = urls_data["label"]
model.fit(X, y)

# Define the home route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        url = request.form["url"]
        # Vectorize the input URL
        X_predict = vectorizer.transform([url])
        # Make a prediction
        prediction = model.predict(X_predict)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

