# EDA Packages
import pandas as pd
import numpy as np
import random
# Machine Learning Packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load Url Data
urls_data = pd.read_csv("C:/Users/rakes/Desktop/urldet1/urldata.csv")

# Custom Tokenizer Function
def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')  # Make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')  # Make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0, len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')  # Make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))  # Remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')  # Removing .com since it occurs a lot of times and should not be included in features
    return total_Tokens

# Labels
y = urls_data["label"]

# Features
url_list = urls_data["url"]

# Using Custom Tokenizer
vectorizer = TfidfVectorizer(tokenizer=makeTokens)
# Store vectors into X variable as our XFeatures
X = vectorizer.fit_transform(url_list)

# Split into training and testing dataset 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
logit = LogisticRegression()  # Using logistic regression
logit.fit(X_train, y_train)

# Accuracy of Our Model
print("Accuracy: ", logit.score(X_test, y_test))

# Predictions
X_predict = ["google.com/search=jcharistech",
              "google.com/search=faizanahmad",
              "pakistanifacebookforever.com/getpassword.php/",
              "www.radsport-voggel.de/wp-admin/includes/log.exe",
              "ahrenhei.without-transfer.ru/nethost.exe ",
              "www.itidea.it/centroesteticosothys/img/_notes/gum.exe"]

X_predict = vectorizer.transform(X_predict)
New_predict = logit.predict(X_predict)
print("Predictions 1:", New_predict)

X_predict1 = ["www.buyfakebillsonlinee.blogspot.com",
               "www.unitedairlineslogistics.com",
               "www.stonehousedelivery.com",
               "www.google.com"]

X_predict1 = vectorizer.transform(X_predict1)
New_predict1 = logit.predict(X_predict1)
print("Predictions 2:", New_predict1)
