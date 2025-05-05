#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
warnings.filterwarnings('ignore')
from feature import FeatureExtraction

#file = open("pickle/model.pkl","rb")
#gbc = pickle.load(file)
#file.close()
#from sklearn.ensemble import GradientBoostingClassifier

#from xgboost import XGBClassifier
#data = pd.read_csv("phishing.csv")
#X = data.drop(["class"],axis =1)
#y = data["class"]
from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#X_train.shape, y_train.shape, X_test.shape, y_test.shape
#data = data.drop(['Index'],axis = 1)
# instantiate the model
#gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

# fit the model 
#gbc.fit(X_train,y_train)
from sklearn.ensemble import GradientBoostingClassifier
data = pd.read_csv("phishing.csv")
data = data.drop(['Index'],axis = 1)
X = data.drop(["class"],axis =1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)
gbc.fit(X_train,y_train)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 

        y_pred =gbc.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
    return render_template("index.html", xx =-1)


if __name__ == "__main__":
    app.run(debug=True)