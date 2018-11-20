from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


app = Flask(__name__,template_folder='templates')
app.config['DEBUG']=True

@app.route('/')
def index():
   return render_template('index.html')

list  =[]
colnames=['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20']


@app.route('/uploader', methods = ['GET', 'POST'])
def submit():
    if request.method == 'POST':
        
        result = request.form
        
        for key, value in result.items():
            list.append(value)
            
        with open("df.csv", 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(list)

        sample = pd.read_csv('df.csv', names=colnames)
        sample.to_csv("samplee.csv")
        y_pred = predict(sample)     
        return render_template("result.html", result = y_pred)

if __name__ == '__main__':
    app.run(threaded=True, use_reloader=False)



def predict(x_test_sample):
    
    
    data = pd.read_csv('German_Categorical.csv')


    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    data.iloc[:,0:1] = labelencoder.fit_transform(data.iloc[:,0:1])
    data.iloc[:,2:3] = labelencoder.fit_transform(data.iloc[:,2:3])
    data.iloc[:,3:4] = labelencoder.fit_transform(data.iloc[:,3:4])
    data.iloc[:,5:6] = labelencoder.fit_transform(data.iloc[:,5:6])
    data.iloc[:,6:7] = labelencoder.fit_transform(data.iloc[:,6:7])
    data.iloc[:,9:10] = labelencoder.fit_transform(data.iloc[:,9:10])
    data.iloc[:,8:9] = labelencoder.fit_transform(data.iloc[:,8:9])
    data.iloc[:,11:12] = labelencoder.fit_transform(data.iloc[:,11:12])
    data.iloc[:,13:14] = labelencoder.fit_transform(data.iloc[:,13:14])
    data.iloc[:,14:15] = labelencoder.fit_transform(data.iloc[:,14:15])
    data.iloc[:,16:17] = labelencoder.fit_transform(data.iloc[:,16:17])
    data.iloc[:,18:19] = labelencoder.fit_transform(data.iloc[:,18:19])
    data.iloc[:,19:20] = labelencoder.fit_transform(data.iloc[:,19:20])



    x=data.iloc[:,0:20]
    y=data.iloc[:,20:21]
    y.loc[y['V21'] > 1, 'V21'] = 0


    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3,shuffle = False)


    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    #from sklearn.tree import DecisionTreeClassifier

    logisticRegr = LogisticRegression()
    #svclassifier = SVC(kernel='linear')
    #classifier = KNeighborsClassifier(n_neighbors=5)
    #classifier1 = DecisionTreeClassifier()


    #Logistic Regression
    logisticRegr.fit(x_train,y_train)

    #SVM Model
    #svclassifier.fit(x_train,y_train)

    #KNN Model
    #classifier.fit(x_train, y_train)

    #Decision tree model

    #classifier1.fit(x_train, y_train)






    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    #Accuracy for logisticRegression
    #accuracy=logisticRegr.score(x_test,y_test)
    #accuracy

    y_pred=logisticRegr.predict(x_test_sample)
    
    #ConfusionMatrix for Logistic
    #confusion_matrix(y_test, y_pred)
    #Precision-recall
    #classification_report(y_test, y_pred)


    #accuracy=svclassifier.score(x_test,y_test)
    #print(accuracy)

    #y_pred = svclassifier.predict(x_test)
    #confusion_matrix(y_test, y_pred)


    #accuracy=classifier.score(x_test,y_test)
    #print(accuracy)

    #y_pred = classifier.predict(x_test)
    #confusion_matrix(y_test, y_pred)
    
    return y_pred

