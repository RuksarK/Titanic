import math
import pickle
import numpy as np 
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def MarvellousTitanicLogistic():
    # step 1: Load data
    titanic_data = pd.read_csv('MarvellousTitanicDataset.csv')

    print("First 5 entries from loaded dataset")
    print(titanic_data.head())

    print("Number of passangers are "+str(len(titanic_data)))
    
    #Step 2: Analyze data
    print("Visualisation: Survived and non survied passangers")
    figure()
    target = "Survived"
    
    countplot(data=titanic_data,x=target).set_title("Marvellous Infosystems : Survived and non surviedpassangers")
    show()
    
    print("Visualisation: Survived and non survied passangers based on Gender")
    figure()
    target = "Survived"
    
    countplot(data=titanic_data,x=target, hue="Sex").set_title("Marvellous Infosystems: Survived and non survied passangers based on Gender")
    show()
    
    print("Visualisation: Survived and non survied passangers based on the Passanger class")
    figure()
    target = "Survived"
    
    countplot(data=titanic_data,x=target, hue="Pclass").set_title("Marvellous Infosystems: Survived and non survied passangers based on the Passanger class")
    show()

    print("Visualisation: Survived and non survied passangers based on Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Marvellous Infosystems: Survived and non survied passangers basedon Age")
    show()

    print("Visualisation: Survived and non survied passangers based on the Fare") 
    figure()
    titanic_data["Fare"].plot.hist().set_title("Marvellous Infosystems: Survived and non survied passangers based on Fare")
    show()

    # Step 3: Data Cleaning
    titanic_data.drop("zero", axis = 1, inplace = True)
    
    print("First 5 entries from loaded dataset after removing zero column")
    print(titanic_data.head(5))

    print("Values of data set after removing irrelevent columns") 
    titanic_data.drop(["Sex","sibsp", "Parch", "Embarked"], axis = 1, inplace = True) 
    print(titanic_data.head(5))

    x = titanic_data.drop("Survived",axis = 1)
    y = titanic_data["Survived"]

    # Convert features names to strings
    x.columns = x.columns.astype(str)

    # Step 4: Data Training
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.5,random_state=42)
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    logmodel = LogisticRegression(max_iter=1000)
    
    logmodel.fit(xtrain,ytrain)

    # Step 4: Data Testing of trained model
    x_trained_prediction = logmodel.predict(xtrain)   

    # Step 5: Calculate Accuracy of trained model
    print("Confusion Matrix of Logistic Regression is: ") 
    print(confusion_matrix(ytrain,x_trained_prediction))

    training_data_accuracy = accuracy_score(ytrain,x_trained_prediction)
    print("Accuracy score of training data :",training_data_accuracy)

    # Step 4: Data Testing of model
    x_test_prediction = logmodel.predict(xtest)

    # Step 5: Calculate Accuracy of trained model
    print("Confusion Matrix of Logistic Regression is: ") 
    print(confusion_matrix(ytest,x_test_prediction))

    test_data_accuracy = accuracy_score(ytest,x_test_prediction)
    print("Accurace score of test dagta :",test_data_accuracy)

    

    # Testing the model by giving the specific passenger input

    # input_data = (2,38,71.28,1)  
    # input_data_as_numpy_array = np.asarray(input_data)

    # input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # prediction = logmodel.predict(input_data_reshaped)
    # #print(prediction)
        # if prediction[0]==0:
    #     print("Dead")
    # if prediction[0]==1:
    #     print("Alive")


    pickle.dump(logmodel,open('classifier.pkl','wb'))

    model = pickle.load(open('classifier.pkl','rb'))

def main():
    print("---- Titanic Survival Predictor By Satyam Kashid and Ruksar Khan-----")

    print("Suervised Machine Learning")
    
    print("Logistic Regression on Titanic data set")

    MarvellousTitanicLogistic()

if __name__ == "__main__":
    main()
