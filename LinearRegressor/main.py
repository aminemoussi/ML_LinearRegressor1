import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

insurance_dataset = pd.read_csv('medical_insurance/insurance.csv')  #putting everything in a data frame

#print(insurance_dataset.tail())
#print(insurance_dataset.shape)
#print(insurance_dataset.info())
#print(insurance_dataset.isnull().sum())
#print(insurance_dataset.describe())

                                    ##Visualizing the data
sns.set()
plt.figure(figsize=(6, 6))
                ##AGE
#sns.displot(insurance_dataset["age"])

                ##SEX
#print(insurance_dataset["sex"].value_counts())  #number of components of every instance of variable
#sns.countplot(data = insurance_dataset, x = "sex")

                ##BMI
#sns.distplot(insurance_dataset["bmi"])
#plt.title("BMI distribution")

                ##KIDS
#print(insurance_dataset["children"].value_counts())
#sns.countplot(data = insurance_dataset, x = "children")

                ##SMOKER
#sns.countplot(data = insurance_dataset, x = "smoker")

#sns.barplot(data = insurance_dataset, x = "smoker", y = "charges", hue = "sex")   #who pays more smoker or non-smoker

#plt.show()


                                    ##pre-processing the data/ changing alphabetucal values to numerical ones

insurance_dataset.replace({ "sex" : {"male": 0, "female": 1}}, inplace=True)
insurance_dataset.replace({ "smoker": {"yes": 0, "no": 1}}, inplace=True)
insurance_dataset.replace( {"region": {"northwest": 3, "northeast": 2, "southeast": 0, "southwest": 1}}, inplace=True)
#print(insurance_dataset.head())


                                    ##splitting the data

x = insurance_dataset.drop(columns="charges", axis = 1)
y = insurance_dataset["charges"]
#print(x.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.14, random_state=2)
#print(x.shape, x_train.shape, x_test.shape)

                                    ##MODEL
regressor = LinearRegression()
regressor.fit(x_train, y_train)

predicted_data_train = regressor.predict(x_train)

score = metrics.r2_score(y_train, predicted_data_train)

#print("The accuracy for trainning data is :" , (score*100) , "%")

predicted_data = regressor.predict(x_test)

score = metrics.r2_score(y_test, predicted_data)

print("The accuracy for testing data is :" , (score*100) , "%")


                                    ##Building a prediction system

input = (25,1,28.595,0,1,2)

input = np.asarray(input)   #setting up a testing value
#print(input)


input_reshaped = input.reshape(1, -1)#reshaping it to one row many colimns
#print(input_reshaped)

input_prediction = regressor.predict(input_reshaped)

print("The predicted value is :$", input_prediction[0])

