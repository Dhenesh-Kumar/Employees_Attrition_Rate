

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve

"""1. Problem Statement: Business Understand"""

# Load and read The Data Set
dataset = pd.read_excel("Attrition Rate-Dataset.xlsx")
Output = pd.read_excel("Attrition Rate-Dataset.xlsx")
dataset
"""2. Data Cleaning:

Clean the dataset by handling missing values, removing duplicates, correcting inconsistencies, and formatting data appropriately.
"""

dataset.isnull().sum()

dataset.info()

dataset.value_counts()

# Describe the Summary 
dataset.describe()

#Drop the unwanted columns : "employee Name" and "Employee ID":
DataFrame = dataset.drop(["EmployeeName","EmployeeID"], axis = 1)
DataFrame

# Converting categorical values into numerical values using Label Encoding : 
LabelEncoder = LabelEncoder()
data = DataFrame.apply(LabelEncoder.fit_transform)
data

"""4.EDA - Exploratory Data Analysis:

Perform exploratory data analysis to gain insights into the dataset. This can involve statistical summaries, data visualization,(plotting) and identifying patterns or correlations within the data.
"""

# Split the data into dependent variable or Target Variable and independent Variable
X = data.drop(['Attrition'],axis = 1)
y = data.Attrition
X.head()

# Variance and Standard deviation
var = data.Attrition.var()
print('Varriance : ', var)

# Standard deviation :
std = data.Attrition.std()
print('Standard Deviation : ', std)

# Skewness 
skew = data.Attrition.skew()
print('Skewness : ', skew)

# kurtossis:
kurt = data.Attrition.kurt()
print('Kurtossis : ', kurt)

# Univariant plot
plt.figure(figsize=(8,6))
plt.hist(data.Designation)
plt.xlabel("Designation")
plt.ylabel("Training Hours")
plt.show()

plt.figure(figsize=(8,6))
plt.hist(data.PercentSalaryHike)
plt.xlabel("Designation")
plt.ylabel("Percent Salary Hike")
plt.show()

plt.figure(figsize=(8,6))
plt.hist(data.TraininginHours)
plt.xlabel("Designation")
plt.ylabel("Training in Hours")
plt.show()

plt.figure(figsize=(8,6))
plt.hist(data.MonthlySalary)
plt.xlabel("Designation")
plt.ylabel("Monthly Salary")
plt.show()

plt.boxplot(data.Tenure)
plt.show()

# Data Destribution :
from scipy import stats
from matplotlib import pylab, mlab, pyplot
stats.probplot(data.Designation, plot=pylab)
stats.probplot(data.PercentSalaryHike, dist="norm",plot=pylab)
stats.probplot(data.TraininginHours, dist="norm",plot=pylab)
stats.probplot(data.TraininginHours, dist="norm",plot=pylab)

# Heatmap for the attrition data
sns.heatmap(data.corr(), annot = True, fmt = '.0%')

# Splitting the data into two types train data  and test data.
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.30, random_state = 0)

"""Decision Tree Classifier()"""

classifier = DecisionTreeClassifier()
classifier.fit(train_x,train_y)
predictions = classifier.predict(test_x)
print("Accuracy of the model using Decision Tree Clasifier on test Data:", accuracy_score(test_y,predictions))

predictions = classifier.predict(train_x)
print("Accuracy of the model using Decision Tree Clasifier on train Data:", accuracy_score(train_y,predictions))

"""Cross Value Score()"""

from sklearn.model_selection import cross_val_score
y1 = np.mean(cross_val_score(classifier,test_x, test_y, cv=10))
y2 = np.mean(cross_val_score(classifier,train_x, train_y, cv=10))

print("Trian Data - RF with CV : ", y2*100)
print("Test Data  - RF with CV : ", y1*100)

"""XGB Claasifier()"""

# import xgboost as xgb
# model = xgb.XGBClassifier()
# model.fit(train_x,train_y)
# predictions = model.predict(test_x)
# print("Accuaracy of the model using XG Classifier on test data: ", accuracy_score(test_y,predictions))

y3 = np.mean(cross_val_score(classifier,test_x, test_y, cv=10))
y4 = np.mean(cross_val_score(classifier,train_x, train_y, cv=10))

print("Trian Data - RF with CV : ", y4*100)
print("Test Data  - RF with CV : ", y3*100)

"""Building the model using RandomForestClassifier()"""

# Building the model using RandomForestClassifier :
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=2, n_estimators=15, criterion="entropy")

# model fitting on the train data
rf.fit(train_x, train_y)

# Predicting on the test data
pred_test_rf = rf.predict(test_x)
pred_test_rf

# Predicting on the test data
pred_train_rf = rf.predict(train_x)
pred_train_rf

# fpr, tpr and theshold values
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
fpr_test_rf, tpr_test_rf, thresholds_test_rf = roc_curve(test_y, pred_test_rf)

# Accuracy for test data
accuracy_test_RF = np.mean(pred_test_rf == test_y)
accuracy_test_RF

# Accuracy for train data
accuracy_train_RF = np.mean(pred_train_rf == train_y)
accuracy_train_RF

i = np.arange(len(tpr_test_rf))
roc_test_rf = pd.DataFrame({'fpr_test_rf' : pd.Series(fpr_test_rf, index=i),
                            'tpr_test_rf' : pd.Series(tpr_test_rf, index = i), 
                            '1-fpr_test_rf' : pd.Series(1-fpr_test_rf, index = i), 
                            'tf_test_rf' : pd.Series(tpr_test_rf - (1-fpr_test_rf), index = i), 
                            'thresholds_test_rf' : pd.Series(thresholds_test_rf, index = i)})
roc_test_rf.iloc[(roc_test_rf['tf_test_rf']-0).abs().argsort()[:1]]

import pylab as pl

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc_test_rf['tpr_test_rf'], color = 'red')

# pl.plot(roc_test_rf['1-fpr_test_rf'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

# Areaunder curve
roc_auc = auc(fpr_test_rf, tpr_test_rf)
print("Area under the ROC curve : ", roc_auc)

# Confusion matrics
from sklearn.metrics import accuracy_score, confusion_matrix
cm_test_rf = confusion_matrix(pred_test_rf, test_y)
cm_test_rf

# Sencitivity(True positive rate)
Sensitivity = cm_test_rf[0,0]/(cm_test_rf[0,0] + cm_test_rf[0,1])
print('sensitivity:', Sensitivity)

# Specificity(True nagative)
specitivity = cm_test_rf[1,1]/(cm_test_rf[1,0] + cm_test_rf[1,1])
print('Specificity:', specitivity)

# Model predict on train data
pred_train_rf = rf.predict(train_x)
pred_train_rf

y5 = np.mean(cross_val_score(classifier,test_x, test_y, cv=10))
y6 = np.mean(cross_val_score(classifier,train_x, train_y, cv=10))

print("Trian Data - RF with CV : ", y6*100)
print("Test Data  - RF with CV : ", y5*100)

classifier.fit(train_x,train_y)

# making prediction with our model 
predictions = classifier.predict(train_x)
predictions

# Output = pd.read_excel(r"/content/Attrition Rate-Dataset.xlsx")

# For Building model obeject ( UI ) 

import pickle
train_x['pred'] = y2
test_x['pred']  = y1

newTable =  pd.concat([train_x,test_x],axis=0)  

df4 = pd.merge(newTable, Output[['EmployeeID','EmployeeName']], left_index=True, right_index=True)
with open('finalModel_randForest.pkl', 'wb') as f:
    pickle.dump(df4, f)