import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import auc,roc_curve,roc_auc_score


sns.set()
# Used to increase display size of columns to max
#pd.set_option('display.max_columns', None)

# Loading and checking data
df = pd.read_csv('Chronic_Kidney_Dsease_data.csv')
#print(df.head())
#print(df.tail())
#print(df.shape)

# null and duplicate values check
#print("Is there a null value check")
#print(df.isnull().sum())
#print("-------------\nIs there a duplicate value check")
#print(df.duplicated().sum())

# statistical summary
#print("Statistical summary")
#print(df.describe())

# Dropping patientID and DoctorinCharge columns
df = df.drop(["PatientID", "DoctorInCharge"], axis=1)
#print(df.head())

# Visualization of patients with kidney disease. Also target distribution
#category_count = df['Diagnosis'].value_counts()

#plt.pie(category_count, labels = category_count.index, autopct = '%.0f%%')
#plt.axis('equal')

#plt.title('Target Distribution', pad = 30)
#plt.tick_params(labelbottom = False, labeltop = False)
#plt.show()

# Age visualization
#sns.histplot(x='Age', data=df)
#plt.show()

# Gender visualization
#sns.countplot(x='Gender', data=df)
#plt.show()

# Correlation heatmap visualization
#corr = df.corr()
#plt.figure(figsize = (20, 16))
#sns.heatmap(data = corr, cmap = 'coolwarm', annot = True, fmt = '.2f', annot_kws = {'fontsize': 6} )
#plt.title('C͟o͟r͟r͟e͟l͟a͟t͟i͟o͟n͟ ͟H͟e͟a͟t͟m͟a͟p͟ ͟V͟i͟s͟u͟a͟l͟i͟z͟e͟d͟', fontsize = 20, pad = 20)
#plt.xlabel('Features', fontsize = 15)
#plt.ylabel('Features', fontsize = 15)
#plt.show()

X = df.drop(['Diagnosis'], axis=1)
Y = df['Diagnosis']
#print(X.shape , Y.shape)

# Normalization (transforming all x values to a value between 0 and 1)
min_max_scaler = MinMaxScaler()
x_scaled_minmax = min_max_scaler.fit_transform(X)
X = pd.DataFrame(x_scaled_minmax, columns = X.columns)
#print(X.head())

# Splitting the data
xtrain,xtest,ytrain,ytest = train_test_split(X,Y, test_size = 0.30 , random_state=10, shuffle=True)
#print(xtrain.shape, ytrain.shape)
#print(xtest.shape, ytest.shape)

# Creating model and training
lr_model = LogisticRegression(random_state=10)
lr_model.fit(xtrain,ytrain)

ypred = lr_model.predict(xtest)
#print(ypred[0:20])
#print(list(ytest[0:20]))

# Predicting probability of 0 and 1
pred_prb = lr_model.predict_proba(xtest)
#print(pred_prb[0:5])
lr_pred_prb = lr_model.predict_proba(xtest)[:,1]
#print(lr_pred_prb)
# Accuracy
accuracy_lr = accuracy_score(ytest,ypred)
print("Accuracy by built-in function: {}".format(accuracy_lr))

# Classification Report
print(classification_report(ytest,ypred))

# ROC AUC Curve
fpr,tpr,threshold=roc_curve(ytest,lr_pred_prb)
auc_lr=roc_auc_score(ytest,lr_pred_prb)
print(auc_lr)

def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8,6))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='best')

sns.set_context('poster')
plot_roc_curve(fpr,tpr,label='AUC = %0.3f'% auc_lr)
plt.show()
