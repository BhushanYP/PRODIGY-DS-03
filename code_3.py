import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

df = pd.read_csv('bank-additional.csv', delimiter = ';')
df.rename(columns = {'y':'deposit'},inplace = True)

# Understanding The Dataset
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.dtypes.value_counts())
print(df.info())

# Data Cleaning and Data Preprocessing :

# Handling Duplicate values
print(df.duplicated().sum())
# Handling Null/Missing values
print(df.isna().sum())

# Extracting Numerical and Categorical Columns

cat_cols = df.select_dtypes(include = 'object').columns
print(cat_cols)
num_cols = df.select_dtypes(exclude = 'object').columns
print(num_cols)

# Descriptive Statistical Analysis

print(df.describe())
print(df.describe(include = 'object'))

# Data Visualization :

# Visualizing Numerical Columns using Histplot
df.hist(figsize=(10,10), color = 'red')
plt.show()

# Visualizing Categorial Data using Barplot

for feature in cat_cols:
    plt.figure(figsize=(5,5))
    sns.countplot(x = feature, data = df, palette = 'crest')
    plt.title(f'Bar Plot of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation = 90)
    plt.show()

# Plotting Boxplot and Checking for outliners

df.plot(kind = 'box',subplots = True,layout = (2,5),figsize = (20,10),color = 'blue')
plt.show()

# Removing Outliers 

column = df[['age','campaign','duration']]
q1 = np.quantile(column,0.25)
q3 = np.quantile(column,0.75)
iqr = q3 - q1
lower_bound = q1 - (1.5*iqr)
upper_bound = q3 + (1.5*iqr)
print(iqr,upper_bound,lower_bound)
df[['age','campaign','duration']] = column[(column > lower_bound) & (column < upper_bound)]

# Plotting Boxplot after removing outliners

df.plot(kind = 'box',subplots = True,layout = (2,5),figsize = (20,10),color = '#808000')
plt.show()

# Feature selection using Correlation

high_corr_cols = ['emp.var.rate','euribor3m','nr.employed']

df1 = df.copy()
print(df1.columns)

df1.drop(high_corr_cols,inplace = True,axis = 1)
print(df1.columns)

print(df1.shape)

# Label Encoding :

# Coversion of catogorial columns into numerical columns using label encoder

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df_encoded = df1.apply(lb.fit_transform)
print(df_encoded)

print(df_encoded['deposit'].value_counts())

#Selecting Independent and Dependent Variables

x = df_encoded.drop('deposit',axis = 1)
y = df_encoded['deposit']
print(x.shape)
print(y.shape)
print(type(x))
print(type(y))

# Training and Testing Dataset

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25 , random_state = 1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#Function to compute Confusion Matrix,Classification Report and to generate training and testing scores

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

def eval_model(y_test,y_pred):
    acc = accuracy_score(y_test,y_pred)
    print("\nAccuracy Score : ",acc)
    cm = confusion_matrix(y_test,y_pred)
    print("\nConfusion Matrix : ",cm)
    print("\nClassification report\n",classification_report(y_test,y_pred))
    
def mscore(model):
    train_score = model.score(x_train,y_train)
    test_score = model.score(x_test,y_test)
    print("\nTraining Score : ",train_score)
    print("\nTesting Score : ",test_score)

# Decision Tree Classifier :

# Importing Decision tree Library
from sklearn.tree import DecisionTreeClassifier

# Building Decision Tree Classifier Model
dt = DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_split=10)
dt.fit(x_train,y_train)

#Evaluating Training and Testing Accuracy
print(mscore(dt))

#Generating Predictions
ypred_dt = dt.predict(x_test)
print(ypred_dt)

#Evaluate the Model ---- Confusion Matrix , Classification report , Accuracy
print(eval_model(y_test,ypred_dt))

#Plotting Decision Tree
from sklearn.tree import plot_tree

cn = ['no','yes'] #class names
fn = x_train.columns #feature names
print(cn)
print(fn)

feature_names = df.columns.tolist()
class_names = ["class_0","class_1"]
plt.figure(figsize = (30,20))
plot_tree(dt,feature_names=feature_names,class_names=class_names,filled = True,fontsize = 7)
plt.show()

# Decision Tree Classifier 2 (using Entropy Criteria) :

#Building Decision Tree Classifier Model
dt1 = DecisionTreeClassifier(criterion='entropy',max_depth=4,min_samples_split=15)
dt1.fit(x_train,y_train)

#Evaluating Training and Testing Accuracy
print(mscore(dt1))

#Generating Predictions
ypred_dt1 = dt1.predict(x_test)
print(ypred_dt1)

#Evaluate the Model ---- Confusion Matrix , Classification report , Accuracy
print(eval_model(y_test,ypred_dt1))

#Plotting Decision classifier tree
plt.figure(figsize = (25,20))
plot_tree(dt1,feature_names=fn.tolist(),class_names=cn,filled = True,fontsize = 10)
plt.show()