from tracemalloc import start
import numpy as np

#pipeline
import pandas as pd

#classifier
from sklearn.linear_model import LogisticRegression

#model evaluation & selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#hold out
from sklearn.model_selection import train_test_split

#pipeline
from imblearn.pipeline import make_pipeline

#dimension reduction
from sklearn.decomposition import PCA

#standardization which standardize all items in {0,1}
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#calculate execution time
import time
start_time = float(time.time())


#import data

df = pd.read_csv('C:/Users/Utente/Desktop/Data Science/Data Science Academy/Saro/ML_exercise/Titanic/train_pipeline')


train = df.drop(['Survived'], axis = 1)
target = df.Survived
test = pd.read_csv('C:/Users/Utente/Desktop/Data Science/Data Science Academy/Saro/ML_exercise/Titanic/test_pipeline')


#hold out 
X_train,X_test, y_train, y_test = train_test_split(train, target ,random_state = 1)

print('Starting pipeline: ready to discover predictions')


pipe = make_pipeline(
    PCA(n_components = 4),
    LogisticRegression()

)

#TO-DO: I'll have to understand logit hyper parameters
params = [{
        'logisticregression__penalty': ['none','l2'],


}]


#logit_pipe = GridSearchCV(pipe, param_grid = params, cv=5, scoring = 'accuracy')
logit_pipe = GridSearchCV(pipe,param_grid= params, cv=5, scoring = 'accuracy')
logit_pipe.fit(X_train, y_train)

print('-' * 50)

report_score = {
    'pipe_logit':['Logistic Regression', logit_pipe.scoring, logit_pipe.best_score_]
}

#Creates DataFrame
df_score = pd.DataFrame(report_score)

#Print the data
print(report_score)


#Submission file
predictions = logit_pipe.predict(test)
df_sub = pd.DataFrame(predictions, index=test["PassengerId"], columns=["Survived"])
df_sub.to_csv("Submission.csv")


#print execution time
execution_time = float(time.time() - start_time)
print(f"---{execution_time}  seconds ---"  )

