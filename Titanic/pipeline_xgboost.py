import numpy as np

#pipeline
import pandas as pd

# classifier
from xgboost import XGBClassifier

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

df = pd.read_csv('train_pipeline')


train = df.drop(['Survived'], axis = 1)
target = df.Survived
test = pd.read_csv('test_pipeline')


#hold out 
X_train,X_test, y_train, y_test = train_test_split(train, target ,random_state = 1)

print('Starting pipeline: ready to discover predictions')



   

#XGBoost parameters based on kaggle lecture: https://www.kaggle.com/prashant111/a-guide-on-xgboost-hyperparameters-tuning



xgb_pipe = make_pipeline(
    PCA(n_components=4),
    XGBClassifier() 


)

params = [{
           "xgbclassifier__booster": ["gbtree", "dart"],
           "xgbclassifier__eta": [0.01, 0.2],  # It suggest these values
           "xgbclassifier__tree_method": ["auto", "exact"],
           "xgbclassifier__eval_metric": ["error", "auc"],

}]




gs_pipe = GridSearchCV(xgb_pipe,param_grid= params, cv=5, scoring = 'accuracy')
gs_pipe.fit(X_train, y_train)

print('-' * 50)

report_score = {
    'pipe_xgb':['XGBoost', gs_pipe.scoring, gs_pipe.best_score_]
}

#Creates DataFrame
df_score = pd.DataFrame(report_score)

#Print the data
print(report_score)


#Submission file
predictions = gs_pipe.predict(test)
df_sub = pd.DataFrame(predictions, index=test["PassengerId"], columns=["Survived"])
df_sub.to_csv("Submission.csv")


#print execution time
execution_time = float(time.time() - start_time)
print(f"---{execution_time}  seconds ---"  )

