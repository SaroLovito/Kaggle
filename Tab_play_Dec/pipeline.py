import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
# classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# model valuation & selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# hold out
from sklearn.model_selection import train_test_split
# pipeline
from imblearn.pipeline import make_pipeline
# dimension reduction
from sklearn.decomposition import PCA
# standardization
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from my library
from lib import ReduceMemoryUsage
from lib import CapAngularValues
from lib import ColumnAggregation
from lib import ManhattanDistance
from lib import RemoveUnusefulFeature
# calculate execution time
import time
start_time = time.time()

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# import data
df = pd.read_csv('train.csv').sample(1000)


X = df.drop(['Cover_Type'], axis=1)
target = df['Cover_Type']
test = pd.read_csv('test.csv')






# reduce size of the dataframe

ReduceMemoryUsage(df)


# hold out
X_train, X_test, y_train, y_test = train_test_split(X, target, random_state=1)


# Transform dataframe following kaggle instruction on feature engineering (TODO: explain it on notebook)

# execute model to use them as baseline
lr = LogisticRegression()
cv_lr = cross_val_score(lr, X_train, y_train, scoring="accuracy", cv=20)

knn = KNeighborsClassifier()
cv_knn = cross_val_score(knn, X_train, y_train, scoring="accuracy", cv=20)


dt = DecisionTreeClassifier()
cv_dt = cross_val_score(dt, X_train, y_train, scoring="accuracy", cv=20)


xgb = XGBClassifier(verbosity='0')
cv_xgb = cross_val_score(xgb, X_train, y_train, scoring="accuracy", cv=20)

mlp = MLPClassifier()
cv_mlp = cross_val_score(mlp, X_train, scoring="accuracy", cv=20)
baseline_summary = {"LogisticRegression": cv_lr.mean(),
                    "KNearestNeighborhood": cv_knn.mean(),
                    "DecisionTree": cv_dt.mean(),
                    "XGBoost": cv_xgb.mean(),
                    "MPLClassifier": cv_mlp.mean()
                    }

baseline_df = pd.DataFrame(baseline_summary, index=[0])

print(f"Baseline summary\n{baseline_df}")


# model with pipeline using Logistic Regression



pipe = make_pipeline( ReduceMemoryUsage(df),
                      CapAngularValues(column="Aspect"),
                      ManhattanDistance(column='Horizontal_Distance_To_Hydrology'),
                      ManhattanDistance(column='Vertical_Distance_To_Hydrology'),
                      ManhattanDistance(column='Horizontal_Distance_To_Roadways'),
                      ManhattanDistance(column='Horizontal_Distance_To_Fire_Points'),
                      ColumnAggregation(name_new_column='Soil_Type_count', start_with='Soil_Type'),
                      ColumnAggregation(name_new_column='Wilderness_count', start_with='Wilderness'),
                      RemoveUnusefulFeature(column='Soil_Type7', axis=1),
                      RemoveUnusefulFeature(column='Soil_Type15', axis=1),
                      PCA(n_components=10),
                      MinMaxScaler(),
                      LogisticRegression()

                      )

params = [{'logisticregression__C': [0.5, 0.6]
           }]

gs_pipe = GridSearchCV(pipe, param_grid=params, cv=20, scoring='accuracy')
gs_pipe.fit(X_train, y_train)
best_estimators = gs_pipe.best_estimator_

# Best parameters which resulted in the best score
print('Cross Validation Score:', gs_pipe.best_score_)
print('Best Parameters:', gs_pipe.best_params_)

# 2nd pipeline with Decision Tree


pipe_2 = make_pipeline(
    ReduceMemoryUsage(df),
    CapAngularValues(column='Aspect'),
    ManhattanDistance(column='Horizontal_Distance_To_Hydrology'),
    ManhattanDistance(column='Vertical_Distance_To_Hydrology'),
    ManhattanDistance(column='Horizontal_Distance_To_Roadways'),
    ManhattanDistance(column='Horizontal_Distance_To_Fire_Points'),
    ColumnAggregation(name_new_column='Soil_Type_count', start_with='Soil_Type'),
    ColumnAggregation(name_new_column='Wilderness_count', start_with='Wilderness'),
    RemoveUnusefulFeature(column='Soil_Type7',axis=1),
    RemoveUnusefulFeature(column='Soil_Type15',axis=1),
    PCA(n_components=10),
    MinMaxScaler(),
    DecisionTreeClassifier()
)

params = [{
    'decisiontreeclassifier__criterion': ['gini', 'entropy'],
    'decisiontreeclassifier__splitter': ['random', 'best']
}]

gs_pipe2 = GridSearchCV(pipe_2, param_grid=params, cv=20, scoring='accuracy')
gs_pipe2.fit(X_train, y_train)
best_estimators_decision = gs_pipe2.best_estimator_

# Best parameters which resulted in the best score
print('Cross Validation Score:', gs_pipe2.best_score_)
print('Best Parameters:', gs_pipe2.best_params_)

pipe_3 = make_pipeline(
    ReduceMemoryUsage(df),
    CapAngularValues(column = 'Aspect'),
    ManhattanDistance(column = 'Horizontal_Distance_To_Hydrology'),
    ManhattanDistance(column = 'Vertical_Distance_To_Hydrology'),
    ManhattanDistance(column = 'Horizontal_Distance_To_Roadways'),
    ManhattanDistance(column = 'Horizontal_Distance_To_Fire_Points'),
    ColumnAggregation(name_new_column = 'Soil_Type_count', start_with = 'Soil_Type'),
    ColumnAggregation(name_new_column = 'Wilderness_count', start_with = 'Wilderness'),
    RemoveUnusefulFeature(column= 'Soil_Type7',axis=1),
    RemoveUnusefulFeature(column= 'Soil_Type15',axis=1),
    PCA(n_components=10),
    MinMaxScaler(),
    XGBClassifier()
)

params = [{
}]


gs_pipe3 = GridSearchCV(pipe_3,param_grid=params, cv=20, scoring='accuracy')
gs_pipe3.fit(X_train, y_train)
best_estimators_xgb = gs_pipe3.best_estimator_

pipe_4 = make_pipeline(
    ReduceMemoryUsage(df),
    CapAngularValues(column = 'Aspect'),
    ManhattanDistance(column = 'Horizontal_Distance_To_Hydrology'),
    ManhattanDistance(column = 'Vertical_Distance_To_Hydrology'),
    ManhattanDistance(column = 'Horizontal_Distance_To_Roadways'),
    ManhattanDistance(column = 'Horizontal_Distance_To_Fire_Points'),
    ColumnAggregation(name_new_column = 'Soil_Type_count', start_with = 'Soil_Type'),
    ColumnAggregation(name_new_column = 'Wilderness_count', start_with = 'Wilderness'),
    RemoveUnusefulFeature(column= 'Soil_Type7',axis=1),
    RemoveUnusefulFeature(column= 'Soil_Type15',axis=1),
    PCA(n_components=10),
    MinMaxScaler(),
    MLPClassifier()
)

params = [{
}]

gs_pipe4 = GridSearchCV(pipe_4,param_grid=params, cv=20, scoring='accuracy')
gs_pipe4.fit(X_train, y_train)
best_estimators_mlp = gs_pipe3.best_estimator_
report_score = {'pipe': ['LogisticRegression', gs_pipe.scoring, gs_pipe.best_score_],
                'pipe_2': ['DecisionTreeClassifier', gs_pipe2.scoring,gs_pipe2.best_score_],
                'pipe_3': ['XGBClassifier', gs_pipe3.scoring, gs_pipe3.best_score_],
                'pipe_4': ['MLP', gs_pipe4.scoring, gs_pipe4.best_score_]
                }

# Creates DataFrame.
df_score = pd.DataFrame(report_score)
# Print the data
print(df_score)

# Submission file

predictions = gs_pipe3.predict(test)
df_sub = pd.DataFrame(predictions, index=test["Id"], columns=["Cover_Type"])
df_sub.to_csv("Submission_SL.csv")

# print execution time
print("--- %s seconds ---" % (time.time() - start_time))

