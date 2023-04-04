import numpy as np
import pandas as pd
import sys
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

df_train = pd.read_csv('data/airbnb_train.csv')
df_test = pd.read_csv('data/airbnb_test.csv')

X_train = df_train.drop(columns=['log_price'])
y_train = df_train['log_price']
X_test = df_test.drop(columns=['log_price'])
y_test = df_test['log_price']

if sys.argv[1] == 'elasticnet':
    pipe = Pipeline(steps =[\
        ('scaler', StandardScaler()),\
        ('model', ElasticNet())
    ])
    params = {'model__alpha': 10.**np.arange(-3, 4),\
            'model__l1_ratio': np.linspace(0., 1., 6),\
            'model__max_iter': [8000]}
    save_name = 'enet.joblib'
elif sys.argv[1] == 'randomforest':
    pipe = Pipeline(steps =[\
    ('model', RandomForestRegressor())
    ])
    params = {'model__n_estimators': np.arange(500, 1100, 100),
              'model__max_depth': np.arange(500, 1100, 100)}
    save_name = 'rfr.joblib'
elif sys.argv[1] == 'xgboost':
    pipe = Pipeline(steps =[\
        ('model', XGBRegressor())
    ])
    params = {'model__n_estimators': np.arange(300, 650, 50),\
            'model__max_depth': np.arange(6, 10),
            'model__learning_rate': [0.1]}
    save_name = 'xgb.joblib'

grid_search = GridSearchCV(pipe, param_grid=params, cv=5) # Set GridSearchCV with 5-fold cross-validation
fitted_search = grid_search.fit(X_train, y_train) # Fit the GridSearchCV with the data
best_estimator = fitted_search.best_estimator_
best_cv_score = fitted_search.best_score_
joblib.dump((best_estimator, best_cv_score), 'model/' + save_name)
