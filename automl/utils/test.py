from automl.utils.FeatureEngineer import FeatureEngineer
from automl.utils.Preprocess import Preprocess
from automl.utils.Enums import Task, Metric
from automl.utils.HPOptimizer import BayesianHPOptimizer
import pandas as pd
from sklearn.metrics import accuracy_score
from automl.utils.HPOptimizer.AcquistionFunctions import ExpectedImprovement
from automl.utils.HPOptimizer.SurrogateModels import GaussianProcessSurrogate
import os

df = pd.read_csv('automl/utils/diabetes.csv')
target_variable = 'Outcome'
df =  Preprocess(target_variable=target_variable).fit_transform(df)
df = FeatureEngineer(target_variable=target_variable).fit_transform(df)

bhpo = BayesianHPOptimizer(Task.CLASSIFICATION, 60, Metric.ACCURACY, True, True, surrogate_model=GaussianProcessSurrogate(), acquisition_function=ExpectedImprovement())
bhpo.fit(df.drop(target_variable, axis=1),df[target_variable])
model = bhpo.getOptimalModel()

predictions = model.predict(df.drop(target_variable, axis=1))
print(accuracy_score(df[target_variable], predictions))

