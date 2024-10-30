from sklearn.pipeline import Pipeline
from pandas import DataFrame
from .FeatureEngineer import FeatureEngineer
from .Preprocess import Preprocess


def createPipeline(df: DataFrame, target_variable: str) -> Pipeline:
    print("Creating the pipeline")
    pipeline = Pipeline([
        ('preprocess', Preprocess(target_variable=target_variable)),
        ('feature_engineer', FeatureEngineer(target_variable=target_variable)),
    ])

    # Fit the pipeline on the dataset
    pipeline.fit(df, df[target_variable])

    # Apply the transformations
    transformed_df = pipeline.transform(df)

    return pipeline
