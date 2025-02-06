from enum import Enum

class Task(Enum):
    REGRESSION = 1
    CLASSIFICATION = 2
    CLUSTERING = 3
    TIME_SERIES = 4
    @classmethod
    def parse(cls, display_name: str) -> 'Task':
        display_to_enum = {
            "Regression": cls.REGRESSION,
            "Classification": cls.CLASSIFICATION,
            "Clustering": cls.CLUSTERING,
            "Time Series": cls.TIME_SERIES,
        }
        return display_to_enum[display_name]

class Metric(Enum):
    RMSE = 1
    MAE = 2
    ACCURACY = 3
    F1 = 4
    PRECISION = 5
    RECALL = 6
    ROC_AUC = 7
    LOG_LOSS = 8
    SILHOUETTE = 9
    DAVIES_BOULDIN = 10
    CALINSKI_HARABASZ = 11
    BIC = 12
    AIC = 13
    BIC_AIC = 14
    MSE = 15
    R2 = 16