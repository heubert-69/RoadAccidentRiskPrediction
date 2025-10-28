import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer



def feature_engineer(df):
    df = df.copy()

    df["num_reported_accidents_log"] = np.log1p(df["num_reported_accidents"])
    df["traffic_density"] = df["num_lanes"] / df["speed_limit"]
    df["curvature_intenisty"] = df["curvature"] * df["speed_limit"]
    df["risk_exposure"] = df["num_reported_accidents"] / df["num_lanes"]
    df["congestion_risk"] = np.where((df["holiday"] == 0) & (df["school_season"] == 1), 1, 0)
    df["mean_accidents"] = df.groupby("road_type")["num_reported_accidents"].transform("mean")
    df["mean_accidents_deviation"] = df["num_reported_accidents"] - df["mean_accidents"]

    
    return df