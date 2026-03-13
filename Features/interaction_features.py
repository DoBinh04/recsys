import pandas as pd

class InteractionFeatureBuilder:

    def __init__(self, interactions):
        self.df = interactions


    def build_time_features(self):

        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], unit="ms")

        self.df["hour"] = self.df["timestamp"].dt.hour
        self.df["dayofweek"] = self.df["timestamp"].dt.dayofweek

        return self.df


    def build(self):

        return self.build_time_features()