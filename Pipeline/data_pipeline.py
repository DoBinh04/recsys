from interactions import build_interactions
from load_data import load_events
from preprocess import preprocess_events
from split import temporal_split


class DataPipeline:

    def __init__(self, data_path):
        self.data_path = data_path

    def run(self):

        df = load_events(self.data_path)

        df = preprocess_events(df)

        interactions = build_interactions(df)

        train, val, test = temporal_split(interactions)

        return train, val, test