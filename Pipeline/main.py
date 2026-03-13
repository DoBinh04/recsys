from data_pipeline import DataPipeline

pipeline = DataPipeline("Data/raw/events.csv")

train, val, test = pipeline.run()

train.to_csv("Data/processed/train.csv", index=False)
val.to_csv("Data/processed/val.csv", index=False)
test.to_csv("Data/processed/test.csv", index=False)