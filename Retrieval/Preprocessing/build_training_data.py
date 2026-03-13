import pandas as pd
from pathlib import Path

from Retrieval.Features.user_features import UserFeatureBuilder
from Retrieval.Features.item_features import ItemFeatureBuilder


class TrainingDataBuilder:

    def __init__(self, train_path, val_path, test_path, output_dir):

        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_interactions(self):

        print("Loading interaction data...")

        train = pd.read_csv(self.train_path)
        val = pd.read_csv(self.val_path)
        test = pd.read_csv(self.test_path)
        # ko sợ bị data leak vì item properties chỉ để tìm category của item, 
        # ko dùng để tính toán đặc trưng nào liên quan đến tương tác của user với item cả.
        p1 = pd.read_csv("Retrieval/data/item_properties_part1.csv")
        p2 = pd.read_csv("Retrieval/data/item_properties_part2.csv")

        item_props = pd.concat([p1, p2])
        item_props = item_props.rename(columns={
            "itemid": "item_id"
        })

        category_tree = pd.read_csv("Retrieval/data/category_tree.csv")
        category_tree = category_tree.rename(columns={
            "parentid": "parent_id"
        })

        return train, val, test, item_props, category_tree

    def build_features(self, interactions, item_props, category_tree):

        print("Building user features...")
        user_builder = UserFeatureBuilder(interactions)
        user_features = user_builder.build()

        print("Building item features...")
        item_builder = ItemFeatureBuilder(interactions, category_tree, item_props)
        item_features = item_builder.build()

        return user_features, item_features

    def merge_features(self, df, user_features, item_features):

        print("Merging user features...")
        #df = df.merge(user_features, on="user_id", how="left")
        df = df.merge(
            user_features,
            on=["user_id", "item_id", "timestamp"],
            how="left"
        )

        print("Merging item features...")
        df = df.merge(item_features, on="item_id", how="left")

        return df
    

    def handle_missing(self, df):

        print("Handling missing values...")

        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        if "recent_items" in df.columns:
            df["recent_items"] = df["recent_items"].apply(self.pad_items)

        return df
    

    def pad_items(self, x):
        MAX_SEQ = 5

        if not isinstance(x, list):
            x = []

        x = x[-MAX_SEQ:]

        if len(x) < MAX_SEQ:
            x = [0]*(MAX_SEQ-len(x)) + x

        return x

    def save_dataset(self, df, filename):

        output_path = self.output_dir / filename

        print(f"Saving dataset -> {output_path}")

        df.to_parquet(output_path, index=False)

    def build(self):

        train, val, test, item_props, category_tree = self.load_interactions()

        # item features chỉ cần build từ train
        print("Building item features...")
        item_builder = ItemFeatureBuilder(train, category_tree, item_props)
        item_features = item_builder.build()

        # -------- TRAIN --------
        print("Processing train set...")

        user_builder = UserFeatureBuilder(train)
        train_user = user_builder.build()

        train = train.merge(
            train_user,
            on=["user_id", "item_id", "timestamp"],
            how="left"
        )

        # bỏ các sample không có history
        train = train[train["recent_items"].map(len) > 0]

        train = train.merge(item_features, on="item_id", how="left")
        train = self.handle_missing(train)


        # -------- VAL --------
        print("Processing validation set...")

        train_val = pd.concat([train, val]).sort_values("timestamp")

        user_builder = UserFeatureBuilder(train_val)
        val_user = user_builder.build()

        val = val.merge(
            val_user,
            on=["user_id", "item_id", "timestamp"],
            how="left"
        )

        val = val.merge(item_features, on="item_id", how="left")
        val = self.handle_missing(val)


        # -------- TEST --------
        print("Processing test set...")

        train_val_test = pd.concat([train_val, test]).sort_values("timestamp")

        user_builder = UserFeatureBuilder(train_val_test)
        test_user = user_builder.build()

        test = test.merge(
            test_user,
            on=["user_id", "item_id", "timestamp"],
            how="left"
        )

        test = test.merge(item_features, on="item_id", how="left")
        test = self.handle_missing(test)


        self.save_dataset(train, "train_ready.parquet")
        self.save_dataset(val, "val_ready.parquet")
        self.save_dataset(test, "test_ready.parquet")

        print("Training dataset built successfully!")

if __name__ == "__main__":

    builder = TrainingDataBuilder(
        train_path="Retrieval/data/train.csv",
        val_path="Retrieval/data/val.csv",
        test_path="Retrieval/data/test.csv",
        output_dir="Retrieval/data/"
    )

    builder.build()