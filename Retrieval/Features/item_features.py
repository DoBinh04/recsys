import pandas as pd

class ItemFeatureBuilder:

    def __init__(self, interactions, category_tree, item_properties):

        self.interactions = interactions
        self.category_tree = category_tree
        self.item_properties = item_properties


    def build_popularity_features(self):
        """
        Tính toán các đặc trưng về độ phổ biến của sản phẩm.
        
        Returns:
            pandas.DataFrame: Một bảng dữ liệu chứa các cột:
                - item_id: ID của sản phẩm.
                - total_views: Tổng lượt xem.
                - total_addtocart: Tổng lượt thêm vào giỏ.
                - total_transactions: Tổng lượt mua thành công.
                - cart_rate: Tỷ lệ thêm vào giỏ (addtocart / views).
                - purchase_rate: Tỷ lệ mua hàng (transactions / views).
        """

        popularity = (
            self.interactions
            .groupby("item_id")
            .agg(
                total_views=("event", lambda x: (x == "view").sum()),
                total_addtocart=("event", lambda x: (x == "addtocart").sum()),
                total_transactions=("event", lambda x: (x == "transaction").sum())
            )
            .reset_index()
        )

        popularity["cart_rate"] = popularity["total_addtocart"] / (popularity["total_views"])
        popularity["purchase_rate"] = popularity["total_transactions"] / (popularity["total_views"])

        popularity = popularity.rename(columns={
            "total_views": "total_views_item",
            "total_addtocart": "total_addtocart_item",
            "total_transactions": "total_transactions_item",
            "purchase_rate": "purchase_rate_item"
        })
        
        return popularity
    
    def build_category_path_map(self):
        parent_map = dict(
            zip(self.category_tree["categoryid"], self.category_tree["parent_id"])
        )

        path_map = {}

        for cat in parent_map:
            path = [cat]
            parent = parent_map.get(cat)

            while pd.notna(parent):
                path.append(parent)
                parent = parent_map.get(parent)

            path_map[cat] = path[::-1]  # root -> leaf

        return path_map    

    def build_category_features(self):

        item_category = self.item_properties[
            self.item_properties["property"] == "categoryid"
        ][["item_id", "value"]]

        item_category = item_category.rename(columns={"value": "categoryid"})

        path_map = self.build_category_path_map()

        item_category["categoryid"] = item_category["categoryid"].astype(int)
        item_category["category_path"] = item_category["categoryid"].map(path_map)

        # root
        item_category["root"] = item_category["category_path"].str[0]

        # leaf
        item_category["leaf"] = item_category["categoryid"]

        # depth
        item_category["depth"] = item_category["category_path"].str.len()
        
        # xử lý missing values
        item_category["root"] = item_category["root"].fillna("unknown")
        item_category["leaf"] = item_category["leaf"].fillna("unknown") 
        item_category["depth"] = item_category["depth"].fillna(0)

        # chỉ giữ các cột cần thiết và loại bỏ duplicates
        item_category = item_category[["item_id", "root", "leaf", "depth"]]
        item_category.drop_duplicates(subset="item_id", inplace=True)
        return item_category[["item_id", "root", "leaf", "depth"]]


    def build(self):
        """
        Hàm tổng hợp: Kết hợp đặc trưng độ phổ biến và đặc trưng danh mục.
        """
        # Lấy các chỉ số thống kê (views, cart rate,...)
        popularity = self.build_popularity_features()
        
        # Lấy các chỉ số phân cấp (root, leaf, depth)
        category = self.build_category_features()

        # Gộp 2 bảng lại dựa trên item_id
        # Dùng how="left" để giữ lại toàn bộ sản phẩm trong bảng popularity
        item_features = popularity.merge(category, on="item_id", how="left")

        return item_features