import pandas as pd
from collections import deque

class UserFeatureBuilder:
    """
    Class này dùng để xây dựng các đặc trưng (features) đại diện cho người dùng (User Tower).
    Dựa trên lịch sử tương tác để hiểu hành vi và sở thích gần đây của họ.
    """

    def __init__(self, interactions):
        """
        Khởi tạo builder với dữ liệu tương tác thô.

        Đầu vào (Input):
            interactions (pd.DataFrame): Bảng chứa lịch sử tương tác người dùng.

            Cần có các cột: 'user_id', 'item_id', 'event', 'timestamp'.
        """
        self.df = interactions

    # def build_recent_item_features(self, k=10):
    #     """
    #     Tạo danh sách các sản phẩm mà người dùng đã tương tác gần đây nhất.

    #     Đầu vào (Input):
    #         k (int): Số lượng sản phẩm cuối cùng muốn giữ lại. Mặc định là 10.

    #     Đầu ra (Output):
    #         pd.DataFrame: Bảng gồm 2 cột:
    #             - 'user_id': ID người dùng.
    #             - 'recent_items': Một danh sách (List) chứa tối đa k ID sản phẩm mới nhất.
    #     """
    #     recent_items = (
    #         self.df
    #         .sort_values("timestamp")  # Sắp xếp theo thời gian
    #         .groupby("user_id")["item_id"]  # Nhóm theo user
    #         .apply(lambda x: list(x.tail(k)))  # Lấy k item cuối cùng
    #         .reset_index()
    #     )

    #     # Đổi tên cột
    #     recent_items.rename(columns={"item_id": "recent_items"}, inplace=True)

    #     return recent_items

    from collections import deque

    def build_recent_item_features(self, k=5):

        df = self.df.sort_values(
            ["user_id", "timestamp"]
        ).reset_index(drop=True)

        recent_items = []

        for user, group in df.groupby("user_id"):

            history = deque(maxlen=k)

            for item in group["item_id"]:

                recent_items.append(list(history))
                history.append(item)

        df["recent_items"] = recent_items

        return df[["user_id", "item_id", "timestamp", "recent_items"]]
    
    def build_activity_features(self):
        """
        Tính toán các chỉ số thống kê về mức độ hoạt động của người dùng.

        Đầu ra (Output):
            pd.DataFrame: Bảng chứa các cột đặc trưng hành vi:
                - 'user_id'
                - 'total_views'
                - 'total_addtocart'
                - 'total_transactions'
                - 'unique_items'
                - 'addtocart_rate'
                - 'purchase_rate'
        """

        activity = (
            self.df
            .groupby("user_id")
            .agg(
                total_views=("event", lambda x: (x == "view").sum()),
                total_addtocart=("event", lambda x: (x == "addtocart").sum()),
                total_transactions=("event", lambda x: (x == "transaction").sum()),
                unique_items=("item_id", "nunique")
            )
            .reset_index()
        )

        # Tính tỷ lệ chuyển đổi
        activity["addtocart_rate"] = activity["total_addtocart"] / (activity["total_views"] + 1)
        activity["purchase_rate"] = activity["total_transactions"] / (activity["total_views"] + 1)

        return activity
    
    def build(self):

        recent = self.build_recent_item_features()

        activity = self.build_activity_features()

        recent = recent.merge(activity, on="user_id", how="left")

        return recent

    # def build(self):
    #     """
    #     Hàm tổng hợp chính để chạy tất cả các hàm thành phần.

    #     Đầu ra (Output):
    #         pd.DataFrame: Bảng tổng hợp tất cả đặc trưng của người dùng (User Profile),
    #                       sẵn sàng để đưa vào User Tower của mô hình Retrieval.
    #     """

    #     # 1. Lấy chuỗi item gần đây
    #     recent = self.build_recent_item_features()

    #     # 2. Lấy thống kê hoạt động
    #     activity = self.build_activity_features()

    #     # 3. Merge hai bảng
    #     user_features = recent.merge(activity, on="user_id")

    #     return user_features