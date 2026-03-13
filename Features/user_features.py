import pandas as pd

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
            Cần có các cột: 'visitorid', 'itemid', 'event', 'timestamp'.
        """
        self.df = interactions

    def build_recent_item_features(self, k=5):        
        """
        Tạo danh sách các sản phẩm mà người dùng đã tương tác gần đây nhất.

        Đầu vào (Input):
            k (int): Số lượng sản phẩm cuối cùng muốn giữ lại. Mặc định là 10.

        Đầu ra (Output):
            pd.DataFrame: Bảng gồm 2 cột:
                - 'visitorid': ID người dùng.
                - 'recent_items': Một danh sách (List) chứa tối đa k ID sản phẩm mới nhất.
        """
        recent_items = (
            self.df
            .sort_values("timestamp") # Sắp xếp theo thời gian để lấy đúng thứ tự
            .groupby("visitorid")["itemid"] # Nhóm theo người dùng, chọn cột itemid
            .apply(lambda x: list(x.tail(k))) # Lấy k item cuối cùng của mỗi người dùng và bỏ vào list
            .reset_index()
        )

        # Đổi tên cột từ 'itemid' sang 'recent_items' để phản ánh đúng kiểu dữ liệu (list)
        recent_items.rename(columns={"itemid": "recent_items"}, inplace=True)

        return recent_items

    def build_activity_features(self):
        """
        Tính toán các chỉ số thống kê về mức độ hoạt động của người dùng.

        Đầu vào (Input):
            Không có (sử dụng dữ liệu self.df truyền vào lúc khởi tạo).

        Đầu ra (Output):
            pd.DataFrame: Bảng chứa các cột đặc trưng hành vi:
                - 'visitorid': ID người dùng.
                - 'total_views/addtocart/transactions': Tổng số lượt xem/giỏ hàng/mua.
                - 'unique_items': Số lượng sản phẩm khác nhau mà người dùng từng xem.
                - 'addtocart_rate/purchase_rate': Tỷ lệ chuyển đổi hành vi.
        """
        activity = (
            self.df
            .groupby("visitorid")
            .agg(
                total_views=("event", lambda x: (x == "view").sum()),
                total_addtocart=("event", lambda x: (x == "addtocart").sum()),
                total_transactions=("event", lambda x: (x == "transaction").sum()),
                unique_items=("itemid", "nunique") # Đếm số lượng sản phẩm duy nhất
            )
            .reset_index()
        )

        # Tính tỷ lệ (cộng 1 vào mẫu số để tránh lỗi chia cho 0 nếu user chưa có lượt view nào)
        activity["addtocart_rate"] = activity["total_addtocart"] / (activity["total_views"] + 1)
        activity["purchase_rate"] = activity["total_transactions"] / (activity["total_views"] + 1)

        return activity

    def build(self):
        """
        Hàm tổng hợp chính để chạy tất cả các hàm thành phần.

        Đầu vào (Input):
            Không có.

        Đầu ra (Output):
            pd.DataFrame: Bảng tổng hợp tất cả đặc trưng của người dùng (User Profile),
                          sẵn sàng để đưa vào User Tower của mô hình Retrieval.
        """
        # 1. Lấy danh sách chuỗi sản phẩm gần đây
        recent = self.build_recent_item_features()
        
        # 2. Lấy các chỉ số thống kê hoạt động
        activity = self.build_activity_features()

        # 3. Kết hợp (Merge) hai bảng lại thành một dựa trên visitorid
        user_features = recent.merge(activity, on="visitorid")

        return user_features