# this file for fake DB

from datetime import datetime

from faker import Faker
import random
from pymongo import MongoClient

fake = Faker('vi_VN')

# Kết nối đến MongoDB
client = MongoClient('localhost', 27017)
db = client['final']
first_names = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Phan", "Võ", "Đặng", "Bùi", "Đỗ"]
middle_names = ["Văn", "Thị", "Đình", "Hữu", "Quốc", "Thịnh", "Như", "Tường", "Hoàng", "Văn"]
last_names = ["Nam", "Trang", "Dũng", "Hương", "Hải", "Hà", "Thịnh", "Thu", "Hoài", "Anh"]

# tinh_thanh_pho_vietnam = [
#     "Thành phố Hồ Chí Minh", "Thành phố Hà Nội", "Thành phố Hải Phòng", "Thành phố Đà Nẵng", "Thành phố Cần Thơ", 
#     "Quảng Ninh", "Bà Rịa - Vũng Tàu", "Hải Dương", "Hưng Yên", "Thái Bình", 
#     "Hà Nam", "Nam Định", "Ninh Bình", "Thanh Hóa", "Nghệ An", "Hà Tĩnh", 
#     "Quảng Bình", "Quảng Trị", "Thừa Thiên Huế", "Đắk Lắk", "Đắk Nông", 
#     "Gia Lai", "Kon Tum", "Lâm Đồng", "Bình Phước", "Bình Dương", "Đồng Nai", 
#     "Tây Ninh", "Long An", "Tiền Giang", "Bến Tre", "Trà Vinh", "Vĩnh Long", 
#     "Đồng Tháp", "An Giang", "Kiên Giang", "Cà Mau", "Sóc Trăng", "Bạc Liêu", 
#     "Cần Thơ", "Hậu Giang", "Lào Cai", "Điện Biên", "Lai Châu", "Lào Cai", 
#     "Yên Bái", "Sơn La", "Phú Thọ", "Vĩnh Phúc", "Bắc Ninh", "Bắc Giang", 
#     "Hải Dương", "Hưng Yên", "Thái Nguyên", "Lạng Sơn", "Bắc Kạn", "Cao Bằng", 
#     "Lai Châu", "Điện Biên", "Sơn La", "Lào Cai", "Yên Bái", "Tuyên Quang"
# ]
tinh_thanh_pho_vietnam = [
    "Thành phố Hồ Chí Minh", "Thành phố Hà Nội", "Thành phố Hải Phòng", "Thành phố Đà Nẵng", "Thành phố Cần Thơ", 
    "Quảng Ninh", "Bà Rịa - Vũng Tàu", "Hải Dương", "Hưng Yên", "Thái Bình", 
    "Hà Nam", "Nam Định", "Ninh Bình", "Thanh Hóa", "Nghệ An", "Hà Tĩnh", 
    "Quảng Bình", "Quảng Trị", "Thừa Thiên Huế", "Đắk Lắk", "Đắk Nông", 
    "Gia Lai", "Kon Tum", "Lâm Đồng", "Bình Phước", "Bình Dương", "Đồng Nai", 
    "Tây Ninh", "Long An", "Tiền Giang", "Bến Tre", "Txrà Vinh", "Vĩnh Long", 
    "Đồng Tháp", "An Giang", "Kiên Giang", "Cà Mau", "Sóc Trăng", "Bạc Liêu", 
    "Cần Thơ", "Hậu Giang", "Lào Cai", "Điện Biên", "Lai Châu", "Lào Cai", 
    "Yên Bái", "Sơn La", "Phú Thọ", "Vĩnh Phúc", "Bắc Ninh", "Bắc Giang", 
    "Hải Dương", "Hưng Yên", "Thái Nguyên", "Lạng Sơn", "Bắc Kạn", "Cao Bằng", 
    "Lai Châu", "Điện Biên", "Sơn La", "Lào Cai", "Yên Bái", "Tuyên Quang",
      # Add your additional provinces here
]

# Add "Tỉnh" before each province
tinh_thanh_pho_vietnam = [f"Tỉnh {province}" if "Thành phố" not in province else province for province in tinh_thanh_pho_vietnam]


# Sinh ngẫu nhiên dữ liệu và thêm vào collections
for _ in range(50):
    # name = fake.name()
    # name = [f"{random.choice(first_names)} {random.choice(middle_names)} {random.choice(last_names)}" for _ in range(1000)]
    name = f"{random.choice(first_names)} {random.choice(middle_names)} {random.choice(last_names)}"
    age = str(fake.random_int(min=1, max=100))

    region = random.choice(tinh_thanh_pho_vietnam)
    # profile_pic = "url_to_profile_picture"  # You should replace this with actual URLs
    # profile_pic = fake.url()

    profile_pic = fake.file_path(depth=1, extension='jpeg')

    result = random.choice(["Pneumonia", "Normal"])

    # Thêm vào collection bacsi
    db.bacsi.insert_one({
        "name": name,
        "age": age,
        "region": region,
        "profile_pic": profile_pic,
        "result": result
    })

# Đóng kết nối đến MongoDB
client.close()
