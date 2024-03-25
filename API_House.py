import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Khởi tạo FastAPI
app = FastAPI()

@app.get('/')
async def index():
    return {"message": "First Route"}

class PropertyData(BaseModel):
    Dia_chi: str
    Quận: str
    Huyện: str
    Loai_nha: str
    Giay_to: str
    Dien_tich: float
    Dài: float
    Rộng: float

# Load mô hình đã lưu, và đảm bảo dấu ngoặc đóng chính xác
model = joblib.load('best_model.pkl')

# Định nghĩa route để tiếp nhận dữ liệu người dùng và trả về dự đoán giá nhà
@app.post("/predict")
def predict_property_price(data: PropertyData):
    # Chuyển dữ liệu người dùng thành pandas DataFrame để dễ xử lý
    input_data = pd.DataFrame([data.dict()])
    # Thực hiện dự đoán
    prediction = model.predict(input_data)
    # Trả về kết quả
    return {"predicted_price": prediction[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)