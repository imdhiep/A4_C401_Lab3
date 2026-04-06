import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Khởi tạo client OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    # Sử dụng model gpt-4o-mini (bản mới và rẻ nhất hiện nay)
    # Lưu ý: Không có model gpt-5.4-mini đâu nhé!
    response = client.chat.completions.create(
        model="gpt-5.4-mini", 
        messages=[
            {"role": "system", "content": "Bạn là trợ lý đặt vé xem phim."},
            {"role": "user", "content": "Đặt cho tôi 2 vé xem phim Zootopia 2 tại rạp CGV TP.HCM vào tối nay nhé!"}
        ]
    )
    
    print("-" * 30)
    print("AI trả lời (Baseline - OpenAI):")
    print(response.choices[0].message.content)
    print("-" * 30)

except Exception as e:
    print(f"Lỗi rồi: {e}")