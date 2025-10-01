import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# --- Config ---
CSV_PATH = "data/Optima/train.csv"       # file của bạn
FAISS_FOLDER = "db/faiss_index"

# --- API Keys Management ---
API_KEYS = [
    os.getenv("GOOGLE_API_KEY_9"),
    os.getenv("GOOGLE_API_KEY_2"), 
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),
    os.getenv("GOOGLE_API_KEY_5"),
    os.getenv("GOOGLE_API_KEY_6"),
    os.getenv("GOOGLE_API_KEY_8")
]

# Rate limiting config
REQUESTS_PER_MINUTE = 100
REQUESTS_PER_KEY = 1000

# --- Load data ---
df = pd.read_csv(CSV_PATH)
texts = df['INFOR'].fillna("").astype(str).tolist()
metadatas = [{"id": int(row["ID"]), "choice": int(row["CHOICE"])} for _, row in df.iterrows()]

# --- Rate Limiting Manager ---
class RateLimitManager:
    def __init__(self, api_keys, requests_per_minute=100, requests_per_key=1000):
        self.api_keys = [key for key in api_keys if key is not None]
        self.requests_per_minute = requests_per_minute
        self.requests_per_key = requests_per_key
        
        # Tracking variables
        self.current_key_index = 0
        self.key_usage_count = {i: 0 for i in range(len(self.api_keys))}
        self.key_request_times = {i: [] for i in range(len(self.api_keys))}  # Track per key
        self.embedding_models = {}
        self.rate_limited_keys = set()  # Keys đang bị rate limit
        
        print(f"✅ Khởi tạo với {len(self.api_keys)} API keys")
    
    def get_embedding_model(self, key_index):
        """Lấy embedding model cho key cụ thể"""
        if key_index not in self.embedding_models:
            self.embedding_models[key_index] = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=self.api_keys[key_index]
            )
        return self.embedding_models[key_index]
    
    def get_current_embedding_model(self):
        """Lấy embedding model hiện tại"""
        return self.get_embedding_model(self.current_key_index)
    
    def is_key_available(self, key_index):
        """Kiểm tra key có sẵn sàng sử dụng không"""
        current_time = datetime.now()
        
        # Kiểm tra rate limit per minute cho key này
        key_times = self.key_request_times[key_index]
        key_times = [t for t in key_times if current_time - t < timedelta(minutes=1)]
        self.key_request_times[key_index] = key_times
        
        # Kiểm tra usage per key
        if self.key_usage_count[key_index] >= self.requests_per_key:
            return False, "quota_exceeded"
        
        # Kiểm tra rate limit per minute
        if len(key_times) >= self.requests_per_minute:
            return False, "rate_limited"
        
        return True, "available"
    
    def find_best_available_key(self):
        """Tìm key tốt nhất có thể sử dụng"""
        available_keys = []
        
        for key_index in range(len(self.api_keys)):
            is_available, reason = self.is_key_available(key_index)
            if is_available:
                # Ưu tiên key có ít requests nhất trong phút vừa rồi
                usage_score = len(self.key_request_times[key_index])
                available_keys.append((key_index, usage_score))
        
        if available_keys:
            # Sắp xếp theo usage score (ít nhất trước)
            available_keys.sort(key=lambda x: x[1])
            best_key = available_keys[0][0]
            self.current_key_index = best_key
            return self.get_embedding_model(best_key)
        
        return None  # Không có key nào available
    
    def switch_to_next_key(self):
        """Chuyển sang key tiếp theo có sẵn"""
        # Thử tìm key tốt nhất
        model = self.find_best_available_key()
        if model:
            print(f"🔄 Chuyển sang API key {self.current_key_index + 1}")
            return model
        
        # Nếu không có key nào available, chờ key đầu tiên
        print("⏳ Tất cả keys đều bị rate limit, chờ key đầu tiên...")
        first_key = 0
        self.current_key_index = first_key
        
        # Tính thời gian chờ cho key đầu tiên
        key_times = self.key_request_times[first_key]
        if key_times:
            oldest_request = min(key_times)
            wait_time = 60 - (datetime.now() - oldest_request).total_seconds()
            if wait_time > 0:
                print(f"⏳ Chờ {wait_time:.1f}s cho key {first_key + 1}...")
                time.sleep(wait_time)
                self.key_request_times[first_key] = []
        
        return self.get_embedding_model(first_key)
    
    def check_rate_limits(self):
        """Kiểm tra và xử lý rate limits - tối ưu để chuyển key ngay lập tức"""
        # Kiểm tra key hiện tại có available không
        is_available, reason = self.is_key_available(self.current_key_index)
        
        if not is_available:
            if reason == "rate_limited":
                print(f"⚡ Key {self.current_key_index + 1} bị rate limit, chuyển ngay sang key khác...")
            elif reason == "quota_exceeded":
                print(f"🔄 Key {self.current_key_index + 1} đã hết quota, chuyển sang key khác...")
            
            return self.switch_to_next_key()
        
        return self.get_current_embedding_model()
    
    def record_request(self):
        """Ghi nhận một request cho key hiện tại"""
        current_time = datetime.now()
        self.key_request_times[self.current_key_index].append(current_time)
        self.key_usage_count[self.current_key_index] += 1
    
    def get_status(self):
        """Lấy thông tin trạng thái hiện tại"""
        current_time = datetime.now()
        key_status = {}
        
        for key_idx in range(len(self.api_keys)):
            # Tính requests trong phút vừa rồi cho mỗi key
            recent_requests = [t for t in self.key_request_times[key_idx] 
                             if current_time - t < timedelta(minutes=1)]
            
            is_available, reason = self.is_key_available(key_idx)
            key_status[key_idx] = {
                "usage": self.key_usage_count[key_idx],
                "requests_last_minute": len(recent_requests),
                "available": is_available,
                "reason": reason if not is_available else "available"
            }
        
        return {
            "current_key": self.current_key_index + 1,
            "key_status": key_status,
            "total_available_keys": sum(1 for status in key_status.values() if status["available"])
        }

# Khởi tạo rate limit manager
rate_manager = RateLimitManager(API_KEYS, REQUESTS_PER_MINUTE, REQUESTS_PER_KEY)

# --- Build FAISS in batches with rate limiting ---
def build_faiss_in_batches(texts, metadatas, rate_manager, faiss_folder, batch_size=20):
    db = None
    total_requests = 0
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        
        # Kiểm tra rate limits trước khi xử lý batch
        embedding_model = rate_manager.check_rate_limits()
        
        try:
            # Xử lý batch với retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    batch_db = FAISS.from_texts(
                        texts=batch_texts,
                        embedding=embedding_model,
                        metadatas=batch_metas
                    )
                    
                    # Ghi nhận requests (ước tính số lượng requests dựa trên batch size)
                    for _ in range(len(batch_texts)):
                        rate_manager.record_request()
                        total_requests += 1
                    
                    break  # Thành công, thoát khỏi retry loop
                    
                except Exception as e:
                    print(f"⚠️ Lỗi batch {i//batch_size + 1}, attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        # Chuyển sang key khác nếu có lỗi
                        embedding_model = rate_manager.switch_to_next_key()
                        time.sleep(2)  # Chờ 2s trước khi retry
                    else:
                        print(f"❌ Thất bại sau {max_retries} attempts, bỏ qua batch này")
                        continue
            
            # Merge batch vào database chính
            if 'batch_db' in locals():
                if db is None:
                    db = batch_db
                else:
                    db.merge_from(batch_db)
            
            # Hiển thị progress và status
            status = rate_manager.get_status()
            current_key_status = status['key_status'][status['current_key']-1]
            print(f"✅ Processed {i+len(batch_texts)}/{len(texts)} rows | "
                  f"Key: {status['current_key']} | "
                  f"Usage: {current_key_status['usage']}/{REQUESTS_PER_KEY} | "
                  f"Requests/min: {current_key_status['requests_last_minute']}/{REQUESTS_PER_MINUTE} | "
                  f"Available keys: {status['total_available_keys']}/{len(rate_manager.api_keys)}")
            
        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng tại batch {i//batch_size + 1}: {str(e)}")
            continue
    
    if db is not None:
        db.save_local(faiss_folder)
        print(f"✅ FAISS index saved to '{faiss_folder}'")
        print(f"📊 Tổng số requests đã sử dụng: {total_requests}")
        
        # Hiển thị thống kê cuối cùng
        final_status = rate_manager.get_status()
        print(f"📈 Thống kê cuối cùng:")
        for key_idx, key_status in final_status['key_status'].items():
            status_icon = "✅" if key_status['available'] else "⏸️"
            print(f"   {status_icon} Key {key_idx + 1}: {key_status['usage']}/{REQUESTS_PER_KEY} requests "
                  f"(last min: {key_status['requests_last_minute']}/{REQUESTS_PER_MINUTE}) - {key_status['reason']}")
    else:
        print("❌ Không thể tạo FAISS index")
    
    return db

# --- Run ---
if __name__ == "__main__":
    print(f"🚀 Bắt đầu xử lý {len(texts)} texts với {len(rate_manager.api_keys)} API keys")
    print(f"⚙️ Cấu hình: {REQUESTS_PER_MINUTE} requests/phút, {REQUESTS_PER_KEY} requests/key")
    
    db = build_faiss_in_batches(texts, metadatas, rate_manager, FAISS_FOLDER, batch_size=10)
