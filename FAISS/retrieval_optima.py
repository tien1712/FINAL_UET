import os
import re
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict
from sentence_transformers import CrossEncoder
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

# --- Cấu hình ---
faiss_folder = "db/faiss_index_paperoptima(70:30)"  # FAISS database từ main.py
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Multiple API Keys & Key Rotation (1000 embeds/day per key) ---
API_KEYS = [
    os.getenv("GOOGLE_API_KEY_9"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),
    os.getenv("GOOGLE_API_KEY_5"),
    os.getenv("GOOGLE_API_KEY_6"),
    os.getenv("GOOGLE_API_KEY_8"),
    os.getenv("GOOGLE_API_KEY_10"),
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_7"),
]
API_KEYS = [k for k in API_KEYS if k]

REQUESTS_PER_KEY_PER_DAY = 1000
REQUESTS_PER_MINUTE = 100

class EmbeddingKeyManager:
    def __init__(self, api_keys, requests_per_key_per_day=1000, requests_per_minute=100):
        self.api_keys = list(api_keys)
        if not self.api_keys:
            raise RuntimeError("Không có GOOGLE_API_KEY nào trong môi trường")
        self.requests_per_key_per_day = requests_per_key_per_day
        self.requests_per_minute = requests_per_minute
        self.current_key_index = 0
        self.models_cache = {}
        self.daily_usage_counts = {i: 0 for i in range(len(self.api_keys))}
        self.minute_request_times = {i: [] for i in range(len(self.api_keys))}
        self.current_date = datetime.now().date()

    def _reset_daily_if_needed(self):
        today = datetime.now().date()
        if today != self.current_date:
            self.current_date = today
            self.daily_usage_counts = {i: 0 for i in range(len(self.api_keys))}
            self.minute_request_times = {i: [] for i in range(len(self.api_keys))}

    def _get_model_for_index(self, idx: int):
        if idx not in self.models_cache:
            self.models_cache[idx] = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=self.api_keys[idx]
            )
        return self.models_cache[idx]

    def _is_key_available(self, idx: int):
        self._reset_daily_if_needed()
        # Daily quota
        if self.daily_usage_counts[idx] >= self.requests_per_key_per_day:
            return False, "quota_exceeded"
        # Per-minute
        now = datetime.now()
        recent = [t for t in self.minute_request_times[idx] if now - t < timedelta(minutes=1)]
        self.minute_request_times[idx] = recent
        if len(recent) >= self.requests_per_minute:
            return False, "rate_limited"
        return True, "available"

    def _find_next_available_key_index(self):
        candidates = []
        for idx in range(len(self.api_keys)):
            ok, _ = self._is_key_available(idx)
            if ok:
                # Prefer the key with fewest requests in the last minute
                candidates.append((idx, len(self.minute_request_times[idx])))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def _switch_to_available_key(self):
        next_idx = self._find_next_available_key_index()
        if next_idx is None:
            # Wait for the soonest per-minute window to clear on the current key
            now = datetime.now()
            times = self.minute_request_times[self.current_key_index]
            if times:
                oldest = min(times)
                wait_sec = max(0.0, 60 - (now - oldest).total_seconds())
                if wait_sec > 0:
                    time.sleep(wait_sec)
                    self.minute_request_times[self.current_key_index] = []
                    return self._get_model_for_index(self.current_key_index)
        else:
            self.current_key_index = next_idx
            return self._get_model_for_index(self.current_key_index)
        # If still none, raise to caller
        raise RuntimeError("Không có API key nào khả dụng để embed vào lúc này")

    def get_current_model(self):
        ok, _ = self._is_key_available(self.current_key_index)
        if not ok:
            return self._switch_to_available_key()
        return self._get_model_for_index(self.current_key_index)

    def embed_query(self, text: str):
        # Try current key; on error/quota switch and retry
        attempts = 0
        last_err = None
        while attempts < len(self.api_keys) + 1:
            model = self.get_current_model()
            try:
                vector = model.embed_query(text)
                now = datetime.now()
                self.minute_request_times[self.current_key_index].append(now)
                self.daily_usage_counts[self.current_key_index] += 1
                return vector
            except Exception as e:
                msg = str(e)
                last_err = e
                # Mark quota exhausted if relevant
                if "429" in msg or "quota" in msg.lower() or "exceeded" in msg.lower():
                    self.daily_usage_counts[self.current_key_index] = self.requests_per_key_per_day
                # Switch to another key if possible
                try:
                    self._switch_to_available_key()
                except Exception:
                    break
                attempts += 1
        raise last_err or RuntimeError("Embed thất bại sau khi thử luân phiên API keys")


choice_map = {
    0: "Public transports",
    1: "Private modes",
    2: "Soft modes",
}


# --- Khởi tạo Embedding Key Manager & model ban đầu ---
key_manager = EmbeddingKeyManager(API_KEYS, REQUESTS_PER_KEY_PER_DAY, REQUESTS_PER_MINUTE)
embedding_model = key_manager.get_current_model()

# --- Load FAISS index ---
db = FAISS.load_local(
    faiss_folder,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)


# --- FAISS-based Retrieval Functions ---
def get_similar_vectors_by_id(query_id: int, query: str, examples_id: int = 2) -> List[Dict]:
    """Lấy tất cả vectors có cùng ID với query_id, re-rank theo query, trả về top n."""
    # Thu thập tất cả Document có cùng ID
    collected_docs = []
    for value in db.docstore._dict.values():
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            doc = item[0] if isinstance(item, tuple) else item
            if hasattr(doc, "metadata") and doc.metadata.get("id") == query_id:
                collected_docs.append(doc)

    if not collected_docs:
        return []
    if len(collected_docs) > examples_id:
        # 🔹 Cross-Encoder re-ranking theo query
        pairs = [(query, d.page_content) for d in collected_docs]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(collected_docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in ranked[:examples_id]]
    else:
        top_docs = collected_docs

    final_situations = []
    for i, doc in enumerate(top_docs, 1):
        content = getattr(doc, "page_content", "") or ""
        choice = doc.metadata.get("choice", 0)
        choice_desc = choice_map.get(choice, "unknown")
        # Tìm và chỉ lấy phần thông tin về chuyến đi, loại bỏ thông tin cá nhân
        match = re.search(r"(.*?free of charge)", content, re.DOTALL)
        result = match.group(1) if match else "No trip information found"
        example = (
            f"Situation {i}: {result}. That person chose {choice_desc}. "
        )
        final_situations.append(example)
    return final_situations

def balanced_retrieval_with_rerank(query: str, query_id: int, k_per_label: int = 5, top_k: int = 9):
    """
    Balanced Retrieval theo label + Cross-Encoder re-ranking.
    Trả về: 1 chuỗi text gồm các ví dụ.
    """
    query_vector = key_manager.embed_query(query)
    candidates = []
    
    # 🔹 Balanced Retrieval
    hits = db.similarity_search_by_vector(query_vector, k=100)

    # Lọc bỏ các document có id trùng với query_id
    hits = [doc for doc in hits if doc.metadata.get("id") != query_id]

    labels = set(doc.metadata.get("choice") for doc in hits)
    for label in labels:
        docs_label = [doc for doc in hits if doc.metadata.get("choice") == label]
        candidates.extend(docs_label[:k_per_label])

    # 🔹 Cross-Encoder re-ranking
    pairs = [(query, cand.page_content) for cand in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    final_docs = [doc for doc, s in ranked[:top_k]]

    examples = []
    for i, doc in enumerate(final_docs, 1):
        choice = doc.metadata.get("choice", 0)
        choice_desc = choice_map.get(choice, "unknown")
        example = (
            f"Example {i}: {doc.page_content} Their choice was {choice_desc}. "
        )
        examples.append(example)

    return examples


def retrieval(query: str, id: int):
    situations = get_similar_vectors_by_id(id, query)
    examples = balanced_retrieval_with_rerank(query, id)
    return situations, examples

#test
if __name__ == "__main__":
    pd = pd.read_csv("data/Optima/test.csv")
    query = pd.iloc[1]["INFOR"]
    id = pd.iloc[1]["ID"]
    situations, examples = retrieval(query, id)
    print(query)
    print(id)
    
    print("Situations:")
    for doc in situations:
       print(doc)
    print("Examples:")
    for doc in examples:
        print(doc)






