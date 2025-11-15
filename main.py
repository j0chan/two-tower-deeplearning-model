import numpy as np
import faiss
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import os

# --- 실행 ---
# uvicorn main:app
# --- 테스트 ---
# http://127.0.0.1:8000/docs


# --- 설정 및 모델/인덱스 로드 ---
# FastAPI 앱 초기화
app = FastAPI(title="가게 추천 API", description="Two-Tower 모델과 FAISS를 이용한 실시간 가게 추천 시스템")

# 모델 및 인덱스 파일 경로 설정
SAVE_DIR = './saved_models/final_model'

# 애플리케이션 시작 시 모델과 인덱스 로드
try:
    print("서버 시작: 모델 및 FAISS 인덱스를 로드합니다...")
    
    # 1. 학습된 User Tower 모델 로드
    user_model = tf.saved_model.load(os.path.join(SAVE_DIR, 'user_model'))
    
    # 2. FAISS 인덱스 로드
    faiss_index = faiss.read_index(os.path.join(SAVE_DIR, 'faiss_index.bin'))

    # 3. FAISS 인덱스와 매핑될 가게 ID 배열 로드
    all_store_ids = np.load(os.path.join(SAVE_DIR, 'all_store_ids.npy'))
    
    print(f"로드 완료: User Model, FAISS 인덱스({faiss_index.ntotal}개 벡터), 가게 ID")

except Exception as e:
    print(f"오류: 모델 또는 인덱스 로드에 실패했습니다. 경로를 확인하세요. - {e}")
    user_model = None
    faiss_index = None
    all_store_ids = None

# --- API 입출력 데이터 모델 정의 ---

class UserRequest(BaseModel):
    """추천 요청 시 받을 사용자 정보 모델"""
    user_id: int
    meal_time_code: int
    day_of_week: int
    top_k: int = 10  # 추천 받을 가게 수 (기본값 10)

class RecommendationResponse(BaseModel):
    """추천 결과로 반환할 가게 ID 리스트 모델"""
    recommended_store_ids: list[int]


# --- API 엔드포인트 생성 ---

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: UserRequest):
    """
    사용자 정보를 입력받아 개인화된 가게 목록을 추천합니다.
    - **user_id**: 사용자 고유 ID
    - **meal_time_code**: 식사 시간 코드 (0: 아침, 1: 점심, 2: 저녁, 3: 야식)
    - **day_of_week**: 요일 코드 (0: 월요일, ..., 6: 일요일)
    - **top_k**: 추천할 가게의 수
    """
    if not all([user_model, faiss_index, all_store_ids is not None]):
        return {"error": "오류: 모델 로드 실패."}

    # --- 3. 추천 로직 실행 (User Vector 생성 -> FAISS 검색) ---
    
    # 1. 입력 데이터를 모델이 이해할 수 있는 TensorFlow 텐서 형태로 변환
    input_features = {
        "user_id": tf.constant([request.user_id], dtype=tf.int64),
        "meal_time_code": tf.constant([request.meal_time_code], dtype=tf.int64),
        "day_of_week": tf.constant([request.day_of_week], dtype=tf.int64)
    }

    # 2. User Tower 모델을 사용해 실시간으로 User Vector 생성
    user_vector = user_model(input_features).numpy()

    # 3. 생성된 User Vector를 FAISS 인덱스에서 검색하여 가장 유사한 top_k개의 아이템 찾기
    # faiss_index.search(검색할 벡터, 개수) -> (거리 배열, 인덱스 배열) 반환
    _, top_k_indices = faiss_index.search(user_vector, request.top_k)
    
    # 4. 검색된 인덱스를 실제 가게 ID로 변환
    recommended_ids = [int(all_store_ids[i]) for i in top_k_indices[0]]

    return {"recommended_store_ids": recommended_ids}

@app.get("/")
def read_root():
    return {"message": "추천 API 서버가 정상적으로 실행 중입니다. /docs 로 이동하여 API를 테스트하세요."}