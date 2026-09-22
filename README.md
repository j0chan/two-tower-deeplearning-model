# Two-Tower 음식점 추천 모델

지역화폐 가맹 음식점을 사용자의 취향과 상황에 맞게 추천하는 모델입니다.
데이터 수집부터 모델 설계·실험·서빙까지 전 과정을 직접 진행했으며, 학습된 모델은 FastAPI로 서빙해 NestJS 백엔드와 연동했습니다.

- **과제**: 홍익대학교 세종캠퍼스 종합설계(캡스톤)
- **기간**: 2025.09 - 2025.11
- **개발 기록**: [기술 블로그 시리즈](#개발-기록)

---

## 1. 배경

지역화폐 플랫폼에 "유사 서비스 대신 이 앱을 쓸 이유"가 필요했습니다.
사용자가 스스로도 잘 몰랐던 취향과 현재 상황(시간대·요일)을 반영한 음식점 추천 기능을 제안하고 전담했습니다.

공개된 추천 데이터셋을 구할 수 없어 **주변 가맹점 115곳과 리뷰 약 3,700건을 직접 수집**했습니다. 수집 가능한 양에 한계가 있었고, 이 제약이 이후 모든 설계 판단의 전제가 되었습니다.

## 2. 모델 선정

### LSTM을 쓰지 않은 이유

처음에는 수업에서 다룬 LSTM으로 방문 순서를 학습시키려 했으나 세 가지 한계가 있었습니다.

| 한계 | 내용 |
|---|---|
| 부정 평가 미반영 | 방문 순서만 학습해 "갔지만 별로였던 가게"를 구분하지 못함 |
| 신규 가게 추천 불가 | 과거 방문 패턴에만 의존 |
| 리뷰 미활용 | 확보한 데이터 중 가장 양이 많은 리뷰를 쓰지 못함 |

### Two-Tower를 택한 이유

- User Tower와 Item Tower를 분리 학습하므로 **한쪽 데이터만 바뀌어도 전체 재학습이 불필요**
- 입력 피처를 자유롭게 추가·제거할 수 있어 **피처별 기여도를 비교 실험하기 적합**
- Item Vector를 사전 계산해 두면 추론 시 사용자 벡터만 생성하면 되므로 **서빙 속도에 유리**

### 폐기한 설계: 리뷰 내용 벡터화

리뷰 텍스트를 그대로 벡터화해 Item Tower에 넣는 방식을 먼저 검토했다가 폐기했습니다.
이 방식은 "오일 파스타", "해산물" 같은 **키워드 등장 빈도**를 학습합니다. 부정적으로 평가한 가게의 리뷰에도 그 키워드가 들어 있으므로, **싫어한 가게를 다시 추천하는 결과**로 이어집니다.

### 채택한 설계: 감성 라벨

리뷰 내용 대신 **리뷰의 긍정·부정 여부를 학습 라벨로** 사용했습니다.
"이 사용자가 이 가게를 좋아했는가"라는 선호 신호만 남기므로, 키워드가 아닌 선호 패턴을 학습합니다.

## 3. 데이터

| 항목 | 내용 |
|---|---|
| 원본 | 가맹점 정보 115곳, 리뷰 약 3,700건 (사용자 약 370명) |
| 라벨 | `is_positive` — [daekeun-ml/koelectra-small-v3-nsmc](https://huggingface.co/daekeun-ml/koelectra-small-v3-nsmc)로 리뷰 감성 분석 |
| 파생 피처 | `meal_time_code`(아침·점심·저녁·야식), `day_of_week`(0-6), `is_weekend`, `is_holiday`(holidays 라이브러리) |
| 분할 | 학습 : 테스트 = 8 : 2 |

수집이 진행 중이던 시점에 중간 데이터로 먼저 실험을 돌렸고, 수집 완료 후 **모든 실험을 같은 조건으로 다시 실행**했습니다. 아래 결과는 전부 최종 데이터 기준입니다.

## 4. 모델 구조

```
User Tower   : user_id, meal_time_code, day_of_week  ─┐
                                                       ├─→ 32차원 임베딩 → 내적 유사도
Item Tower   : store_id, category                    ─┘
```

- 프레임워크: TensorFlow 2.15 + TensorFlow Recommenders 0.7.3
- 평가 지표: Top-10 Accuracy (`FactorizedTopK`)
- 베이스라인은 `user_id`, `store_id`만 사용 — 피처를 하나씩 더해가며 **각 피처의 기여도를 분리해 측정**하고, 학습이 실패했을 때 원인이 피처인지 ID 관계인지 구분하기 위함

## 5. 실험

### 실험 조건 통제

`meal_time_code`를 추가한 실험에서 **같은 코드인데 실행할 때마다 정확도 편차가 크게** 나타났습니다.
데이터셋이 작아 shuffle 결과와 초기 가중치에 따라 수렴 경로가 달라지는 것이 원인이었습니다.

→ TensorFlow와 NumPy의 **난수 시드를 1로 고정**하고 모든 실험을 처음부터 다시 실행했습니다. 피처 비교가 성립하려면 비교 조건부터 같아야 하기 때문입니다.

### 결과

| 모델 | 추가 피처 | Top-10 Accuracy | 변화 | 판정 |
|---|---|---:|---:|---|
| Random | — | 8.77% | — | 비교 그래프 기준선 |
| Baseline | `user_id` + `store_id` | 6.79% | — | 기준 |
| Exp 2 | + `category` | **12.25%** | +5.46%p | 채택 |
| Exp 3 | + `meal_time_code` | **15.73%** | +3.48%p | 채택 |
| Exp 4 | + `day_of_week` | **16.39%** | +0.66%p | **최종 채택** |
| Exp 5-1 | + `is_holiday` | 14.74% | -1.65%p | 기각 |
| Exp 5-2 | + `is_weekend` | 14.90% | -1.49%p | 기각 |

### 채택·기각 근거

- **`category`** — 가장 기여도가 큰 피처. 사용자의 음식 종류 선호가 가장 강한 신호
- **`meal_time_code`** — 중간 데이터에서는 오히려 성능을 떨어뜨렸으나, 데이터가 늘자 유의미한 패턴으로 학습됨. **적은 데이터로 낸 판단을 확정으로 쓰면 안 된다는 것을 확인한 지점**
- **`day_of_week`** — "금요일 저녁", "월요일 점심" 같은 구체적 문맥을 만들어 소폭 향상
- **`is_holiday` 기각** — 공휴일은 연 10-15일뿐이라 표본이 절대적으로 부족했고 노이즈로 작용
- **`is_weekend` 기각** — `day_of_week`에 완전히 종속된 정보라 새로운 정보를 주지 못함

Exp 4 이후 개선을 멈추고 서빙 단계로 넘어갔습니다. 리뷰를 고정 차원 벡터로 변환해 Item Tower에 추가하는 등 개선 여지는 남아 있었지만, **발표 기한 안에 동작하는 기능을 완성하는 것**을 우선했습니다.

## 6. 서빙

```
NestJS (Controller → Service)
  └─ 로그인 사용자 ID + 현재 시각
     └─ _getCodesFromDate() → meal_time_code, day_of_week 변환
        └─ POST http://127.0.0.1:8000/recommend
           └─ FastAPI
              ├─ User Tower로 사용자 벡터 실시간 생성
              ├─ FAISS 인덱스에서 top-k 검색
              └─ store_id 목록 반환
```

| 항목 | 내용 |
|---|---|
| 인덱스 | FAISS `IndexFlatIP` (내적 기반 완전 탐색) |
| 벡터 | 32차원 × 115개 가게 |
| 입력 | `user_id`, `meal_time_code`, `day_of_week`, `top_k`(기본 10) |
| 출력 | `recommended_store_ids` |

가게가 115개뿐이라 완전 탐색으로 충분했습니다. IVF·HNSW 같은 **근사 탐색은 적용하지 않았고, 대규모 검색 성능은 검증하지 않았습니다.**

## 7. 저장소 구조

```
├── notebooks/
│   ├── 1_data_preprocessing.ipynb            전처리·감성 라벨링·분할
│   ├── 2_model_implementation.ipynb          Baseline
│   ├── 3_experiment_2_category.ipynb         + category
│   ├── 4_experiment_3_meal_time.ipynb        + meal_time_code
│   ├── 5_experiment_4_day_of_week.ipynb      + day_of_week (최종)
│   ├── 6_experiment_5-1_is_holiday.ipynb     + is_holiday (기각)
│   ├── 7_experiment_5-2_is_weekend.ipynb     + is_weekend (기각)
│   ├── 8_accuracy_comparison_graph.ipynb     실험 비교 그래프
│   ├── 9_deployment_preparation_faiss.ipynb  FAISS 인덱스 구축
│   └── 11_nestjs.ipynb                       백엔드 연동 테스트
├── data/
│   ├── raw/                                  수집 원본
│   └── processed/                            전처리 결과 (train/test)
├── saved_models/                             실험별 User/Item Tower, 임베딩
│   └── final_model/                          Exp 4 + FAISS 인덱스
├── main.py                                   FastAPI 추천 API
└── requirements.txt
```

## 8. 실행

```bash
pip install -r requirements.txt

# 노트북을 1 → 9 순서로 실행하면 전처리부터 FAISS 인덱스 구축까지 재현됩니다.

# 추천 API 실행
uvicorn main:app
# http://127.0.0.1:8000/docs 에서 테스트
```

TensorFlow 2.15 / Keras 2.15 / TensorFlow Recommenders 0.7.3 조합을 씁니다. 최신 버전에서는 `metrics` 인자 형식과 `FactorizedTopK` 입력 처리에서 API 불일치가 발생해 이 조합으로 고정했습니다.

## 9. 한계

정직하게 남겨 둡니다.

- **처음 목표했던 "사용자가 쓸 만하다고 체감할 품질"에는 도달하지 못했습니다.** 최종 Top-10 정확도 16.39%는 베이스라인 대비 개선이지만 실사용 품질을 보장하는 수치가 아닙니다
- 데이터가 115개 가게·약 3,700건 리뷰 규모라 결과의 일반화 가능성이 제한적입니다
- 완수 범위는 **앱 백엔드 연동 후 동작 확인까지**입니다. 서버는 로컬(`127.0.0.1`)에서 실행했고 운영 배포·부하 검증은 하지 않았습니다
- 감성 라벨은 사전학습 모델 출력을 표본 검토로 확인했을 뿐, 라벨 정확도를 정량 측정하지 않았습니다
- 긍정·부정 라벨의 불균형을 별도로 보정하지 않았습니다

## 개발 기록

| 글 | 내용 |
|---|---|
| [모델 선정 및 설계](https://chan-code.tistory.com/64) | LSTM 한계, Two-Tower 선정, 설계 1 폐기 |
| [데이터 전처리](https://chan-code.tistory.com/67) | 파생 피처 생성, 감성 라벨링, 분할 |
| [모델 구현 및 평가 1](https://chan-code.tistory.com/68) | Baseline 설계, 버전 호환 문제 |
| [모델 구현 및 평가 2](https://chan-code.tistory.com/69) | 학습 편차, 시드 고정, 피처 실험 |
| [모델 구현 마무리](https://chan-code.tistory.com/70) | 전체 데이터 재실험, 피처 채택·기각 |
| [FAISS · FastAPI](https://chan-code.tistory.com/71) | 인덱스 구축, 추천 API |
| [백엔드 연동](https://chan-code.tistory.com/72) | NestJS 연동, E2E 테스트 |
