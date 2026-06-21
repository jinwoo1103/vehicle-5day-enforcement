# 🚗 차량 5부제 자동 단속 시스템

YOLOv8, ResNet18, EasyOCR을 활용한 차량 5부제 자동 단속 시스템입니다.

- **모드 1 (CCTV 이미지 분석)** — YOLOv8이 차량과 번호판 위치를 탐지하고 crop 이미지를 반환합니다.
- **모드 2 (번호판 직접 입력)** — ResNet18로 친환경 여부를 분류하고 EasyOCR로 끝자리를 추출하여 5부제 위반 여부를 판정합니다.

## 📌 데모
🔗 https://jinwoooh30-vehicle-5day-enforcement-system.hf.space

## 🛠 사용 기술
- **YOLOv8** - 차량 및 번호판 탐지
- **ResNet18** - 친환경 번호판 분류
- **EasyOCR** - 번호판 끝자리 추출
- **Streamlit** - 웹 인터페이스 구현
- **Hugging Face Spaces** - 배포

## 📁 파일 구성
| 파일 | 설명 |
|------|------|
| `app.py` | Streamlit 구현 코드 |
| `best.pt` | YOLOv8 학습 가중치 |
| `resnet18.pth` | ResNet18 학습 가중치 |
| `requirements.txt` | 필요 라이브러리 목록 |
| `YOLOv8.ipynb` | YOLOv8 학습 코드 |
| `ResNet18.ipynb` | ResNet18 학습 코드 |

## 🚀 실행 방법
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 모델 성능
| 모델 | 지표 | 결과 |
|------|------|------|
| YOLOv8 | mAP@0.5 | 0.930 |
| YOLOv8 | Precision | 0.938 |
| YOLOv8 | Recall | 0.904 |
| ResNet18 | Accuracy | 0.951 |
| ResNet18 | ECO Recall | 0.981 |

## 👤 개발자
- 이름: 오진우
- 학번: 22012193
- 과목: 딥러닝실습