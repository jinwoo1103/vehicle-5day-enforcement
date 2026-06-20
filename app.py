# app_v2.py
import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import easyocr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image
from datetime import datetime

st.set_page_config(page_title="차량 5부제 단속 시스템", layout="wide")

st.markdown("""
<style>
    /* 다크 모드 강제 */
    html, body, [data-testid="stApp"] {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] { background-color: #161b22 !important; }
    section[data-testid="stMain"] { background-color: #0e1117 !important; }
    div[data-testid="stHorizontalBlock"] .stButton > button {
        width: 100%; height: 52px;
        font-size: 1rem; font-weight: 600;
        border-radius: 10px;
        border: 2px solid #30363d;
        background-color: #161b22;
        color: #c9d1d9;
        transition: all 0.2s;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        border-color: #58a6ff;
        color: #58a6ff;
    }
    .stat-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-label { font-size: 0.8rem; color: #8b949e; }
    .stat-value { font-size: 1.8rem; font-weight: 700; color: #c9d1d9; }
    .result-card {
        background-color: #161b22;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.8rem;
        border: 1px solid #30363d;
    }
    .violation { border-left: 5px solid #f85149; }
    .normal    { border-left: 5px solid #3fb950; }
    .eco       { border-left: 5px solid #58a6ff; }
    .unknown   { border-left: 5px solid #d29922; }
    .schedule-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        line-height: 1.9;
    }
    .schedule-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #8b949e;
        margin-bottom: 0.4rem;
    }
    .today-row { color: #58a6ff; font-weight: 700; }
    .other-row { color: #8b949e; }
    [data-testid="stFileUploader"] {
        background-color: #161b22 !important;
        border: 1px dashed #30363d !important;
        border-radius: 10px;
    }
    hr { border-color: #30363d !important; }
    .tbl-header { font-size: 0.8rem; color: #8b949e; padding-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🚗 차량 5부제 자동 단속 시스템")
st.markdown("<span style='color:#8b949e'>📷 <b style='color:#c9d1d9'>CCTV 이미지 분석</b> — YOLOv8이 차량과 번호판 위치를 탐지하여 crop 이미지로 반환합니다. &nbsp;|&nbsp; 🔍 <b style='color:#c9d1d9'>번호판 이미지 입력</b> — ResNet18과 EasyOCR로 친환경 여부를 분류하고 5부제 위반 여부를 판정합니다.</span>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ───────────────── 모델 로드 ──────────────────────────────
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo   = YOLO("best.pt")                          # YOLOv8 탐지 모델
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.load_state_dict(torch.load("resnet18.pth", map_location=device))
    resnet = resnet.to(device)
    resnet.eval()                                      # ResNet18 분류 모델
    ocr = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())  # EasyOCR
    return yolo, resnet, ocr, device

yolo_model, resnet_model, reader, device = load_models()

# ResNet18 입력 전처리 (224x224 리사이즈 + 정규화)
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── 번호판 이미지를 ResNet18에 입력하여 eco/normal 분류 ──
def classify_plate(crop_rgb):
    pil  = Image.fromarray(crop_rgb)
    x    = resnet_transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(resnet_model(x), dim=1)[0]
        pred = prob.argmax().item()
    return ("eco" if pred == 0 else "normal"), float(prob[pred])

# ── OCR 전 번호판 이미지 품질 개선 (업스케일 + 샤프닝) ──
def enhance_plate(crop_rgb):
    h, w = crop_rgb.shape[:2]
    if w < 100:  # 너비 100px 미만이면 3배 확대
        crop_rgb = cv2.resize(crop_rgb, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])  # 샤프닝 필터
    return cv2.filter2D(crop_rgb, -1, kernel)

# ── EasyOCR로 번호판 끝자리 숫자 추출 ────────────────────
def ocr_last_digit(crop_rgb, conf_thr=0.3):
    enhanced  = enhance_plate(crop_rgb)
    results   = reader.readtext(enhanced)
    filtered  = [(t, c) for (_, t, c) in results if c >= conf_thr]  # 신뢰도 필터링
    full_text = "".join([t for t, _ in filtered])
    digits    = [c for c in full_text if c.isdigit()]
    return (digits[-1] if digits else None), (full_text if full_text else "-")

# ── OCR 텍스트에서 숫자 마지막 4자리 추출 ────────────────
def last4(text):
    digits = [c for c in text if c.isdigit()]
    return "".join(digits[-4:]) if len(digits) >= 4 else text

# ── 끝자리 숫자와 요일을 비교하여 5부제 위반 여부 판정 ──
def check_restriction(digit, weekday):
    rules = {0:['1','6'], 1:['2','7'], 2:['3','8'], 3:['4','9'], 4:['5','0']}
    days  = {0:'월', 1:'화', 2:'수', 3:'목', 4:'금', 5:'토', 6:'일'}
    if weekday >= 5:
        return "해당없음", days[weekday]
    return ("위반" if digit in rules[weekday] else "정상"), days[weekday]

# ── 요일별 단속 번호 HTML 생성 (모드 2 좌측 패널용) ──────
def schedule_html(weekday):
    schedule = [
        (0, "월", "1, 6"),
        (1, "화", "2, 7"),
        (2, "수", "3, 8"),
        (3, "목", "4, 9"),
        (4, "금", "5, 0"),
    ]
    rows = ""
    for wd, name, nums in schedule:
        cls = "today-row" if wd == weekday else "other-row"
        mark = " ◀ 오늘" if wd == weekday else ""
        rows += f'<div class="{cls}">{name} &nbsp; {nums}{mark}</div>'
    weekend = '<div class="other-row">토 · 일 &nbsp; 전 차량 면제</div>'
    if weekday >= 5:
        weekend = '<div class="today-row">토 · 일 &nbsp; 전 차량 면제 ◀ 오늘</div>'
    return f"""
    <div class="schedule-box">
        <div class="schedule-title">금일 단속 번호</div>
        {rows}
        {weekend}
    </div>
    """

# ── 모드 선택 버튼 (session_state로 상태 유지) ───────────
if "mode" not in st.session_state:
    st.session_state.mode = "CCTV 이미지 분석"

btn1, btn2 = st.columns(2)
with btn1:
    if st.button("📷  CCTV 이미지 분석", use_container_width=True):
        st.session_state.mode = "CCTV 이미지 분석"
with btn2:
    if st.button("🔍  번호판 이미지 입력", use_container_width=True):
        st.session_state.mode = "번호판 이미지 입력"

mode = st.session_state.mode
st.markdown(f"<span style='color:#8b949e; font-size:0.85rem'>현재 모드: <b style='color:#58a6ff'>{mode}</b></span>", unsafe_allow_html=True)
st.divider()

weekday = 3  # 테스트용 목요일 고정 (실제 운영 시 datetime.now().weekday()로 변경)

# ── 모드 1: CCTV 이미지 분석 (YOLOv8 탐지만 수행) ───────
if mode == "CCTV 이미지 분석":
    left, right = st.columns([1, 2.5])

    with left:
        st.markdown("#### 업로드")
        uploaded_list = st.file_uploader("CCTV 이미지 (여러 장 가능)",
                                         type=["jpg","jpeg","png"],
                                         accept_multiple_files=True,
                                         label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="schedule-box">
            <div class="schedule-title">안내</div>
            <div class="other-row">차량이 있는 CCTV 이미지를 입력하면 차량 위치와 번호판 위치를 반환합니다.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if uploaded_list:
            st.success(f"분석 완료 — {len(uploaded_list)}장 업로드됨")
        else:
            st.info("이미지를 업로드해주세요.")

    with right:
        if uploaded_list:
            total_cars = 0
            total_plates = 0
            all_crops = []

            for uploaded in uploaded_list:
                file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
                img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w       = img_rgb.shape[:2]

                with st.spinner("탐지 중..."):
                    results = yolo_model.predict(img_rgb, conf=0.25, iou=0.5, verbose=False)
                boxes = results[0].boxes
                car_boxes   = [b for b in boxes if int(b.cls) == 0]
                plate_boxes = [b for b in boxes if int(b.cls) == 1]
                total_cars   += len(car_boxes)
                total_plates += len(plate_boxes)

                for cb in car_boxes:
                    x1,y1,x2,y2 = map(int, cb.xyxy[0].tolist())
                    x1,y1 = max(0,x1), max(0,y1)
                    x2,y2 = min(w,x2), min(h,y2)
                    car_crop = img_rgb[y1:y2, x1:x2]

                    # 차량 bbox 내부에서 가장 큰 번호판 bbox 매칭
                    plate_crop  = None
                    plate_conf  = None
                    best_area   = 0
                    for pb in plate_boxes:
                        px1,py1,px2,py2 = map(int, pb.xyxy[0].tolist())
                        if px1 >= x1 and py1 >= y1 and px2 <= x2 and py2 <= y2:
                            area = (px2-px1) * (py2-py1)
                            if area > best_area:
                                best_area  = area
                                plate_conf = float(pb.conf)
                                plate_crop = img_rgb[max(0,py1):min(h,py2),
                                                     max(0,px1):min(w,px2)]
                    all_crops.append({
                        "origin":     img_rgb,
                        "car":        car_crop,
                        "plate":      plate_crop,
                        "car_conf":   float(cb.conf),
                        "plate_conf": plate_conf,
                        "filename":   uploaded.name,
                    })

            # 요약 통계
            c1, c2, c3 = st.columns(3)
            for col, label, val, color in zip(
                [c1, c2, c3],
                ["분석 이미지", "검출 차량", "검출 번호판"],
                [len(uploaded_list), total_cars, total_plates],
                ["#c9d1d9", "#3fb950", "#58a6ff"]
            ):
                with col:
                    st.markdown(
                        f'<div class="stat-box"><div class="stat-label">{label}</div>'
                        f'<div class="stat-value" style="color:{color}">{val}</div></div>',
                        unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**탐지 결과 목록**")

            header = st.columns([2, 1.5, 1.5, 1, 1])
            for col, label in zip(header, ["원본 사진", "인식된 차량", "인식된 번호판", "차량 신뢰도", "번호판 신뢰도"]):
                col.markdown(f"<span class='tbl-header'>{label}</span>", unsafe_allow_html=True)
            st.markdown("<hr style='margin:4px 0; border-color:#30363d'>", unsafe_allow_html=True)

            for r in all_crops:
                row = st.columns([2, 1.5, 1.5, 1, 1])
                with row[0]:
                    st.image(r["origin"], width=160)
                with row[1]:
                    st.image(r["car"], width=100)
                with row[2]:
                    if r["plate"] is not None:
                        st.image(r["plate"], width=100)
                    else:
                        st.markdown("<span style='color:#8b949e; font-size:0.85rem'>미검출</span>", unsafe_allow_html=True)
                row[3].write(f"{r['car_conf']:.1%}")
                row[4].write(f"{r['plate_conf']:.1%}" if r['plate_conf'] else "—")
                st.markdown("<hr style='margin:2px 0; border-color:#30363d; opacity:0.4'>", unsafe_allow_html=True)

# ── 모드 2: 번호판 직접 입력 (ResNet18 + EasyOCR 단속 판정) ──
else:
    left, right = st.columns([1, 2.5])

    with left:
        st.markdown("#### 업로드")
        uploaded_list = st.file_uploader("번호판 이미지 (여러 장 가능)",
                                         type=["jpg","jpeg","png"],
                                         accept_multiple_files=True,
                                         label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(schedule_html(weekday), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if uploaded_list:
            st.success(f"분석 완료 — {len(uploaded_list)}장 업로드됨")
        else:
            st.info("이미지를 업로드해주세요.")

    with right:
        if uploaded_list:
            results_log = []
            for uploaded in uploaded_list:
                file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
                img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # ResNet18 친환경 분류 → 일반이면 EasyOCR 끝자리 추출 → 5부제 판정
                plate_type, type_conf = classify_plate(img_rgb)

                if plate_type == "eco":
                    result  = "단속 면제"
                    digit   = "-"
                else:
                    digit, ocr_txt = ocr_last_digit(img_rgb)
                    if digit:
                        r, _ = check_restriction(digit, weekday)
                        result = "단속 대상" if r == "위반" else "정상 통과"
                    else:
                        result = "판단 불가"
                        digit  = "—"

                results_log.append({
                    "crop":    img_rgb,
                    "종류":    "친환경" if plate_type == "eco" else "일반",
                    "끝자리":  digit,
                    "판정":    result,
                    "신뢰도":  type_conf,
                })

            # 요약 통계
            total     = len(results_log)
            violation = sum(1 for r in results_log if r["판정"] == "단속 대상")
            normal    = sum(1 for r in results_log if r["판정"] == "정상 통과")
            eco       = sum(1 for r in results_log if r["판정"] == "단속 면제")

            c1, c2, c3, c4 = st.columns(4)
            for col, label, val, color in zip(
                [c1, c2, c3, c4],
                ["총 번호판", "단속 대상", "정상 통과", "친환경 면제"],
                [total, violation, normal, eco],
                ["#c9d1d9", "#f85149", "#3fb950", "#58a6ff"]
            ):
                with col:
                    st.markdown(
                        f'<div class="stat-box"><div class="stat-label">{label}</div>'
                        f'<div class="stat-value" style="color:{color}">{val}</div></div>',
                        unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**단속 결과 목록**")

            header = st.columns([1, 1, 1, 1.5, 1.5])
            for col, label in zip(header, ["캡처", "종류", "끝자리", "판정", "분류 신뢰도"]):
                col.markdown(f"<span class='tbl-header'>{label}</span>", unsafe_allow_html=True)
            st.markdown("<hr style='margin:4px 0; border-color:#30363d'>", unsafe_allow_html=True)

            for r in results_log:
                판정 = r["판정"]
                if 판정 == "단속 대상":
                    badge = '<span style="background:#f85149;color:#fff;padding:2px 10px;border-radius:6px;font-size:0.8rem">위반</span>'
                elif 판정 == "단속 면제":
                    badge = '<span style="background:#58a6ff;color:#fff;padding:2px 10px;border-radius:6px;font-size:0.8rem">면제</span>'
                elif 판정 == "정상 통과":
                    badge = '<span style="background:#3fb950;color:#fff;padding:2px 10px;border-radius:6px;font-size:0.8rem">정상</span>'
                else:
                    badge = '<span style="background:#d29922;color:#fff;padding:2px 10px;border-radius:6px;font-size:0.8rem">불가</span>'

                row = st.columns([1, 1, 1, 1.5, 1.5])
                with row[0]:
                    st.image(r["crop"], width=80)
                row[1].write(r["종류"])
                row[2].write(r["끝자리"])
                row[3].markdown(badge, unsafe_allow_html=True)
                row[4].write(f"{r['신뢰도']:.1%}")
                st.markdown("<hr style='margin:2px 0; border-color:#30363d; opacity:0.4'>", unsafe_allow_html=True)