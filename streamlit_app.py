# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1Tqr2znfekEJYzZBnm1UIT7QU3lkuVbv7")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {

     labels[0]: {
       "texts": ["발로란트 가디언", "2250원", "헤드 DMG 195"],
       "images": ["data:image/jpeg;base64,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", "https://static.inven.co.kr/image_2011/site_image/valorant/skinimage/skinimage_102002001.jpg?v=200428a"],
       "videos": ["https://www.youtube.com/watch?v=j5UOdqtOudc"]
     },
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
