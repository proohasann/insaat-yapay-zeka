# -*- coding: utf-8 -*-
"""
İnşaat Proje Maliyeti Tahmin Aracı
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Sayfa ayarları ve stil
# ------------------------------------------------------------------
st.set_page_config(
    page_title="İnşaat Maliyet Tahmin Aracı",
    page_icon="🏗️",
    layout="centered",
)

st.markdown("""
<style>
    /* 1. ANA ARKA PLAN - MAVİ */
    .stApp, .main {
        background-color: #1D4ED8 !important; 
    }
    
    /* Mavi arka planda okunabilmesi için tüm genel metinler beyaz */
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp span, .stApp div {
        color: #F8FAFC !important;
    }

    /* 2. BAŞLIK KUTUSU (Lacivert) */
    .app-header {
        background-color: #0F172A !important; 
        padding: 1.3rem 1.6rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        border-left: 5px solid #38BDF8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .app-header h1 { color: white !important; margin: 0; font-size: 1.6rem; }
    .app-header p { color: #93C5FD !important; margin: 0.4rem 0 0 0; font-size: 1.05rem; font-weight: 500; }

    /* 3. SAYI GİRİLEN YERLER - Arka plan Beyaz, Sayılar Siyah */
    div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        border: 2px solid #94A3B8 !important;
    }
    div[data-baseweb="input"] input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: bold !important;
    }
    div[data-baseweb="input"] svg {
        fill: #000000 !important;
    }
    
    /* Seçim Kutusu (Gelişmiş Model İçin) */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
    }
    div[data-baseweb="select"] span {
        color: #000000 !important;
    }

    /* 4. TAHMİN ET BUTONU - YEŞİL */
    .stButton > button {
        background-color: #10B981 !important; 
        color: white !important;
        border: none !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        padding: 0.75rem 0 !important;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #059669 !important; 
        transform: scale(1.02);
    }

    /* 5. SONUÇ KUTULARI */
    .result-box {
        background-color: #0F172A !important;
        color: white !important;
        padding: 1.4rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #10B981 !important;
    }
    .result-box .value { font-size: 2.3rem !important; font-weight: 700 !important; color: #10B981 !important; }
    .result-box .label { font-size: 0.95rem !important; opacity: 0.9 !important; color: white !important; }
    
    .warn-box {
        background-color: #FEF2F2 !important;
        border: 1.5px solid #DC2626 !important;
        color: #DC2626 !important;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        font-size: 0.92rem;
        margin-top: 0.6rem;
    }
    .ok-box {
        background-color: #ECFDF5 !important;
        border: 1.5px solid #10B981 !important;
        color: #059669 !important;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        font-size: 0.92rem;
        margin-top: 0.6rem;
    }

    /* 6. YAN MENÜ (SIDEBAR) - MAVİYE ZIT KİREMİT / TURUNCU */
    [data-testid="stSidebar"] {
        background-color: #C2410C !important; /* Sıcak Pas / Kiremit Turuncusu */
    }
    
    /* Yan menüdeki metinlerin okunabilirliği */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.25) !important;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Başlık ve Kişisel Bilgiler
# ------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>🏗️ İnşaat Proje Maliyeti Tahmin Aracı</h1>
    <p>2023232064 - Hasan Karadağ - +90 544 971 50 52</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Model ve metaveriyi yükle
# ------------------------------------------------------------------
@st.cache_resource
def yukle():
    basit = joblib.load("maliyet_modeli_basit.pkl")
    gelismis = joblib.load("maliyet_modeli_gelismis.pkl")
    with open("meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return basit, gelismis, meta

try:
    model_basit, model_gelismis, meta = yukle()
except FileNotFoundError:
    st.error(
        "Model dosyaları bulunamadı. Lütfen 'maliyet_modeli_basit.pkl', "
        "'maliyet_modeli_gelismis.pkl' ve 'meta.json' dosyalarının GitHub'da "
        "app.py ile aynı yerde (ana dizinde) yüklü olduğundan emin olun."
    )
    st.stop()

# ------------------------------------------------------------------
# Kenar çubuğu: model seçimi + bilgi
# ------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Model Seçimi")
model_secimi = st.sidebar.radio(
    "Hangi modeli kullanmak istersiniz?",
    ["Basit Model (Hücre 5)", "Gelişmiş Model (Hücre 5 — Devam)"],
    help="Basit model sadece alan ve kat sayısını kullanır. Gelişmiş model "
         "zemin sınıfı ve inşaat yılını da ekler.",
)
gelismis_mi = model_secimi.startswith("Gelişmiş")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Bilgisi")
if gelismis_mi:
    m = meta["gelismis_model"]
    st.sidebar.metric("Test R²", f"{m['r2']:.3f}")
    st.sidebar.metric("Train R²", f"{m['train_r2']:.3f}")
    st.sidebar.metric("Test MAE", f"{m['mae']:,.0f} TL")
    st.sidebar.caption(
        "⚠️ Train R² ile Test R² arasındaki büyük fark, bu modelin "
        "**aşırı öğrenme (overfitting)** riski taşıdığını gösterir."
    )
else:
    m = meta["basit_model"]
    st.sidebar.metric("Test R²", f"{m['r2']:.3f}")
    st.sidebar.metric("Test MAE", f"{m['mae']:,.0f} TL")
    st.sidebar.caption(
        "⚠️ R² negatif — bu, modelin sadece 2 değişkenle (alan, kat) "
        "yeterince açıklayıcı olmadığının işaretidir."
    )
st.sidebar.markdown("---")
st.sidebar.caption(f"Eğitim verisi: {meta['n_proje']} proje kaydı")

# ------------------------------------------------------------------
# Girdi formu
# ------------------------------------------------------------------
st.markdown("### Proje Bilgilerini Girin")

col1, col2 = st.columns(2)
with col1:
    alan_m2 = st.number_input(
        "Taban Alanı (m²)", min_value=50, max_value=10000, value=1200, step=50,
    )
with col2:
    kat_sayisi = st.number_input(
        "Kat Sayısı", min_value=1, max_value=40, value=8, step=1,
    )

zemin_sinifi = None
insaat_yili = None
if gelismis_mi:
    col3, col4 = st.columns(2)
    with col3:
        zemin_sinifi = st.selectbox("Zemin Sınıfı (TBDY'ye göre)", meta["zemin_siniflari"], index=0)
    with col4:
        insaat_yili = st.number_input(
            "İnşaat (Bitiş) Yılı", min_value=2015, max_value=2030,
            value=meta["insaat_yili"]["max"], step=1,
        )

# ------------------------------------------------------------------
# Ekstrapolasyon kontrolü
# ------------------------------------------------------------------
def araligin_disinda_mi(deger, anahtar):
    lo, hi = meta[anahtar]["min"], meta[anahtar]["max"]
    return deger < lo or deger > hi, lo, hi

uyarilar = []
disi, lo, hi = araligin_disinda_mi(alan_m2, "alan_m2")
if disi:
    uyarilar.append(f"**Alan** ({alan_m2:,} m²) eğitim verisinin aralığının ({lo:,}-{hi:,} m²) dışında.")
disi, lo, hi = araligin_disinda_mi(kat_sayisi, "kat_sayisi")
if disi:
    uyarilar.append(f"**Kat sayısı** ({kat_sayisi}) eğitim verisinin aralığının ({lo}-{hi}) dışında.")
if gelismis_mi:
    disi, lo, hi = araligin_disinda_mi(insaat_yili, "insaat_yili")
    if disi:
        uyarilar.append(f"**İnşaat yılı** ({insaat_yili}) eğitim verisinin aralığının ({lo}-{hi}) dışında.")

# ------------------------------------------------------------------
# Tahmin
# ------------------------------------------------------------------
if st.button("💰 Maliyeti Tahmin Et", type="primary", use_container_width=True):
    if gelismis_mi:
        row = {"alan_m2": alan_m2, "kat_sayisi": kat_sayisi, "insaat_yili": insaat_yili,
               "zemin_sinifi_B": 0, "zemin_sinifi_C": 0, "zemin_sinifi_D": 0}
        if zemin_sinifi != "A":
            row[f"zemin_sinifi_{zemin_sinifi}"] = 1
        X_yeni = pd.DataFrame([row])[meta["gelismis_model"]["features"]]
        tahmin = model_gelismis.predict(X_yeni)[0]
    else:
        X_yeni = pd.DataFrame([{"alan_m2": alan_m2, "kat_sayisi": kat_sayisi}])
        tahmin = model_basit.predict(X_yeni)[0]

    if uyarilar:
        st.markdown(f"""
        <div class="result-box" style="background-color:#FEF2F2 !important; border: 2px solid #DC2626 !important;">
            <div class="value" style="color: #DC2626 !important;">{tahmin:,.0f} TL</div>
            <div class="label" style="color: #DC2626 !important;">Tahmini Toplam Maliyet — GÜVENİLİR DEĞİL</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            '<div class="warn-box"><b>⚠️ Ekstrapolasyon uyarısı:</b> ' +
            " ".join(uyarilar) +
            " Model bu bölgede hiçbir şey öğrenmemiştir, tahmin güvenilir değildir.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"""
        <div class="result-box">
            <div class="value">{tahmin:,.0f} TL</div>
            <div class="label">Tahmini Toplam Maliyet</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            '<div class="ok-box">✅ Girdi değerleri eğitim verisinin aralığı içinde — tahmin makul bir güvenilirlik taşıyor.</div>',
            unsafe_allow_html=True,
        )

    with st.expander("📐 Model bu tahmini nasıl hesapladı?"):
        if gelismis_mi:
            c = meta["gelismis_model"]["coefs"]
            st.markdown(f"""
Gelişmiş model, her özelliğin katsayısını (diğerleri sabitken) şu şekilde kullanır:

- Alan katsayısı: **{c['alan_m2']:,.0f} TL/m²**
- Kat katsayısı: **{c['kat_sayisi']:,.0f} TL/kat**
- Yıl katsayısı: **{c['insaat_yili']:,.0f} TL/yıl**
- Zemin B/C/D etkisi: **{c['zemin_sinifi_B']:,.0f}** / **{c['zemin_sinifi_C']:,.0f}** / **{c['zemin_sinifi_D']:,.0f}** TL (A zeminine göre farkı)
            """)
        else:
            b = meta["basit_model"]
            st.markdown(f"""
Basit model şu formülü kullanır:

**Maliyet = {b['alan_katsayisi']:,.0f} × Alan + {b['kat_katsayisi']:,.0f} × Kat + sabit**
            """)
        st.caption(
            "Not: Bu bir karar destek aracıdır, karar verici değil. Nihai kararı her zaman mühendis verir."
        )

st.markdown("---")
st.caption(
    "2023232064 - Hasan Karadağ - +90 544 971 50 52"
)
