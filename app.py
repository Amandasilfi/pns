"""
=========================================================
 Simulator Kebijakan Keuntungan Toko - Minggu 14
 Pemodelan & Simulasi - Analisis What-If
=========================================================
Cara menjalankan:
    pip install streamlit pandas numpy scikit-learn matplotlib
    streamlit run app.py
=========================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# KONFIGURASI HALAMAN
# ---------------------------------------------------------
st.set_page_config(
    page_title="Simulator Kebijakan Toko",
    page_icon="🚀",
    layout="wide"
)

# ---------------------------------------------------------
# 1. LOAD / LATIH MODEL (Mesin Replika / Digital Twin)
#    @st.cache_resource -> model tidak dilatih ulang setiap slider digeser
# ---------------------------------------------------------
@st.cache_resource
def train_model():
    # Fitur: [Iklan (Juta), Diskon (%)]
    X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]])
    # Target: Keuntungan (Juta)
    y_train = np.array([50, 80, 110, 90, 150])
    model = LinearRegression().fit(X_train, y_train)
    return model

model = train_model()

# ---------------------------------------------------------
# 2. DEFINE BASELINE (Skenario Bisnis Saat Ini)
# ---------------------------------------------------------
BASELINE_IKLAN = 10
BASELINE_DISKON = 10
baseline_input = np.array([[BASELINE_IKLAN, BASELINE_DISKON]])
baseline_pred = model.predict(baseline_input)[0]

# Asumsi error model (RMSE) untuk analisis ketidakpastian (Soal Umpan Balik No.9)
RMSE_MODEL = 10  # Juta

# ---------------------------------------------------------
# 3. SIMULATION ENGINE (Logika What-If)
# ---------------------------------------------------------
def run_simulation(new_iklan, new_diskon):
    intervention_input = np.array([[new_iklan, new_diskon]])
    prediction = model.predict(intervention_input)[0]
    delta_y = prediction - baseline_pred
    return prediction, delta_y

def sensitivity_analysis(base_iklan, base_diskon, step=5):
    """Analisis sensitivitas: uji +step pada masing-masing variabel kontrol."""
    pred_iklan_up, _ = run_simulation(base_iklan + step, base_diskon)
    pred_diskon_up, _ = run_simulation(base_iklan, base_diskon + step)
    base_pred, _ = run_simulation(base_iklan, base_diskon)
    sens_iklan = pred_iklan_up - base_pred
    sens_diskon = pred_diskon_up - base_pred
    return sens_iklan, sens_diskon

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; padding: 10px; border-bottom: 2px solid #4F46E5; margin-bottom: 20px;">
        <span style="letter-spacing: 2px; font-weight: 600; color: #6B7280; font-size: 0.85rem;">
            Amanda Silfiana | 2313020186 | PEMODELAN DAN SIMULASI
        </span>
    </div>
    """, 
    unsafe_allow_html=True
)
st.title("🚀 Simulator Kebijakan Keuntungan Toko")
st.write(
    "Gunakan **Tuas Kebijakan** di sidebar untuk menguji skenario *What-If* "
    "dan lihat dampaknya secara real-time terhadap prediksi keuntungan toko."
)
st.divider()

# ---------------------------------------------------------
# SIDEBAR: VARIABEL KONTROL (Controllable Input)
# ---------------------------------------------------------
st.sidebar.header("🎛️ Tuas Kebijakan (Intervensi)")
iklan_slider = st.sidebar.slider("📢 Anggaran Iklan (Juta Rp)", 0, 50, BASELINE_IKLAN)
diskon_slider = st.sidebar.slider("🏷️ Besaran Diskon (%)", 0, 50, BASELINE_DISKON)

st.sidebar.divider()
st.sidebar.caption(
    "ℹ️ Slider dibatasi maksimal 50 agar simulasi tidak melakukan "
    "**ekstrapolasi** jauh di luar rentang data historis (lihat materi Minggu 4)."
)

# Variabel Kontekstual (Non-Controllable) - hiasan kontekstual realisme pasar
st.sidebar.divider()
st.sidebar.header("🌦️ Kondisi Eksternal (Non-Controllable)")
harga_kompetitor = st.sidebar.select_slider(
    "Harga Kompetitor",
    options=["Jauh Lebih Murah", "Lebih Murah", "Setara", "Lebih Mahal"],
    value="Setara",
)
st.sidebar.caption(
    "Variabel ini tidak masuk ke model (di luar scope data latih), namun "
    "ditampilkan sebagai konteks naratif bagi pengambil keputusan."
)

# ---------------------------------------------------------
# ENGINE: JALANKAN SIMULASI
# ---------------------------------------------------------
hasil_pred, delta = run_simulation(iklan_slider, diskon_slider)
sens_iklan, sens_diskon = sensitivity_analysis(iklan_slider, diskon_slider, step=5)

# ===========================================================
# >>> KEKHASAN KAMI #1: MOOD INDICATOR TOKO <<<
# Emoji & warna berubah otomatis sesuai besar/kecilnya delta
# ===========================================================
def get_mood(delta_value):
    if delta_value >= 20:
        return "🤩", "Luar Biasa!", "#0E8C3A"
    elif delta_value >= 5:
        return "😀", "Cukup Baik", "#3CB371"
    elif delta_value > -5:
        return "😐", "Netral / Stagnan", "#B8860B"
    elif delta_value > -20:
        return "😟", "Perlu Waspada", "#D2691E"
    else:
        return "😱", "Berisiko Tinggi!", "#C0392B"

mood_emoji, mood_label, mood_color = get_mood(delta)

st.markdown(
    f"""
    <div style="
        background-color:{mood_color}1A;
        border-left: 6px solid {mood_color};
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 18px;">
        <span style="font-size:34px;">{mood_emoji}</span>
        <span style="font-size:20px; font-weight:600; color:{mood_color};">
            &nbsp; Mood Toko: {mood_label}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# UI: TAMPILKAN HASIL UTAMA
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("📌 Baseline Saat Ini", f"Rp {baseline_pred:.2f} Jt")
col2.metric("🔮 Prediksi Skenario Baru", f"Rp {hasil_pred:.2f} Jt", f"{delta:.2f} Jt")
col3.metric(
    "⚠️ Rentang Ketidakpastian (±RMSE)",
    f"Rp {RMSE_MODEL} Jt",
    help="Akurasi model memiliki margin error RMSE sebesar ini. "
         "Anggap prediksi sebagai rentang, bukan angka pasti."
)

st.info(
    f"Skenario ini (Iklan Rp{iklan_slider} Jt, Diskon {diskon_slider}%) menghasilkan "
    f"perubahan sebesar **{delta:.2f} Juta** dibandingkan kondisi baseline "
    f"(Iklan Rp{BASELINE_IKLAN} Jt, Diskon {BASELINE_DISKON}%)."
)

st.divider()

# ---------------------------------------------------------
# VISUALISASI 1: PERBANDINGAN BASELINE VS INTERVENSI
# ---------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("📊 Perbandingan Skenario")
    data_plot = pd.DataFrame({
        'Skenario': ['Baseline', 'Intervensi'],
        'Keuntungan': [baseline_pred, hasil_pred]
    })
    st.bar_chart(data=data_plot, x='Skenario', y='Keuntungan')

# ===========================================================
# >>> KEKHASAN KAMI #2: TORNADO CHART SENSITIVITAS OTOMATIS <<<
# Membandingkan dampak +5 unit pada Iklan vs Diskon
# ===========================================================
with right:
    st.subheader("🌪️ Tornado Chart: Analisis Sensitivitas")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    vars_ = ["Diskon (+5%)", "Iklan (+5Jt)"]
    values = [sens_diskon, sens_iklan]
    colors = ["#4C72B0" if v >= 0 else "#C0392B" for v in values]
    bars = ax.barh(vars_, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ Keuntungan (Juta Rp)")
    for bar, v in zip(bars, values):
        ax.text(v, bar.get_y() + bar.get_height()/2, f" {v:.1f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    st.pyplot(fig)

    paling_sensitif = "Diskon" if abs(sens_diskon) > abs(sens_iklan) else "Iklan"
    st.caption(
        f"🔑 **Tuas kebijakan paling sensitif saat ini: `{paling_sensitif}`** — "
        f"perubahan kecil pada variabel ini memberi dampak terbesar terhadap keuntungan."
    )

st.divider()

# >>> KEKHASAN KAMI #3: NARASI REKOMENDASI OTOMATIS <<< 
st.subheader("🧠 Narasi Rekomendasi Otomatis (Auto-Insight)") 


if delta >= 5: 
    rekomendasi = ( 
        f"Skenario ini **direkomendasikan**. Kombinasi Iklan Rp{iklan_slider} Jt dan " 
        f"Diskon {diskon_slider}% berpotensi menaikkan keuntungan sebesar **{delta:.2f} Juta** " 
        f"dibanding kondisi saat ini. Pertimbangkan untuk menerapkan kebijakan ini secara bertahap " 
        f"sambil memantau {paling_sensitif.lower()} sebagai indikator utama." 
    ) 
    st.success(rekomendasi) 
elif delta <= -5: 
    rekomendasi = ( 
        f"⚠️ Skenario ini **berisiko menurunkan keuntungan** sebesar **{abs(delta):.2f} Juta**. " 
        f"Disarankan untuk **tidak** menerapkan kombinasi ini, atau melakukan uji coba kecil " 
        f"(pilot test) terlebih dahulu sebelum rollout penuh." 
    ) 
    st.error(rekomendasi) 
else: 
    rekomendasi = ( 
        "Skenario ini menghasilkan dampak yang **relatif netral**. Tidak ada alasan kuat untuk " 
        "mengubah kebijakan saat ini kecuali ada pertimbangan non-finansial lain " 
        "(misalnya brand awareness atau loyalitas pelanggan)." 
    ) 
    st.warning(rekomendasi) 


st.caption( 
    f"Catatan: Harga kompetitor saat ini diasumsikan **{harga_kompetitor}** — " 
    "faktor ini tidak dimodelkan secara numerik, namun sebaiknya jadi bahan pertimbangan kualitatif manajer." 
) 

st.divider() 
st.caption( 
    "💡 Dibangun untuk Praktikum Pemodelan & Simulasi — Minggu 14 (Simulator Interaktif & Analisis What-If). " 
    "Model: Regresi Linear sederhana (Digital Twin), Engine: model.predict() real-time via Streamlit." 
) 
