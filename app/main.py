import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from prometheus_api_client import PrometheusConnect

st.set_page_config(
    page_title="OTel-AIOps",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("OTel-AIOps")
st.markdown("**OpenTelemetry Verisi Üzerinde Anomali Tespiti ve Kök Neden Analizi**")
st.divider()

# ── Modelleri Yükle ────────────────────────────────────────────────
@st.cache_resource
def modelleri_yukle():
    sonuc = {}

    # NAB Modeli
    try:
        with open("models/nab_model.pkl",     "rb") as f: sonuc["nab_model"]     = pickle.load(f)
        with open("models/nab_scaler.pkl",    "rb") as f: sonuc["nab_scaler"]    = pickle.load(f)
        with open("models/nab_threshold.pkl", "rb") as f: sonuc["nab_threshold"] = pickle.load(f)
        with open("models/nab_features.pkl",  "rb") as f: sonuc["nab_features"]  = pickle.load(f)
        sonuc["nab_yuklendi"] = True
        print("NAB modeli yüklendi!")
    except Exception as e:
        sonuc["nab_yuklendi"] = False

    # Prometheus Modeli
    try:
        with open("models/prometheus_model.pkl",    "rb") as f: sonuc["prom_model"]    = pickle.load(f)
        with open("models/prometheus_scaler.pkl",   "rb") as f: sonuc["prom_scaler"]   = pickle.load(f)
        with open("models/prometheus_features.pkl", "rb") as f: sonuc["prom_features"] = pickle.load(f)
        with open("models/prometheus_threshold.pkl","rb") as f: sonuc["prom_threshold"]= pickle.load(f)
        with open("models/prometheus_metrics.pkl",  "rb") as f: sonuc["prom_metrics"]  = pickle.load(f)
        sonuc["prom_yuklendi"] = True
        print("Prometheus modeli yüklendi!")
    except Exception as e:
        sonuc["prom_yuklendi"] = False

    return sonuc

modeller = modelleri_yukle()

# Sidebar
st.sidebar.title("Ayarlar")
mod = st.sidebar.radio("Mod Seç", ["CSV Yükle ve Analiz Et", "Gerçek Zamanlı İzleme"])

# Model durumu
if modeller.get("nab_yuklendi"):
    st.sidebar.success("NAB modeli yüklendi!")
if modeller.get("prom_yuklendi"):
    st.sidebar.success("Prometheus modeli yüklendi!")

contamination    = st.sidebar.slider("Anomali Orani (%)", 1, 10, 2) / 100
threshold_manual = st.sidebar.slider("Esik Degeri", 0.10, 0.90, 0.35)
pencere          = st.sidebar.slider("Zaman Penceresi", 3, 15, 5)

# ── Yardımcı Fonksiyonlar ──────────────────────────────────────────
def anomali_tespit_nab(df, threshold):
    """NAB modeli — tek boyutlu metrik."""
    df = df.copy().reset_index(drop=True)
    df["rolling_mean"] = df["value"].rolling(pencere).mean()
    df["rolling_std"]  = df["value"].rolling(pencere).std()
    df["rolling_max"]  = df["value"].rolling(pencere).max()
    df["diff"]         = df["value"].diff()
    df = df.dropna().copy()

    if len(df) < 10:
        df["hybrid_score"] = 0
        df["anomaly_pred"] = 0
        return df

    features = ["value", "rolling_mean", "rolling_std", "rolling_max", "diff"]
    X        = df[features].values

    if modeller.get("nab_yuklendi"):
        X_scaled = modeller["nab_scaler"].transform(X)
        if_score = -modeller["nab_model"].decision_function(X_scaled)
    else:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model    = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
        model.fit(X_scaled)
        if_score = -model.decision_function(X_scaled)

    z_scores = np.abs(zscore(X))
    max_z    = np.nan_to_num(z_scores.max(axis=1), nan=0.0)
    if_norm  = (if_score - if_score.min()) / (if_score.max() - if_score.min() + 1e-9)
    z_norm   = (max_z   - max_z.min())    / (max_z.max()    - max_z.min()    + 1e-9)

    df["hybrid_score"] = 0.4 * if_norm + 0.6 * z_norm
    df["anomaly_pred"] = (df["hybrid_score"] >= threshold).astype(int)
    return df

def anomali_tespit_prometheus(df_pivot, threshold):
    """Prometheus modeli — çok boyutlu, kendi sistemine özel."""
    df_pivot = df_pivot.copy().reset_index(drop=True)

    metrik_sutunlar = modeller["prom_metrics"]
    pencere_p       = 5

    for col in metrik_sutunlar:
        if col in df_pivot.columns:
            df_pivot[f"{col}_rolling_mean"] = df_pivot[col].rolling(pencere_p).mean()
            df_pivot[f"{col}_rolling_std"]  = df_pivot[col].rolling(pencere_p).std()
            df_pivot[f"{col}_diff"]         = df_pivot[col].diff()

    df_pivot = df_pivot.dropna().reset_index(drop=True)

    if len(df_pivot) < 10:
        df_pivot["hybrid_score"] = 0
        df_pivot["anomaly_pred"] = 0
        return df_pivot

    feature_cols = modeller["prom_features"]
    mevcut_cols  = [c for c in feature_cols if c in df_pivot.columns]

    if len(mevcut_cols) < len(feature_cols):
        eksik = [c for c in feature_cols if c not in df_pivot.columns]
        for c in eksik:
            df_pivot[c] = 0

    X        = df_pivot[feature_cols].values
    X_scaled = modeller["prom_scaler"].transform(X)
    if_score = -modeller["prom_model"].decision_function(X_scaled)

    z_scores = np.abs(zscore(X_scaled))
    max_z    = np.nan_to_num(z_scores.max(axis=1), nan=0.0)
    if_norm  = (if_score - if_score.min()) / (if_score.max() - if_score.min() + 1e-9)
    z_norm   = (max_z   - max_z.min())    / (max_z.max()    - max_z.min()    + 1e-9)

    df_pivot["hybrid_score"] = 0.4 * if_norm + 0.6 * z_norm
    df_pivot["anomaly_pred"] = (df_pivot["hybrid_score"] >= threshold).astype(int)
    return df_pivot

def servis_belirle(metrik_adi):
    if "cpu"  in metrik_adi.lower() or "CPU" in metrik_adi: return "ec2-compute-service"
    if "bellek" in metrik_adi.lower() or "Bellek" in metrik_adi: return "ec2-memory-service"
    if "network" in metrik_adi.lower() or "Network" in metrik_adi: return "ec2-network-service"
    if "disk" in metrik_adi.lower() or "Disk" in metrik_adi: return "ec2-storage-service"
    return "unknown-service"

def rca_yap(df, metrik_adi):
    anomaliler = df[df["anomaly_pred"] == 1].copy()
    if len(anomaliler) == 0:
        return pd.DataFrame()
    servis = servis_belirle(metrik_adi)
    anomaliler["servis"]      = servis
    anomaliler["hata_mesaji"] = anomaliler.get("value", anomaliler.iloc[:, 1]).apply(
        lambda v: f"AnomalyDetected [{metrik_adi}]: spike ({v:.4f})"
    )
    cols = ["timestamp", "servis", "hybrid_score", "hata_mesaji"]
    if "value" in anomaliler.columns:
        cols = ["timestamp", "value", "hybrid_score", "servis", "hata_mesaji"]
    return anomaliler[cols]

def prometheus_veri_cek(prom, sorgu, sure_dk=10):
    try:
        simdi     = pd.Timestamp.now()
        baslangic = simdi - pd.Timedelta(minutes=sure_dk)
        sonuc     = prom.custom_query_range(
            query      = sorgu,
            start_time = baslangic.to_pydatetime(),
            end_time   = simdi.to_pydatetime(),
            step       = "15s"
        )
        if not sonuc:
            return pd.DataFrame()

        kayitlar = []
        for seri in sonuc:
            for ts, val in seri["values"]:
                try:
                    kayitlar.append({
                        "timestamp": pd.Timestamp(ts, unit="s"),
                        "value"    : float(val)
                    })
                except:
                    continue

        df = pd.DataFrame(kayitlar)
        if len(df) > 0:
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["value"] = df["value"].clip(lower=0)
        return df
    except Exception as e:
        st.error(f"Prometheus hatasi: {e}")
        return pd.DataFrame()

def prometheus_tum_metrikleri_cek(prom, sure_dk=10):
    """Tüm metrikleri pivot formatta çeker."""
    SORGULAR = {
        "cpu":         'avg(rate(node_cpu_seconds_total{mode!="idle", job="node-exporter"}[2m])) / avg(rate(node_cpu_seconds_total{job="node-exporter"}[2m])) * 100',
        "bellek":      '100 - ((avg(node_memory_MemAvailable_bytes{job="node-exporter"}) / avg(node_memory_MemTotal_bytes{job="node-exporter"})) * 100)',
        "network_in":  'sum(rate(node_network_receive_bytes_total{job="node-exporter"}[2m]))',
        "network_out": 'sum(rate(node_network_transmit_bytes_total{job="node-exporter"}[2m]))',
        "disk":        'sum(rate(node_disk_read_bytes_total{job="node-exporter"}[2m]))',
    }

    simdi     = pd.Timestamp.now()
    baslangic = simdi - pd.Timedelta(minutes=sure_dk)
    df_liste  = []

    for metrik_adi, sorgu in SORGULAR.items():
        try:
            sonuc = prom.custom_query_range(
                query      = sorgu,
                start_time = baslangic.to_pydatetime(),
                end_time   = simdi.to_pydatetime(),
                step       = "15s"
            )
            if sonuc:
                for seri in sonuc:
                    for ts, val in seri["values"]:
                        try:
                            df_liste.append({
                                "timestamp": pd.Timestamp(ts, unit="s"),
                                "metrik"   : metrik_adi,
                                "value"    : float(val)
                            })
                        except:
                            continue
        except:
            continue

    if not df_liste:
        return pd.DataFrame()

    df_raw   = pd.DataFrame(df_liste)
    df_raw["value"] = df_raw["value"].clip(lower=0)
    df_pivot = df_raw.pivot_table(
        index="timestamp", columns="metrik", values="value"
    ).reset_index()
    df_pivot.columns.name = None
    return df_pivot.dropna().reset_index(drop=True)

# Metrik tanımları (tek boyutlu izleme için)
METRIKLER = {
    "CPU (%)":             'avg(rate(node_cpu_seconds_total{mode!="idle", job="node-exporter"}[2m])) / avg(rate(node_cpu_seconds_total{job="node-exporter"}[2m])) * 100',
    "Bellek (%)":          '100 - ((avg(node_memory_MemAvailable_bytes{job="node-exporter"}) / avg(node_memory_MemTotal_bytes{job="node-exporter"})) * 100)',
    "Network In (bytes/s)":'sum(rate(node_network_receive_bytes_total{job="node-exporter"}[2m]))',
    "Network Out (bytes/s)":'sum(rate(node_network_transmit_bytes_total{job="node-exporter"}[2m]))',
    "Disk Okuma (bytes/s)":'sum(rate(node_disk_read_bytes_total{job="node-exporter"}[2m]))',
}

# ── MOD 1: CSV Yükle ──────────────────────────────────────────────
if mod == "CSV Yükle ve Analiz Et":
    st.header("CSV Yükle ve Analiz Et")
    st.info(f"Aktif Model: **NAB (AWS CloudWatch)** | Esik: **{modeller.get('nab_threshold', 0.35)}**")

    yuklenen = st.file_uploader(
        "NAB formatında CSV yükle (timestamp, value sütunlari olmalı)",
        type=["csv"],
        accept_multiple_files=True
    )

    if yuklenen:
        tum_df = []
        for dosya in yuklenen:
            df_tmp = pd.read_csv(dosya, parse_dates=["timestamp"])
            df_tmp["dosya"] = dosya.name
            tum_df.append(df_tmp)

        df_all = pd.concat(tum_df, ignore_index=True)
        st.success(f"{len(yuklenen)} dosya yüklendi — {len(df_all):,} veri noktasi")

        with st.spinner("NAB modeli ile analiz yapılıyor..."):
            sonuclar = []
            for dosya_adi, grup in df_all.groupby("dosya"):
                try:
                    sonuc = anomali_tespit_nab(
                        grup, modeller.get("nab_threshold", threshold_manual)
                    )
                    sonuc["dosya"] = dosya_adi
                    sonuclar.append(sonuc)
                except Exception as e:
                    st.warning(f"{dosya_adi} analiz edilemedi: {e}")

            df_result = pd.concat(sonuclar, ignore_index=True)

        toplam       = len(df_result)
        n_anomali    = int(df_result["anomaly_pred"].sum())
        anomali_oran = df_result["anomaly_pred"].mean() * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Toplam Veri Noktası", f"{toplam:}")
        col2.metric("Tespit Edilen Anomali", n_anomali)
        col3.metric("Anomali Oranı", f"%{anomali_oran:.2f}")
        col4.metric("Model", "NAB")

        st.divider()
        st.subheader("Metrik Grafikleri")

        for dosya_adi, grup in df_result.groupby("dosya"):
            with st.expander(f"📊 {dosya_adi}", expanded=True):
                fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
                fig.patch.set_facecolor("#0F1117")
                for ax in axes:
                    ax.set_facecolor("#1A1D27")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#333")

                axes[0].plot(grup["timestamp"], grup["value"],
                             color="steelblue", linewidth=0.7, alpha=0.8, label="Metrik")
                anomaliler = grup[grup["anomaly_pred"] == 1]
                if len(anomaliler) > 0:
                    axes[0].scatter(anomaliler["timestamp"], anomaliler["value"],
                                    color="red", s=20, zorder=5,
                                    label=f"Anomali ({len(anomaliler)})")
                axes[0].set_ylabel("Değer", color="white")
                axes[0].tick_params(colors="white")
                axes[0].legend(fontsize=8, facecolor="#1A1D27", labelcolor="white")
                axes[0].grid(True, alpha=0.2, color="white")
                axes[0].set_title(dosya_adi, color="white", fontsize=9)

                axes[1].fill_between(grup["timestamp"], grup["hybrid_score"],
                                     alpha=0.4, color="orange")
                axes[1].plot(grup["timestamp"], grup["hybrid_score"],
                             color="orange", linewidth=0.8)
                axes[1].axhline(modeller.get("nab_threshold", threshold_manual),
                                color="red", linestyle="--",
                                label=f"Eşik ({modeller.get('nab_threshold', threshold_manual)})")
                axes[1].set_ylabel("Anomali Skoru", color="white")
                axes[1].set_xlabel("Zaman", color="white")
                axes[1].tick_params(colors="white")
                axes[1].legend(fontsize=8, facecolor="#1A1D27", labelcolor="white")
                axes[1].grid(True, alpha=0.2, color="white")
                axes[1].set_ylim(0, 1.1)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.divider()
        st.subheader("Kök Neden Analizi (RCA)")

        rca_listesi = []
        for dosya_adi, grup in df_result.groupby("dosya"):
            rca = rca_yap(grup, dosya_adi)
            if len(rca) > 0:
                rca_listesi.append(rca)

        if rca_listesi:
            df_rca = pd.concat(rca_listesi, ignore_index=True)
            st.dataframe(
                df_rca.sort_values("hybrid_score", ascending=False).head(50),
                use_container_width=True
            )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Servis Bazında Anomali")
                st.bar_chart(df_rca["servis"].value_counts())
            with col2:
                st.subheader("Anomali Skoru Dağılımı")
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                fig2.patch.set_facecolor("#0F1117")
                ax2.set_facecolor("#1A1D27")
                ax2.hist(df_rca["hybrid_score"], bins=30,
                         color="steelblue", edgecolor="white", alpha=0.8)
                ax2.axvline(modeller.get("nab_threshold", threshold_manual),
                            color="red", linestyle="--")
                ax2.tick_params(colors="white")
                ax2.grid(True, alpha=0.2, color="white")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

            csv = df_rca.to_csv(index=False).encode("utf-8")
            st.download_button("RCA Sonuçlarını İndir", csv,
                               "rca_sonuclari.csv", "text/csv")

# ── MOD 2: Gerçek Zamanlı Prometheus İzleme ───────────────────────
else:
    st.header("Gerçek Zamanlı Prometheus İzleme")

    if modeller.get("prom_yuklendi"):
        st.info("Prometheus modeli aktif — Kendi sistemine özel model kullanılıyor!")
    else:
        st.warning("Prometheus modeli yüklenemedi — NAB modeli kullanılıyor.")

    prom_url = st.sidebar.text_input("Prometheus URL", "http://localhost:9090")
    sure_dk  = st.sidebar.slider("Geçmiş veri (dakika)", 5, 60, 10)
    yenileme = st.sidebar.slider("Yenileme süresi (sn)", 5, 60, 15)

    col_btn1, col_btn2 = st.columns(2)
    baslat = col_btn1.button("İzlemeyi Başlat", type="primary")
    durdur = col_btn2.button("Durdur")

    if "izleme_aktif" not in st.session_state:
        st.session_state.izleme_aktif = False
    if baslat:
        st.session_state.izleme_aktif = True
    if durdur:
        st.session_state.izleme_aktif = False

    if st.session_state.izleme_aktif:
        try:
            prom = PrometheusConnect(url=prom_url, disable_ssl=True)
            st.success(f"Prometheus bağlantısı kuruldu: {prom_url}")
        except Exception as e:
            st.error(f"Prometheus bağlantısı kurulamadı: {e}")
            st.stop()

        placeholder = st.empty()

        while st.session_state.izleme_aktif:
            simdi = pd.Timestamp.now()

            with placeholder.container():
                st.caption(f"Son güncelleme: {simdi.strftime('%H:%M:%S')}")

                # Tüm metrikleri çek
                df_pivot = prometheus_tum_metrikleri_cek(prom, sure_dk)

                if df_pivot.empty:
                    st.warning("Prometheus'tan veri alınamadı.")
                    time.sleep(yenileme)
                    st.rerun()
                    continue

                # Prometheus modeli ile analiz
                if modeller.get("prom_yuklendi"):
                    sonuc_pivot = anomali_tespit_prometheus(
                        df_pivot, modeller["prom_threshold"]
                    )
                    model_adi = "Prometheus (Sisteme Özel)"
                    threshold_goster = modeller["prom_threshold"]
                else:
                    # Fallback: NAB ile CPU analizi
                    df_cpu = df_pivot[["timestamp", "cpu"]].rename(columns={"cpu": "value"})
                    sonuc_pivot = anomali_tespit_nab(df_cpu, modeller.get("nab_threshold", 0.35))
                    model_adi = "NAB (Fallback)"
                    threshold_goster = modeller.get("nab_threshold", 0.35)

                tum_anomali = int(sonuc_pivot["anomaly_pred"].sum())

                # KPI kartları
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Veri Noktası", f"{len(sonuc_pivot):,}")
                col2.metric("Toplam Anomali", tum_anomali,
                            delta=f"+{tum_anomali}" if tum_anomali > 0 else None,
                            delta_color="inverse")
                col3.metric("Model", model_adi.split("(")[0].strip())
                col4.metric("Prometheus", "Bağlı", delta="canlı")

                # Her metrik için grafik
                metrik_sutunlar = ["cpu", "bellek", "network_in", "network_out", "disk"]
                metrik_renkleri = {
                    "cpu"        : "steelblue",
                    "bellek"     : "orange",
                    "network_in" : "green",
                    "network_out": "red",
                    "disk"       : "purple"
                }

                for metrik in metrik_sutunlar:
                    if metrik not in sonuc_pivot.columns:
                        continue

                    n_anomali = int(sonuc_pivot["anomaly_pred"].sum())

                    with st.expander(f"📊 {metrik.upper()} — Anomali: {n_anomali}", expanded=True):
                        fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
                        fig.patch.set_facecolor("#0F1117")
                        for ax in axes:
                            ax.set_facecolor("#1A1D27")
                            for spine in ax.spines.values():
                                spine.set_edgecolor("#333")

                        renk = metrik_renkleri.get(metrik, "steelblue")
                        axes[0].plot(sonuc_pivot["timestamp"], sonuc_pivot[metrik],
                                     color=renk, linewidth=0.9, alpha=0.9, label=metrik)

                        anomaliler = sonuc_pivot[sonuc_pivot["anomaly_pred"] == 1]
                        if len(anomaliler) > 0:
                            axes[0].scatter(
                                anomaliler["timestamp"], anomaliler[metrik],
                                color="red", s=40, zorder=5,
                                label=f"Anomali ({len(anomaliler)})"
                            )

                        axes[0].set_ylabel(metrik, color="white")
                        axes[0].tick_params(colors="white")
                        axes[0].legend(fontsize=8, facecolor="#1A1D27", labelcolor="white")
                        axes[0].grid(True, alpha=0.2, color="white")
                        axes[0].set_title(f"{metrik.upper()} [{model_adi}]",
                                          color="white", fontsize=10)

                        axes[1].fill_between(sonuc_pivot["timestamp"],
                                             sonuc_pivot["hybrid_score"],
                                             alpha=0.4, color="orange")
                        axes[1].plot(sonuc_pivot["timestamp"], sonuc_pivot["hybrid_score"],
                                     color="orange", linewidth=0.8)
                        axes[1].axhline(threshold_goster, color="red", linestyle="--",
                                        label=f"Esik ({threshold_goster})")
                        axes[1].set_ylabel("Anomali Skoru", color="white")
                        axes[1].set_xlabel("Zaman", color="white")
                        axes[1].tick_params(colors="white")
                        axes[1].legend(fontsize=8, facecolor="#1A1D27", labelcolor="white")
                        axes[1].grid(True, alpha=0.2, color="white")
                        axes[1].set_ylim(0, 1.1)

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                # RCA tablosu
                st.divider()
                if tum_anomali > 0:
                    st.subheader("Canlı RCA Raporu")
                    anomali_satirlar = sonuc_pivot[sonuc_pivot["anomaly_pred"] == 1].copy()
                    rca_rows = []
                    for _, row in anomali_satirlar.iterrows():
                        for metrik in metrik_sutunlar:
                            if metrik in row:
                                rca_rows.append({
                                    "timestamp"   : row["timestamp"],
                                    "metrik"      : metrik,
                                    "value"       : round(row[metrik], 4),
                                    "hybrid_score": round(row["hybrid_score"], 4),
                                    "servis"      : servis_belirle(metrik),
                                    "hata_mesaji" : f"AnomalyDetected [{metrik}]: {row[metrik]:.4f}"
                                })

                    if rca_rows:
                        df_rca = pd.DataFrame(rca_rows)
                        st.dataframe(
                            df_rca.sort_values("hybrid_score", ascending=False).head(20),
                            use_container_width=True
                        )
                else:
                    st.success("Aktif anomali tespit edilmedi.")

            time.sleep(yenileme)
            st.rerun()