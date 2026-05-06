import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy import stats
from scipy.stats import boxcox, shapiro
from io import BytesIO

# ──────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Transformación Box-Cox",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
#  CSS PERSONALIZADO
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fuentes */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Header principal */
    .main-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 4.2rem;
        font-weight: 600;
        color: #0f4c81;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .main-subtitle {
        font-size: 1rem;
        color: #5f7a99;
        margin-top: 0.2rem;
        font-weight: 300;
    }

    /* Tarjetas de métricas */
    div[data-testid="metric-container"] {
        background: #f0f6ff;
        border: 1px solid #c8ddf5;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.78rem !important;
        color: #5f7a99 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem !important;
        color: #0f4c81 !important;
        font-weight: 600;
    }

    /* Secciones */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: #0f4c81;
        border-left: 4px solid #2980e8;
        padding-left: 10px;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* Cajas informativas */
    .info-box {
        background: #edf4ff;
        border: 1px solid #b3d0f5;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.91rem;
        color: #1a3a5c;
        line-height: 1.6;
    }
    .formula-box {
        background: #1a1a2e;
        color: #e0e8ff;
        font-family: 'IBM Plex Mono', monospace;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.88rem;
        line-height: 1.9;
    }
    .result-ok  { background:#e8f5e9; border-left:4px solid #4caf50; padding:10px 14px; border-radius:6px; }
    .result-bad { background:#fff3e0; border-left:4px solid #ff9800; padding:10px 14px; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
#  ENCABEZADO
# ──────────────────────────────────────────────────────────────────
col_hdr, col_logo = st.columns([4, 1])
with col_hdr:
    st.markdown('<p class="main-title">📊 Transformación Box-Cox</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Herramienta interactiva para normalización de datos estadísticos</p>',
                unsafe_allow_html=True)
st.divider()

# ──────────────────────────────────────────────────────────────────
#  BARRA LATERAL — CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parámetros de análisis")

    lambda_min_s = st.slider("λ mínimo", -5.0, -0.1, -3.0, 0.1,
                              help="Límite inferior del rango de búsqueda de lambda")
    lambda_max_s = st.slider("λ máximo",  0.1,  5.0,  3.0, 0.1,
                              help="Límite superior del rango de búsqueda de lambda")
    n_pts        = st.slider("Resolución de la curva", 100, 1000, 400, 50,
                              help="Cantidad de valores de λ evaluados")
    bins_n       = st.slider("Intervalos del histograma", 5, 50, 12, 1)

    st.divider()
    st.markdown("### 📐 Fórmula de transformación")
    st.markdown("""
<div class="formula-box">
  λ ≠ 0 →  y' = (yᵏ − 1) / (λ · Gᵏ⁻¹)<br>
  λ = 0  →  y' = G · ln(y)<br><br>
  <span style="color:#8bb8e8">G = media geométrica</span>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
<div class="info-box">
<b>¿Qué buscamos?</b><br>
El valor de <b>λ</b> que <b>minimiza la desviación estándar</b> de los datos transformados,
logrando que su distribución se aproxime a la normal.
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
#  SECCIÓN 1 — INGRESO DE DATOS
# ──────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">① Ingreso de Datos</p>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown("""
<div class="info-box">
<b>Instrucciones:</b>
<ul style="margin:6px 0 0 0; padding-left:18px;">
  <li>Edite la tabla como si fuera Excel</li>
  <li>Use el botón <b>+</b> para agregar filas</li>
  <li>Todos los valores deben ser <b>positivos</b></li>
  <li>Se requieren al menos <b>3 datos</b></li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader("📂 Importar desde CSV o Excel",
                                type=["csv", "xlsx"],
                                help="La primera columna numérica se usará como datos")

with col_right:
    # Datos por defecto (20 observaciones con sesgo positivo típico)
    _default_vals = [1.8, 2.3, 4.5, 1.2, 8.7, 3.4, 12.1, 5.6, 2.8,
                     9.3, 4.1, 6.7, 3.2, 11.4, 2.1, 7.8, 4.9, 1.8, 6.3, 3.7]

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                _df_up = pd.read_csv(uploaded)
            else:
                _df_up = pd.read_excel(uploaded)
            # Toma la primera columna numérica
            _num_cols = _df_up.select_dtypes(include="number").columns
            _init_df  = pd.DataFrame({"Valores": _df_up[_num_cols[0]].dropna().values})
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            _init_df = pd.DataFrame({"Valores": _default_vals})
    else:
        _init_df = pd.DataFrame({"Valores": _default_vals})

    edited_df = st.data_editor(
        _init_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Valores": st.column_config.NumberColumn(
                "Valores",
                help="Ingrese valores positivos (> 0)",
                format="%.4f",
                step=0.0001,
            )
        },
        key="tabla_datos",
    )

# ──────────────────────────────────────────────────────────────────
#  VALIDACIÓN
# ──────────────────────────────────────────────────────────────────
raw = edited_df["Valores"].dropna().values
data = raw[raw > 0]

if len(data) < 3:
    st.error("⚠️  Ingrese al menos **3 valores positivos** para continuar.")
    st.stop()

if len(data) < len(raw):
    st.warning(f"⚠️  Se ignoraron {len(raw)-len(data)} valor(es) no positivo(s).")

# ──────────────────────────────────────────────────────────────────
#  CÁLCULO CENTRAL
# ──────────────────────────────────────────────────────────────────
lambdas = np.linspace(lambda_min_s, lambda_max_s, n_pts)

# Media geométrica (para la transformación normalizada)
geo_mean = np.exp(np.mean(np.log(data)))

# Desviación estándar normalizada para cada λ
def boxcox_normalized(y, lam, G):
    """Transformación Box-Cox normalizada para comparar std entre λ's."""
    if abs(lam) < 1e-10:
        return G * np.log(y)
    return (y**lam - 1.0) / (lam * G**(lam - 1.0))

std_curve = np.array([
    np.std(boxcox_normalized(data, lam, geo_mean), ddof=1)
    for lam in lambdas
])

# λ óptimo por mínima desviación estándar
idx_opt    = int(np.argmin(std_curve))
lambda_opt = lambdas[idx_opt]
std_opt    = std_curve[idx_opt]

# λ óptimo por MLE (scipy)
_, lambda_mle = boxcox(data)

# Datos transformados con λ óptimo (mín std)
data_trans = boxcox_normalized(data, lambda_opt, geo_mean)

# Datos transformados con λ MLE (para comparación)
data_trans_mle = boxcox(data, lmbda=lambda_mle)

# Prueba de normalidad Shapiro-Wilk
sw_stat_orig,  sw_p_orig  = shapiro(data)
sw_stat_trans, sw_p_trans = shapiro(data_trans)
sw_stat_mle,   sw_p_mle   = shapiro(data_trans_mle)


def norm_badge(p):
    icon  = "✅" if p > 0.05 else "❌"
    label = "Normal" if p > 0.05 else "No normal"
    return f"{icon} {label} (p = {p:.4f})"


# ──────────────────────────────────────────────────────────────────
#  SECCIÓN 2 — MÉTRICAS PRINCIPALES
# ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-header">② Resultados del Análisis</p>', unsafe_allow_html=True)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("N° de datos",          f"{len(data)}")
m2.metric("Media geométrica (G)", f"{geo_mean:.4f}")
m3.metric("λ óptimo (mín. std)",  f"{lambda_opt:.4f}")
m4.metric("Std dev mínima",       f"{std_opt:.4f}")
m5.metric("λ óptimo (MLE)",       f"{lambda_mle:.4f}")
m6.metric("Skewness original",    f"{stats.skew(data):.4f}")

# Interpretación de λ
def interpret_lambda(lam):
    if abs(lam) < 0.05:           return "Transformación **logarítmica** — ln(y)"
    if abs(lam - 0.5)  < 0.08:   return "Transformación **raíz cuadrada** — √y"
    if abs(lam - 0.333) < 0.08:  return "Transformación **raíz cúbica** — ∛y"
    if abs(lam + 1.0)  < 0.08:   return "Transformación **inversa** — 1/y"
    if abs(lam - 1.0)  < 0.08:   return "**Sin transformación** — datos originales"
    if abs(lam - 2.0)  < 0.08:   return "Transformación **cuadrática** — y²"
    return f"Transformación de potencia — y^{lam:.3f}"

st.markdown("<br>", unsafe_allow_html=True)
col_interp, col_norm = st.columns(2)
with col_interp:
    st.info(f"**Interpretación del λ óptimo:**  {interpret_lambda(lambda_opt)}")
with col_norm:
    orig_ok  = sw_p_orig  > 0.05
    trans_ok = sw_p_trans > 0.05
    if trans_ok and not orig_ok:
        st.success(f"🎉 La transformación **mejoró** la normalidad  |  Original: {norm_badge(sw_p_orig)}  →  Transformado: {norm_badge(sw_p_trans)}")
    elif trans_ok:
        st.success(f"Los datos ya eran normales y se mantienen  |  {norm_badge(sw_p_trans)}")
    else:
        st.warning(f"La transformación no logró normalidad perfecta  |  {norm_badge(sw_p_trans)}")

# ──────────────────────────────────────────────────────────────────
#  SECCIÓN 3 — GRÁFICAS
# ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-header">③ Gráficas de Análisis</p>', unsafe_allow_html=True)

# Paleta de colores
C_ORIG  = "#2980e8"   # azul
C_TRANS = "#e84c2e"   # rojo-naranja
C_LINE  = "#1a1a2e"   # casi negro
C_OPT   = "#27ae60"   # verde
C_MLE   = "#e67e22"   # naranja
C_GRID  = "#e8ecf0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        C_GRID,
    "grid.linewidth":    0.8,
    "axes.facecolor":    "#fafcff",
    "figure.facecolor":  "#ffffff",
})

# ── Figura 1: Datos Originales ─────────────────────────────────
st.markdown("##### 📉 Comportamiento de los datos originales")

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5))
fig1.suptitle("Datos Originales", fontsize=13, fontweight="bold", color=C_LINE, y=1.02)

# 1a. Histograma original
ax = axes1[0]
n, bins_h, patches = ax.hist(data, bins=bins_n, color=C_ORIG, edgecolor="white",
                              alpha=0.85, linewidth=0.8)
# Curva normal superpuesta
xr = np.linspace(data.min(), data.max(), 200)
ax.plot(xr,
        stats.norm.pdf(xr, np.mean(data), np.std(data)) * len(data) * (bins_h[1]-bins_h[0]),
        "k--", linewidth=1.5, alpha=0.6, label="Normal teórica")
ax.set_title("Histograma + Curva Normal", fontsize=11, fontweight="bold")
ax.set_xlabel("Valor")
ax.set_ylabel("Frecuencia")
ax.text(0.97, 0.97, norm_badge(sw_p_orig), transform=ax.transAxes,
        ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9, edgecolor="#ccc"))
ax.legend(fontsize=8)

# 1b. Q-Q plot original
ax = axes1[1]
(osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
ax.scatter(osm, osr, color=C_ORIG, s=35, alpha=0.8, zorder=3, label="Datos")
ax.plot(osm, slope*np.array(osm) + intercept, "k--", linewidth=1.5, alpha=0.7, label="Línea normal")
ax.set_title("Gráfico Q-Q", fontsize=11, fontweight="bold")
ax.set_xlabel("Cuantiles teóricos (Normal)")
ax.set_ylabel("Cuantiles observados")
ax.text(0.05, 0.95, f"R² = {r**2:.4f}", transform=ax.transAxes, fontsize=9,
        va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax.legend(fontsize=8)

# 1c. Box-Whisker
ax = axes1[2]
bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                medianprops=dict(color="white", linewidth=2.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker="o", markersize=6, markerfacecolor=C_ORIG, alpha=0.7))
bp["boxes"][0].set_facecolor(C_ORIG)
bp["boxes"][0].set_alpha(0.75)
ax.set_title("Box-Whisker Plot", fontsize=11, fontweight="bold")
ax.set_ylabel("Valor")
ax.set_xticks([])
# Estadísticos en el margen
stats_txt = (f"Media: {np.mean(data):.3f}\n"
             f"Mediana: {np.median(data):.3f}\n"
             f"Asimetría: {stats.skew(data):.3f}\n"
             f"Curtosis: {stats.kurtosis(data):.3f}")
ax.text(1.35, 0.5, stats_txt, transform=ax.transAxes, fontsize=8.5,
        va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f6ff", edgecolor="#c8ddf5"))

plt.tight_layout()
st.pyplot(fig1, use_container_width=True)
plt.close(fig1)


# ── Figura 2: Std Dev vs Lambda (GRÁFICA PRINCIPAL) ───────────────
st.markdown("##### 📈 Desviación estándar vs exponente λ  ← gráfica clave")

fig2, ax2 = plt.subplots(figsize=(13, 5.5))

# Curva principal
ax2.plot(lambdas, std_curve, color=C_ORIG, linewidth=2.5, zorder=3)
ax2.fill_between(lambdas, std_curve, std_curve.max(),
                 alpha=0.07, color=C_ORIG)

# Punto mínimo
ax2.scatter([lambda_opt], [std_opt], color=C_OPT, s=200, zorder=6,
            marker="*", edgecolors="white", linewidth=1.2,
            label=f"λ mín. std = {lambda_opt:.4f}  (std = {std_opt:.4f})")

# Línea vertical λ óptimo (std)
ax2.axvline(lambda_opt, color=C_OPT, linestyle="--", linewidth=2, alpha=0.9,
            label=f"λ óptimo (mín std) = {lambda_opt:.4f}")

# Línea vertical λ MLE
ax2.axvline(lambda_mle, color=C_MLE, linestyle=":", linewidth=2, alpha=0.9,
            label=f"λ óptimo (MLE scipy) = {lambda_mle:.4f}")

# Anotación del mínimo
ax2.annotate(
    f"  Mínimo\n  λ = {lambda_opt:.4f}\n  std = {std_opt:.4f}",
    xy=(lambda_opt, std_opt),
    xytext=(lambda_opt + (lambda_max_s - lambda_min_s)*0.08,
            std_opt + (std_curve.max() - std_curve.min())*0.12),
    fontsize=9, color=C_OPT, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_OPT, lw=1.5),
)

# Referencias de λ comunes
ref_lambdas = {-1: "−1\n(1/y)", 0: "0\nln(y)", 0.5: "½\n(√y)",
               0.333: "⅓\n(∛y)", 1: "1\n(sin cambio)", 2: "2\n(y²)"}
y_ref = std_curve.min() - (std_curve.max() - std_curve.min()) * 0.14
for lv, ltxt in ref_lambdas.items():
    if lambda_min_s < lv < lambda_max_s:
        ax2.axvline(lv, color="#aab8cc", linestyle=":", linewidth=1, alpha=0.6)
        ax2.text(lv, y_ref, ltxt, ha="center", va="top", fontsize=7.5,
                 color="#7a8da0", fontweight="bold")

ax2.set_xlabel("Exponente λ (Lambda)", fontsize=13, labelpad=8)
ax2.set_ylabel("Desviación Estándar (normalizada)", fontsize=13, labelpad=8)
ax2.set_title("Búsqueda del λ óptimo: Mínima Desviación Estándar",
              fontsize=14, fontweight="bold", pad=12)
ax2.legend(fontsize=10, loc="upper right",
           frameon=True, fancybox=True, framealpha=0.92)
ax2.set_xlim(lambda_min_s, lambda_max_s)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

plt.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)

# Tabla de valores clave alrededor del mínimo
with st.expander("🔎 Ver tabla de valores λ alrededor del mínimo"):
    half = 10
    lo = max(0, idx_opt - half)
    hi = min(n_pts, idx_opt + half + 1)
    df_curve = pd.DataFrame({
        "λ (Lambda)":            lambdas[lo:hi].round(4),
        "Desviación Estándar":   std_curve[lo:hi].round(6),
        "¿Mínimo?": ["⭐ Mínimo" if i == idx_opt else ""
                     for i in range(lo, hi)],
    })
    st.dataframe(df_curve, use_container_width=True, hide_index=True)


# ── Figura 3: Datos transformados ────────────────────────────────
st.markdown(f"##### ✅ Datos transformados  (λ = {lambda_opt:.4f})")

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
fig3.suptitle(f"Datos Transformados  (λ = {lambda_opt:.4f})",
              fontsize=13, fontweight="bold", color=C_LINE, y=1.02)

# 3a. Histograma transformado
ax = axes3[0]
n3, bins3, _ = ax.hist(data_trans, bins=bins_n, color=C_TRANS,
                        edgecolor="white", alpha=0.85, linewidth=0.8)
xr3 = np.linspace(data_trans.min(), data_trans.max(), 200)
ax.plot(xr3,
        stats.norm.pdf(xr3, np.mean(data_trans), np.std(data_trans))
        * len(data_trans) * (bins3[1]-bins3[0]),
        "k--", linewidth=1.5, alpha=0.6, label="Normal teórica")
ax.set_title("Histograma + Curva Normal", fontsize=11, fontweight="bold")
ax.set_xlabel("Valor transformado")
ax.set_ylabel("Frecuencia")
ax.text(0.97, 0.97, norm_badge(sw_p_trans), transform=ax.transAxes,
        ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#e8f5e9" if sw_p_trans > 0.05 else "lightyellow",
                  alpha=0.9, edgecolor="#ccc"))
ax.legend(fontsize=8)

# 3b. Q-Q plot transformado
ax = axes3[1]
(osm3, osr3), (sl3, ic3, r3) = stats.probplot(data_trans, dist="norm")
ax.scatter(osm3, osr3, color=C_TRANS, s=35, alpha=0.8, zorder=3)
ax.plot(osm3, sl3*np.array(osm3) + ic3, "k--", linewidth=1.5, alpha=0.7)
ax.set_title("Gráfico Q-Q", fontsize=11, fontweight="bold")
ax.set_xlabel("Cuantiles teóricos (Normal)")
ax.set_ylabel("Cuantiles observados")
ax.text(0.05, 0.95, f"R² = {r3**2:.4f}", transform=ax.transAxes, fontsize=9,
        va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 3c. Box plots comparados (normalizados 0-1)
ax = axes3[2]
def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

bp2 = ax.boxplot([minmax(data), minmax(data_trans)],
                 patch_artist=True,
                 labels=["Original\n(normalizado)", f"Transformado\n(λ={lambda_opt:.3f})"],
                 medianprops=dict(color="white", linewidth=2.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5),
                 flierprops=dict(marker="o", markersize=5, alpha=0.7))
bp2["boxes"][0].set_facecolor(C_ORIG);  bp2["boxes"][0].set_alpha(0.75)
bp2["boxes"][1].set_facecolor(C_TRANS); bp2["boxes"][1].set_alpha(0.75)
ax.set_title("Comparación Box Plots", fontsize=11, fontweight="bold")
ax.set_ylabel("Valor normalizado [0–1]")

plt.tight_layout()
st.pyplot(fig3, use_container_width=True)
plt.close(fig3)


# ──────────────────────────────────────────────────────────────────
#  SECCIÓN 4 — TABLA ESTADÍSTICA COMPARATIVA
# ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-header">④ Tabla Estadística Comparativa</p>', unsafe_allow_html=True)

def estadisticos(arr, sw_p):
    return {
        "Media":                  f"{np.mean(arr):.6f}",
        "Mediana":                f"{np.median(arr):.6f}",
        "Desviación Estándar":    f"{np.std(arr, ddof=1):.6f}",
        "Varianza":               f"{np.var(arr, ddof=1):.6f}",
        "Mínimo":                 f"{arr.min():.6f}",
        "Máximo":                 f"{arr.max():.6f}",
        "Asimetría (Skewness)":   f"{stats.skew(arr):.6f}",
        "Curtosis (Fisher)":      f"{stats.kurtosis(arr):.6f}",
        "Shapiro-Wilk (estadístico)": f"{shapiro(arr)[0]:.6f}",
        "Shapiro-Wilk (p-valor)":     f"{sw_p:.6f}",
        "¿Distribución normal?":  "✅ Sí (p > 0.05)" if sw_p > 0.05 else "❌ No (p ≤ 0.05)",
    }

df_stats = pd.DataFrame({
    "Estadístico":        list(estadisticos(data, sw_p_orig).keys()),
    "Datos Originales":   list(estadisticos(data, sw_p_orig).values()),
    f"Transformados λ={lambda_opt:.4f}":  list(estadisticos(data_trans, sw_p_trans).values()),
    f"Transformados MLE λ={lambda_mle:.4f}": list(estadisticos(data_trans_mle, sw_p_mle).values()),
})

st.dataframe(df_stats, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────
#  SECCIÓN 5 — EXPORTAR RESULTADOS
# ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-header">⑤ Exportar Resultados</p>', unsafe_allow_html=True)

col_e1, col_e2 = st.columns(2)

# CSV con datos y transformación
with col_e1:
    df_export = pd.DataFrame({
        "Datos_Originales":              data,
        f"Transformados_lambda_{lambda_opt:.4f}":  data_trans,
        f"Transformados_MLE_lambda_{lambda_mle:.4f}": data_trans_mle,
    })
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar datos transformados (CSV)",
        data=csv_bytes,
        file_name="boxcox_datos.csv",
        mime="text/csv",
        use_container_width=True,
    )

# CSV con curva std vs lambda
with col_e2:
    df_curve_full = pd.DataFrame({
        "Lambda":              lambdas.round(6),
        "Desviacion_Estandar": std_curve.round(8),
    })
    csv_curve = df_curve_full.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar curva λ vs std dev (CSV)",
        data=csv_curve,
        file_name="boxcox_curva_lambda.csv",
        mime="text/csv",
        use_container_width=True,
    )
