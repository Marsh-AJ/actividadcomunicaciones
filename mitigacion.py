"""
mitigacion.py
Módulo de Técnicas de Mitigación y Mejora del Rendimiento para Comunicaciones Móviles.

Desarrollado para la asignatura Electiva de Comunicaciones Móviles (Python 3.13.5).
Fase 7: Filtrado Digital FIR, Ecualización de Canal Zero-Forcing y Diversidad de Recepción.

Contenido:
1. Filtrado Digital FIR:
   - Diseño de filtros FIR paso bajo y paso banda con ventana Hamming (scipy.signal).
   - Medición analítica de la atenuación fuera de banda (frecuencia de muestreo fs=16 kHz, corte fc=1 kHz).
   - Aplicación con compensación exacta de retardo de grupo temporal.
2. Ecualización de Canal Zero-Forcing (ZF):
   - Ecualizador FIR de longitud finita (Nw coeficientes) y retardo de decisión explícito Delta.
   - Cálculo de la respuesta combinada canal-ecualizador (c = h * w) y medición de ISI residual.
   - Cuantificación analítica de la amplificación de ruido (Noise Enhancement Factor, NEF = ||w||^2).
3. Diversidad de Recepción (SIMO 1x2):
   - Generación de canales y ruidos Rayleigh independientes e idénticamente distribuidos.
   - Combinación de Máxima Relación (MRC - Maximal Ratio Combining).
   - Combinación por Selección (SC - Selection Combining).
   - Comparación con recepción SISO de una sola rama bajo igual potencia transmitida.
"""

from __future__ import annotations
import math
from typing import Any
import numpy as np
from scipy import signal
from scipy.linalg import toeplitz


# =====================================================================
# 1. FILTRADO DIGITAL FIR
# =====================================================================

def disenar_filtro_fir_pasabajos(
    fs: float = 16000.0,
    f_corte: float = 1000.0,
    orden: int = 48
) -> np.ndarray:
    """
    Diseña un filtro FIR paso bajo de fase lineal con ventana Hamming:
    - Frecuencia de muestreo: fs (por defecto 16000 Hz).
    - Frecuencia de corte a -6 dB: f_corte (por defecto 1000 Hz = Rs).
    - Orden del filtro: orden (48 coeficientes, longitud 49).
    - Normalización: ganancia unitaria a frecuencia continua (DC = 0 Hz).

    Retorna:
        np.ndarray: Coeficientes del filtro FIR (1D, tipo float).
    """
    if fs <= 0 or f_corte <= 0:
        raise ValueError("fs y f_corte deben ser estrictamente positivos.")
    if f_corte >= fs / 2.0:
        raise ValueError("f_corte debe ser menor a la frecuencia de Nyquist (fs / 2).")
    if orden <= 0:
        raise ValueError("El orden del filtro debe ser positivo.")

    numtaps = int(orden) + 1
    coeficientes = signal.firwin(
        numtaps=numtaps,
        cutoff=float(f_corte),
        fs=float(fs),
        window="hamming",
        pass_zero="lowpass"
    )
    # Normalización para ganancia exacta 1.0 en DC (sum(h) = 1.0)
    coeficientes = coeficientes / np.sum(coeficientes)
    return coeficientes


def medir_atenuacion_filtro(
    coeficientes: np.ndarray,
    fs: float,
    f_interferencia: float
) -> dict[str, float]:
    """
    Calcula analíticamente la respuesta en frecuencia del filtro y la atenuación
    proporcionada a la frecuencia de interferencia especificada (ej. f_interf = 1500 Hz):
        Atenuacion_dB = 20 · log10( |H(0)| / |H(f_interf)| )

    Retorna:
        dict con ganancia_dc_db, ganancia_interf_db y atenuacion_relativa_db.
    """
    w, h = signal.freqz(coeficientes, 1.0, worN=[0.0, float(f_interferencia)], fs=float(fs))
    mag_dc = abs(h[0])
    mag_int = abs(h[1])

    g_dc_db = float(20.0 * np.log10(max(mag_dc, 1e-12)))
    g_int_db = float(20.0 * np.log10(max(mag_int, 1e-12)))
    atenuacion_db = float(g_dc_db - g_int_db)

    return {
        "ganancia_dc_db": g_dc_db,
        "ganancia_interf_db": g_int_db,
        "atenuacion_relativa_db": atenuacion_db,
        "f_interferencia_hz": float(f_interferencia),
        "fs_hz": float(fs)
    }


def aplicar_filtro_fir(
    senal: np.ndarray,
    coeficientes: np.ndarray,
    compensar_retardo: bool = True
) -> np.ndarray:
    """
    Filtra una señal unidimensional (real o compleja) mediante el filtro FIR.
    Si compensar_retardo es True, compensa el retardo de grupo constante tau = orden // 2
    para mantener el alineamiento temporal estricto con los instantes de decisión.

    Parámetros:
        senal: Arreglo 1D con la señal a filtrar.
        coeficientes: Coeficientes del filtro FIR.
        compensar_retardo: Si es True, recorta el transitorio inicial y alinea la señal.

    Retorna:
        np.ndarray: Señal filtrada con la misma longitud que la entrada.
    """
    s = np.asarray(senal)
    if s.ndim != 1 or len(s) == 0:
        raise ValueError("La señal debe ser un arreglo 1D no vacío.")

    n_taps = len(coeficientes)
    retardo = (n_taps - 1) // 2

    # Filtrado causal mediante ecuación en diferencias
    s_filtrada = signal.lfilter(coeficientes, 1.0, s)

    if compensar_retardo and retardo > 0:
        # Desplazar hacia atrás en 'retardo' muestras para alinear fase
        # Completar el final con el último valor para preservar longitud
        s_alineada = np.empty_like(s_filtrada)
        if len(s_filtrada) > retardo:
            s_alineada[: -retardo] = s_filtrada[retardo:]
            s_alineada[-retardo:] = s_filtrada[-1]
        else:
            s_alineada = s_filtrada
        return s_alineada

    return s_filtrada


# =====================================================================
# 2. ECUALIZACIÓN ZERO-FORCING (ZF) DE LONGITUD FINITA
# =====================================================================

def disenar_ecualizador_zf(
    h_canal: np.ndarray,
    longitud_w: int = 7,
    retardo_decision: int = 0
) -> dict[str, Any]:
    """
    Diseña un ecualizador FIR Zero-Forcing (ZF) de longitud finita Nw para un canal
    dispersivo discreto h de longitud Lh:
        c[n] = (h * w)[n] ≈ delta[n - Delta]

    El vector de coeficientes w_ZF se obtiene resolviendo el sistema de convolución
    por mínimos cuadrados:
        H · w = e_Delta  ==>  w = (H^H · H)^(-1) · H^H · e_Delta

    Parámetros:
        h_canal: Respuesta impulsional del canal discreto a tasa de símbolo (1D).
        longitud_w: Número de coeficientes del ecualizador FIR (por defecto 7).
        retardo_decision: Retardo deseado Delta en muestras (por defecto 0 para canales de fase mínima).

    Retorna:
        dict con coeficientes w, respuesta combinada c, retardo Delta,
        amplificación de ruido (NEF) y métrica de ISI residual.
    """
    h = np.asarray(h_canal, dtype=complex).ravel()
    lh = len(h)
    nw = int(longitud_w)

    if lh == 0 or nw <= 0:
        raise ValueError("h_canal y longitud_w deben ser positivos y no vacíos.")

    lc = lh + nw - 1
    delta = int(retardo_decision)
    if delta < 0 or delta >= lc:
        raise ValueError(f"retardo_decision ({delta}) fuera de rango [0, {lc - 1}].")

    # Matriz de convolución de Toeplitz H de dimensión (Lc x Nw)
    columna_h = np.zeros(lc, dtype=complex)
    columna_h[:lh] = h
    fila_h = np.zeros(nw, dtype=complex)
    fila_h[0] = h[0]
    matriz_h = toeplitz(columna_h, fila_h)

    # Vector objetivo e_Delta
    e_delta = np.zeros(lc, dtype=complex)
    e_delta[delta] = 1.0

    # Solución de Mínimos Cuadrados Zero-Forcing
    w_zf, _, _, _ = np.linalg.lstsq(matriz_h, e_delta, rcond=None)

    # Respuesta impulsional combinada canal-ecualizador: c = h * w
    c_combinada = matriz_h @ w_zf

    # Factor de amplificación de ruido: NEF = ||w||^2 = sum(|w_n|^2)
    nef_lineal = float(np.sum(np.abs(w_zf) ** 2))
    nef_db = float(10.0 * np.log10(max(nef_lineal, 1e-12)))

    # Métrica de ISI residual: energía de los taps secundarios frente al tap deseado
    pot_deseada = float(np.abs(c_combinada[delta]) ** 2)
    pot_isi = float(np.sum(np.abs(c_combinada) ** 2) - pot_deseada)
    isi_residual_db = float(10.0 * np.log10(max(pot_isi, 1e-12) / max(pot_deseada, 1e-12)))

    return {
        "coeficientes_w": w_zf,
        "respuesta_combinada_c": c_combinada,
        "retardo_decision": delta,
        "nef_lineal": nef_lineal,
        "nef_db": nef_db,
        "isi_residual_db": isi_residual_db,
        "tap_principal": float(np.abs(c_combinada[delta])),
        "longitud_w": nw
    }


def aplicar_canal_dispersivo_isi(
    simbolos_tx: np.ndarray,
    h_canal: np.ndarray
) -> np.ndarray:
    """
    Aplica la distorsión de un canal multitrayecto con dispersión temporal e ISI
    sobre una secuencia de símbolos discretos:
        r[k] = sum_m h[m] · s[k - m]
    """
    s = np.asarray(simbolos_tx, dtype=complex).ravel()
    h = np.asarray(h_canal, dtype=complex).ravel()
    # Convolución lineal preservando la longitud original
    r = signal.lfilter(h, 1.0, s)
    return r


def aplicar_ecualizador_zf(
    simbolos_rx: np.ndarray,
    coeficientes_w: np.ndarray,
    retardo_decision: int = 0
) -> np.ndarray:
    """
    Filtra los símbolos recibidos con el ecualizador FIR ZF y compensa el retardo
    de decisión Delta:
        s_eq[k] = (w * r)[k + Delta]
    """
    r = np.asarray(simbolos_rx, dtype=complex).ravel()
    w = np.asarray(coeficientes_w, dtype=complex).ravel()
    delta = int(retardo_decision)

    salida_filtrada = signal.lfilter(w, 1.0, r)

    if delta > 0 and len(salida_filtrada) > delta:
        s_eq = np.empty_like(salida_filtrada)
        s_eq[: -delta] = salida_filtrada[delta:]
        s_eq[-delta:] = salida_filtrada[-1]
        return s_eq

    return salida_filtrada


# =====================================================================
# 3. DIVERSIDAD DE RECEPCIÓN (SIMO 1x2: MRC Y SC)
# =====================================================================

def generar_canales_rayleigh_independientes(
    n_simbolos: int,
    n_ramas: int = 2,
    seed: int | None = None
) -> np.ndarray:
    """
    Genera ganancias de canal complejas Rayleigh independientes para cada rama de recepción:
        h_i = (X + j Y) / sqrt(2),  con X, Y ~ N(0, 1)  ==> E[|h_i|^2] = 1.0

    Retorna:
        np.ndarray de dimensión (n_ramas, n_simbolos) tipo complex128.
    """
    rng = np.random.default_rng(seed)
    h_ramas = np.empty((n_ramas, n_simbolos), dtype=complex)
    for i in range(n_ramas):
        i_comp = rng.normal(0.0, 1.0 / np.sqrt(2.0), size=n_simbolos)
        q_comp = rng.normal(0.0, 1.0 / np.sqrt(2.0), size=n_simbolos)
        h_ramas[i] = i_comp + 1j * q_comp
    return h_ramas


def transmitir_simo_rayleigh(
    simbolos_tx: np.ndarray,
    canales_h: np.ndarray,
    snr_lineal: float,
    seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simula la recepción en un enlace SIMO de L ramas bajo potencia de transmisión idéntica (Es = 1.0):
        r_i[k] = h_i[k] · s[k] + w_i[k],  para i = 1, ..., L.
    donde w_i son ruidos blancos gaussianos complejos independientes entre ramas con:
        sigma_w^2 = N0 = 1.0 / snr_lineal  ==>  sigma_I^2 = sigma_Q^2 = N0 / 2.

    Retorna:
        tuple[np.ndarray, np.ndarray]: (r_ramas, w_ramas) de forma (L, n_simbolos).
    """
    s = np.asarray(simbolos_tx, dtype=complex).ravel()
    h = np.asarray(canales_h, dtype=complex)
    n_ramas, n_sim = h.shape

    if len(s) != n_sim:
        raise ValueError("La longitud de simbolos_tx no coincide con el número de muestras de canal.")

    rng = np.random.default_rng(seed)
    n0 = 1.0 / float(snr_lineal)
    sigma_comp = np.sqrt(n0 / 2.0)

    r_ramas = np.empty_like(h)
    w_ramas = np.empty_like(h)

    for i in range(n_ramas):
        ruido = rng.normal(0.0, sigma_comp, size=n_sim) + 1j * rng.normal(0.0, sigma_comp, size=n_sim)
        w_ramas[i] = ruido
        r_ramas[i] = h[i] * s + ruido

    return r_ramas, w_ramas


def combinar_diversidad_mrc(
    r_ramas: np.ndarray,
    h_ramas: np.ndarray
) -> np.ndarray:
    """
    Combina coherentemente las L ramas de recepción mediante Maximal Ratio Combining (MRC):
        y_MRC[k] = sum_i ( h_i*[k] · r_i[k] ) / sum_i ( |h_i[k]|^2 )
                 = s[k] + sum_i ( h_i*[k] · w_i[k] ) / sum_i ( |h_i[k]|^2 )

    Esta combinación maximiza la SNR instantánea resultante:
        gamma_MRC[k] = sum_i gamma_i[k].

    Retorna:
        np.ndarray: Símbolos combinados 1D listos para demodulación.
    """
    r = np.asarray(r_ramas, dtype=complex)
    h = np.asarray(h_ramas, dtype=complex)

    denominador = np.sum(np.abs(h) ** 2, axis=0)
    # Evitar divisiones por cero en desvanecimientos extremos
    denominador = np.where(denominador == 0.0, 1e-12, denominador)

    numerador = np.sum(np.conj(h) * r, axis=0)
    y_mrc = numerador / denominador
    return y_mrc


def combinar_diversidad_sc(
    r_ramas: np.ndarray,
    h_ramas: np.ndarray
) -> np.ndarray:
    """
    Combina las L ramas mediante Selection Combining (SC):
    Para cada símbolo k, selecciona la rama con mayor ganancia de canal instantánea |h_i[k]|:
        i*(k) = argmax_i |h_i[k]|
        y_SC[k] = r_{i*}[k] / h_{i*}[k] = s[k] + w_{i*}[k] / h_{i*}[k]

    Retorna:
        np.ndarray: Símbolos combinados 1D listos para demodulación.
    """
    r = np.asarray(r_ramas, dtype=complex)
    h = np.asarray(h_ramas, dtype=complex)

    magnitudes = np.abs(h)
    mejor_rama = np.argmax(magnitudes, axis=0)  # Índices de 0 a L-1 por cada símbolo
    n_sim = h.shape[1]

    y_sc = np.empty(n_sim, dtype=complex)
    for k in range(n_sim):
        idx = mejor_rama[k]
        h_val = h[idx, k]
        if abs(h_val) == 0.0:
            h_val = 1e-12
        y_sc[k] = r[idx, k] / h_val

    return y_sc


def demodular_siso_rayleigh(
    r_rama1: np.ndarray,
    h_rama1: np.ndarray
) -> np.ndarray:
    """
    Demodula la rama única SISO (L=1) ecualizando coherentemente la ganancia de canal:
        y_SISO[k] = r_1[k] / h_1[k]
    """
    r = np.asarray(r_rama1, dtype=complex).ravel()
    h = np.asarray(h_rama1, dtype=complex).ravel()

    h_seguro = np.where(np.abs(h) == 0.0, 1e-12, h)
    return r / h_seguro
