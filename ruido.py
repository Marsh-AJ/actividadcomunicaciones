"""
ruido.py
Módulo de Modelos Estocásticos de Ruido para el Simulador de Modulaciones Digitales.

Contiene los siguientes modelos de ruido:
1. AWGN (Ruido Blanco Gaussiano Aditivo en símbolos, banda base sobremuestreada y RF).
2. Ruido impulsivo Bernoulli-Gaussiano.
3. Ruido térmico basado en física de radio (N = kTB con figura de ruido).
4. Ruido complejo con distribución de Laplace (colas pesadas no gaussianas).

Desarrollado para la asignatura Electiva de Comunicaciones Móviles (Python 3.13.5).
"""

from __future__ import annotations
import numpy as np
from typing import Union


# Constante fundamental de Boltzmann en J/K (definición exacta SI)
CONSTANTE_BOLTZMANN_K = 1.380649e-23
# Temperatura estándar física de ruido de referencia IEEE (Kelvin)
TEMPERATURA_ESTANDAR_T0 = 290.0


def _validar_senal_unidimensional(senal: np.ndarray, nombre: str = "señal") -> np.ndarray:
    """Valida que el arreglo no esté vacío, sea 1D y no contenga NaN ni Inf."""
    arr = np.asarray(senal)
    if arr.size == 0:
        raise ValueError(f"El arreglo de {nombre} no puede estar vacío.")
    if arr.ndim != 1:
        raise ValueError(f"El arreglo de {nombre} debe ser unidimensional (ndim=1). Se recibió ndim={arr.ndim}.")
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError(f"El arreglo de {nombre} contiene valores inválidos (NaN o Inf).")
    return arr


# =====================================================================
# 1. AWGN (RUIDO BLANCO GAUSSIANO ADITIVO)
# =====================================================================

def aplicar_awgn_simbolos(
    simbolos: np.ndarray,
    snr_db: float,
    seed: int | None = None
) -> np.ndarray:
    """
    Aplica ruido AWGN complejo sobre símbolos discretos a tasa de símbolo (1 muestra/símbolo).

    Convención analítica:
        Es = 1.0 (energía de símbolo normalizada).
        SNR_lineal = 10^(snr_db / 10) = Es / N0.
        N0 = 1.0 / SNR_lineal.
        Varianza compleja total: σ_sym² = N0.
        Varianza por componente ortogonal: σ_I² = σ_Q² = N0 / 2.

    Parámetros:
        simbolos: Arreglo 1D de símbolos complejos.
        snr_db: Relación señal a ruido Es/N0 en decibelios [dB].
        seed: Semilla para el generador pseudoaleatorio.

    Retorna:
        np.ndarray: Símbolos complejos perturbados por AWGN.
    """
    s = _validar_senal_unidimensional(simbolos, "símbolos")
    rng = np.random.default_rng(seed)

    snr_lineal = 10.0 ** (float(snr_db) / 10.0)
    n0 = 1.0 / snr_lineal
    sigma_componente = np.sqrt(n0 / 2.0)

    ruido_i = rng.normal(0.0, sigma_componente, size=len(s))
    ruido_q = rng.normal(0.0, sigma_componente, size=len(s))
    ruido_complejo = ruido_i + 1j * ruido_q

    return s + ruido_complejo


def aplicar_awgn_banda_base(
    senal_bb: np.ndarray,
    snr_db: float,
    sps: int,
    seed: int | None = None
) -> np.ndarray:
    """
    Aplica ruido AWGN complejo sobre la envolvente en banda base sobremuestreada (fs, Nsps).

    Convención analítica:
        Para un filtro receptor promediador de ganancia continua G0 = 1.0 y energía Eh = 1/Nsps,
        la varianza por muestra de banda base debe ser:
            σ_BB² = Nsps · N0  ==>  σ_BB,I² = σ_BB,Q² = (Nsps · N0) / 2.
        De este modo, a la salida del filtro promediador:
            σ_salida,I² = σ_BB,I² · Eh = (Nsps · N0 / 2) · (1 / Nsps) = N0 / 2,
        coincidiendo exactamente con la dispersión del modelo de símbolos discretos.

    Parámetros:
        senal_bb: Envolvente compleja sobremuestreada (1D).
        snr_db: Es/N0 objetivo en decibelios [dB].
        sps: Muestras por símbolo (Nsps). Debe ser entero positivo.
        seed: Semilla para el generador pseudoaleatorio.

    Retorna:
        np.ndarray: Señal en banda base sobremuestreada con ruido AWGN.
    """
    s = _validar_senal_unidimensional(senal_bb, "senal_bb")
    if int(sps) <= 0:
        raise ValueError(f"sps debe ser un entero estrictamente positivo. Se recibió {sps}.")
    nsps = int(sps)

    rng = np.random.default_rng(seed)
    snr_lineal = 10.0 ** (float(snr_db) / 10.0)
    n0 = 1.0 / snr_lineal

    # Varianza por componente en banda base
    sigma_bb_comp = np.sqrt((nsps * n0) / 2.0)

    ruido_i = rng.normal(0.0, sigma_bb_comp, size=len(s))
    ruido_q = rng.normal(0.0, sigma_bb_comp, size=len(s))
    ruido_complejo = ruido_i + 1j * ruido_q

    return s + ruido_complejo


def aplicar_awgn_rf(
    senal_rf: np.ndarray,
    snr_rf_db: float,
    pot_senal: float = 0.5,
    seed: int | None = None
) -> np.ndarray:
    """
    Aplica ruido AWGN real sobre la señal pasobanda de radiofrecuencia (RF) según la
    convención de SNR medida directamente en el canal RF.

    Convención analítica:
        SNR_RF = P_s,RF / σ_RF² ==> σ_RF² = P_s,RF / 10^(SNR_RF_dB / 10).
        Para señales I/Q moduladas sobre portadora unitaria, P_s,RF = 0.50 W.

    Parámetros:
        senal_rf: Señal real pasobanda (1D).
        snr_rf_db: SNR medida directamente en RF en decibelios [dB].
        pot_senal: Potencia media de la señal en RF (por defecto 0.5 W).
        seed: Semilla para el generador pseudoaleatorio.

    Retorna:
        np.ndarray: Señal pasobanda perturbada con ruido gaussiano real.
    """
    s = _validar_senal_unidimensional(senal_rf, "senal_rf")
    if pot_senal <= 0.0:
        raise ValueError(f"pot_senal debe ser estrictamente positiva. Se recibió {pot_senal}.")

    rng = np.random.default_rng(seed)
    snr_lineal = 10.0 ** (float(snr_rf_db) / 10.0)
    sigma_rf = np.sqrt(pot_senal / snr_lineal)

    ruido_rf = rng.normal(0.0, sigma_rf, size=len(s))
    return s + ruido_rf


def aplicar_awgn_rf_equivalente_esn0(
    senal_rf: np.ndarray,
    es_n0_db: float,
    sps: int,
    seed: int | None = None
) -> np.ndarray:
    """
    Aplica ruido AWGN real en RF utilizando la equivalencia analítica estricta con
    la relación Es/N0 en símbolos y banda base sobremuestreada.

    Convención analítica obligatoria (diseño aprobado):
        σ_RF² = (Nsps · N0) / 4.
        Tras el receptor coherente (producto por 2·cos(2π fc t) y promediador discreto
        de Nsps muestras), la varianza a la salida de cada rama resulta:
            σ_I,salida² = (2 · σ_RF²) / Nsps = 2 · (Nsps · N0 / 4) / Nsps = N0 / 2.

    Parámetros:
        senal_rf: Señal real pasobanda (1D).
        es_n0_db: Es/N0 objetivo en decibelios [dB].
        sps: Muestras por símbolo (Nsps).
        seed: Semilla para el generador pseudoaleatorio.

    Retorna:
        np.ndarray: Señal RF perturbada bajo equivalencia Es/N0.
    """
    s = _validar_senal_unidimensional(senal_rf, "senal_rf")
    if int(sps) <= 0:
        raise ValueError(f"sps debe ser un entero estrictamente positivo. Se recibió {sps}.")
    nsps = int(sps)

    rng = np.random.default_rng(seed)
    snr_lineal = 10.0 ** (float(es_n0_db) / 10.0)
    n0 = 1.0 / snr_lineal

    # Varianza en RF para equivalencia perfecta tras demodulación coherente
    sigma_rf = np.sqrt((nsps * n0) / 4.0)

    ruido_rf = rng.normal(0.0, sigma_rf, size=len(s))
    return s + ruido_rf


# =====================================================================
# 2. RUIDO IMPULSIVO (BERNOULLI-GAUSS)
# =====================================================================

def aplicar_ruido_impulsivo(
    senal: np.ndarray,
    p_imp: float = 0.01,
    factor_amplitud: float = 5.0,
    es_compleja: bool | None = None,
    seed: int | None = None
) -> np.ndarray:
    """
    Aplica ruido impulsivo no gaussiano según el modelo de Bernoulli-Gauss:
        w_imp[n] = b[n] · g[n]
    donde b[n] ∈ {0, 1} con Pr(b=1) = p_imp, y g[n] ~ N(0, σ_imp²) con
    potencia σ_imp² = factor_amplitud² · P_s.

    Parámetros:
        senal: Señal de entrada (real o compleja, 1D).
        p_imp: Probabilidad de ocurrencia de impulso por muestra (0 < p_imp < 1).
        factor_amplitud: Multiplicador de amplitud del impulso respecto a la señal rms.
        es_compleja: Si es None, se infiere del tipo de la señal.
        seed: Semilla pseudoaleatoria.

    Retorna:
        np.ndarray: Señal perturbada con impulsos aditivos.
    """
    s = _validar_senal_unidimensional(senal, "señal")
    if not (0.0 < float(p_imp) < 1.0):
        raise ValueError(f"p_imp debe estar en el intervalo abierto (0, 1). Se recibió {p_imp}.")
    if float(factor_amplitud) <= 0.0:
        raise ValueError(f"factor_amplitud debe ser estrictamente positivo. Se recibió {factor_amplitud}.")

    compleja = np.iscomplexobj(s) if es_compleja is None else bool(es_compleja)
    rng = np.random.default_rng(seed)

    # Potencia media de la señal
    pot_s = np.mean(np.abs(s) ** 2)
    if pot_s == 0.0:
        pot_s = 1.0
    sigma_imp = float(factor_amplitud) * np.sqrt(pot_s)

    # Proceso de Bernoulli (detección de impulsos)
    ocurre_impulso = rng.random(len(s)) < float(p_imp)

    if compleja:
        g_i = rng.normal(0.0, sigma_imp / np.sqrt(2.0), size=len(s))
        g_q = rng.normal(0.0, sigma_imp / np.sqrt(2.0), size=len(s))
        impulsos = ocurre_impulso * (g_i + 1j * g_q)
    else:
        g = rng.normal(0.0, sigma_imp, size=len(s))
        impulsos = ocurre_impulso * g

    return s + impulsos


# =====================================================================
# 3. RUIDO TÉRMICO FÍSICO (N = kTB)
# =====================================================================

def calcular_ruido_termico_ktb(
    temperatura_k: float = TEMPERATURA_ESTANDAR_T0,
    ancho_banda_hz: float = 200e3,
    figura_ruido_db: float = 5.0
) -> dict[str, float]:
    """
    Calcula analíticamente los parámetros del ruido térmico físico a partir de la termodinámica:
        N_fisico = k · T · B · F  [Watts]

    A temperatura T = 290 K, la densidad espectral física es:
        N0 = k · T0 = 4.00388e-21 W/Hz ≈ -173.975 dBm/Hz (aproximada como -174 dBm/Hz).

    Parámetros:
        temperatura_k: Temperatura absoluta en Kelvin [K] (por defecto 290 K).
        ancho_banda_hz: Ancho de banda de recepción en Hertz [Hz].
        figura_ruido_db: Figura de ruido del receptor en decibelios [dB].

    Retorna:
        dict: Diccionario con métricas físicas (N0_W_Hz, N0_dBm_Hz, N_W, N_dBm, F_lineal).
    """
    if temperatura_k <= 0.0:
        raise ValueError(f"temperatura_k debe ser positiva. Se recibió {temperatura_k}.")
    if ancho_banda_hz <= 0.0:
        raise ValueError(f"ancho_banda_hz debe ser positivo. Se recibió {ancho_banda_hz}.")

    k = CONSTANTE_BOLTZMANN_K
    t = float(temperatura_k)
    b = float(ancho_banda_hz)
    f_lin = 10.0 ** (float(figura_ruido_db) / 10.0)

    # Densidad espectral de potencia física de ruido
    n0_w_hz = k * t
    n0_dbm_hz = 10.0 * np.log10(n0_w_hz / 1e-3)

    # Potencia total física de ruido recibida
    n_w = k * t * b * f_lin
    n_dbm = 10.0 * np.log10(n_w / 1e-3)

    return {
        "k_joule_k": k,
        "temperatura_k": t,
        "ancho_banda_hz": b,
        "figura_ruido_db": float(figura_ruido_db),
        "f_lineal": f_lin,
        "n0_w_hz": n0_w_hz,
        "n0_dbm_hz": n0_dbm_hz,
        "potencia_ruido_w": n_w,
        "potencia_ruido_dbm": n_dbm,
    }


def aplicar_ruido_termico(
    senal: np.ndarray,
    potencia_rx_dbm: float = -80.0,
    temperatura_k: float = TEMPERATURA_ESTANDAR_T0,
    ancho_banda_hz: float = 200e3,
    figura_ruido_db: float = 5.0,
    pot_ref_digital: float = 1.0,
    es_compleja: bool | None = None,
    seed: int | None = None
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Aplica ruido térmico físico mapeándolo rigurosamente al dominio digital normalizado.

    Fundamento:
        1. Se calcula la potencia física de ruido N_fisico = k · T · B · F [W].
        2. Se calcula la SNR lineal física a partir de la potencia recibida:
           P_rx [W] = 10^((potencia_rx_dbm - 30) / 10).
           SNR_lineal = P_rx / N_fisico.
        3. Para una señal digital con potencia media normalizada pot_ref_digital (1.0 W para BB,
           0.5 W para RF pasobanda):
           Varianza digital requerida: σ_digital² = pot_ref_digital / SNR_lineal.
           Para señales complejas: σ_I² = σ_Q² = σ_digital² / 2 = pot_ref_digital / (2 · SNR_lineal).
           Para señales reales: σ_real² = pot_ref_digital / SNR_lineal.

    Retorna:
        tuple[np.ndarray, dict]: (senal_perturbada, métricas_enlace).
    """
    s = _validar_senal_unidimensional(senal, "señal")
    compleja = np.iscomplexobj(s) if es_compleja is None else bool(es_compleja)
    rng = np.random.default_rng(seed)

    # 1. Parámetros físicos
    metricas = calcular_ruido_termico_ktb(temperatura_k, ancho_banda_hz, figura_ruido_db)
    n_fisico_w = metricas["potencia_ruido_w"]

    # 2. Potencia física de recepción y SNR lineal
    p_rx_w = 1e-3 * (10.0 ** (float(potencia_rx_dbm) / 10.0))
    snr_lineal = p_rx_w / n_fisico_w
    snr_db = 10.0 * np.log10(snr_lineal)

    # 3. Mapeo a varianza digital
    sigma_digital_sq = float(pot_ref_digital) / snr_lineal

    if compleja:
        sigma_comp = np.sqrt(sigma_digital_sq / 2.0)
        ruido = rng.normal(0.0, sigma_comp, size=len(s)) + 1j * rng.normal(0.0, sigma_comp, size=len(s))
    else:
        sigma_real = np.sqrt(sigma_digital_sq)
        ruido = rng.normal(0.0, sigma_real, size=len(s))

    metricas_completas = {
        **metricas,
        "potencia_rx_dbm": float(potencia_rx_dbm),
        "potencia_rx_w": p_rx_w,
        "snr_lineal": snr_lineal,
        "snr_db": snr_db,
        "pot_ref_digital": float(pot_ref_digital),
        "varianza_digital_total": sigma_digital_sq,
        "varianza_por_rama": (sigma_digital_sq / 2.0) if compleja else sigma_digital_sq,
    }

    return s + ruido, metricas_completas


# =====================================================================
# 4. RUIDO DE DISTRIBUCIÓN LAPLACE (COLAS PESADAS)
# =====================================================================

def aplicar_ruido_laplace(
    senal: np.ndarray,
    snr_db: float,
    pot_senal: float | None = None,
    es_compleja: bool | None = None,
    seed: int | None = None
) -> np.ndarray:
    """
    Aplica ruido no gaussiano con función de densidad de probabilidad de Laplace (colas pesadas):
        f(x) = (1 / (2b)) · exp(-|x| / b), con varianza σ² = 2 · b².

    Para mantener la misma potencia de ruido que un canal con SNR dada:
        σ_total² = P_s / SNR_lineal.
        - En caso real: b = sqrt(σ_total² / 2).
        - En caso complejo (I y Q independientes):
          σ_I² = σ_Q² = σ_total² / 2 = 2 · b²  ==>  b = σ_total / 2.

    Parámetros:
        senal: Señal 1D de entrada.
        snr_db: Relación señal a ruido en dB.
        pot_senal: Potencia de señal (si None, se calcula empíricamente).
        es_compleja: Si es None, se infiere del arreglo.
        seed: Semilla pseudoaleatoria.

    Retorna:
        np.ndarray: Señal perturbada con ruido de Laplace.
    """
    s = _validar_senal_unidimensional(senal, "señal")
    compleja = np.iscomplexobj(s) if es_compleja is None else bool(es_compleja)
    rng = np.random.default_rng(seed)

    if pot_senal is None:
        p_s = np.mean(np.abs(s) ** 2)
        if p_s == 0.0:
            p_s = 1.0
    else:
        p_s = float(pot_senal)
        if p_s <= 0.0:
            raise ValueError(f"pot_senal debe ser positiva. Se recibió {pot_senal}.")

    snr_lineal = 10.0 ** (float(snr_db) / 10.0)
    sigma_total_sq = p_s / snr_lineal

    if compleja:
        # Varianza por componente σ_I² = σ_total² / 2 = 2 · b² ==> b = sqrt(σ_total² / 4)
        escala_b = np.sqrt(sigma_total_sq / 4.0)
        ruido_i = rng.laplace(0.0, escala_b, size=len(s))
        ruido_q = rng.laplace(0.0, escala_b, size=len(s))
        ruido = ruido_i + 1j * ruido_q
    else:
        # Varianza real σ² = 2 · b² ==> b = sqrt(σ² / 2)
        escala_b = np.sqrt(sigma_total_sq / 2.0)
        ruido = rng.laplace(0.0, escala_b, size=len(s))

    return s + ruido

