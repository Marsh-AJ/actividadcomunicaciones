"""
efectos_canal.py
Módulo de Efectos de Canal de Propagación y Movilidad.

Contiene los siguientes modelos:
1. Ruido de fase (aproximación simplificada de ruido blanco independiente por muestra en banda base).
2. Efecto Doppler (fórmula cinemática, rotación compleja continua en banda base,
   reconstrucción de portadora pasobanda en RF y desplazamiento de tonos BFSK).

Desarrollado para la asignatura Electiva de Comunicaciones Móviles (Python 3.13.5).
"""

from __future__ import annotations
import numpy as np


# Velocidad de propagación de la luz en el vacío (m/s)
VELOCIDAD_LUZ_C = 299792458.0


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
# 1. RUIDO DE FASE (APROXIMACIÓN ALEATORIA POR MUESTRA EN BANDA BASE)
# =====================================================================

def aplicar_ruido_fase(
    senal_bb: np.ndarray,
    desviacion_grados: float,
    seed: int | None = None
) -> np.ndarray:
    """
    Aplica una fluctuación aleatoria de fase independiente por muestra sobre la
    envolvente compleja en banda base sobremuestreada:
        r_BB[n] = s_BB[n] · exp(j · θ[n])
    donde θ[n] ~ N(0, σ_θ²) con σ_θ = desviacion_grados · (π / 180) en radianes.

    Identificación y limitaciones del modelo:
        - Representa una aproximación simplificada de ruido de fase blanco muestral.
        - No constituye un modelo espectral continuo L(f) de un lazo enganchado en fase (PLL).
        - No es aplicable a BFSK (debe rechazarse por regla de despacho al carecer de constelación I/Q).

    Parámetros:
        senal_bb: Envolvente compleja en banda base (1D, complejo).
        desviacion_grados: Desviación estándar de fase en grados (>= 0).
        seed: Semilla pseudoaleatoria.

    Retorna:
        np.ndarray: Señal en banda base con dispersión angular de fase.
    """
    s = _validar_senal_unidimensional(senal_bb, "senal_bb")
    if float(desviacion_grados) < 0.0:
        raise ValueError(f"desviacion_grados debe ser mayor o igual a cero. Se recibió {desviacion_grados}.")

    if not np.iscomplexobj(s):
        s = s.astype(complex)

    if float(desviacion_grados) == 0.0:
        return np.copy(s)

    rng = np.random.default_rng(seed)
    sigma_rad = float(desviacion_grados) * (np.pi / 180.0)

    # Generar ángulo de fase aleatorio gaussiano por cada muestra
    theta = rng.normal(0.0, sigma_rad, size=len(s))
    rotacion_compleja = np.exp(1j * theta)

    return s * rotacion_compleja


# =====================================================================
# 2. EFECTO DOPPLER (MOVILIDAD Y DESPLAZAMIENTO ESPECTRAL)
# =====================================================================

def calcular_frecuencia_doppler(
    velocidad_kmh: float,
    fc: float,
    angulo_rad: float = 0.0
) -> float:
    """
    Calcula el desplazamiento de frecuencia Doppler máximo a partir de la cinemática:
        f_D = (v / c) · f_c · cos(θ_arr)

    Parámetros:
        velocidad_kmh: Velocidad relativa móvil en kilómetros por hora [km/h].
        fc: Frecuencia de portadora en Hertz [Hz] (> 0).
        angulo_rad: Ángulo de llegada entre la dirección de movimiento y la onda incidente en radianes.

    Retorna:
        float: Desplazamiento Doppler en Hertz [Hz].
    """
    if fc <= 0.0:
        raise ValueError(f"fc debe ser estrictamente positiva. Se recibió {fc}.")

    # Conversión de km/h a m/s: v [m/s] = v [km/h] / 3.6
    v_ms = float(velocidad_kmh) / 3.6
    c = VELOCIDAD_LUZ_C

    f_doppler = (v_ms / c) * float(fc) * np.cos(float(angulo_rad))
    return float(f_doppler)


def aplicar_doppler_banda_base(
    senal_bb: np.ndarray,
    fs: float,
    f_doppler: float
) -> np.ndarray:
    """
    Aplica el desplazamiento Doppler sobre una envolvente compleja en banda base
    mediante una rotación angular compleja continua en el tiempo:
        r_BB[n] = s_BB[n] · exp(j · 2π · f_D · t_n), con t_n = n / fs.

    Parámetros:
        senal_bb: Señal compleja en banda base (1D).
        fs: Frecuencia de muestreo en Hz (> 0).
        f_doppler: Frecuencia Doppler en Hz.

    Retorna:
        np.ndarray: Señal en banda base rotada por el desplazamiento Doppler.
    """
    s = _validar_senal_unidimensional(senal_bb, "senal_bb")
    if fs <= 0.0:
        raise ValueError(f"fs debe ser estrictamente positiva. Se recibió {fs}.")

    if not np.iscomplexobj(s):
        s = s.astype(complex)

    if float(f_doppler) == 0.0:
        return np.copy(s)

    t = np.arange(len(s)) / float(fs)
    rotacion_doppler = np.exp(1j * 2.0 * np.pi * float(f_doppler) * t)

    return s * rotacion_doppler


def aplicar_doppler_rf(
    i_bb: np.ndarray,
    q_bb: np.ndarray,
    fs: float,
    fc: float,
    f_doppler: float
) -> np.ndarray:
    """
    Aplica el efecto Doppler en radiofrecuencia (pasobanda real) mediante la
    reconstrucción rigurosa de las portadoras sobre la frecuencia desplazada:
        f_c' = f_c + f_D
        s_RF,doppler[n] = I[n] · cos(2π (fc + f_D) t_n) - Q[n] · sin(2π (fc + f_D) t_n)

    Evita la multiplicación errónea de una señal real por una exponencial compleja.

    Parámetros:
        i_bb: Componente En Fase (I) en banda base.
        q_bb: Componente en Cuadratura (Q) en banda base.
        fs: Frecuencia de muestreo en Hz.
        fc: Frecuencia de portadora original en Hz.
        f_doppler: Desplazamiento Doppler en Hz.

    Retorna:
        np.ndarray: Señal real pasobanda desplazada en frecuencia.
    """
    i_arr = _validar_senal_unidimensional(i_bb, "i_bb")
    q_arr = _validar_senal_unidimensional(q_bb, "q_bb")
    if len(i_arr) != len(q_arr):
        raise ValueError(f"Las ramas I y Q deben tener igual longitud. Longitudes: I={len(i_arr)}, Q={len(q_arr)}.")
    if fs <= 0.0 or fc <= 0.0:
        raise ValueError("fs y fc deben ser estrictamente positivas.")

    fc_desplazada = float(fc) + float(f_doppler)
    if fc_desplazada <= 0.0 or fc_desplazada >= (fs / 2.0):
        raise ValueError(
            f"La portadora desplazada fc'={fc_desplazada:.1f} Hz está fuera del rango "
            f"permitido (0, fs/2) = (0, {fs/2.0:.1f} Hz)."
        )

    t = np.arange(len(i_arr)) / float(fs)
    portadora_i = np.cos(2.0 * np.pi * fc_desplazada * t)
    portadora_q = np.sin(2.0 * np.pi * fc_desplazada * t)

    return i_arr * portadora_i - q_arr * portadora_q


def aplicar_doppler_bfsk(
    bits: np.ndarray,
    fs: float,
    fc: float,
    f0: float,
    f1: float,
    sps: int,
    f_doppler: float,
    amplitud: float = 1.0
) -> np.ndarray:
    """
    Aplica el efecto Doppler a una modulación BFSK desplazando físicamente los tonos:
        f_0' = f_0 + f_D
        f_1' = f_1 + f_D
    Genera la señal pasobanda real directamente con fase continua (CPFSK),
    conservando la integridad del demodulador pasobanda sin modelos I/Q artificiales.

    Parámetros:
        bits: Arreglo de bits (0 y 1).
        fs: Frecuencia de muestreo en Hz.
        fc: Portadora de referencia en Hz.
        f0: Tono para bit 0 en Hz.
        f1: Tono para bit 1 en Hz.
        sps: Muestras por símbolo.
        f_doppler: Desplazamiento Doppler en Hz.
        amplitud: Amplitud de la señal (1.0 para potencia 0.5 W).

    Retorna:
        np.ndarray: Señal real pasobanda BFSK con frecuencias desplazadas por Doppler.
    """
    b_arr = _validar_senal_unidimensional(bits, "bits")
    f0_d = float(f0) + float(f_doppler)
    f1_d = float(f1) + float(f_doppler)

    if min(f0_d, f1_d) <= 0.0 or max(f0_d, f1_d) >= (fs / 2.0):
        raise ValueError(
            f"Los tonos BFSK desplazados ({f0_d:.1f}, {f1_d:.1f} Hz) violan el criterio de Nyquist "
            f"o son no positivos para fs={fs} Hz."
        )

    n_simbolos = len(b_arr)
    num_muestras = n_simbolos * int(sps)
    s_rf_doppler = np.zeros(num_muestras, dtype=float)

    fase_actual = 0.0
    dt = 1.0 / float(fs)

    for k, bit in enumerate(b_arr):
        f_tono = f1_d if bit == 1 else f0_d
        idx_ini = k * sps
        idx_fin = (k + 1) * sps

        # Vector de tiempo local relativo al símbolo
        t_loc = np.arange(sps) * dt
        fase_tramo = fase_actual + 2.0 * np.pi * f_tono * t_loc
        s_rf_doppler[idx_ini:idx_fin] = float(amplitud) * np.cos(fase_tramo)

        # Actualizar fase inicial para el siguiente símbolo (CPFSK continua)
        fase_actual = (fase_tramo[-1] + 2.0 * np.pi * f_tono * dt) % (2.0 * np.pi)

    return s_rf_doppler

