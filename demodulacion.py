"""
demodulacion.py
Módulo de demodulación y recuperación de bits para sistemas de comunicaciones digitales.
Implementa receptores para BFSK, BPSK, QPSK y 16-QAM en canales ideales y con distorsión.
Totalmente compatible con los moduladores de modulaciones.py y fundamentos.py.
Incluye validación exhaustiva de entradas, manejo de excepciones y cálculo de métricas.
"""

import os
from typing import Union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Importar estructuras y funciones de fases anteriores
from modulaciones import (
    ModulacionResultado,
    obtener_constelacion_bpsk,
    obtener_constelacion_qpsk,
    obtener_constelacion_16qam
)

# Configuración de directorio de caché local para matplotlib
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = cache_dir


# =====================================================================
# FUNCIONES AUXILIARES DE VALIDACIÓN INTERNA
# =====================================================================

def _validar_arreglo_simbolos(simbolos: np.ndarray, nombre_mod: str) -> np.ndarray:
    """
    Valida que el arreglo de símbolos no sea vacío, sea unidimensional y no contenga NaN ni Inf.
    """
    arr = np.asarray(simbolos)
    if arr.size == 0:
        raise ValueError(f"El arreglo de símbolos para {nombre_mod} no puede estar vacío.")
    if arr.ndim != 1:
        raise ValueError(f"El arreglo de símbolos para {nombre_mod} debe ser unidimensional (ndim=1). Se recibió ndim={arr.ndim}.")
    if np.any(np.isnan(arr)):
        raise ValueError(f"El arreglo de símbolos para {nombre_mod} contiene valores no numéricos (NaN).")
    if np.any(np.isinf(arr)):
        raise ValueError(f"El arreglo de símbolos para {nombre_mod} contiene valores infinitos (Inf).")
    return arr


# =====================================================================
# DEMODULADORES DIGITALES
# =====================================================================

def demodular_bfsk(
    entrada: Union[ModulacionResultado, np.ndarray],
    f0: float | None = None,
    f1: float | None = None,
    sps: int | None = None,
    fs: float | None = None
) -> np.ndarray:
    """
    Demodula una señal BFSK utilizando un receptor de correlación en cuadratura (banco de filtros adaptados).
    Es invariante a desfases de portadora desconocidos o acumulados (CPFSK).

    Parámetros:
        entrada: Objeto ModulacionResultado o arreglo 1D con la señal real modulada en pasobanda (RF).
        f0 (float): Frecuencia del tono para el bit 0 en Hz.
        f1 (float): Frecuencia del tono para el bit 1 en Hz.
        sps (int): Muestras por símbolo.
        fs (float): Frecuencia de muestreo en Hz.

    Retorna:
        np.ndarray: Secuencia binaria recuperada (valores 0 y 1) de tipo np.int8.
    """
    if isinstance(entrada, ModulacionResultado):
        s_rf = entrada.senal_rf
        sps_val = entrada.sps
        fs_val = entrada.fs
        f0_val = f0 if f0 is not None else (entrada.fc - 1000.0)
        f1_val = f1 if f1 is not None else (entrada.fc + 1000.0)
    else:
        s_rf = np.asarray(entrada, dtype=float)
        if None in (f0, f1, sps, fs):
            raise ValueError("Cuando la entrada es un arreglo numpy, se deben especificar f0, f1, sps y fs.")
        f0_val = float(f0)
        f1_val = float(f1)
        sps_val = int(sps)
        fs_val = float(fs)

    # Validaciones exhaustivas de parámetros de BFSK
    if s_rf.size == 0:
        raise ValueError("La señal BFSK de entrada no puede estar vacía.")
    if s_rf.ndim != 1:
        raise ValueError(f"La señal BFSK debe ser un vector unidimensional (ndim=1). Se recibió ndim={s_rf.ndim}.")
    if np.any(np.isnan(s_rf)) or np.any(np.isinf(s_rf)):
        raise ValueError("La señal BFSK contiene valores no válidos (NaN o Infinito).")
    if sps_val <= 0:
        raise ValueError(f"sps debe ser estrictamente positivo. Se recibió {sps_val}.")
    if fs_val <= 0:
        raise ValueError(f"fs debe ser estrictamente positivo. Se recibió {fs_val}.")
    if f0_val <= 0 or f1_val <= 0:
        raise ValueError(f"Las frecuencias de tono deben ser positivas. Se recibieron f0={f0_val}, f1={f1_val}.")
    if np.isclose(f0_val, f1_val):
        raise ValueError(f"Las frecuencias f0 y f1 no pueden ser iguales ({f0_val} Hz).")
    if max(f0_val, f1_val) * 2 > fs_val:
        raise ValueError(f"Frecuencias ({f0_val}, {f1_val} Hz) violan el criterio de Nyquist para fs={fs_val} Hz.")

    num_muestras = len(s_rf)
    if num_muestras % sps_val != 0:
        raise ValueError(
            f"La longitud de la señal ({num_muestras}) no es divisible por sps ({sps_val})."
        )

    n_simbolos = num_muestras // sps_val
    bits_recuperados = np.zeros(n_simbolos, dtype=np.int8)

    # Base temporal para un intervalo de símbolo
    t_simbolo = np.arange(sps_val) / fs_val
    base_i0 = np.cos(2 * np.pi * f0_val * t_simbolo)
    base_q0 = np.sin(2 * np.pi * f0_val * t_simbolo)
    base_i1 = np.cos(2 * np.pi * f1_val * t_simbolo)
    base_q1 = np.sin(2 * np.pi * f1_val * t_simbolo)

    for k in range(n_simbolos):
        tramo = s_rf[k * sps_val : (k + 1) * sps_val]

        # Correlación en cuadratura para el tono f0
        i0 = np.sum(tramo * base_i0)
        q0 = np.sum(tramo * base_q0)
        energia_f0 = i0**2 + q0**2

        # Correlación en cuadratura para el tono f1
        i1 = np.sum(tramo * base_i1)
        q1 = np.sum(tramo * base_q1)
        energia_f1 = i1**2 + q1**2

        # Regla de decisión de máxima energía
        bits_recuperados[k] = 1 if energia_f1 > energia_f0 else 0

    return bits_recuperados


def demodular_bpsk(entrada: Union[ModulacionResultado, np.ndarray]) -> np.ndarray:
    """
    Demodula una señal BPSK con umbral óptimo en el eje En Fase (I).

    Mapeo de Fase 3:
        Bit 0 -> Símbolo -1.0
        Bit 1 -> Símbolo +1.0

    Regla de decisión:
        Re{s} >= 0.0 -> Bit 1
        Re{s} <  0.0 -> Bit 0

    Parámetros:
        entrada: Objeto ModulacionResultado o arreglo 1D con los símbolos complejos recibidos.

    Retorna:
        np.ndarray: Secuencia binaria recuperada de tipo np.int8.
    """
    if isinstance(entrada, ModulacionResultado):
        simbolos = entrada.simbolos
    else:
        simbolos = np.asarray(entrada, dtype=complex)

    simbolos = _validar_arreglo_simbolos(simbolos, "BPSK")

    # Detección por umbral sobre la componente real (I)
    bits_recuperados = np.where(simbolos.real >= 0.0, 1, 0).astype(np.int8)
    return bits_recuperados


def demodular_qpsk(
    entrada: Union[ModulacionResultado, np.ndarray],
    metodo: str = "umbral"
) -> np.ndarray:
    """
    Demodula símbolos QPSK respetando rigurosamente el código Gray de Fase 3:
        '00' -> (+1 + j) / sqrt(2)  [I > 0, Q > 0]
        '01' -> (-1 + j) / sqrt(2)  [I < 0, Q > 0]
        '11' -> (-1 - j) / sqrt(2)  [I < 0, Q < 0]
        '10' -> (+1 - j) / sqrt(2)  [I > 0, Q < 0]

    Parámetros:
        entrada: Objeto ModulacionResultado o arreglo 1D de símbolos complejos.
        metodo (str): 'umbral' (slicing directo) o 'distancia' (mínima distancia euclídea).

    Retorna:
        np.ndarray: Secuencia binaria recuperada con longitud 2 * len(simbolos).
    """
    if metodo not in ("umbral", "distancia"):
        raise ValueError(f"Método de demodulación '{metodo}' no reconocido. Debe ser 'umbral' o 'distancia'.")

    if isinstance(entrada, ModulacionResultado):
        simbolos = entrada.simbolos
    else:
        simbolos = np.asarray(entrada, dtype=complex)

    simbolos = _validar_arreglo_simbolos(simbolos, "QPSK")
    n_simbolos = len(simbolos)

    if metodo == "distancia":
        const_ref = obtener_constelacion_qpsk(normalizar=True)
        puntos = np.array(list(const_ref.values()), dtype=complex)
        etiquetas = list(const_ref.keys())

        bits_rec = []
        for s in simbolos:
            distancias = np.abs(s - puntos) ** 2
            idx_min = int(np.argmin(distancias))
            etiqueta = etiquetas[idx_min]
            bits_rec.extend([int(etiqueta[0]), int(etiqueta[1])])
        return np.array(bits_rec, dtype=np.int8)

    # Método de umbral óptimo (slicing directo en I y Q)
    # Bit 0 está determinado por el signo de Q
    b0 = np.where(simbolos.imag >= 0.0, 0, 1).astype(np.int8)
    # Bit 1 está determinado por el signo de I
    b1 = np.where(simbolos.real >= 0.0, 0, 1).astype(np.int8)

    # Entrelazar bits para reconstruir el flujo serie: [b0_0, b1_0, b0_1, b1_1, ...]
    bits_recuperados = np.empty(2 * n_simbolos, dtype=np.int8)
    bits_recuperados[0::2] = b0
    bits_recuperados[1::2] = b1

    return bits_recuperados


def demodular_16qam(
    entrada: Union[ModulacionResultado, np.ndarray],
    metodo: str = "umbral"
) -> np.ndarray:
    """
    Demodula símbolos 16-QAM respetando la grilla Gray rectangular de Fase 3:
    Niveles teóricos sin normalizar: {-3, -1, +1, +3} con factor K = 1 / sqrt(10).

    Límites de decisión normalizados:
        d1 = -2.0 / sqrt(10) ≈ -0.632456
        d2 =  0.0
        d3 = +2.0 / sqrt(10) ≈ +0.632456

    Parámetros:
        entrada: Objeto ModulacionResultado o arreglo 1D de símbolos complejos.
        metodo (str): 'umbral' o 'distancia'.

    Retorna:
        np.ndarray: Secuencia binaria recuperada con longitud 4 * len(simbolos).
    """
    if metodo not in ("umbral", "distancia"):
        raise ValueError(f"Método de demodulación '{metodo}' no reconocido. Debe ser 'umbral' o 'distancia'.")

    if isinstance(entrada, ModulacionResultado):
        simbolos = entrada.simbolos
    else:
        simbolos = np.asarray(entrada, dtype=complex)

    simbolos = _validar_arreglo_simbolos(simbolos, "16-QAM")
    n_simbolos = len(simbolos)

    if metodo == "distancia":
        const_ref = obtener_constelacion_16qam(normalizar=True)
        puntos = np.array(list(const_ref.values()), dtype=complex)
        etiquetas = list(const_ref.keys())

        bits_rec = []
        for s in simbolos:
            distancias = np.abs(s - puntos) ** 2
            idx_min = int(np.argmin(distancias))
            etiqueta = etiquetas[idx_min]
            bits_rec.extend([int(b) for b in etiqueta])
        return np.array(bits_rec, dtype=np.int8)

    # Método de umbrales analíticos (Slicing óptimo de Máxima Verosimilitud)
    norm = np.sqrt(10.0)
    umbral_inf = -2.0 / norm
    umbral_sup = 2.0 / norm

    def desmodular_eje_2bits(valores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bit_a = np.empty(len(valores), dtype=np.int8)
        bit_b = np.empty(len(valores), dtype=np.int8)

        # Bit A: es 1 si x >= 0, es 0 si x < 0
        bit_a[:] = np.where(valores >= 0.0, 1, 0)

        # Bit B: es 1 en la región central (-2/norm <= x < +2/norm), 0 en extremos
        region_central = (valores >= umbral_inf) & (valores < umbral_sup)
        bit_b[:] = np.where(region_central, 1, 0)

        return bit_a, bit_b

    # Desmodular 2 bits de la componente en fase (I)
    b0, b1 = desmodular_eje_2bits(simbolos.real)
    # Desmodular 2 bits de la componente en cuadratura (Q)
    b2, b3 = desmodular_eje_2bits(simbolos.imag)

    # Entrelazar los 4 bits por símbolo: [b0, b1, b2, b3]
    bits_recuperados = np.empty(4 * n_simbolos, dtype=np.int8)
    bits_recuperados[0::4] = b0
    bits_recuperados[1::4] = b1
    bits_recuperados[2::4] = b2
    bits_recuperados[3::4] = b3

    return bits_recuperados


# =====================================================================
# EVALUACIÓN DE TASA DE ERROR (BER / SER)
# =====================================================================

def calcular_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> tuple[float, int, int]:
    """
    Calcula la Tasa de Error de Bits (BER):
        BER = bits_erroneos / total_bits

    Parámetros:
        bits_tx (np.ndarray): Secuencia binaria original transmitida.
        bits_rx (np.ndarray): Secuencia binaria recuperada por el demodulador.

    Retorna:
        tuple[float, int, int]: (ber, numero_errores, total_bits).
    """
    tx = np.asarray(bits_tx)
    rx = np.asarray(bits_rx)

    if tx.size == 0 or rx.size == 0:
        raise ValueError("Los vectores de bits no pueden estar vacíos.")
    if tx.ndim != 1 or rx.ndim != 1:
        raise ValueError(f"Los vectores de bits deben ser unidimensionales (ndim=1). Se recibió TX={tx.ndim}, RX={rx.ndim}.")
    if len(tx) != len(rx):
        raise ValueError(
            f"La longitud de bits transmitidos ({len(tx)}) difiere de la recuperada ({len(rx)})."
        )
    if np.any(np.isnan(tx)) or np.any(np.isnan(rx)):
        raise ValueError("Los vectores de bits contienen valores NaN.")
    if not np.all(np.isin(tx, [0, 1])) or not np.all(np.isin(rx, [0, 1])):
        raise ValueError("Los vectores de bits deben contener únicamente valores binarios (0 o 1).")

    errores = int(np.sum(tx != rx))
    total = len(tx)
    ber = float(errores / total)
    return ber, errores, total


# =====================================================================
# FUNCIONES DE VISUALIZACIÓN Y REPORTES GRÁFICOS
# =====================================================================

def graficar_demodulacion_comparativa(
    bits_tx: np.ndarray,
    bits_rx: np.ndarray,
    tipo_modulacion: str,
    n_bits_mostrar: int = 16,
    ruta_guardado: str | None = None
) -> None:
    """
    Genera un gráfico comparativo de coincidencia bit a bit entre la señal transmitida y la recuperada.
    """
    n = min(n_bits_mostrar, len(bits_tx))
    tx_sub = bits_tx[:n]
    rx_sub = bits_rx[:n]
    indices = np.arange(n)
    errores_sub = tx_sub != rx_sub

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.step(indices, tx_sub + 0.05, where='mid', color='#1f77b4', linewidth=2.5, label='Bits Transmitidos (TX)')
    ax.step(indices, rx_sub - 0.05, where='mid', color='#2ca02c', linewidth=2.0, linestyle='--', label='Bits Recuperados (RX)')

    for i in range(n):
        color_borde = '#d62728' if errores_sub[i] else '#2ca02c'
        ax.text(i, tx_sub[i] + 0.15, f"{tx_sub[i]}", ha='center', va='bottom', fontweight='bold',
                color='#1f77b4', fontsize=9)
        ax.text(i, rx_sub[i] - 0.25, f"{rx_sub[i]}", ha='center', va='top', fontweight='bold',
                color=color_borde, fontsize=9)

    ber, err, tot = calcular_ber(bits_tx, bits_rx)
    estado_texto = f"BER = {ber:.4e} ({err} errores en {tot} bits)" if err > 0 else f"BER = 0.000 (0 errores en {tot} bits) - ¡Coincidencia 100%!"

    ax.set_ylim(-0.45, 1.45)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', '1'], fontweight='bold')
    ax.set_xticks(indices)
    ax.set_xlabel("Índice de Bit", fontsize=10, fontweight='bold')
    ax.set_ylabel("Valor Lógico", fontsize=10, fontweight='bold')
    ax.set_title(f"Demodulación {tipo_modulacion} - Comparación TX vs RX\n{estado_texto}", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    if ruta_guardado:
        os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
        plt.savefig(ruta_guardado, dpi=300)
        print(f"    [GRÁFICO GUARDADO] {ruta_guardado}")
    plt.close(fig)


def graficar_resumen_fase4(
    metricas: dict[str, dict],
    ruta_guardado: str | None = None
) -> None:
    """
    Genera un panel visual integrador con el resumen de la Fase 4 para las 4 modulaciones.
    """
    modulaciones = list(metricas.keys())
    total_bits = [metricas[m]["total_bits"] for m in modulaciones]
    bits_erroneos = [metricas[m]["errores"] for m in modulaciones]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colores = ['#1f77b4', '#2ca02c', '#9467bd', '#2ca25f']

    # Subplot 1: Total de bits probados sin error
    barras = ax1.bar(modulaciones, total_bits, color=colores, width=0.55, edgecolor='black', alpha=0.85)
    ax1.set_ylabel("Cantidad de Bits Evaluados", fontsize=10, fontweight='bold')
    ax1.set_title("Bits Evaluados en Canal Ideal (Sin Ruido)", fontsize=11, fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    for bar in barras:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{int(yval)} bits\n(100% OK)", ha='center', va='bottom',
                 fontsize=9, fontweight='bold')
    ax1.set_ylim(0, max(total_bits) * 1.25)

    # Subplot 2: BER Observada
    ax2.bar(modulaciones, [0]*len(modulaciones), color='green', edgecolor='black', width=0.55)
    ax2.set_ylabel("Tasa de Error de Bits (BER)", fontsize=10, fontweight='bold')
    ax2.set_title("BER en Canal Ideal (Cero Errores)", fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 0.1)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    for i, mod_nombre in enumerate(modulaciones):
        ax2.text(i, 0.02, f"BER = 0.0\nErrores: {bits_erroneos[i]}", ha='center', va='bottom',
                 fontweight='bold', color='darkgreen',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#d4edda", ec="#28a745"))

    plt.suptitle("FASE 4: Verificación Integral de Demodulación y Recuperación de Bits",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    if ruta_guardado:
        os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
        plt.savefig(ruta_guardado, dpi=300)
        print(f"    [GRÁFICO GUARDADO] {ruta_guardado}")
    plt.close(fig)
