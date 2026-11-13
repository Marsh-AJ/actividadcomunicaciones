"""
modulaciones.py
Módulo de modulaciones digitales: BFSK, BPSK, QPSK y 16-QAM.
Implementa mapeo Gray, normalización de constelaciones y síntesis de señales.
Reutiliza funciones de fundamentos.py.
"""

import os
from dataclasses import dataclass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Importar funciones de fundamentos de la Fase 2
import fundamentos as fnd

# Asegurar configuración de caché de matplotlib
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = cache_dir


@dataclass
class ModulacionResultado:
    """
    Estructura de datos con el resultado completo de una modulación digital.
    """
    tipo: str                          # 'BFSK', 'BPSK', 'QPSK', '16-QAM'
    bits: np.ndarray                   # Bits originales transmitidos
    simbolos: np.ndarray               # Símbolos complejos discretos en constelación
    senal_bb: np.ndarray               # Envolvente compleja en banda base sobremuestreada
    senal_rf: np.ndarray               # Señal real pasobanda (RF)
    t: np.ndarray                      # Vector de tiempo continuo muestreado
    fs: float                          # Frecuencia de muestreo [Hz]
    fc: float                          # Frecuencia de portadora [Hz]
    sps: int                           # Muestras por símbolo (Samples Per Symbol)
    bits_por_simbolo: int              # Cantidad de bits por símbolo (k = log2(M))
    constelacion_ref: dict[str, complex]  # Diccionario {bits_str: simbolo_complejo}


# =====================================================================
# DICCIONARIOS DE CONSTELACIONES TEÓRICAS CON MAPEO GRAY Y NORMALIZACIÓN
# =====================================================================

def obtener_constelacion_bpsk(normalizar: bool = True) -> dict[str, complex]:
    """
    Constelación BPSK (M=2, k=1 bit/símbolo):
    Bit '0' -> -1.0 + 0j
    Bit '1' -> +1.0 + 0j
    Energía promedio E_avg = 1.0.
    """
    return {
        "0": complex(-1.0, 0.0),
        "1": complex(1.0, 0.0)
    }


def obtener_constelacion_qpsk(normalizar: bool = True) -> dict[str, complex]:
    """
    Constelación QPSK con asignación Gray (M=4, k=2 bits/símbolo):
    '00' -> (+1 + j) / sqrt(2)  (Fase: 45°)
    '01' -> (-1 + j) / sqrt(2)  (Fase: 135°)
    '11' -> (-1 - j) / sqrt(2)  (Fase: 225°)
    '10' -> (+1 - j) / sqrt(2)  (Fase: 315°)
    Símbolos adyacentes difieren en solo 1 bit (Código Gray).
    Factor de normalización = 1 / sqrt(2) -> E_avg = 1.0.
    """
    norm = np.sqrt(2.0) if normalizar else 1.0
    return {
        "00": complex(1.0 / norm, 1.0 / norm),
        "01": complex(-1.0 / norm, 1.0 / norm),
        "11": complex(-1.0 / norm, -1.0 / norm),
        "10": complex(1.0 / norm, -1.0 / norm)
    }


def obtener_constelacion_16qam(normalizar: bool = True) -> dict[str, complex]:
    """
    Constelación 16-QAM con asignación Gray rectangular (M=16, k=4 bits/símbolo).
    Niveles en I y Q en {-3, -1, +1, +3}.
    Mapeo Gray por dimensión: '00'->-3, '01'->-1, '11'->+1, '10'->+3.
    Energía promedio sin normalizar: E_avg = 10.
    Factor de normalización = 1 / sqrt(10) -> E_avg = 1.0.
    """
    mapeo_2bits = {
        "00": -3.0,
        "01": -1.0,
        "11": 1.0,
        "10": 3.0
    }
    norm = np.sqrt(10.0) if normalizar else 1.0
    constelacion = {}
    for b_i, val_i in mapeo_2bits.items():
        for b_q, val_q in mapeo_2bits.items():
            cuarteto = b_i + b_q
            constelacion[cuarteto] = complex(val_i / norm, val_q / norm)
    return constelacion


def obtener_factor_normalizacion(tipo: str) -> float:
    """
    Retorna el factor de escala analítico K para normalizar la energía promedio
    de los símbolos a exactamente 1.0 (E_avg = E_s = 1.0):
    - BPSK: K = 1.0 (símbolos en {-1, +1})
    - QPSK: K = 1 / sqrt(2) ≈ 0.707107 (símbolos en {±1 ± j} / sqrt(2))
    - 16-QAM: K = 1 / sqrt(10) ≈ 0.316228 (símbolos en {±1, ±3} / sqrt(10))
    - BFSK: K = 1.0 (amplitud unitaria)
    """
    tipo_upper = tipo.upper().strip()
    if tipo_upper == "BPSK":
        return 1.0
    elif tipo_upper == "QPSK":
        return float(1.0 / np.sqrt(2.0))
    elif tipo_upper in ["16-QAM", "16QAM", "QAM16"]:
        return float(1.0 / np.sqrt(10.0))
    elif tipo_upper == "BFSK":
        return 1.0
    else:
        raise ValueError(f"Tipo de modulación '{tipo}' no reconocido para normalización.")


def calcular_energia_promedio(simbolos: np.ndarray | dict[str, complex] | list) -> float:
    """
    Calcula la energía promedio de símbolo E_s (E_avg) de una constelación teórica o de una secuencia transmitida:
    E_s = E_avg = (1 / N) * sum(|s_i|^2)

    Parámetros:
        simbolos: Diccionario {patron_bits: punto_complejo}, arreglo NumPy o lista de símbolos complejos.

    Retorna:
        float: Energía promedio calculada.
    """
    if isinstance(simbolos, dict):
        valores = np.array(list(simbolos.values()), dtype=complex)
    else:
        valores = np.asarray(simbolos, dtype=complex)
    if len(valores) == 0:
        raise ValueError("El conjunto de símbolos no puede estar vacío.")
    return float(np.mean(np.abs(valores) ** 2))


def calcular_energia_bit(simbolos: np.ndarray | dict[str, complex] | list, bits_por_simbolo: int) -> float:
    """
    Calcula la energía promedio por bit E_b a partir de los símbolos y su orden de modulación:
    E_b = E_s / k = E_avg / k, donde k = log2(M) es la cantidad de bits por símbolo.

    Parámetros:
        simbolos: Diccionario de constelación o arreglo de símbolos complejos.
        bits_por_simbolo (int): Bits por símbolo (k=1 para BPSK/BFSK, k=2 para QPSK, k=4 para 16-QAM).

    Retorna:
        float: Energía promedio por bit E_b.
    """
    if bits_por_simbolo <= 0:
        raise ValueError("bits_por_simbolo debe ser estrictamente positivo.")
    e_s = calcular_energia_promedio(simbolos)
    return e_s / float(bits_por_simbolo)


# =====================================================================
# MODULADORES DIGITALES
# =====================================================================

def modular_bfsk(
    bits: np.ndarray,
    f0: float = 2000.0,
    f1: float = 4000.0,
    sps: int = 100,
    fs: float = 100000.0,
    fase_continua: bool = True
) -> ModulacionResultado:
    """
    Modula una secuencia binaria utilizando BFSK (Binary Frequency Shift Keying).

    Parámetros:
        bits (np.ndarray): Secuencia de bits (0 y 1).
        f0 (float): Frecuencia asociada al bit 0 en Hz.
        f1 (float): Frecuencia asociada al bit 1 en Hz.
        sps (int): Muestras por símbolo/bit.
        fs (float): Frecuencia de muestreo en Hz.
        fase_continua (bool): Si es True, mantiene continuidad de fase (CPFSK).

    Retorna:
        ModulacionResultado: Objeto con las señales y parámetros de la modulación.
    """
    if not np.all(np.isin(bits, [0, 1])):
        raise ValueError("La secuencia debe contener únicamente bits con valores 0 o 1.")
    if sps <= 0 or fs <= 0:
        raise ValueError("sps y fs deben ser estrictamente positivos.")
    if max(f0, f1) * 2 > fs:
        raise ValueError(f"Frecuencias ({f0}, {f1} Hz) exceden el límite de Nyquist para fs={fs} Hz.")

    n_bits = len(bits)
    num_muestras = n_bits * sps
    t = np.arange(num_muestras) / fs

    # En BFSK cada bit es un símbolo binario
    # Generamos la envolvente compleja en banda base donde bit 0 -> e^(j 2pi (f0-fc) t)
    fc = (f0 + f1) / 2.0  # Portadora central de referencia
    dev_f = (f1 - f0) / 2.0  # Desviación de frecuencia

    # Frecuencia instantánea por muestra
    frecuencias_por_bit = np.where(bits == 1, f1, f0)
    frec_instantanea = np.repeat(frecuencias_por_bit, sps)

    if fase_continua:
        # CPFSK: Fase es la integral temporal de la frecuencia instantánea
        # theta[n] = 2 * pi * sum(f[k] * dt)
        fase_instantanea = 2.0 * np.pi * np.cumsum(frec_instantanea) / fs
        s_rf = np.cos(fase_instantanea)
        # Envolvente compleja en banda base respecto a fc
        fase_bb = 2.0 * np.pi * np.cumsum(frec_instantanea - fc) / fs
        s_bb = np.exp(1j * fase_bb)
    else:
        # FSK con fase independiente por símbolo
        fase_acumulada = 0.0
        s_rf = np.zeros(num_muestras)
        s_bb = np.zeros(num_muestras, dtype=complex)
        for i, b in enumerate(bits):
            f_actual = f1 if b == 1 else f0
            idx_ini = i * sps
            idx_fin = (i + 1) * sps
            t_local = np.arange(sps) / fs
            s_rf[idx_ini:idx_fin] = np.cos(2 * np.pi * f_actual * t_local)
            s_bb[idx_ini:idx_fin] = np.exp(1j * 2 * np.pi * (f_actual - fc) * t_local)

    # Símbolos discretos representados en el espacio de frecuencia (+1 para f1, -1 para f0)
    simbolos = np.where(bits == 1, complex(0.0, 1.0), complex(1.0, 0.0))

    const_ref = {
        "0": complex(1.0, 0.0),  # Representación ortogonal bit 0
        "1": complex(0.0, 1.0)   # Representación ortogonal bit 1
    }

    return ModulacionResultado(
        tipo="BFSK",
        bits=bits.copy(),
        simbolos=simbolos,
        senal_bb=s_bb,
        senal_rf=s_rf,
        t=t,
        fs=fs,
        fc=fc,
        sps=sps,
        bits_por_simbolo=1,
        constelacion_ref=const_ref
    )


def modular_bpsk(
    bits: np.ndarray,
    fc: float = 5000.0,
    sps: int = 100,
    fs: float = 100000.0,
    normalizar: bool = True
) -> ModulacionResultado:
    """
    Modula una secuencia binaria utilizando BPSK (Binary Phase Shift Keying).

    Parámetros:
        bits (np.ndarray): Secuencia binaria (0 y 1).
        fc (float): Frecuencia de portadora en Hz.
        sps (int): Muestras por símbolo.
        fs (float): Frecuencia de muestreo en Hz.
        normalizar (bool): Normaliza la energía de constelación a 1.0.

    Retorna:
        ModulacionResultado: Objeto con señales moduladas en banda base y RF.
    """
    if not np.all(np.isin(bits, [0, 1])):
        raise ValueError("La secuencia debe contener únicamente bits con valores 0 o 1.")
    if len(bits) == 0:
        raise ValueError("La secuencia de bits no puede estar vacía.")

    const_ref = obtener_constelacion_bpsk(normalizar=normalizar)

    # Mapeo de bits a símbolos complejos
    simbolos = np.array([const_ref[str(b)] for b in bits], dtype=complex)

    # Envolvente compleja en banda base (pulso rectangular sobremuestreado)
    s_bb = np.repeat(simbolos, sps)

    # Síntesis pasobanda mediante reutilización de fundamentos.modular_iq_pasobanda
    t, s_rf = fnd.modular_iq_pasobanda(s_bb.real, s_bb.imag, fc=fc, fs=fs)

    return ModulacionResultado(
        tipo="BPSK",
        bits=bits.copy(),
        simbolos=simbolos,
        senal_bb=s_bb,
        senal_rf=s_rf,
        t=t,
        fs=fs,
        fc=fc,
        sps=sps,
        bits_por_simbolo=1,
        constelacion_ref=const_ref
    )


def modular_qpsk(
    bits: np.ndarray,
    fc: float = 5000.0,
    sps: int = 100,
    fs: float = 100000.0,
    normalizar: bool = True
) -> ModulacionResultado:
    """
    Modula una secuencia binaria utilizando QPSK (Quadrature Phase Shift Keying) con mapeo Gray.

    Parámetros:
        bits (np.ndarray): Secuencia binaria (debe ser múltiplo de 2).
        fc (float): Frecuencia de portadora en Hz.
        sps (int): Muestras por símbolo.
        fs (float): Frecuencia de muestreo en Hz.
        normalizar (bool): Normaliza la constelación para energía promedio E_avg = 1.0.

    Retorna:
        ModulacionResultado: Objeto con señales moduladas.
    """
    if not np.all(np.isin(bits, [0, 1])):
        raise ValueError("La secuencia debe contener únicamente bits con valores 0 o 1.")
    if len(bits) % 2 != 0:
        raise ValueError(
            f"La longitud de bits ({len(bits)}) no es múltiplo de 2 (bits por símbolo para QPSK)."
        )

    const_ref = obtener_constelacion_qpsk(normalizar=normalizar)

    # Agrupamiento de bits en pares (dibits)
    dibits = bits.reshape(-1, 2)
    simbolos = np.array([
        const_ref[f"{b[0]}{b[1]}"] for b in dibits
    ], dtype=complex)

    # Banda base sobremuestreada
    s_bb = np.repeat(simbolos, sps)

    # Modulación RF pasobanda I/Q
    t, s_rf = fnd.modular_iq_pasobanda(s_bb.real, s_bb.imag, fc=fc, fs=fs)

    return ModulacionResultado(
        tipo="QPSK",
        bits=bits.copy(),
        simbolos=simbolos,
        senal_bb=s_bb,
        senal_rf=s_rf,
        t=t,
        fs=fs,
        fc=fc,
        sps=sps,
        bits_por_simbolo=2,
        constelacion_ref=const_ref
    )


def modular_16qam(
    bits: np.ndarray,
    fc: float = 5000.0,
    sps: int = 100,
    fs: float = 100000.0,
    normalizar: bool = True
) -> ModulacionResultado:
    """
    Modula una secuencia binaria utilizando 16-QAM con mapeo Gray bidimensional.

    Parámetros:
        bits (np.ndarray): Secuencia binaria (debe ser múltiplo de 4).
        fc (float): Frecuencia de portadora en Hz.
        sps (int): Muestras por símbolo.
        fs (float): Frecuencia de muestreo en Hz.
        normalizar (bool): Aplica factor 1/sqrt(10) para E_avg = 1.0.

    Retorna:
        ModulacionResultado: Objeto con señales moduladas.
    """
    if not np.all(np.isin(bits, [0, 1])):
        raise ValueError("La secuencia debe contener únicamente bits con valores 0 o 1.")
    if len(bits) % 4 != 0:
        raise ValueError(
            f"La longitud de bits ({len(bits)}) no es múltiplo de 4 (bits por símbolo para 16-QAM)."
        )

    const_ref = obtener_constelacion_16qam(normalizar=normalizar)

    # Agrupamiento en bloques de 4 bits (cuartetos)
    cuartetos = bits.reshape(-1, 4)
    simbolos = np.array([
        const_ref[f"{b[0]}{b[1]}{b[2]}{b[3]}"] for b in cuartetos
    ], dtype=complex)

    # Banda base sobremuestreada
    s_bb = np.repeat(simbolos, sps)

    # Modulación RF pasobanda I/Q
    t, s_rf = fnd.modular_iq_pasobanda(s_bb.real, s_bb.imag, fc=fc, fs=fs)

    return ModulacionResultado(
        tipo="16-QAM",
        bits=bits.copy(),
        simbolos=simbolos,
        senal_bb=s_bb,
        senal_rf=s_rf,
        t=t,
        fs=fs,
        fc=fc,
        sps=sps,
        bits_por_simbolo=4,
        constelacion_ref=const_ref
    )


# =====================================================================
# FUNCIONES DE VISUALIZACIÓN Y GENERACIÓN DE GRÁFICOS
# =====================================================================

def graficar_modulacion_bfsk(
    res: ModulacionResultado,
    n_bits_mostrar: int = 8,
    ruta_guardado: str | None = None
) -> None:
    """
    Genera gráfico temporal de BFSK mostrando bits, señal modulada y variación de frecuencia.
    """
    n_bits_mostrar = min(n_bits_mostrar, len(res.bits))
    muestras_mostrar = n_bits_mostrar * res.sps
    t_sub = res.t[:muestras_mostrar] * 1e3
    s_rf_sub = res.senal_rf[:muestras_mostrar]
    bits_sub = res.bits[:n_bits_mostrar]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    # 1. Secuencia de bits en formato NRZ
    tb_ms = (res.sps / res.fs) * 1e3
    nrz = np.repeat(bits_sub, res.sps)
    ax1.plot(t_sub, nrz, color='#1f77b4', linewidth=2.5, label="Flujo de bits original")
    ax1.set_ylabel("Nivel Lógico", fontsize=10, fontweight='bold')
    ax1.set_title(f"Modulación BFSK - Secuencia Binaria ({n_bits_mostrar} bits)", fontsize=11, fontweight='bold')
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, linestyle="--", alpha=0.6)

    for i, b in enumerate(bits_sub):
        t_centro = (i + 0.5) * tb_ms
        ax1.text(t_centro, 0.5, f"b={b}", ha='center', va='center', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f77b4"))
        if i > 0:
            ax1.axvline(i * tb_ms, color='gray', linestyle=':', alpha=0.7)
            ax2.axvline(i * tb_ms, color='gray', linestyle=':', alpha=0.7)

    # 2. Señal BFSK modulada
    ax2.plot(t_sub, s_rf_sub, color='#d62728', linewidth=1.8, label=r"Señal BFSK $s(t)$")
    ax2.set_xlabel("Tiempo [ms]", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Amplitud [V]", fontsize=10, fontweight='bold')
    ax2.set_title(f"Señal Modulada BFSK en el Dominio del Tiempo (f0={res.fc - 1000:.0f}Hz, f1={res.fc + 1000:.0f}Hz)",
                  fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    if ruta_guardado:
        os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
        plt.savefig(ruta_guardado, dpi=300)
        print(f"    [GRÁFICO GUARDADO] {ruta_guardado}")
    plt.close(fig)


def graficar_modulacion_bpsk(
    res: ModulacionResultado,
    n_bits_mostrar: int = 8,
    ruta_guardado: str | None = None
) -> None:
    """
    Genera gráfico temporal y diagrama de constelación de BPSK.
    """
    n_bits_mostrar = min(n_bits_mostrar, len(res.bits))
    muestras_mostrar = n_bits_mostrar * res.sps
    t_sub = res.t[:muestras_mostrar] * 1e3
    s_rf_sub = res.senal_rf[:muestras_mostrar]
    bits_sub = res.bits[:n_bits_mostrar]
    tb_ms = (res.sps / res.fs) * 1e3

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.2])

    # 1. Bits en NRZ
    ax1 = fig.add_subplot(gs[0, 0])
    nrz = np.repeat(bits_sub, res.sps)
    ax1.plot(t_sub, nrz, color='#1f77b4', linewidth=2)
    ax1.set_ylabel("Bit Lógico", fontsize=9, fontweight='bold')
    ax1.set_title(f"BPSK: Bits Transmitidos ({n_bits_mostrar} bits)", fontsize=10, fontweight='bold')
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, linestyle="--", alpha=0.5)

    for i, b in enumerate(bits_sub):
        ax1.text((i + 0.5) * tb_ms, 0.5, f"{b}", ha='center', va='center', fontweight='bold',
                 bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="#1f77b4"))
        if i > 0:
            ax1.axvline(i * tb_ms, color='gray', linestyle=':', alpha=0.6)

    # 2. Señal RF con cambios de fase de 180°
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(t_sub, s_rf_sub, color='#2ca02c', linewidth=1.8, label=r"$s_{BPSK}(t)$")
    ax2.set_xlabel("Tiempo [ms]", fontsize=9, fontweight='bold')
    ax2.set_ylabel("Amplitud [V]", fontsize=9, fontweight='bold')
    ax2.set_title("Señal BPSK Pasobanda (Inversiones de Fase de 180°)", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    for i in range(1, n_bits_mostrar):
        ax2.axvline(i * tb_ms, color='gray', linestyle=':', alpha=0.6)

    # 3. Constelación BPSK
    ax_const = fig.add_subplot(gs[:, 1])
    ax_const.axhline(0, color='black', linewidth=1)
    ax_const.axvline(0, color='black', linewidth=1)
    circulo = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle=':', alpha=0.7)
    ax_const.add_patch(circulo)

    for patron, pto in res.constelacion_ref.items():
        ax_const.scatter([pto.real], [pto.imag], color='#2ca02c', s=140, zorder=5)
        ax_const.text(pto.real, pto.imag + 0.25, f"'{patron}'\n({pto.real:+.1f}, {pto.imag:+.1f})",
                      ha='center', va='bottom', fontweight='bold', fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.4))

    ax_const.set_xlim(-1.8, 1.8)
    ax_const.set_ylim(-1.8, 1.8)
    ax_const.set_aspect('equal', adjustable='box')
    ax_const.set_xlabel("En Fase (I) [Real]", fontsize=9, fontweight='bold')
    ax_const.set_ylabel("Cuadratura (Q) [Imag]", fontsize=9, fontweight='bold')
    ax_const.set_title("Constelación BPSK (M=2)\n$E_{avg} = 1.0$", fontsize=10, fontweight='bold')
    ax_const.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if ruta_guardado:
        os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
        plt.savefig(ruta_guardado, dpi=300)
        print(f"    [GRÁFICO GUARDADO] {ruta_guardado}")
    plt.close(fig)


def graficar_modulacion_qpsk(
    res: ModulacionResultado,
    n_simbolos_mostrar: int = 8,
    ruta_guardado: str | None = None
) -> None:
    """
    Genera gráfico temporal y diagrama de constelación de QPSK con codificación Gray.
    """
    n_simbolos_mostrar = min(n_simbolos_mostrar, len(res.simbolos))
    muestras_mostrar = n_simbolos_mostrar * res.sps
    t_sub = res.t[:muestras_mostrar] * 1e3
    s_rf_sub = res.senal_rf[:muestras_mostrar]
    ts_ms = (res.sps / res.fs) * 1e3

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.2])

    # 1. Señal en banda base I y Q
    ax1 = fig.add_subplot(gs[0, 0])
    s_i_sub = res.senal_bb.real[:muestras_mostrar]
    s_q_sub = res.senal_bb.imag[:muestras_mostrar]
    ax1.plot(t_sub, s_i_sub, color='blue', label="Rama I (En Fase)")
    ax1.plot(t_sub, s_q_sub, color='red', linestyle='--', label="Rama Q (Cuadratura)")
    ax1.set_ylabel("Amplitud I/Q [V]", fontsize=9, fontweight='bold')
    ax1.set_title(f"QPSK: Componentes I y Q ({n_simbolos_mostrar} símbolos = {n_simbolos_mostrar*2} bits)",
                  fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)

    dibits = res.bits[:n_simbolos_mostrar * 2].reshape(-1, 2)
    for i, dibit in enumerate(dibits):
        t_centro = (i + 0.5) * ts_ms
        ax1.text(t_centro, 0.0, f"{dibit[0]}{dibit[1]}", ha='center', va='center', fontweight='bold',
                 bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="black"))
        if i > 0:
            ax1.axvline(i * ts_ms, color='gray', linestyle=':', alpha=0.6)

    # 2. Señal RF modulada
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(t_sub, s_rf_sub, color='purple', linewidth=1.8, label=r"$s_{QPSK}(t)$")
    ax2.set_xlabel("Tiempo [ms]", fontsize=9, fontweight='bold')
    ax2.set_ylabel("Amplitud RF [V]", fontsize=9, fontweight='bold')
    ax2.set_title("Señal RF Modulada QPSK", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    for i in range(1, n_simbolos_mostrar):
        ax2.axvline(i * ts_ms, color='gray', linestyle=':', alpha=0.6)

    # 3. Constelación QPSK
    ax_const = fig.add_subplot(gs[:, 1])
    ax_const.axhline(0, color='black', linewidth=1)
    ax_const.axvline(0, color='black', linewidth=1)
    circulo = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle=':', alpha=0.7)
    ax_const.add_patch(circulo)

    for patron, pto in res.constelacion_ref.items():
        ax_const.scatter([pto.real], [pto.imag], color='purple', s=140, zorder=5)
        ax_const.text(pto.real * 1.35, pto.imag * 1.35, f"'{patron}'\n$\\theta={np.rad2deg(np.angle(pto)):.0f}^\\circ$",
                      ha='center', va='center', fontweight='bold', fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.4))

    ax_const.set_xlim(-1.8, 1.8)
    ax_const.set_ylim(-1.8, 1.8)
    ax_const.set_aspect('equal', adjustable='box')
    ax_const.set_xlabel("En Fase (I) [Real]", fontsize=9, fontweight='bold')
    ax_const.set_ylabel("Cuadratura (Q) [Imag]", fontsize=9, fontweight='bold')
    ax_const.set_title("Constelación QPSK (M=4)\nMapeo Gray | $E_{avg} = 1.0$", fontsize=10, fontweight='bold')
    ax_const.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if ruta_guardado:
        os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
        plt.savefig(ruta_guardado, dpi=300)
        print(f"    [GRÁFICO GUARDADO] {ruta_guardado}")
    plt.close(fig)


def graficar_modulacion_16qam(
    res: ModulacionResultado,
    n_simbolos_mostrar: int = 8,
    ruta_guardado: str | None = None
) -> None:
    """
    Genera gráfico temporal y diagrama de constelación de 16-QAM.
    """
    n_simbolos_mostrar = min(n_simbolos_mostrar, len(res.simbolos))
    muestras_mostrar = n_simbolos_mostrar * res.sps
    t_sub = res.t[:muestras_mostrar] * 1e3
    s_rf_sub = res.senal_rf[:muestras_mostrar]
    ts_ms = (res.sps / res.fs) * 1e3

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.3])

    # 1. Señal en banda base I y Q multinivel
    ax1 = fig.add_subplot(gs[0, 0])
    s_i_sub = res.senal_bb.real[:muestras_mostrar]
    s_q_sub = res.senal_bb.imag[:muestras_mostrar]
    ax1.plot(t_sub, s_i_sub, color='blue', label="Rama I (4 niveles)")
    ax1.plot(t_sub, s_q_sub, color='orange', linestyle='--', label="Rama Q (4 niveles)")
    ax1.set_ylabel("Amplitud I/Q [V]", fontsize=9, fontweight='bold')
    ax1.set_title(f"16-QAM: Niveles Multinivel I y Q ({n_simbolos_mostrar} símbolos = {n_simbolos_mostrar*4} bits)",
                  fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)

    cuartetos = res.bits[:n_simbolos_mostrar * 4].reshape(-1, 4)
    for i, c in enumerate(cuartetos):
        t_centro = (i + 0.5) * ts_ms
        ax1.text(t_centro, 0.0, f"{c[0]}{c[1]}{c[2]}{c[3]}", ha='center', va='center', fontweight='bold',
                 fontsize=8, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="black"))
        if i > 0:
            ax1.axvline(i * ts_ms, color='gray', linestyle=':', alpha=0.6)

    # 2. Señal RF modulada
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(t_sub, s_rf_sub, color='darkgreen', linewidth=1.8, label=r"$s_{16QAM}(t)$")
    ax2.set_xlabel("Tiempo [ms]", fontsize=9, fontweight='bold')
    ax2.set_ylabel("Amplitud RF [V]", fontsize=9, fontweight='bold')
    ax2.set_title("Señal RF Modulada 16-QAM (Variación Simultánea de Fase y Amplitud)", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    for i in range(1, n_simbolos_mostrar):
        ax2.axvline(i * ts_ms, color='gray', linestyle=':', alpha=0.6)

    # 3. Constelación 16-QAM (grilla 4x4)
    ax_const = fig.add_subplot(gs[:, 1])
    ax_const.axhline(0, color='black', linewidth=1)
    ax_const.axvline(0, color='black', linewidth=1)

    circulo = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle=':', alpha=0.6)
    ax_const.add_patch(circulo)

    for patron, pto in res.constelacion_ref.items():
        ax_const.scatter([pto.real], [pto.imag], color='darkgreen', s=90, zorder=5)
        ax_const.text(pto.real, pto.imag + 0.12, f"{patron}",
                      ha='center', va='bottom', fontweight='bold', fontsize=7.5,
                      bbox=dict(boxstyle="round,pad=0.1", fc="yellow", alpha=0.4))

    ax_const.set_xlim(-1.6, 1.6)
    ax_const.set_ylim(-1.6, 1.6)
    ax_const.set_aspect('equal', adjustable='box')
    ax_const.set_xlabel("En Fase (I) [Real]", fontsize=9, fontweight='bold')
    ax_const.set_ylabel("Cuadratura (Q) [Imag]", fontsize=9, fontweight='bold')
    ax_const.set_title("Constelación 16-QAM (M=16)\nGrilla Gray | $E_{avg} = 1.0$", fontsize=10, fontweight='bold')
    ax_const.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if ruta_guardado:
        os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
        plt.savefig(ruta_guardado, dpi=300)
        print(f"    [GRÁFICO GUARDADO] {ruta_guardado}")
    plt.close(fig)


def graficar_comparacion_constelaciones(ruta_guardado: str | None = None) -> None:
    """
    Genera un panel comparativo de las constelaciones BPSK, QPSK y 16-QAM lado a lado,
    mostrando normalización de energía unitaria y asignación de bits Gray.
    """
    bpsk_ref = obtener_constelacion_bpsk(normalizar=True)
    qpsk_ref = obtener_constelacion_qpsk(normalizar=True)
    qam16_ref = obtener_constelacion_16qam(normalizar=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Subplot 1: BPSK
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].axvline(0, color='black', linewidth=1)
    c1 = plt.Circle((0, 0), 1.0, color='blue', fill=False, linestyle='--', alpha=0.6, label="Círculo $E=1.0$")
    axes[0].add_patch(c1)
    for bit, pto in bpsk_ref.items():
        axes[0].scatter([pto.real], [pto.imag], color='#1f77b4', s=130, zorder=5)
        axes[0].text(pto.real, pto.imag + 0.25, f"'{bit}'", ha='center', va='bottom',
                     fontweight='bold', fontsize=10, bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="#1f77b4"))
    axes[0].set_title(r"BPSK (M=2, 1 bit/símbolo)" + "\n" + r"2 fases: $\{0^\circ, 180^\circ\}$", fontsize=11, fontweight='bold')
    axes[0].set_xlim(-1.6, 1.6)
    axes[0].set_ylim(-1.6, 1.6)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlabel("I [Real]", fontweight='bold')
    axes[0].set_ylabel("Q [Imag]", fontweight='bold')
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower left", fontsize=8)

    # Subplot 2: QPSK
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].axvline(0, color='black', linewidth=1)
    c2 = plt.Circle((0, 0), 1.0, color='purple', fill=False, linestyle='--', alpha=0.6, label="Círculo $E=1.0$")
    axes[1].add_patch(c2)
    for dibit, pto in qpsk_ref.items():
        axes[1].scatter([pto.real], [pto.imag], color='purple', s=130, zorder=5)
        axes[1].text(pto.real * 1.35, pto.imag * 1.35, f"'{dibit}'", ha='center', va='center',
                     fontweight='bold', fontsize=9.5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="purple"))
    axes[1].set_title(r"QPSK (M=4, 2 bits/símbolo)" + "\n" + r"4 fases: $\{45^\circ, 135^\circ, 225^\circ, 315^\circ\}$",
                      fontsize=11, fontweight='bold')
    axes[1].set_xlim(-1.6, 1.6)
    axes[1].set_ylim(-1.6, 1.6)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlabel("I [Real]", fontweight='bold')
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower left", fontsize=8)

    # Subplot 3: 16-QAM
    axes[2].axhline(0, color='black', linewidth=1)
    axes[2].axvline(0, color='black', linewidth=1)
    c3 = plt.Circle((0, 0), 1.0, color='darkgreen', fill=False, linestyle='--', alpha=0.6, label="Círculo $E_{avg}=1.0$")
    axes[2].add_patch(c3)
    for cuarteto, pto in qam16_ref.items():
        axes[2].scatter([pto.real], [pto.imag], color='darkgreen', s=70, zorder=5)
        axes[2].text(pto.real, pto.imag + 0.11, f"{cuarteto}", ha='center', va='bottom',
                     fontweight='bold', fontsize=7.5)
    axes[2].set_title("16-QAM (M=16, 4 bits/símbolo)\n16 puntos en grilla rectangular Gray", fontsize=11, fontweight='bold')
    axes[2].set_xlim(-1.6, 1.6)
    axes[2].set_ylim(-1.6, 1.6)
    axes[2].set_aspect('equal', adjustable='box')
    axes[2].set_xlabel("I [Real]", fontweight='bold')
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="lower left", fontsize=8)

    plt.suptitle("Comparación de Constelaciones Normalizadas en el Espacio de Señales (Fase 3)",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    if ruta_guardado:
        os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
        plt.savefig(ruta_guardado, dpi=300)
        print(f"    [GRÁFICO GUARDADO] {ruta_guardado}")
    plt.close(fig)
