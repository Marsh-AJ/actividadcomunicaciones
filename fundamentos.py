"""
fundamentos.py
Módulo de fundamentos de simulación de comunicaciones digitales.
Implementa generación de secuencias binarias, formas de onda NRZ,
señales sinusoidales parametrizadas, muestreo y representación I/Q.
"""

import os
import sys

# Configurar directorio de caché local ANTES de importar matplotlib para evitar advertencias de permisos
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = cache_dir

import numpy as np
import matplotlib
# Configurar backend no interactivo para asegurar guardado de gráficos en servidor/headless
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generar_bits(n_bits: int, semilla: int | None = None) -> np.ndarray:
    """
    Genera una secuencia pseudoaleatoria de bits binarios (0 y 1).

    Parámetros:
        n_bits (int): Número de bits a generar. Debe ser entero positivo.
        semilla (int | None): Semilla para el generador pseudoaleatorio (reproducibilidad).

    Retorna:
        np.ndarray: Vector 1D con valores binarios 0 o 1 de tipo np.int8.
    """
    if n_bits <= 0:
        raise ValueError("El número de bits debe ser un entero estrictamente positivo.")
    rng = np.random.default_rng(semilla)
    return rng.integers(0, 2, size=n_bits, dtype=np.int8)


def bits_a_nrz(bits: np.ndarray, muestras_por_bit: int, polar: bool = False) -> np.ndarray:
    """
    Convierte una secuencia de bits discretos en una señal en banda base NRZ
    (Non-Return-to-Zero) sobremuestreada.

    Parámetros:
        bits (np.ndarray): Secuencia binaria de entrada (valores 0 y 1).
        muestras_por_bit (int): Número de muestras discretas por cada intervalo de bit (N_sps).
        polar (bool): Si es False, genera NRZ unipolar (0 -> 0V, 1 -> +1V).
                      Si es True, genera NRZ polar/bipolar (0 -> -1V, 1 -> +1V).

    Retorna:
        np.ndarray: Vector 1D de la señal escalonada en el tiempo con longitud len(bits) * muestras_por_bit.
    """
    if muestras_por_bit <= 0:
        raise ValueError("Las muestras por bit deben ser mayores a cero.")
    
    if polar:
        # Mapeo polar: bit 1 -> +1, bit 0 -> -1
        niveles = np.where(bits == 1, 1.0, -1.0)
    else:
        # Mapeo unipolar: bit 1 -> 1.0, bit 0 -> 0.0
        niveles = bits.astype(float)
        
    return np.repeat(niveles, muestras_por_bit)


def generar_senoidal(
    frecuencia: float,
    amplitud: float = 1.0,
    fase_rad: float = 0.0,
    duracion: float = 1e-3,
    fs: float = 100e3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera una señal sinusoidal en tiempo discreto: s[n] = A * cos(2*pi*f*t[n] + phi).

    Parámetros:
        frecuencia (float): Frecuencia de la señal en Hertz (Hz).
        amplitud (float): Amplitud pico de la señal (V).
        fase_rad (float): Fase inicial en radianes.
        duracion (float): Duración total en segundos.
        fs (float): Frecuencia de muestreo en muestras/segundo (Hz).

    Retorna:
        tuple[np.ndarray, np.ndarray]: (vector_tiempo t, vector_senal s).
    """
    if fs <= 0 or duracion <= 0:
        raise ValueError("Frecuencia de muestreo y duración deben ser positivas.")
    if fs < 2 * frecuencia:
        print(f"[ADVERTENCIA NYQUIST] fs ({fs} Hz) es menor que 2*f ({2*frecuencia} Hz). Ocurrirá aliasing.")

    num_muestras = int(np.round(duracion * fs))
    t = np.arange(num_muestras) / fs
    senal = amplitud * np.cos(2 * np.pi * frecuencia * t + fase_rad)
    return t, senal


def modular_iq_pasobanda(
    I_bb: np.ndarray,
    Q_bb: np.ndarray,
    fc: float,
    fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Modula componentes en banda base (I y Q) a una portadora pasobanda en frecuencia fc:
    s_RF(t) = I(t) * cos(2*pi*fc*t) - Q(t) * sin(2*pi*fc*t).

    Parámetros:
        I_bb (np.ndarray): Señal en fase (In-Phase) en banda base.
        Q_bb (np.ndarray): Señal en cuadratura (Quadrature) en banda base.
        fc (float): Frecuencia de portadora de RF en Hz.
        fs (float): Frecuencia de muestreo en Hz.

    Retorna:
        tuple[np.ndarray, np.ndarray]: (vector_tiempo t, senal_RF).
    """
    if len(I_bb) != len(Q_bb):
        raise ValueError("Las componentes I y Q deben tener exactamente la misma longitud.")
    
    num_muestras = len(I_bb)
    t = np.arange(num_muestras) / fs
    portadora_i = np.cos(2 * np.pi * fc * t)
    portadora_q = np.sin(2 * np.pi * fc * t)

    s_rf = (I_bb * portadora_i) - (Q_bb * portadora_q)
    return t, s_rf


def calcular_potencia(senal: np.ndarray) -> float:
    """
    Calcula la potencia media de una señal discreta:
    P = (1/N) * sum(|s[n]|^2)
    """
    return float(np.mean(np.abs(senal) ** 2))


# =====================================================================
# FUNCIONES DE VISUALIZACIÓN Y GENERACIÓN DE GRÁFICOS
# =====================================================================

def graficar_secuencia_bits(
    bits: np.ndarray,
    muestras_por_bit: int,
    fs: float,
    ruta_guardado: str | None = None
) -> None:
    """
    Genera un gráfico comparativo del flujo binario en formato de pulsos NRZ unipolar y polar.
    """
    tb = muestras_por_bit / fs
    duracion = len(bits) * tb
    nrz_uni = bits_a_nrz(bits, muestras_por_bit, polar=False)
    nrz_pol = bits_a_nrz(bits, muestras_por_bit, polar=True)
    t = np.arange(len(nrz_uni)) / fs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # Gráfico 1: Unipolar
    ax1.plot(t * 1e3, nrz_uni, color='#1f77b4', linewidth=2, label="NRZ Unipolar (0V / 1V)")
    ax1.set_ylabel("Amplitud [V]", fontsize=10)
    ax1.set_title("Secuencia Digital: Pulsos NRZ Unipolar", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_ylim(-0.3, 1.3)

    # Gráfico 2: Bipolar
    ax2.plot(t * 1e3, nrz_pol, color='#2ca02c', linewidth=2, label="NRZ Bipolar (-1V / +1V)")
    ax2.set_xlabel("Tiempo [ms]", fontsize=10)
    ax2.set_ylabel("Amplitud [V]", fontsize=10)
    ax2.set_title("Secuencia Digital: Pulsos NRZ Polar / Bipolar", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_ylim(-1.4, 1.4)

    # Dibujar límites de bit y etiquetas con los bits
    for i, b in enumerate(bits):
        t_centro = (i + 0.5) * tb * 1e3
        ax1.text(t_centro, 0.5, f"b={b}", ha='center', va='center', fontweight='bold',
                 bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="#1f77b4", lw=1))
        ax2.text(t_centro, 0.0, f"b={b}", ha='center', va='center', fontweight='bold',
                 bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="#2ca02c", lw=1))
        if i > 0:
            ax1.axvline(i * tb * 1e3, color='gray', linestyle=':', alpha=0.7)
            ax2.axvline(i * tb * 1e3, color='gray', linestyle=':', alpha=0.7)

    plt.tight_layout()
    if ruta_guardado:
        os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
        plt.savefig(ruta_guardado, dpi=300)
        print(f"    [GRÁFICO GUARDADO] {ruta_guardado}")
    plt.close(fig)


def graficar_ondas_sinusoidales(
    fs: float = 100e3,
    duracion: float = 2e-3,
    ruta_guardado: str | None = None
) -> None:
    """
    Genera un gráfico demostrativo de variación de frecuencia, amplitud y fase.
    """
    t, s_ref = generar_senoidal(frecuencia=1000, amplitud=1.0, fase_rad=0.0, duracion=duracion, fs=fs)
    _, s_amp = generar_senoidal(frecuencia=1000, amplitud=2.0, fase_rad=0.0, duracion=duracion, fs=fs)
    _, s_fase = generar_senoidal(frecuencia=1000, amplitud=1.0, fase_rad=np.pi/2, duracion=duracion, fs=fs)
    _, s_frec = generar_senoidal(frecuencia=2500, amplitud=1.0, fase_rad=0.0, duracion=duracion, fs=fs)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    # 1. Variación de amplitud
    axes[0].plot(t * 1e3, s_ref, label=r"Referencia: $A=1\text{V}, f=1\text{kHz}, \phi=0^\circ$", color='blue')
    axes[0].plot(t * 1e3, s_amp, '--', label=r"Amplitud Doble: $A=2\text{V}, f=1\text{kHz}, \phi=0^\circ$", color='red')
    axes[0].set_title("Efecto de la Amplitud (Base para modulación ASK / QAM)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Amplitud [V]")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=8)

    # 2. Variación de fase
    axes[1].plot(t * 1e3, s_ref, label=r"Referencia: $\phi=0^\circ$", color='blue')
    axes[1].plot(t * 1e3, s_fase, '-.', label=r"Desfase: $\phi=+90^\circ\ (\pi/2\text{ rad})$", color='purple')
    axes[1].set_title("Efecto de la Fase (Base para modulación PSK)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Amplitud [V]")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=8)

    # 3. Variación de frecuencia
    axes[2].plot(t * 1e3, s_ref, label=r"Referencia: $f=1\text{kHz}$", color='blue')
    axes[2].plot(t * 1e3, s_frec, ':', label=r"Mayor frecuencia: $f=2.5\text{kHz}$", color='darkorange', linewidth=2)
    axes[2].set_title("Efecto de la Frecuencia (Base para modulación FSK)", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Tiempo [ms]")
    axes[2].set_ylabel("Amplitud [V]")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    if ruta_guardado:
        os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
        plt.savefig(ruta_guardado, dpi=300)
        print(f"    [GRÁFICO GUARDADO] {ruta_guardado}")
    plt.close(fig)


def graficar_demostracion_iq(
    fc: float = 5000,
    fs: float = 200e3,
    ruta_guardado: str | None = None
) -> None:
    """
    Ilustra didácticamente la síntesis I/Q:
    - Componente I en fase (coseno)
    - Componente Q en cuadratura (seno)
    - Suma modulada s_RF(t)
    - Representación fasorial en el plano complejo
    """
    duracion = 0.8e-3  # 0.8 milisegundos (4 periodos de portadora de 5 kHz)
    t = np.arange(int(duracion * fs)) / fs

    # Asignamos valores constantes escalares para este intervalo
    # Ejemplo: I = 1.0, Q = 1.0 (corresponde a un símbolo de QPSK en el primer cuadrante)
    I_val = 1.0
    Q_val = 1.0

    portadora_i = np.cos(2 * np.pi * fc * t)
    portadora_q = -np.sin(2 * np.pi * fc * t)

    rama_i = I_val * portadora_i
    rama_q = Q_val * portadora_q
    s_rf = rama_i + rama_q  # Equivalente a I*cos - Q*sin

    magnitud_teorica = np.sqrt(I_val**2 + Q_val**2)
    fase_teorica_deg = np.rad2deg(np.arctan2(Q_val, I_val))

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(3, 2)

    # 1. Rama I en el tiempo
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t * 1e3, rama_i, color='blue', label=r"$s_I(t) = I \cdot \cos(2\pi f_c t)$")
    ax1.set_ylabel("Rama I [V]")
    ax1.set_title(f"Componente En Fase (I = {I_val})", fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)

    # 2. Rama Q en el tiempo
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(t * 1e3, rama_q, color='red', label=r"$s_Q(t) = -Q \cdot \sin(2\pi f_c t)$")
    ax2.set_ylabel("Rama Q [V]")
    ax2.set_title(f"Componente en Cuadratura (Q = {Q_val})", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    # 3. Señal RF resultante combinada
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(t * 1e3, s_rf, color='purple', linewidth=2,
             label=rf"$s_{{RF}}(t) = s_I(t) + s_Q(t)$ ($A={magnitud_teorica:.2f}\text{{V}}, \theta={fase_teorica_deg:.0f}^\circ$)")
    ax3.set_xlabel("Tiempo [ms]")
    ax3.set_ylabel("Señal RF [V]")
    ax3.set_title("Señal Paso Banda Transmitida en RF", fontsize=10, fontweight='bold')
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8)

    # 4. Plano Complejo Fasorial I/Q (Subplot derecho ocupando las 3 filas)
    ax_iq = fig.add_subplot(gs[:, 1])
    ax_iq.axhline(0, color='black', linewidth=1)
    ax_iq.axvline(0, color='black', linewidth=1)

    # Dibujar círculo de magnitud
    circulo = plt.Circle((0, 0), magnitud_teorica, color='gray', fill=False, linestyle=':', alpha=0.6)
    ax_iq.add_patch(circulo)

    # Vector fasor
    ax_iq.annotate(
        "", xy=(I_val, Q_val), xytext=(0, 0),
        arrowprops=dict(facecolor='purple', edgecolor='purple', width=2, headwidth=8)
    )
    # Proyecciones punteadas
    ax_iq.plot([I_val, I_val], [0, Q_val], 'b--', alpha=0.7)
    ax_iq.plot([0, I_val], [Q_val, Q_val], 'r--', alpha=0.7)

    # Punto del símbolo
    ax_iq.scatter([I_val], [Q_val], color='purple', s=120, zorder=5, label=f"Símbolo: ({I_val}, {Q_val})")
    info_texto = f"$I={I_val}$\n$Q={Q_val}$\n$|s|={magnitud_teorica:.2f}$\n$\\theta={fase_teorica_deg:.0f}^\\circ$"
    ax_iq.text(I_val + 0.1, Q_val + 0.1, info_texto,
               fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

    ax_iq.set_xlim(-1.8, 1.8)
    ax_iq.set_ylim(-1.8, 1.8)
    ax_iq.set_aspect('equal', adjustable='box')
    ax_iq.set_xlabel("En Fase (I) [Componente Real]", fontsize=10, fontweight='bold')
    ax_iq.set_ylabel("Cuadratura (Q) [Componente Imaginario]", fontsize=10, fontweight='bold')
    ax_iq.set_title("Representación en el Plano Complejo I/Q\n(Fundamento de Constelaciones PSK/QAM)", fontsize=11, fontweight='bold')
    ax_iq.grid(True, linestyle="--", alpha=0.6)
    ax_iq.legend(loc="upper left")

    plt.tight_layout()
    if ruta_guardado:
        os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
        plt.savefig(ruta_guardado, dpi=300)
        print(f"    [GRÁFICO GUARDADO] {ruta_guardado}")
    plt.close(fig)
