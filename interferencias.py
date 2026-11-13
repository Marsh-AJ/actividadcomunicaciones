"""
interferencias.py
Módulo de Modelos de Interferencias Electromagnéticas y Multiusuario.

Contiene los siguientes modelos:
1. ACI (Interferencia de Canal Adyacente con validación del límite de Nyquist).
2. CDMA / MUI (Interferencia multiusuario con matriz Walsh-Hadamard 8, BPSK, zero-padding y correlación teórica).
3. Interferencia tonal monocromática de onda continua (CW).
4. CCI (Interferencia cocanal en la misma frecuencia portadora).

Desarrollado para la asignatura Electiva de Comunicaciones Móviles (Python 3.13.5).
"""

from __future__ import annotations
import numpy as np
from typing import Union


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
# 1. INTERFERENCIA DE CANAL ADYACENTE (ACI)
# =====================================================================

def aplicar_aci(
    senal_rf: np.ndarray,
    fs: float,
    fc: float,
    delta_f: float,
    sir_db: float,
    pot_senal: float = 0.5,
    fase_inicial: float = 0.0
) -> np.ndarray:
    """
    Aplica interferencia de canal adyacente (ACI) mediante un tono sinusoidal adyacente
    centrado en f_adj = fc + delta_f.

    Validación estricta de Nyquist:
        Se exige |fc + delta_f| < fs / 2. Si se viola el criterio, se levanta ValueError.

    Parámetros:
        senal_rf: Señal real pasobanda de entrada (1D).
        fs: Frecuencia de muestreo en Hz (> 0).
        fc: Frecuencia de portadora deseada en Hz (> 0).
        delta_f: Separación en frecuencia del canal adyacente en Hz (puede ser positiva o negativa).
        sir_db: Relación señal a interferencia (Signal-to-Interference Ratio) en dB.
        pot_senal: Potencia de la señal deseada en Watts (por defecto 0.5 W).
        fase_inicial: Fase inicial del tono interferente en radianes.

    Retorna:
        np.ndarray: Señal RF pasobanda con la interferencia adyacente superpuesta.
    """
    s = _validar_senal_unidimensional(senal_rf, "senal_rf")
    if fs <= 0.0:
        raise ValueError(f"fs debe ser positiva. Se recibió {fs}.")
    if fc <= 0.0:
        raise ValueError(f"fc debe ser positiva. Se recibió {fc}.")
    if pot_senal <= 0.0:
        raise ValueError(f"pot_senal debe ser positiva. Se recibió {pot_senal}.")

    f_adj = float(fc) + float(delta_f)
    if abs(f_adj) >= (fs / 2.0):
        raise ValueError(
            f"La frecuencia interferente adyacente f_adj={f_adj:.1f} Hz viola el criterio "
            f"de Nyquist para fs={fs:.1f} Hz (límite: fs/2={fs/2.0:.1f} Hz)."
        )

    # Potencia y amplitud del tono adyacente
    p_int = float(pot_senal) * (10.0 ** (-float(sir_db) / 10.0))
    a_int = np.sqrt(2.0 * p_int)

    t = np.arange(len(s)) / float(fs)
    interferencia = a_int * np.cos(2.0 * np.pi * f_adj * t + float(fase_inicial))

    return s + interferencia


# =====================================================================
# 2. INTERFERENCIA MULTIUSUARIO (MUI) — FLUJO CDMA REAL
# =====================================================================

def generar_codigos_walsh(orden: int = 8) -> np.ndarray:
    """
    Genera la matriz de códigos ortogonales de Walsh-Hadamard de dimensión orden x orden.
    Por construcción recursiva de Sylvester:
        H_1 = [1]
        H_{2N} = [[H_N,  H_N],
                  [H_N, -H_N]]

    Parámetros:
        orden: Tamaño de la matriz (potencia de 2, por defecto 8).

    Retorna:
        np.ndarray: Matriz binaria bipolar {-1, +1} de dimensiones (orden, orden).
    """
    if orden <= 0 or (orden & (orden - 1)) != 0:
        raise ValueError(f"El orden debe ser una potencia positiva de 2. Se recibió {orden}.")

    h = np.array([[1]], dtype=int)
    while h.shape[0] < orden:
        h = np.block([[h, h], [h, -h]])

    return h


def calcular_correlacion_parcial_walsh(
    codigo_u: np.ndarray,
    codigo_0: np.ndarray,
    tau_chips: int
) -> tuple[float, float]:
    """
    Calcula las dos funciones de correlación cruzada aperiódica parcial entre el código
    del usuario interferente (u) y el usuario deseado (0) para un retardo tau_chips ∈ {0, ..., L-1}:
        Γ_{u,0}(τ) = sum_{n=0}^{τ-1} c_u[n - τ + L] · c_0[n]   (solapamiento con símbolo anterior)
        Γ̂_{u,0}(τ) = sum_{n=τ}^{L-1} c_u[n - τ] · c_0[n]       (solapamiento con símbolo actual)

    Parámetros:
        codigo_u: Código Walsh del usuario interferente (longitud L).
        codigo_0: Código Walsh del usuario deseado (longitud L).
        tau_chips: Retardo en chips (0 <= tau_chips < L).

    Retorna:
        tuple[float, float]: (gamma_ant, gamma_act).
    """
    l_chips = len(codigo_0)
    tau = int(tau_chips) % l_chips

    if tau == 0:
        gamma_ant = 0.0
        gamma_act = float(np.sum(codigo_u * codigo_0))
    else:
        # Chips del símbolo anterior: n in [0, tau-1]
        gamma_ant = float(np.sum(codigo_u[l_chips - tau :] * codigo_0[:tau]))
        # Chips del símbolo actual: n in [tau, L-1]
        gamma_act = float(np.sum(codigo_u[: l_chips - tau] * codigo_0[tau:]))

    return gamma_ant, gamma_act


def simular_cdma_multiusuario(
    bits_tx_u0: np.ndarray,
    num_usuarios: int = 4,
    alfa_interf: float | list[float] = 1.0,
    retardos_chips: list[int] | None = None,
    seed: int | None = None
) -> dict[str, Union[np.ndarray, float, list]]:
    """
    Implementa el flujo DS-CDMA real con códigos Walsh-Hadamard de orden 8 y modulación BPSK.

    Reglas obligatorias de diseño:
        - Soporta máximo 8 usuarios simultáneos (U <= 8).
        - Modulación BPSK exclusiva: b_u ∈ {-1, +1}.
        - Expansión a 8 chips por símbolo.
        - Retardos enteros mediante zero-padding (sin wrap circular).
        - α_u representa la amplitud relativa del usuario u.
        - En la potencia teórica de interferencia se utiliza α_u²:
            P_MUI,teor = sum_{u=1}^{U-1} α_u² · (Γ_{u,0}²(τ_u) + Γ̂_{u,0}²(τ_u)) / L²

    Parámetros:
        bits_tx_u0: Secuencia binaria (0 y 1) del usuario deseado (u=0).
        num_usuarios: Cantidad de usuarios simultáneos (1 <= num_usuarios <= 8).
        alfa_interf: Amplitud relativa de los interferentes (escalar o lista de longitud U-1).
        retardos_chips: Lista de retardos enteros en chips para cada usuario u in {0, ..., U-1}.
                        El usuario deseado siempre tiene retardo 0.
        seed: Semilla pseudoaleatoria.

    Retorna:
        dict: Diccionario completo con bits recuperados, señales en chips,
              potencia teórica y medida de MUI, y factores de correlación.
    """
    bits_0 = _validar_senal_unidimensional(bits_tx_u0, "bits_tx_u0")
    if not (1 <= int(num_usuarios) <= 8):
        raise ValueError(
            f"El modelo CDMA con matriz Walsh de orden 8 admite entre 1 y 8 usuarios simultáneos. "
            f"Se solicitó num_usuarios={num_usuarios}."
        )
    u_total = int(num_usuarios)
    l_chips = 8  # Matriz Walsh H8

    rng = np.random.default_rng(seed)
    matriz_walsh = generar_codigos_walsh(orden=l_chips)

    n_simbolos = len(bits_0)

    # 1. Configurar amplitudes relativas alfa_u para u in {1, ..., U-1}
    if isinstance(alfa_interf, (int, float)):
        alfas = [float(alfa_interf)] * (u_total - 1)
    else:
        alfas = [float(a) for a in alfa_interf]
        if len(alfas) != (u_total - 1):
            raise ValueError(f"alfa_interf debe tener {u_total - 1} elementos. Se recibieron {len(alfas)}.")

    # 2. Configurar retardos enteros tau_u
    if retardos_chips is None:
        # Por defecto: retardo 0 para u=0, y retardos fijos no nulos para interferentes
        retardos = [0] + [(u * 2) % l_chips for u in range(1, u_total)]
    else:
        retardos = [int(r) for r in retardos_chips]
        if len(retardos) != u_total:
            raise ValueError(f"retardos_chips debe tener {u_total} elementos. Se recibieron {len(retardos)}.")
        if retardos[0] != 0:
            raise ValueError(f"El usuario deseado (u=0) debe tener retardo 0 chips. Se recibió {retardos[0]}.")

    # 3. Generar bits BPSK (-1, +1) para cada usuario
    # Usuario 0: bits de entrada
    bits_bpsk = np.zeros((u_total, n_simbolos), dtype=float)
    bits_bpsk[0] = np.where(bits_0 == 1, 1.0, -1.0)

    for u in range(1, u_total):
        # Bits aleatorios i.i.d. para usuarios interferentes
        bits_rand = rng.integers(0, 2, size=n_simbolos)
        bits_bpsk[u] = np.where(bits_rand == 1, 1.0, -1.0)

    # 4. Expansión a tasa de chip (L chips por símbolo) y aplicación de zero-padding
    total_chips_base = n_simbolos * l_chips
    max_tau = max(retardos)
    total_chips_trama = total_chips_base + max_tau

    senales_chips = np.zeros((u_total, total_chips_trama), dtype=float)

    for u in range(u_total):
        codigo_u = matriz_walsh[u]
        # Expansión: repetir cada bit L veces y multiplicar por el código
        chips_u = np.repeat(bits_bpsk[u], l_chips) * np.tile(codigo_u, n_simbolos)

        tau = retardos[u]
        # Zero-padding: alinear con ceros al inicio según tau
        senales_chips[u, tau : tau + total_chips_base] = chips_u

    # 5. Señal compuesta recibida
    # Usuario deseado u=0 tiene amplitud 1.0; interferentes tienen amplitud alfa_u
    senal_compuesta = np.copy(senales_chips[0])
    for u in range(1, u_total):
        alfa_u = alfas[u - 1]
        senal_compuesta += alfa_u * senales_chips[u]

    # 6. Receptor correlador / descorrelador para el usuario deseado (u=0)
    # Extraer los chips correspondientes a los n_simbolos completos del usuario 0
    chips_rx_u0 = senal_compuesta[:total_chips_base]
    codigo_0 = matriz_walsh[0]

    # Correlación símbolo a símbolo
    chips_bloques = chips_rx_u0.reshape(n_simbolos, l_chips)
    simbolos_descorrelacionados = np.sum(chips_bloques * codigo_0, axis=1) / float(l_chips)

    # Detección por umbral BPSK (símbolo >= 0 -> bit 1; < 0 -> bit 0)
    bits_recuperados = np.where(simbolos_descorrelacionados >= 0.0, 1, 0).astype(np.int8)

    # 7. Cálculo analítico de la potencia teórica de interferencia esperada
    # Para bits i.i.d. con E[b²]=1, la potencia teórica esperada es:
    # P_MUI,teor = sum_{u=1}^{U-1} alfa_u² · (Gamma_ant² + Gamma_act²) / L²
    p_mui_teor = 0.0
    detalles_interferentes = []

    for u in range(1, u_total):
        alfa_u = alfas[u - 1]
        tau_u = retardos[u]
        codigo_u = matriz_walsh[u]

        g_ant, g_act = calcular_correlacion_parcial_walsh(codigo_u, codigo_0, tau_u)
        rho_u = (g_ant**2 + g_act**2) / float(l_chips**2)
        pot_u_teor = (alfa_u**2) * rho_u
        p_mui_teor += pot_u_teor

        detalles_interferentes.append({
            "usuario": u,
            "alfa_u": alfa_u,
            "tau_chips": tau_u,
            "gamma_ant": g_ant,
            "gamma_act": g_act,
            "rho_u": rho_u,
            "potencia_teorica": pot_u_teor,
        })

    # 8. Medición empírica de interferencia
    # La interferencia instantánea sobre cada símbolo es: I[k] = y_0[k] - b_0[k]
    interferencia_medida = simbolos_descorrelacionados - bits_bpsk[0]
    p_mui_medida = float(np.mean(interferencia_medida**2))

    return {
        "bits_tx_u0": bits_0,
        "bits_rx_u0": bits_recuperados,
        "simbolos_descorrelacionados": simbolos_descorrelacionados,
        "senal_compuesta_chips": senal_compuesta,
        "num_usuarios": u_total,
        "alfas": alfas,
        "retardos": retardos,
        "p_mui_teor": p_mui_teor,
        "p_mui_medida": p_mui_medida,
        "detalles_interferentes": detalles_interferentes,
    }


# =====================================================================
# 3. INTERFERENCIA TONAL (ONDA CONTINUA — CW)
# =====================================================================

def aplicar_interferencia_tonal(
    senal_rf: np.ndarray,
    fs: float,
    f_cw: float,
    sir_db: float,
    pot_senal: float = 0.5,
    fase_inicial: float = 0.0
) -> np.ndarray:
    """
    Aplica una interferencia tonal monocromática pura de onda continua (CW) a frecuencia f_cw:
        i_CW[n] = A_cw · cos(2π f_cw t_n + φ_cw)
    con potencia calibrada según la relación señal a interferencia SIR [dB].

    Parámetros:
        senal_rf: Señal real pasobanda de entrada (1D).
        fs: Frecuencia de muestreo en Hz (> 0).
        f_cw: Frecuencia del tono interferente en Hz (0 < f_cw < fs/2).
        sir_db: Relación señal a interferencia en dB.
        pot_senal: Potencia de la señal deseada (0.5 W por defecto en RF).
        fase_inicial: Fase inicial en radianes.

    Retorna:
        np.ndarray: Señal pasobanda con el tono parásito superpuesto.
    """
    s = _validar_senal_unidimensional(senal_rf, "senal_rf")
    if fs <= 0.0:
        raise ValueError(f"fs debe ser positiva. Se recibió {fs}.")
    if not (0.0 < float(f_cw) < (fs / 2.0)):
        raise ValueError(f"f_cw ({f_cw} Hz) debe estar en el rango (0, fs/2) = (0, {fs/2.0} Hz).")
    if pot_senal <= 0.0:
        raise ValueError(f"pot_senal debe ser positiva. Se recibió {pot_senal}.")

    p_int = float(pot_senal) * (10.0 ** (-float(sir_db) / 10.0))
    a_cw = np.sqrt(2.0 * p_int)

    t = np.arange(len(s)) / float(fs)
    tono = a_cw * np.cos(2.0 * np.pi * float(f_cw) * t + float(fase_inicial))

    return s + tono


# =====================================================================
# 4. INTERFERENCIA COCANAL (CCI)
# =====================================================================

def aplicar_cci(
    senal: np.ndarray,
    sir_db: float,
    senal_interferente: np.ndarray | None = None,
    retardo_muestras: int = 15,
    fase_rad: float = 0.5,
    es_compleja: bool | None = None,
    seed: int | None = None
) -> np.ndarray:
    """
    Aplica interferencia cocanal (CCI) sumando una réplica modulada independiente
    en la misma frecuencia portadora y ancho de banda.

    Parámetros:
        senal: Señal deseada (1D, real o compleja).
        sir_db: Relación señal a interferencia cocanal en dB.
        senal_interferente: Señal interferente independiente opcional. Si es None,
                            se genera a partir de la señal original con retardo circular y fase.
        retardo_muestras: Retardo temporal en muestras del interferente.
        fase_rad: Desfase angular relativo del interferente en radianes.
        es_compleja: Si es None, se infiere de la señal.
        seed: Semilla pseudoaleatoria.

    Retorna:
        np.ndarray: Señal con interferencia cocanal aditiva.
    """
    s = _validar_senal_unidimensional(senal, "señal")
    compleja = np.iscomplexobj(s) if es_compleja is None else bool(es_compleja)
    rng = np.random.default_rng(seed)

    p_s = np.mean(np.abs(s) ** 2)
    if p_s == 0.0:
        p_s = 1.0

    factor_amplitud = np.sqrt(10.0 ** (-float(sir_db) / 10.0))

    if senal_interferente is not None:
        s_int = _validar_senal_unidimensional(senal_interferente, "senal_interferente")
        if len(s_int) < len(s):
            # Repetir o truncar para ajustar longitud
            repeticiones = int(np.ceil(len(s) / len(s_int)))
            s_int = np.tile(s_int, repeticiones)[: len(s)]
        else:
            s_int = s_int[: len(s)]
        p_int_actual = np.mean(np.abs(s_int) ** 2)
        if p_int_actual > 0:
            s_int = s_int / np.sqrt(p_int_actual) * np.sqrt(p_s)
    else:
        # Generar réplica decorrelacionada aplicando retardo y desfase
        ret = int(retardo_muestras) % len(s)
        s_int = np.roll(s, ret)

    if compleja:
        rotacion = np.exp(1j * float(fase_rad))
        interferencia = factor_amplitud * (s_int * rotacion)
    else:
        rotacion = np.cos(float(fase_rad))
        interferencia = factor_amplitud * (s_int * rotacion)

    return s + interferencia

