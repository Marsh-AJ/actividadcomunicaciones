"""
metricas.py
Módulo de Métricas Estadísticas, Calibración Analítica y Rendimiento de Modulaciones.

Desarrollado para la asignatura Electiva de Comunicaciones Móviles (Python 3.13.5).
Fase 6: Métricas y Evaluación del Rendimiento de las Modulaciones.

Contenido:
1. Conversiones analíticas entre dominios (Eb/N0, Es/N0, N0 y varianzas).
2. Cálculo riguroso de BER con intervalos de confianza Wilson al 95% y regla de tres (3/Nb).
3. Cálculo de EVM (Error Vector Magnitude) a nivel de símbolo en instantes de decisión (I/Q).
4. Curvas teóricas de referencia:
   - BPSK y QPSK coherente (exacta).
   - 16-QAM Gray (aproximación clásica).
   - BFSK no coherente (detector de envolvente en cuadratura con tonos ortogonales).
5. Eficiencia espectral con documentación explícita de convenciones (Carson, Nyquist, nulo a nulo).
6. Estimación de complejidad computacional justificada (operaciones elementales por símbolo).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Union
import numpy as np
from scipy import special


# Constante fundamental de Boltzmann en J/K
CONSTANTE_BOLTZMANN_K = 1.380649e-23
# Temperatura física estándar de ruido IEEE (Kelvin)
TEMPERATURA_ESTANDAR_T0 = 290.0
# Impedancia de referencia estándar de radiofrecuencia (Ohmios)
IMPEDANCIA_REFERENCIA_OHMS = 50.0


# =====================================================================
# 1. ESTRUCTURAS DE DATOS DE MÉTRICAS
# =====================================================================

@dataclass(frozen=True)
class MetricasBER:
    """
    Estructura inmutable para reportar la tasa de error de bit (BER)
    junto con sus intervalos estadísticos de confianza.
    """
    ber: float
    errores: int
    total_bits: int
    cota_tres: float | None        # Regla de tres (3 / Nb) cuando errores == 0
    wilson_lower: float            # Límite inferior Wilson al 95%
    wilson_upper: float            # Límite superior Wilson al 95%
    es_cero_errores: bool


# =====================================================================
# 2. CONVERSIONES ANALÍTICAS Y CALIBRACIÓN DE AWGN
# =====================================================================

def ebn0_a_esn0(ebn0_db: float, bits_por_simbolo: int) -> float:
    """
    Convierte la relación energía de bit a densidad de ruido (Eb/N0) [dB]
    a relación energía de símbolo a densidad de ruido (Es/N0) [dB]:
        (Es/N0)_dB = (Eb/N0)_dB + 10 · log10(k)

    Parámetros:
        ebn0_db: Relación Eb/N0 en decibelios [dB].
        bits_por_simbolo (int): Bits por símbolo k = log2(M).
            - BFSK: k = 1
            - BPSK: k = 1
            - QPSK: k = 2
            - 16-QAM: k = 4

    Retorna:
        float: Relación Es/N0 en decibelios [dB].
    """
    if bits_por_simbolo <= 0:
        raise ValueError(f"bits_por_simbolo debe ser > 0. Se recibió {bits_por_simbolo}.")
    return float(ebn0_db + 10.0 * np.log10(float(bits_por_simbolo)))


def esn0_a_ebn0(esn0_db: float, bits_por_simbolo: int) -> float:
    """
    Convierte (Es/N0) [dB] a (Eb/N0) [dB]:
        (Eb/N0)_dB = (Es/N0)_dB - 10 · log10(k)
    """
    if bits_por_simbolo <= 0:
        raise ValueError(f"bits_por_simbolo debe ser > 0. Se recibió {bits_por_simbolo}.")
    return float(esn0_db - 10.0 * np.log10(float(bits_por_simbolo)))


def calcular_n0_desde_ebn0(ebn0_db: float, bits_por_simbolo: int, es: float = 1.0) -> float:
    """
    Calcula la densidad espectral de potencia unilateral de ruido N0
    a partir de Eb/N0 en dB y la energía promedio de símbolo normalizada Es.

    Convención:
        gamma_b = 10^(Eb/N0_dB / 10)
        Eb = Es / k
        N0 = Eb / gamma_b = Es / (k · gamma_b)

    Retorna:
        float: N0 unilateral [W/Hz o adimensional según Es].
    """
    if bits_por_simbolo <= 0:
        raise ValueError("bits_por_simbolo debe ser > 0.")
    if es <= 0.0:
        raise ValueError("es debe ser > 0.0.")
    gamma_b = 10.0 ** (float(ebn0_db) / 10.0)
    eb = float(es) / float(bits_por_simbolo)
    return float(eb / gamma_b)


def calcular_varianza_bb(n0: float, sps: int) -> tuple[float, float]:
    """
    Calcula la varianza total y por componente en banda base sobremuestreada
    para que tras el filtro promediador rectangular (Eh = 1/Nsps) la varianza
    de decisión resulte exactamente N0/2 por rama:
        sigma_BB^2 = Nsps · N0
        sigma_BB,I^2 = sigma_BB,Q^2 = (Nsps · N0) / 2

    Retorna:
        tuple[float, float]: (varianza_total, varianza_por_componente).
    """
    if n0 <= 0.0 or sps <= 0:
        raise ValueError("n0 y sps deben ser estrictamente positivos.")
    var_total = float(sps * n0)
    var_comp = float(var_total / 2.0)
    return var_total, var_comp


def calcular_varianza_rf(n0: float, sps: int) -> float:
    """
    Calcula la varianza de ruido gaussiano real en RF pasobanda (con portadora unitaria, Ps=0.5 W)
    para equivalencia estricta tras demodulación coherente:
        sigma_RF^2 = (Nsps · N0) / 4

    Retorna:
        float: varianza del ruido real en RF.
    """
    if n0 <= 0.0 or sps <= 0:
        raise ValueError("n0 y sps deben ser estrictamente positivos.")
    return float((sps * n0) / 4.0)


def calcular_varianza_simbolo(n0: float) -> tuple[float, float]:
    """
    Calcula la varianza de ruido en el espacio discreto de símbolos (1 muestra/símbolo):
        sigma_sym^2 = N0
        sigma_I^2 = sigma_Q^2 = N0 / 2

    Retorna:
        tuple[float, float]: (varianza_total, varianza_por_componente).
    """
    if n0 <= 0.0:
        raise ValueError("n0 debe ser estrictamente positivo.")
    return float(n0), float(n0 / 2.0)


# =====================================================================
# 3. ESTIMACIÓN DE BER E INTERVALOS DE CONFIANZA
# =====================================================================

def calcular_ber_con_intervalo(
    bits_tx: np.ndarray,
    bits_rx: np.ndarray,
    alpha: float = 0.05
) -> MetricasBER:
    """
    Calcula la tasa de error de bit (BER) empírica y sus intervalos de confianza:
    - Intervalo asintótico score de Wilson al nivel (1 - alpha) (por defecto 95%).
    - Para BER = 0 (cero errores observados): reporta el límite superior de Wilson
      y la cota superior conservadora de la regla de tres (3 / Nb).

    Parámetros:
        bits_tx: Arreglo 1D de bits transmitidos (0 y 1).
        bits_rx: Arreglo 1D de bits recibidos (0 y 1).
        alpha: Nivel de significancia (0.05 para 95% de confianza).

    Retorna:
        MetricasBER con ber, errores, total_bits, cota_tres, wilson_lower, wilson_upper.
    """
    b_tx = np.asarray(bits_tx, dtype=np.int8).ravel()
    b_rx = np.asarray(bits_rx, dtype=np.int8).ravel()

    if len(b_tx) == 0:
        raise ValueError("La secuencia de bits transmitidos no puede estar vacía.")
    if len(b_tx) != len(b_rx):
        raise ValueError(f"Longitudes dispares: tx={len(b_tx)}, rx={len(b_rx)}.")

    n_bits = len(b_tx)
    errores = int(np.sum(b_tx != b_rx))
    ber_estimada = float(errores / n_bits)

    # Valor crítico z para distribución normal estándar bilateral
    # Para alpha = 0.05, z ≈ 1.95996
    z = float(special.ndtri(1.0 - alpha / 2.0))

    denominador = 1.0 + (z**2) / float(n_bits)
    centro = (ber_estimada + (z**2) / (2.0 * n_bits)) / denominador
    margen = (z * np.sqrt((ber_estimada * (1.0 - ber_estimada) / n_bits) + (z**2) / (4.0 * (n_bits**2)))) / denominador

    w_lower = max(0.0, float(centro - margen))
    w_upper = min(1.0, float(centro + margen))

    if errores == 0:
        es_cero = True
        cota_tres = float(3.0 / n_bits)
        w_lower = 0.0
        # Wilson con p_hat = 0
        w_upper = float((z**2) / (n_bits + z**2))
    else:
        es_cero = False
        cota_tres = None

    return MetricasBER(
        ber=ber_estimada,
        errores=errores,
        total_bits=n_bits,
        cota_tres=cota_tres,
        wilson_lower=w_lower,
        wilson_upper=w_upper,
        es_cero_errores=es_cero
    )


# =====================================================================
# 4. EVM (ERROR VECTOR MAGNITUDE) A NIVEL DE SÍMBOLO
# =====================================================================

def calcular_evm_simbolos(
    simbolos_tx: np.ndarray | None,
    simbolos_rx: np.ndarray | None,
    tipo_mod: str
) -> tuple[float | None, float | None]:
    """
    Calcula la Magnitud del Vector de Error (EVM) estrictamente a nivel de símbolo discreto
    (1 muestra por símbolo en el instante de decisión) para modulaciones lineales complejas:
        e[k] = s_rx[k] - s_tx[k]
        EVM_RMS = sqrt( sum(|e[k]|^2) / sum(|s_tx[k]|^2) )
        EVM_% = EVM_RMS · 100%
        EVM_dB = 20 · log10(max(EVM_RMS, 1e-12))

    Regla obligatoria para BFSK:
        BFSK no posee constelación en el plano complejo I/Q y su decisión se realiza
        mediante banco de correladores pasobanda de envolvente. Por ello, para BFSK
        esta función retorna formalmente (None, None), representándose como "N/A".

    Parámetros:
        simbolos_tx: Arreglo 1D de símbolos transmitidos complejos (o None en BFSK).
        simbolos_rx: Arreglo 1D de símbolos recibidos complejos tras filtrado (o None).
        tipo_mod: Cadena con el nombre de la modulación ('BFSK', 'BPSK', 'QPSK', '16-QAM').

    Retorna:
        tuple[float | None, float | None]: (evm_porcentaje, evm_db) o (None, None).
    """
    mod_norm = str(tipo_mod).strip().upper()

    if mod_norm == "BFSK":
        # BFSK no opera sobre constelación compleja I/Q
        return None, None

    if simbolos_tx is None or simbolos_rx is None:
        raise ValueError(f"Para modulación lineal '{mod_norm}', simbolos_tx y simbolos_rx no pueden ser None.")

    tx = np.asarray(simbolos_tx, dtype=complex).ravel()
    rx = np.asarray(simbolos_rx, dtype=complex).ravel()

    if len(tx) == 0:
        raise ValueError("El arreglo de símbolos transmitidos no puede estar vacío.")
    if len(tx) != len(rx):
        raise ValueError(
            f"Vectores de símbolos desalineados o de longitud dispar: tx={len(tx)}, rx={len(rx)}."
        )
    if np.any(np.isnan(tx)) or np.any(np.isnan(rx)) or np.any(np.isinf(tx)) or np.any(np.isinf(rx)):
        raise ValueError("Los vectores de símbolos contienen valores no numéricos (NaN o Inf).")

    potencia_error = float(np.sum(np.abs(rx - tx) ** 2))
    potencia_referencia = float(np.sum(np.abs(tx) ** 2))

    if potencia_referencia <= 0.0:
        potencia_referencia = 1.0

    evm_rms = math.sqrt(potencia_error / potencia_referencia)
    evm_pct = float(evm_rms * 100.0)
    evm_db = float(20.0 * math.log10(max(evm_rms, 1e-12)))

    return evm_pct, evm_db


# =====================================================================
# 5. MODELOS TEÓRICOS DE REFERENCIA
# =====================================================================

def q_function(x: np.ndarray | float) -> np.ndarray | float:
    """
    Función Q de Gauss:
        Q(x) = (1 / sqrt(2pi)) int_x^inf exp(-u^2 / 2) du = 0.5 · erfc(x / sqrt(2))
    """
    arr = np.asarray(x, dtype=float)
    resultado = 0.5 * special.erfc(arr / np.sqrt(2.0))
    if isinstance(x, (int, float)):
        return float(resultado)
    return resultado


def ber_teorica_bpsk_qpsk(ebn0_db: np.ndarray | float) -> np.ndarray | float:
    """
    Probabilidad teórica de error de bit exacta para BPSK y QPSK con mapeo Gray
    en canal AWGN con receptor coherente óptimo:
        P_b = Q( sqrt(2 · Eb/N0) ) = 0.5 · erfc( sqrt(Eb/N0) )

    Parámetros:
        ebn0_db: Relación Eb/N0 en decibelios [dB] (escalar o arreglo NumPy).

    Retorna:
        BER teórica exacta.
    """
    ebn0_lin = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    res = 0.5 * special.erfc(np.sqrt(ebn0_lin))
    if isinstance(ebn0_db, (int, float)):
        return float(res)
    return res


def ber_teorica_16qam_aproximada(ebn0_db: np.ndarray | float) -> np.ndarray | float:
    """
    Aproximación clásica teórica de la probabilidad de error de bit para 16-QAM
    con constelación rectangular y codificación Gray en canal AWGN:
        P_b ≈ (3/4) · Q( sqrt(0.8 · Eb/N0) ) = (3/8) · erfc( sqrt(0.4 · Eb/N0) )

    Documentación de diseño obligatoria:
        Esta curva se rotula de forma explícita en tablas y gráficos como:
        'BER_teorica_aproximada' o '16-QAM Teórica (Aprox.)'.

    Parámetros:
        ebn0_db: Relación Eb/N0 en decibelios [dB].

    Retorna:
        BER teórica aproximada.
    """
    ebn0_lin = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    res = (3.0 / 8.0) * special.erfc(np.sqrt(0.4 * ebn0_lin))
    if isinstance(ebn0_db, (int, float)):
        return float(res)
    return res


def ber_teorica_bfsk_no_coherente(ebn0_db: np.ndarray | float) -> np.ndarray | float:
    """
    Probabilidad teórica de error de bit para BFSK no coherente con tonos ortogonales
    (detector de envolvente en cuadratura / filtro adaptado cuadrático):
        P_b = 0.5 · exp( -0.5 · Eb/N0 )

    Hipótesis del modelo documentadas:
        1. Tonos ortogonales en el intervalo de símbolo Ts:
           Delta_f = |f1 - f0| = 2000 Hz = 2 · Rs (cumple ortogonalidad estricta).
        2. Receptor implementado en demodulacion.demodular_bfsk mediante banco
           de correlación en cuadratura (I0^2 + Q0^2 vs I1^2 + Q1^2).
        3. Ruido blanco gaussiano no correlacionado entre ramas.
        Rotulación obligatoria: 'BFSK Teórica No Coherente (Ortogonal)'.

    Parámetros:
        ebn0_db: Relación Eb/N0 en decibelios [dB].

    Retorna:
        BER teórica para detector no coherente ortogonal.
    """
    ebn0_lin = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    res = 0.5 * np.exp(-0.5 * ebn0_lin)
    if isinstance(ebn0_db, (int, float)):
        return float(res)
    return res


# =====================================================================
# 6. EFICIENCIA ESPECTRAL Y ANCHO DE BANDA
# =====================================================================

def calcular_eficiencia_espectral(
    tipo_mod: str,
    rs: float = 1000.0,
    **kwargs: Any
) -> dict[str, Any]:
    """
    Calcula la eficiencia espectral eta [bps/Hz] documentando explícitamente
    las diferentes convenciones de ancho de banda utilizadas en comunicaciones:

    1. Convención Carson (estándar para FSK):
       B_Carson = Delta_f + 2·Rs = 2000 + 2000 = 4000 Hz.
       eta_Carson = Rb / B_Carson = 1000 / 4000 = 0.25 bps/Hz.

    2. Convención Ancho de Banda de RF Nulo a Nulo:
       Para modulaciones de pulso rectangular de duración Ts = 1/Rs:
       El lóbulo principal pasobanda se extiende en 2·Rs = 2000 Hz.
       - BPSK: Rb = 1000 bps, B = 2000 Hz -> eta = 0.50 bps/Hz.
       - QPSK: Rb = 2000 bps, B = 2000 Hz -> eta = 1.00 bps/Hz.
       - 16-QAM: Rb = 4000 bps, B = 2000 Hz -> eta = 2.00 bps/Hz.

    3. Convención Nyquist Ideal (límite teórico sin sobremuestreo):
       B_Nyquist = Rs = 1000 Hz en banda base.
       - BPSK: eta = 1.00 bps/Hz.
       - QPSK: eta = 2.00 bps/Hz.
       - 16-QAM: eta = 4.00 bps/Hz.

    Retorna:
        dict con valores calculados, fórmulas y convención principal.
    """
    mod_norm = str(tipo_mod).strip().upper()
    rs_val = float(rs)

    if mod_norm == "BFSK":
        delta_f = float(kwargs.get("delta_f", 2000.0))  # |4000 - 2000| Hz
        k = 1
        rb = rs_val * k
        b_carson = delta_f + 2.0 * rs_val  # 4000 Hz
        eta_carson = rb / b_carson          # 0.25 bps/Hz
        b_nulo_rf = delta_f + 2.0 * rs_val # 4000 Hz (f1+Rs - (f0-Rs))
        eta_nulo = rb / b_nulo_rf           # 0.25 bps/Hz

        return {
            "modulacion": "BFSK",
            "bits_por_simbolo": k,
            "tasa_simbolos_baud": rs_val,
            "tasa_bits_bps": rb,
            "convencion_principal": "Regla de Carson",
            "ancho_banda_principal_hz": b_carson,
            "eficiencia_principal_bps_hz": eta_carson,
            "detalle_convenciones": {
                "carson": {"ancho_banda_hz": b_carson, "eficiencia_bps_hz": eta_carson, "formula": "Rb / (Delta_f + 2·Rs)"},
                "nulo_a_nulo_rf": {"ancho_banda_hz": b_nulo_rf, "eficiencia_bps_hz": eta_nulo, "formula": "Rb / (f1+Rs - (f0-Rs))"},
            }
        }

    elif mod_norm == "BPSK":
        k = 1
        rb = rs_val * k
        b_nulo_rf = 2.0 * rs_val    # 2000 Hz
        b_nyquist = rs_val          # 1000 Hz
        eta_nulo = rb / b_nulo_rf   # 0.50 bps/Hz
        eta_nyquist = rb / b_nyquist# 1.00 bps/Hz

        return {
            "modulacion": "BPSK",
            "bits_por_simbolo": k,
            "tasa_simbolos_baud": rs_val,
            "tasa_bits_bps": rb,
            "convencion_principal": "Lóbulo principal RF (nulo a nulo)",
            "ancho_banda_principal_hz": b_nulo_rf,
            "eficiencia_principal_bps_hz": eta_nulo,
            "detalle_convenciones": {
                "nulo_a_nulo_rf": {"ancho_banda_hz": b_nulo_rf, "eficiencia_bps_hz": eta_nulo, "formula": "Rb / (2·Rs)"},
                "nyquist_ideal": {"ancho_banda_hz": b_nyquist, "eficiencia_bps_hz": eta_nyquist, "formula": "Rb / Rs"},
            }
        }

    elif mod_norm == "QPSK":
        k = 2
        rb = rs_val * k
        b_nulo_rf = 2.0 * rs_val    # 2000 Hz
        b_nyquist = rs_val          # 1000 Hz
        eta_nulo = rb / b_nulo_rf   # 1.00 bps/Hz
        eta_nyquist = rb / b_nyquist# 2.00 bps/Hz

        return {
            "modulacion": "QPSK",
            "bits_por_simbolo": k,
            "tasa_simbolos_baud": rs_val,
            "tasa_bits_bps": rb,
            "convencion_principal": "Lóbulo principal RF (nulo a nulo)",
            "ancho_banda_principal_hz": b_nulo_rf,
            "eficiencia_principal_bps_hz": eta_nulo,
            "detalle_convenciones": {
                "nulo_a_nulo_rf": {"ancho_banda_hz": b_nulo_rf, "eficiencia_bps_hz": eta_nulo, "formula": "Rb / (2·Rs)"},
                "nyquist_ideal": {"ancho_banda_hz": b_nyquist, "eficiencia_bps_hz": eta_nyquist, "formula": "Rb / Rs"},
            }
        }

    elif mod_norm in ("16-QAM", "16QAM"):
        k = 4
        rb = rs_val * k
        b_nulo_rf = 2.0 * rs_val    # 2000 Hz
        b_nyquist = rs_val          # 1000 Hz
        eta_nulo = rb / b_nulo_rf   # 2.00 bps/Hz
        eta_nyquist = rb / b_nyquist# 4.00 bps/Hz

        return {
            "modulacion": "16-QAM",
            "bits_por_simbolo": k,
            "tasa_simbolos_baud": rs_val,
            "tasa_bits_bps": rb,
            "convencion_principal": "Lóbulo principal RF (nulo a nulo)",
            "ancho_banda_principal_hz": b_nulo_rf,
            "eficiencia_principal_bps_hz": eta_nulo,
            "detalle_convenciones": {
                "nulo_a_nulo_rf": {"ancho_banda_hz": b_nulo_rf, "eficiencia_bps_hz": eta_nulo, "formula": "Rb / (2·Rs)"},
                "nyquist_ideal": {"ancho_banda_hz": b_nyquist, "eficiencia_bps_hz": eta_nyquist, "formula": "Rb / Rs"},
            }
        }

    else:
        raise ValueError(f"Modulación '{tipo_mod}' no reconocida.")


# =====================================================================
# 7. COMPLEJIDAD OPERACIONAL JUSTIFICADA POR SÍMBOLO
# =====================================================================

def estimar_complejidad_operacional(tipo_mod: str, sps: int = 16) -> dict[str, Any]:
    """
    Estima rigurosamente la complejidad computacional en el receptor contando
    las operaciones elementales aritméticas de punto flotante por símbolo (FLOPs/símbolo):

    - BFSK (banco de correlación en cuadratura sobre RF pasobanda):
      Por cada símbolo de Nsps muestras:
      4 ramas de producto base: 4 · Nsps multiplicaciones.
      4 acumulaciones de suma: 4 · (Nsps - 1) sumas.
      2 elevaciones al cuadrado de envolvente (I^2 + Q^2): 2 multiplicaciones + 2 sumas.
      1 comparador de decisión: 1 comparación.
      Total ops/símbolo ≈ 8 · Nsps + 3.

    - BPSK (decisión de signo coherente en constelación I):
      1 comparación de signo en Re{s}: 1 op/símbolo (o 0.5 ops/bit).

    - QPSK (decisión de signo independiente en I y Q):
      2 comparaciones de signo: 2 ops/símbolo (1 op/bit).

    - 16-QAM (decisión multiumbral en I y Q con mapeo Gray):
      En I: comparaciones con umbrales {-2, 0, 2} / sqrt(10) -> ~3 comparaciones.
      En Q: comparaciones con umbrales {-2, 0, 2} / sqrt(10) -> ~3 comparaciones.
      Lógica de mapeo de cuadrantes: ~2 operaciones lógicas.
      Total ops/símbolo ≈ 8 ops/símbolo (2 ops/bit).

    Retorna:
        dict con desglose analítico y métricas de complejidad.
    """
    mod_norm = str(tipo_mod).strip().upper()
    nsps = int(sps)

    if mod_norm == "BFSK":
        k = 1
        multiplicaciones = 4 * nsps + 2
        sumas = 4 * (nsps - 1) + 2
        comparaciones = 1
        total_ops = multiplicaciones + sumas + comparaciones
        return {
            "modulacion": "BFSK",
            "dominio_receptor": "RF Pasobanda",
            "bits_por_simbolo": k,
            "nsps": nsps,
            "ops_símbolo": total_ops,
            "ops_bit": float(total_ops / k),
            "desglose": f"{multiplicaciones} mults, {sumas} sumas, {comparaciones} comp",
            "nivel_cualitativo": "Complejidad media en RF pasobanda"
        }

    elif mod_norm == "BPSK":
        k = 1
        total_ops = 1  # Signo Re{s} > 0
        return {
            "modulacion": "BPSK",
            "dominio_receptor": "Símbolo Banda Base",
            "bits_por_simbolo": k,
            "nsps": nsps,
            "ops_símbolo": total_ops,
            "ops_bit": float(total_ops / k),
            "desglose": "1 comparación de signo",
            "nivel_cualitativo": "Complejidad mínima elemental"
        }

    elif mod_norm == "QPSK":
        k = 2
        total_ops = 2  # Signos independientes en I y Q
        return {
            "modulacion": "QPSK",
            "dominio_receptor": "Símbolo Banda Base",
            "bits_por_simbolo": k,
            "nsps": nsps,
            "ops_símbolo": total_ops,
            "ops_bit": float(total_ops / k),
            "desglose": "2 comparaciones de signo (I y Q)",
            "nivel_cualitativo": "Muy baja complejidad"
        }

    elif mod_norm in ("16-QAM", "16QAM"):
        k = 4
        total_ops = 8  # 3 umbrales I + 3 umbrales Q + 2 lógicas
        return {
            "modulacion": "16-QAM",
            "dominio_receptor": "Símbolo Banda Base",
            "bits_por_simbolo": k,
            "nsps": nsps,
            "ops_símbolo": total_ops,
            "ops_bit": float(total_ops / k),
            "desglose": "6 comparaciones de umbral + 2 lógicas Gray",
            "nivel_cualitativo": "Complejidad moderada en constelación"
        }

    else:
        raise ValueError(f"Modulación '{tipo_mod}' no reconocida.")

