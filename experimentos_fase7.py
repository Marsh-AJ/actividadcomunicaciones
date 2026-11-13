"""
experimentos_fase7.py
Motor Experimental para Evaluación de Técnicas de Mitigación (Fase 7).

Desarrollado para la asignatura Electiva de Comunicaciones Móviles (Python 3.13.5).
Fase 7: Técnicas de Mitigación y Mejora del Rendimiento.

Experimentos implementados:
1. Experimento 1: Filtrado Digital FIR vs Interferencia de Canal Adyacente (ACI).
   - Comparación Antes (Sin Filtro) vs Después (Con Filtro FIR paso bajo de orden 48).
2. Experimento 2: Ecualización Zero-Forcing (ZF) vs Canal Dispersivo con ISI.
   - Comparación Antes (Sin Ecualizar) vs Después (Con Ecualizador ZF de 7 coeficientes).
3. Experimento 3: Diversidad de Recepción en Canal Rayleigh (SISO vs SC vs MRC).
   - Comparación de 1 rama (SISO L=1) vs 2 ramas con Selección (SC L=2) y Máxima Relación (MRC L=2).
"""

from __future__ import annotations
import math
from typing import Any
import numpy as np

import fundamentos
import modulaciones
import demodulacion
import ruido
import interferencias
import canales
import metricas
import mitigacion


# =====================================================================
# 1. EXPERIMENTO 1: FILTRADO DIGITAL FIR FRENTE A ACI
# =====================================================================

def ejecutar_experimento_1_filtrado(
    sir_valores_db: np.ndarray | None = None,
    delta_f: float = 1500.0,
    ebn0_fondo_db: float = 15.0,
    n_bits: int = 60000,
    seed_base: int = 5000
) -> dict[str, Any]:
    """
    Evalúa el desempeño antes y después del filtrado digital FIR frente a interferencia
    de canal adyacente (ACI) a delta_f = 1500 Hz con fondo AWGN (Eb/N0 = 15 dB):
    - BPSK, QPSK, 16-QAM: Filtro FIR paso bajo Hamming orden 48, fc = 1000 Hz, fs = 16000 Hz,
      aplicado sobre la mezcla coherente I/Q antes del muestreo/promediado de símbolo (ventana central [2:14]).
    - BFSK: Filtro FIR paso banda Hamming orden 72, banda [1800, 4200] Hz centrado en fc = 3000 Hz,
      aplicado en RF antes del banco de correlación en cuadratura.
    - Barrido: SIR de 0 a 16 dB (paso 2 dB).

    Retorna:
        dict con resultados detallados 'sin_filtro', 'con_filtro' y metadatos de los filtros.
    """
    if sir_valores_db is None:
        sir_valores_db = np.arange(0.0, 17.0, 2.0)

    fs = 16000.0
    sps = 16
    fc = 2000.0
    f0_bfsk = 2000.0
    f1_bfsk = 4000.0
    fc_bfsk = 3000.0

    # Parámetros del filtro paso bajo (banda base) y paso banda (BFSK RF)
    f_corte = 1000.0  # Ancho de banda de símbolo Rs = 1000 baud
    orden_filtro = 48
    f_baja_bfsk = 1800.0
    f_alta_bfsk = 4200.0
    orden_bfsk = 72

    # 1. Diseñar filtros y medir atenuaciones reales sobre la interferencia ACI
    h_fir = mitigacion.disenar_filtro_fir_pasabajos(fs=fs, f_corte=f_corte, orden=orden_filtro)
    info_filtro = mitigacion.medir_atenuacion_filtro(h_fir, fs=fs, f_interferencia=delta_f, f_referencia=0.0)

    h_fir_bfsk = mitigacion.disenar_filtro_fir_pasabanda(
        fs=fs, f_baja=f_baja_bfsk, f_alta=f_alta_bfsk, orden=orden_bfsk
    )
    info_filtro_bfsk = mitigacion.medir_atenuacion_filtro(
        h_fir_bfsk, fs=fs, f_interferencia=fc_bfsk + delta_f, f_referencia=fc_bfsk
    )

    n_bits_ajustado = int(math.ceil(n_bits / 4.0) * 4)
    modulaciones_lista = ["BFSK", "BPSK", "QPSK", "16-QAM"]

    resultados: dict[str, dict[str, list[dict[str, Any]]]] = {
        "sin_filtro": {m: [] for m in modulaciones_lista},
        "con_filtro": {m: [] for m in modulaciones_lista}
    }

    isi_limpia_16qam_pct = 0.0

    for m_idx, mod_nombre in enumerate(modulaciones_lista):
        bits_tx = fundamentos.generar_bits(n_bits_ajustado, semilla=seed_base + m_idx * 111)

        # Modulación original limpia
        if mod_nombre == "BFSK":
            mod_res = modulaciones.modular_bfsk(
                bits_tx, f0=f0_bfsk, f1=f1_bfsk, sps=sps, fs=fs, fase_continua=True
            )
        elif mod_nombre == "BPSK":
            mod_res = modulaciones.modular_bpsk(bits_tx, fc=fc, sps=sps, fs=fs, normalizar=True)
        elif mod_nombre == "QPSK":
            mod_res = modulaciones.modular_qpsk(bits_tx, fc=fc, sps=sps, fs=fs, normalizar=True)
        elif mod_nombre in ("16-QAM", "16QAM"):
            mod_res = modulaciones.modular_16qam(bits_tx, fc=fc, sps=sps, fs=fs, normalizar=True)
            # Verificar preservación de componentes útiles en banda base (ISI residual < 2%)
            bb_limpia_f = mitigacion.aplicar_filtro_fir(mod_res.senal_bb, h_fir, compensar_retardo=True)
            n_sym_c = len(bb_limpia_f) // sps
            sym_limpios_f = np.mean(bb_limpia_f[: n_sym_c * sps].reshape(n_sym_c, sps)[:, 2 : sps - 2], axis=1)
            isi_evm, _ = metricas.calcular_evm_simbolos(mod_res.simbolos, sym_limpios_f, "16-QAM")
            isi_limpia_16qam_pct = float(isi_evm or 0.0)
        else:
            continue

        k_bits = mod_res.bits_por_simbolo
        es_n0_fondo = metricas.ebn0_a_esn0(ebn0_fondo_db, k_bits)
        t = mod_res.t
        fc_mod = float(mod_res.fc)

        for s_idx, sir_db in enumerate(sir_valores_db):
            semilla_punto = seed_base + m_idx * 1000 + s_idx * 13

            # A. Aplicar ACI en RF pasobanda + AWGN en RF (exactamente la misma señal entra a ambos receptores)
            s_rf_aci = interferencias.aplicar_aci(
                mod_res.senal_rf,
                fs=fs,
                fc=fc_mod,
                delta_f=delta_f,
                sir_db=float(sir_db),
                pot_senal=0.5,
                fase_inicial=0.0
            )
            s_rf_pert = ruido.aplicar_awgn_rf_equivalente_esn0(
                s_rf_aci,
                es_n0_db=es_n0_fondo,
                sps=sps,
                seed=semilla_punto + 777
            )

            # CASO 1: SIN FILTRO FIR (Antes)
            if mod_nombre == "BFSK":
                bits_rx_sin = demodulacion.demodular_bfsk(
                    s_rf_pert, f0=f0_bfsk, f1=f1_bfsk, sps=sps, fs=fs
                )
                evm_pct_sin, evm_db_sin = None, None
            else:
                s_sym_sin, _ = canales.demodular_envolvente_coherente(
                    s_rf_pert, fc=fc_mod, fs=fs, sps=sps
                )
                if mod_nombre == "BPSK":
                    bits_rx_sin = demodulacion.demodular_bpsk(s_sym_sin)
                elif mod_nombre == "QPSK":
                    bits_rx_sin = demodulacion.demodular_qpsk(s_sym_sin)
                else:
                    bits_rx_sin = demodulacion.demodular_16qam(s_sym_sin)

                evm_pct_sin, evm_db_sin = metricas.calcular_evm_simbolos(
                    mod_res.simbolos, s_sym_sin, mod_nombre
                )

            mber_sin = metricas.calcular_ber_con_intervalo(bits_tx, bits_rx_sin)

            resultados["sin_filtro"][mod_nombre].append({
                "sir_db": float(sir_db),
                "ber": mber_sin.ber,
                "errores": mber_sin.errores,
                "cota_tres": mber_sin.cota_tres,
                "es_cero_errores": mber_sin.es_cero_errores,
                "evm_pct": evm_pct_sin,
                "evm_db": evm_db_sin,
            })

            # CASO 2: CON FILTRO DIGITAL FIR (Después)
            if mod_nombre == "BFSK":
                # Filtro FIR paso banda en RF centrado en la portadora antes del banco de correlación
                s_rf_filtrada = mitigacion.aplicar_filtro_fir(
                    s_rf_pert, h_fir_bfsk, compensar_retardo=True
                )
                bits_rx_con = demodulacion.demodular_bfsk(
                    s_rf_filtrada, f0=f0_bfsk, f1=f1_bfsk, sps=sps, fs=fs
                )
                evm_pct_con, evm_db_con = None, None
            else:
                # Mezcla coherente RF -> banda base SIN promediar aún (conserva el tono ACI a delta_f)
                s_bb_mezcla = (
                    2.0 * s_rf_pert * np.cos(2.0 * np.pi * fc_mod * t)
                    - 1j * 2.0 * s_rf_pert * np.sin(2.0 * np.pi * fc_mod * t)
                )
                # Filtrado FIR paso bajo antes del muestreo/integración de símbolo
                s_bb_filtrada = mitigacion.aplicar_filtro_fir(
                    s_bb_mezcla, h_fir, compensar_retardo=True
                )
                # Promediado en ventana central [2:14] evitando transitorios de borde por el filtro
                n_sym = len(s_bb_filtrada) // sps
                s_sym_filtrados = np.mean(
                    s_bb_filtrada[: n_sym * sps].reshape(n_sym, sps)[:, 2 : sps - 2],
                    axis=1
                )

                if mod_nombre == "BPSK":
                    bits_rx_con = demodulacion.demodular_bpsk(s_sym_filtrados)
                elif mod_nombre == "QPSK":
                    bits_rx_con = demodulacion.demodular_qpsk(s_sym_filtrados)
                else:
                    bits_rx_con = demodulacion.demodular_16qam(s_sym_filtrados)

                evm_pct_con, evm_db_con = metricas.calcular_evm_simbolos(
                    mod_res.simbolos, s_sym_filtrados, mod_nombre
                )

            mber_con = metricas.calcular_ber_con_intervalo(bits_tx, bits_rx_con)

            resultados["con_filtro"][mod_nombre].append({
                "sir_db": float(sir_db),
                "ber": mber_con.ber,
                "errores": mber_con.errores,
                "cota_tres": mber_con.cota_tres,
                "es_cero_errores": mber_con.es_cero_errores,
                "evm_pct": evm_pct_con,
                "evm_db": evm_db_con,
            })

    return {
        "resultados": resultados,
        "info_filtro": info_filtro,
        "info_filtro_bfsk": info_filtro_bfsk,
        "f_corte_hz": f_corte,
        "orden_filtro": orden_filtro,
        "banda_bfsk_hz": (f_baja_bfsk, f_alta_bfsk),
        "orden_bfsk": orden_bfsk,
        "fs_hz": fs,
        "delta_f_hz": delta_f,
        "isi_residual_limpia_16qam_pct": isi_limpia_16qam_pct,
    }


# =====================================================================
# 2. EXPERIMENTO 2: ECUALIZACIÓN ZF FRENTE A CANAL CON ISI
# =====================================================================

def ejecutar_experimento_2_ecualizacion(
    ebn0_valores_db: np.ndarray | None = None,
    h_canal: np.ndarray | None = None,
    longitud_w: int = 7,
    retardo_decision: int = 0,
    n_bits: int = 60000,
    seed_base: int = 6000
) -> dict[str, Any]:
    """
    Evalúa el desempeño antes y después del ecualizador FIR Zero-Forcing (ZF)
    frente a un canal multitrayecto dispersivo con interferencia intersimbólica (ISI):
    - Canal ISI: h = [1.0, 0.35] (camino principal + eco retardado 1 Ts).
    - Ecualizador ZF: FIR de longitud Nw=7 con retardo explicito Delta=0.
    - Modulaciones: BPSK, QPSK y 16-QAM.
    - Barrido: Eb/N0 de 0 a 16 dB (paso 2 dB).

    Retorna:
        dict con 'sin_ecualizar', 'con_ecualizador_zf' y metadatos del ecualizador.
    """
    if ebn0_valores_db is None:
        ebn0_valores_db = np.arange(0.0, 17.0, 2.0)
    if h_canal is None:
        h_canal = np.array([1.0, 0.35])

    # 1. Diseñar ecualizador ZF y verificar respuesta combinada y NEF
    info_zf = mitigacion.disenar_ecualizador_zf(
        h_canal=h_canal,
        longitud_w=longitud_w,
        retardo_decision=retardo_decision
    )
    w_zf = info_zf["coeficientes_w"]

    n_bits_ajustado = int(math.ceil(n_bits / 4.0) * 4)
    modulaciones_lista = ["BPSK", "QPSK", "16-QAM"]

    resultados: dict[str, dict[str, list[dict[str, Any]]]] = {
        "sin_ecualizar": {m: [] for m in modulaciones_lista},
        "con_ecualizador_zf": {m: [] for m in modulaciones_lista}
    }

    for m_idx, mod_nombre in enumerate(modulaciones_lista):
        bits_tx = fundamentos.generar_bits(n_bits_ajustado, semilla=seed_base + m_idx * 222)

        # Modulación limpia en espacio de símbolos discretos
        if mod_nombre == "BPSK":
            mod_res = modulaciones.modular_bpsk(bits_tx, sps=1, fs=1000.0, fc=1000.0, normalizar=True)
        elif mod_nombre == "QPSK":
            mod_res = modulaciones.modular_qpsk(bits_tx, sps=1, fs=1000.0, fc=1000.0, normalizar=True)
        elif mod_nombre in ("16-QAM", "16QAM"):
            mod_res = modulaciones.modular_16qam(bits_tx, sps=1, fs=1000.0, fc=1000.0, normalizar=True)
        else:
            continue

        simbolos_tx = mod_res.simbolos
        assert simbolos_tx is not None
        k_bits = mod_res.bits_por_simbolo

        for snr_idx, ebn0_db in enumerate(ebn0_valores_db):
            semilla_punto = seed_base + m_idx * 2000 + snr_idx * 17
            rng = np.random.default_rng(semilla_punto)

            # 1. Aplicar canal dispersivo ISI
            simbolos_isi = mitigacion.aplicar_canal_dispersivo_isi(simbolos_tx, h_canal)

            # 2. Agregar ruido AWGN según Eb/N0
            es_n0_lin = 10.0 ** (metricas.ebn0_a_esn0(ebn0_db, k_bits) / 10.0)
            n0 = 1.0 / es_n0_lin
            sig_comp = np.sqrt(n0 / 2.0)
            ruido = rng.normal(0.0, sig_comp, size=len(simbolos_isi)) + 1j * rng.normal(0.0, sig_comp, size=len(simbolos_isi))
            r_rx = simbolos_isi + ruido

            # CASO 1: SIN ECUALIZACIÓN (Antes)
            if mod_nombre == "BPSK":
                bits_sin = demodulacion.demodular_bpsk(r_rx)
            elif mod_nombre == "QPSK":
                bits_sin = demodulacion.demodular_qpsk(r_rx)
            elif mod_nombre in ("16-QAM", "16QAM"):
                bits_sin = demodulacion.demodular_16qam(r_rx)

            mber_sin = metricas.calcular_ber_con_intervalo(bits_tx, bits_sin)
            evm_sin, _ = metricas.calcular_evm_simbolos(simbolos_tx, r_rx, mod_nombre)

            resultados["sin_ecualizar"][mod_nombre].append({
                "ebn0_db": float(ebn0_db),
                "ber": mber_sin.ber,
                "errores": mber_sin.errores,
                "cota_tres": mber_sin.cota_tres,
                "es_cero_errores": mber_sin.es_cero_errores,
                "evm_pct": evm_sin,
            })

            # CASO 2: CON ECUALIZADOR ZF (Después)
            s_eq = mitigacion.aplicar_ecualizador_zf(
                simbolos_rx=r_rx,
                coeficientes_w=w_zf,
                retardo_decision=retardo_decision
            )

            if mod_nombre == "BPSK":
                bits_con = demodulacion.demodular_bpsk(s_eq)
            elif mod_nombre == "QPSK":
                bits_con = demodulacion.demodular_qpsk(s_eq)
            elif mod_nombre in ("16-QAM", "16QAM"):
                bits_con = demodulacion.demodular_16qam(s_eq)

            mber_con = metricas.calcular_ber_con_intervalo(bits_tx, bits_con)
            evm_con, _ = metricas.calcular_evm_simbolos(simbolos_tx, s_eq, mod_nombre)

            resultados["con_ecualizador_zf"][mod_nombre].append({
                "ebn0_db": float(ebn0_db),
                "ber": mber_con.ber,
                "errores": mber_con.errores,
                "cota_tres": mber_con.cota_tres,
                "es_cero_errores": mber_con.es_cero_errores,
                "evm_pct": evm_con,
            })

    return {
        "resultados": resultados,
        "info_zf": info_zf,
        "h_canal": list(np.real(h_canal))
    }


# =====================================================================
# 3. EXPERIMENTO 3: DIVERSIDAD DE RECEPCIÓN (SISO VS SC VS MRC)
# =====================================================================

def ejecutar_experimento_3_diversidad(
    ebn0_valores_db: np.ndarray | None = None,
    n_bits: int = 60000,
    seed_base: int = 7000
) -> dict[str, Any]:
    """
    Evalúa la ganancia de diversidad en canal Rayleigh plano independiente:
    - Transmisión con potencia comparable (Es = 1.0) para SISO y SIMO.
    - Ramas Rayleigh i.i.d. y ruido AWGN independiente en cada antena.
    - Comparación:
      1. SISO (L=1): Una sola rama de recepción con ecualización de canal.
      2. SC (L=2): Selección de la mejor antena por símbolo.
      3. MRC (L=2): Combinación óptima de máxima relación.
    - Modulaciones: BPSK y QPSK.
    - Barrido: Eb/N0 medio de 0 a 18 dB (paso 2 dB).

    Retorna:
        dict con resultados empíricos reales de BER para 'SISO', 'SC' y 'MRC'.
    """
    if ebn0_valores_db is None:
        ebn0_valores_db = np.arange(0.0, 19.0, 2.0)

    n_bits_ajustado = int(math.ceil(n_bits / 4.0) * 4)
    modulaciones_lista = ["BPSK", "QPSK"]

    resultados: dict[str, dict[str, list[dict[str, Any]]]] = {
        "SISO": {m: [] for m in modulaciones_lista},
        "SC": {m: [] for m in modulaciones_lista},
        "MRC": {m: [] for m in modulaciones_lista}
    }

    for m_idx, mod_nombre in enumerate(modulaciones_lista):
        bits_tx = fundamentos.generar_bits(n_bits_ajustado, semilla=seed_base + m_idx * 333)

        if mod_nombre == "BPSK":
            mod_res = modulaciones.modular_bpsk(bits_tx, sps=1, fs=1000.0, fc=1000.0, normalizar=True)
        else:
            mod_res = modulaciones.modular_qpsk(bits_tx, sps=1, fs=1000.0, fc=1000.0, normalizar=True)

        simbolos_tx = mod_res.simbolos
        assert simbolos_tx is not None
        n_sim = len(simbolos_tx)
        k_bits = mod_res.bits_por_simbolo

        for snr_idx, ebn0_db in enumerate(ebn0_valores_db):
            semilla_punto = seed_base + m_idx * 3000 + snr_idx * 19

            # 1. Generar 2 canales Rayleigh independientes e idénticamente distribuidos
            h_ramas = mitigacion.generar_canales_rayleigh_independientes(
                n_simbolos=n_sim,
                n_ramas=2,
                seed=semilla_punto
            )

            # 2. Transmisión SIMO bajo igual potencia y ruido independiente
            es_n0_lin = 10.0 ** (metricas.ebn0_a_esn0(ebn0_db, k_bits) / 10.0)
            r_ramas, _ = mitigacion.transmitir_simo_rayleigh(
                simbolos_tx=simbolos_tx,
                canales_h=h_ramas,
                snr_lineal=es_n0_lin,
                seed=semilla_punto + 888
            )

            # CASO A: SISO (Rama 1 únicamente)
            y_siso = mitigacion.demodular_siso_rayleigh(r_ramas[0], h_ramas[0])
            bits_siso = demodulacion.demodular_bpsk(y_siso) if mod_nombre == "BPSK" else demodulacion.demodular_qpsk(y_siso)
            mber_siso = metricas.calcular_ber_con_intervalo(bits_tx, bits_siso)

            resultados["SISO"][mod_nombre].append({
                "ebn0_db": float(ebn0_db),
                "ber": mber_siso.ber,
                "errores": mber_siso.errores,
                "cota_tres": mber_siso.cota_tres,
                "es_cero_errores": mber_siso.es_cero_errores
            })

            # CASO B: SELECTION COMBINING (SC, L=2)
            y_sc = mitigacion.combinar_diversidad_sc(r_ramas, h_ramas)
            bits_sc = demodulacion.demodular_bpsk(y_sc) if mod_nombre == "BPSK" else demodulacion.demodular_qpsk(y_sc)
            mber_sc = metricas.calcular_ber_con_intervalo(bits_tx, bits_sc)

            resultados["SC"][mod_nombre].append({
                "ebn0_db": float(ebn0_db),
                "ber": mber_sc.ber,
                "errores": mber_sc.errores,
                "cota_tres": mber_sc.cota_tres,
                "es_cero_errores": mber_sc.es_cero_errores
            })

            # CASO C: MAXIMAL RATIO COMBINING (MRC, L=2)
            y_mrc = mitigacion.combinar_diversidad_mrc(r_ramas, h_ramas)
            bits_mrc = demodulacion.demodular_bpsk(y_mrc) if mod_nombre == "BPSK" else demodulacion.demodular_qpsk(y_mrc)
            mber_mrc = metricas.calcular_ber_con_intervalo(bits_tx, bits_mrc)

            resultados["MRC"][mod_nombre].append({
                "ebn0_db": float(ebn0_db),
                "ber": mber_mrc.ber,
                "errores": mber_mrc.errores,
                "cota_tres": mber_mrc.cota_tres,
                "es_cero_errores": mber_mrc.es_cero_errores
            })

    return {
        "resultados": resultados,
        "n_ramas": 2,
        "n_bits": n_bits_ajustado
    }

