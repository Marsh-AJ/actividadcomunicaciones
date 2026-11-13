"""
experimentos.py
Motor de Simulación Experimental Monte Carlo para Evaluación de Modulaciones Digitales.

Desarrollado para la asignatura Electiva de Comunicaciones Móviles (Python 3.13.5).
Fase 6: Métricas y Evaluación del Rendimiento de las Modulaciones.

Experimentos implementados:
1. Curvas de BER y EVM vs Eb/N0 en canal AWGN (comparación con modelos teóricos).
2. Impacto del ruido térmico físico (kTB) según potencia recibida Prx [dBm] (50 Ω, 290 K, 2 kHz, NF 4 dB).
3. Sensibilidad frente a Interferencia Cocanal (CCI) con retardo, fase y fondo AWGN.
4. Degradación por ruido de fase en oscilador local e Interferencia de Canal Adyacente (ACI).
5. Matriz de compromiso global (Trade-off) multidimensional entre modulaciones.
"""

from __future__ import annotations
import os
import math
from typing import Any
import numpy as np

import fundamentos
import modulaciones
import demodulacion
import ruido
import interferencias
import efectos_canal
import canales
import metricas


# Parámetros globales por defecto del banco de pruebas
SPS_DEFECTO = 16
FS_DEFECTO = 16000.0
FC_DEFECTO = 2000.0
# Tonos estándar obligatorios BFSK aprobados en Fases 3 y 4
F0_BFSK_DEFECTO = 2000.0
F1_BFSK_DEFECTO = 4000.0


# =====================================================================
# 1. TRANSMISIÓN Y PERTURBACIÓN CALIBRADA PUNTO A PUNTO
# =====================================================================

def ejecutar_transmision_calibrada(
    bits: np.ndarray,
    tipo_mod: str,
    tipo_canal: str,
    params_canal: dict[str, Any],
    sps: int = SPS_DEFECTO,
    fs: float = FS_DEFECTO,
    fc: float = FC_DEFECTO,
    seed: int | None = None
) -> dict[str, Any]:
    """
    Ejecuta una transmisión completa punto a punto:
        Bits TX -> Modulador -> Canal Perturbado Calibrado -> Demodulador -> Bits RX
    calculando BER con intervalos Wilson, cota de tres y EVM a nivel de símbolo.

    Parámetros:
        bits: Arreglo de bits binarios (0 y 1).
        tipo_mod: 'BFSK', 'BPSK', 'QPSK', '16-QAM'.
        tipo_canal: 'awgn', 'termico', 'cocanal', 'aci', 'ruido_fase'.
        params_canal: Diccionario con parámetros específicos del canal.
        sps: Muestras por símbolo.
        fs: Frecuencia de muestreo [Hz].
        fc: Frecuencia de portadora [Hz].
        seed: Semilla para reproducibilidad estricta.

    Retorna:
        dict con bits_tx, bits_rx, metricas_ber, evm_pct, evm_db, mod_res, res_pert.
    """
    mod_norm = str(tipo_mod).strip().upper()
    canal_norm = str(tipo_canal).strip().lower()

    # 1. Modulación
    if mod_norm == "BFSK":
        mod_res = modulaciones.modular_bfsk(
            bits,
            f0=F0_BFSK_DEFECTO,
            f1=F1_BFSK_DEFECTO,
            sps=sps,
            fs=fs,
            fase_continua=True
        )
    elif mod_norm == "BPSK":
        mod_res = modulaciones.modular_bpsk(bits, fc=fc, sps=sps, fs=fs, normalizar=True)
    elif mod_norm == "QPSK":
        mod_res = modulaciones.modular_qpsk(bits, fc=fc, sps=sps, fs=fs, normalizar=True)
    elif mod_norm in ("16-QAM", "16QAM"):
        mod_res = modulaciones.modular_16qam(bits, fc=fc, sps=sps, fs=fs, normalizar=True)
    else:
        raise ValueError(f"Modulación '{tipo_mod}' no soportada.")

    # 2. Inyección calibrada de perturbación
    if canal_norm == "awgn":
        ebn0_db = float(params_canal.get("ebn0_db", 10.0))
        k = mod_res.bits_por_simbolo
        es_n0_db = metricas.ebn0_a_esn0(ebn0_db, k)
        # canales.aplicar_canal recibe snr_db (convención aprobada: snr_db = Es/N0 en dB)
        res_pert = canales.aplicar_canal(mod_res, "awgn", snr_db=es_n0_db, seed=seed)

    elif canal_norm == "termico":
        p_rx_dbm = float(params_canal.get("potencia_rx_dbm", -80.0))
        temp_k = float(params_canal.get("temperatura_k", 290.0))
        bw_hz = float(params_canal.get("ancho_banda_hz", 2000.0))
        nf_db = float(params_canal.get("figura_ruido_db", 4.0))

        # 1. Parámetros físicos rigurosos: P_N,fis = k · T · B · F [W]
        metricas_ktb = ruido.calcular_ruido_termico_ktb(
            temperatura_k=temp_k,
            ancho_banda_hz=bw_hz,
            figura_ruido_db=nf_db
        )
        p_n_fis_dbm = metricas_ktb["potencia_ruido_dbm"]  # ≈ -136.965 dBm
        snr_fis_db = p_rx_dbm - p_n_fis_dbm

        # 2. Relación de energía de símbolo a densidad de ruido (Es/N0):
        # Es / N0 = (Prx / Rs) / (PN / B) = (Prx / PN) · (B / Rs) = SNR_fis · (B / Rs)
        rs = fs / float(sps)  # Tasa de símbolos de la simulación (1000.0 baudios)
        factor_bw_rs_db = 10.0 * np.log10(bw_hz / rs)  # 10·log10(2000 / 1000) = +3.0103 dB
        es_n0_termico_db = snr_fis_db + factor_bw_rs_db

        # 3. Inyección en el simulador mediante el despachador AWGN validado:
        # Esto asegura la varianza de banda base σ_BB² = Nsps · N0 y RF σ_RF² = Nsps · N0 / 4,
        # compensando exactamente el promediador 1/Nsps para que a la entrada de decisión
        # la varianza por rama sea exactamente N0 / 2, coincidiendo con la física de radio.
        res_pert = canales.aplicar_canal(mod_res, "awgn", snr_db=es_n0_termico_db, seed=seed)

    elif canal_norm in ("cocanal", "cci"):
        sir_db = float(params_canal.get("sir_db", 10.0))
        retardo = int(params_canal.get("retardo_muestras", 7))
        fase = float(params_canal.get("fase_rad", np.pi / 4.0))
        ebn0_fondo_db = params_canal.get("ebn0_fondo_db", None)

        # Generar secuencia de bits completamente independiente para el interferente
        seed_interf = (seed + 55555) if seed is not None else None
        bits_interf = fundamentos.generar_bits(len(bits), semilla=seed_interf)

        # Modular la señal interferente con el mismo esquema y parámetros
        if mod_norm == "BFSK":
            mod_int = modulaciones.modular_bfsk(
                bits_interf, f0=F0_BFSK_DEFECTO, f1=F1_BFSK_DEFECTO, sps=sps, fs=fs, fase_continua=True
            )
            s_interf = mod_int.senal_rf
        elif mod_norm == "BPSK":
            mod_int = modulaciones.modular_bpsk(bits_interf, fc=fc, sps=sps, fs=fs, normalizar=True)
            s_interf = mod_int.senal_bb
        elif mod_norm == "QPSK":
            mod_int = modulaciones.modular_qpsk(bits_interf, fc=fc, sps=sps, fs=fs, normalizar=True)
            s_interf = mod_int.senal_bb
        elif mod_norm in ("16-QAM", "16QAM"):
            mod_int = modulaciones.modular_16qam(bits_interf, fc=fc, sps=sps, fs=fs, normalizar=True)
            s_interf = mod_int.senal_bb

        res_cci = canales.aplicar_canal(
            mod_res,
            "cocanal",
            sir_db=sir_db,
            retardo_muestras=retardo,
            fase_rad=fase,
            senal_interferente=s_interf,
            seed=seed
        )

        if ebn0_fondo_db is not None:
            k = mod_res.bits_por_simbolo
            es_n0_fondo = metricas.ebn0_a_esn0(float(ebn0_fondo_db), k)
            seed_fondo = (seed + 9999) if seed is not None else None
            res_pert = canales.aplicar_canal(res_cci, "awgn", snr_db=es_n0_fondo, seed=seed_fondo)
        else:
            res_pert = res_cci

    elif canal_norm in ("ruido_fase", "fase"):
        if mod_norm == "BFSK":
            raise ValueError("Ruido de fase en constelación no aplica a BFSK.")
        desv_grados = float(params_canal.get("desviacion_grados", 5.0))
        ebn0_fondo_db = params_canal.get("ebn0_fondo_db", None)

        res_fase = canales.aplicar_canal(mod_res, "ruido_fase", desviacion_grados=desv_grados, seed=seed)

        if ebn0_fondo_db is not None:
            k = mod_res.bits_por_simbolo
            es_n0_fondo = metricas.ebn0_a_esn0(float(ebn0_fondo_db), k)
            seed_fondo = (seed + 8888) if seed is not None else None
            res_pert = canales.aplicar_canal(res_fase, "awgn", snr_db=es_n0_fondo, seed=seed_fondo)
        else:
            res_pert = res_fase

    elif canal_norm == "aci":
        delta_f = float(params_canal.get("delta_f", 1500.0))
        sir_db = float(params_canal.get("sir_db", 10.0))
        fase_ini = float(params_canal.get("fase_inicial", 0.0))
        ebn0_fondo_db = params_canal.get("ebn0_fondo_db", None)

        res_aci = canales.aplicar_canal(
            mod_res,
            "aci",
            delta_f=delta_f,
            sir_db=sir_db,
            fase_inicial=fase_ini,
            seed=seed
        )

        if ebn0_fondo_db is not None:
            k = mod_res.bits_por_simbolo
            es_n0_fondo = metricas.ebn0_a_esn0(float(ebn0_fondo_db), k)
            seed_fondo = (seed + 7777) if seed is not None else None
            res_pert = canales.aplicar_canal(res_aci, "awgn", snr_db=es_n0_fondo, seed=seed_fondo)
        else:
            res_pert = res_aci

    else:
        raise ValueError(f"Canal '{tipo_canal}' no soportado en transmisión calibrada.")

    # 3. Demodulación
    if mod_norm == "BFSK":
        bits_rx = canales.demodular_resultado_perturbado(
            res_pert,
            f0=F0_BFSK_DEFECTO,
            f1=F1_BFSK_DEFECTO
        )
    else:
        bits_rx = canales.demodular_resultado_perturbado(res_pert)

    # 4. Cálculo de métricas
    metricas_ber = metricas.calcular_ber_con_intervalo(bits, bits_rx)
    evm_pct, evm_db = metricas.calcular_evm_simbolos(
        mod_res.simbolos,
        res_pert.simbolos_pert,
        mod_norm
    )

    return {
        "modulacion": mod_norm,
        "canal": canal_norm,
        "bits_tx": bits,
        "bits_rx": bits_rx,
        "metricas_ber": metricas_ber,
        "evm_pct": evm_pct,
        "evm_db": evm_db,
        "mod_res": mod_res,
        "res_pert": res_pert
    }


# =====================================================================
# 2. EXPERIMENTO 1: CURVAS DE BER Y EVM VS Eb/N0 EN CANAL AWGN
# =====================================================================

def ejecutar_experimento_1_awgn(
    ebn0_valores_db: np.ndarray | None = None,
    n_bits: int = 60000,
    seed_base: int = 1000
) -> dict[str, list[dict[str, Any]]]:
    """
    Ejecuta el barrido de Eb/N0 en canal AWGN para las cuatro modulaciones.
    Rango por defecto: 0 a 12 dB (paso 1 dB).
    Calcula BER empírica, cotas de cero errores, EVM y curvas teóricas.
    """
    if ebn0_valores_db is None:
        ebn0_valores_db = np.arange(0.0, 13.0, 1.0)

    # Ajustar n_bits a múltiplo de 4 para compatibilidad exacta con 16-QAM y QPSK
    n_bits_ajustado = int(math.ceil(n_bits / 4.0) * 4)
    modulaciones_lista = ["BFSK", "BPSK", "QPSK", "16-QAM"]
    resultados: dict[str, list[dict[str, Any]]] = {m: [] for m in modulaciones_lista}

    for m_idx, mod_nombre in enumerate(modulaciones_lista):
        # Semilla determinista pero independiente por modulación
        bits_tx = fundamentos.generar_bits(n_bits_ajustado, semilla=seed_base + m_idx * 100)

        for snr_idx, ebn0_db in enumerate(ebn0_valores_db):
            semilla_punto = seed_base + m_idx * 1000 + snr_idx * 17
            punto = ejecutar_transmision_calibrada(
                bits=bits_tx,
                tipo_mod=mod_nombre,
                tipo_canal="awgn",
                params_canal={"ebn0_db": float(ebn0_db)},
                seed=semilla_punto
            )

            # Cálculo teórico correspondiente
            if mod_nombre in ("BPSK", "QPSK"):
                ber_teorica = float(metricas.ber_teorica_bpsk_qpsk(ebn0_db))
                etiqueta_teorica = "BPSK/QPSK Teórica (Exacta)"
            elif mod_nombre == "16-QAM":
                ber_teorica = float(metricas.ber_teorica_16qam_aproximada(ebn0_db))
                etiqueta_teorica = "BER_teorica_aproximada"
            elif mod_nombre == "BFSK":
                ber_teorica = float(metricas.ber_teorica_bfsk_no_coherente(ebn0_db))
                etiqueta_teorica = "BFSK Teórica No Coherente (Ortogonal)"
            else:
                ber_teorica = 0.0
                etiqueta_teorica = "N/A"

            mber = punto["metricas_ber"]
            resultados[mod_nombre].append({
                "ebn0_db": float(ebn0_db),
                "esn0_db": float(metricas.ebn0_a_esn0(float(ebn0_db), punto["mod_res"].bits_por_simbolo)),
                "ber_simulada": mber.ber,
                "errores": mber.errores,
                "total_bits": mber.total_bits,
                "wilson_lower": mber.wilson_lower,
                "wilson_upper": mber.wilson_upper,
                "cota_tres": mber.cota_tres,
                "es_cero_errores": mber.es_cero_errores,
                "ber_teorica": ber_teorica,
                "etiqueta_teorica": etiqueta_teorica,
                "evm_pct": punto["evm_pct"],
                "evm_db": punto["evm_db"],
            })

    return resultados


# =====================================================================
# 3. EXPERIMENTO 2: IMPACTO DEL RUIDO TÉRMICO FÍSICO (N = kTB)
# =====================================================================

def ejecutar_experimento_2_ruido_termico(
    pot_rx_valores_dbm: np.ndarray | None = None,
    n_bits: int = 60000,
    seed_base: int = 2000
) -> dict[str, list[dict[str, Any]]]:
    """
    Ejecuta el barrido de potencia recibida Prx en canal con ruido térmico físico
    (50 Ω, 290 K, 2 kHz, NF = 4 dB).
    Rango por defecto: -135 a -115 dBm (paso 2 dBm).
    """
    if pot_rx_valores_dbm is None:
        pot_rx_valores_dbm = np.arange(-135.0, -114.0, 2.0)

    n_bits_ajustado = int(math.ceil(n_bits / 4.0) * 4)
    modulaciones_lista = ["BFSK", "BPSK", "QPSK", "16-QAM"]
    resultados: dict[str, list[dict[str, Any]]] = {m: [] for m in modulaciones_lista}

    for m_idx, mod_nombre in enumerate(modulaciones_lista):
        bits_tx = fundamentos.generar_bits(n_bits_ajustado, semilla=seed_base + m_idx * 150)

        for prx_idx, prx_dbm in enumerate(pot_rx_valores_dbm):
            semilla_punto = seed_base + m_idx * 2000 + prx_idx * 19
            punto = ejecutar_transmision_calibrada(
                bits=bits_tx,
                tipo_mod=mod_nombre,
                tipo_canal="termico",
                params_canal={
                    "potencia_rx_dbm": float(prx_dbm),
                    "temperatura_k": 290.0,
                    "ancho_banda_hz": 2000.0,
                    "figura_ruido_db": 4.0,
                },
                seed=semilla_punto
            )

            mber = punto["metricas_ber"]
            resultados[mod_nombre].append({
                "potencia_rx_dbm": float(prx_dbm),
                "temperatura_k": 290.0,
                "ancho_banda_hz": 2000.0,
                "figura_ruido_db": 4.0,
                "ber_simulada": mber.ber,
                "errores": mber.errores,
                "total_bits": mber.total_bits,
                "wilson_lower": mber.wilson_lower,
                "wilson_upper": mber.wilson_upper,
                "cota_tres": mber.cota_tres,
                "es_cero_errores": mber.es_cero_errores,
                "evm_pct": punto["evm_pct"],
                "evm_db": punto["evm_db"],
            })

    return resultados


# =====================================================================
# 4. EXPERIMENTO 3: SENSIBILIDAD A INTERFERENCIA COCANAL (CCI)
# =====================================================================

def ejecutar_experimento_3_interferencia_cci(
    sir_valores_db: np.ndarray | None = None,
    n_bits: int = 60000,
    seed_base: int = 3000
) -> dict[str, list[dict[str, Any]]]:
    """
    Ejecuta el barrido de Relación Señal a Interferencia Cocanal (SIR)
    con réplica retardada en 7 muestras (0.35 Ts), desfase de pi/4 y fondo AWGN (Eb/N0 = 15 dB).
    Rango por defecto: 0 a 20 dB (paso 2 dB).
    """
    if sir_valores_db is None:
        sir_valores_db = np.arange(0.0, 21.0, 2.0)

    n_bits_ajustado = int(math.ceil(n_bits / 4.0) * 4)
    modulaciones_lista = ["BFSK", "BPSK", "QPSK", "16-QAM"]
    resultados: dict[str, list[dict[str, Any]]] = {m: [] for m in modulaciones_lista}

    for m_idx, mod_nombre in enumerate(modulaciones_lista):
        bits_tx = fundamentos.generar_bits(n_bits_ajustado, semilla=seed_base + m_idx * 200)

        for s_idx, sir_db in enumerate(sir_valores_db):
            semilla_punto = seed_base + m_idx * 3000 + s_idx * 23
            punto = ejecutar_transmision_calibrada(
                bits=bits_tx,
                tipo_mod=mod_nombre,
                tipo_canal="cocanal",
                params_canal={
                    "sir_db": float(sir_db),
                    "retardo_muestras": 7,
                    "fase_rad": float(np.pi / 4.0),
                    "ebn0_fondo_db": 15.0,
                },
                seed=semilla_punto
            )

            mber = punto["metricas_ber"]
            resultados[mod_nombre].append({
                "sir_db": float(sir_db),
                "ebn0_fondo_db": 15.0,
                "ber_simulada": mber.ber,
                "errores": mber.errores,
                "total_bits": mber.total_bits,
                "wilson_lower": mber.wilson_lower,
                "wilson_upper": mber.wilson_upper,
                "cota_tres": mber.cota_tres,
                "es_cero_errores": mber.es_cero_errores,
                "evm_pct": punto["evm_pct"],
                "evm_db": punto["evm_db"],
            })

    return resultados


# =====================================================================
# 5. EXPERIMENTO 4: DEGRADACIÓN POR RUIDO DE FASE Y CANAL ADYACENTE (ACI)
# =====================================================================

def ejecutar_experimento_4_aci_y_fase(
    desv_fase_valores: np.ndarray | None = None,
    sir_aci_valores: np.ndarray | None = None,
    n_bits: int = 60000,
    seed_base: int = 4000
) -> dict[str, Any]:
    """
    Ejecuta las dos ramas de perturbación de fase y selectividad de frecuencia:
    - Rama 4A: Ruido de fase angular con sigma_theta en [0, 25] grados (BPSK, QPSK, 16-QAM; BFSK es N/A).
    - Rama 4B: Interferencia ACI con Delta_f = 1500 Hz y SIR en [0, 16] dB (todas las modulaciones).
    """
    if desv_fase_valores is None:
        desv_fase_valores = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    if sir_aci_valores is None:
        sir_aci_valores = np.arange(0.0, 17.0, 2.0)

    n_bits_ajustado = int(math.ceil(n_bits / 4.0) * 4)

    # 4A. Ruido de Fase (solo modulaciones lineales)
    res_fase: dict[str, list[dict[str, Any]]] = {"BPSK": [], "QPSK": [], "16-QAM": []}
    for m_idx, mod_nombre in enumerate(["BPSK", "QPSK", "16-QAM"]):
        bits_tx = fundamentos.generar_bits(n_bits_ajustado, semilla=seed_base + m_idx * 300)
        for f_idx, desv in enumerate(desv_fase_valores):
            semilla_punto = seed_base + m_idx * 4000 + f_idx * 29
            punto = ejecutar_transmision_calibrada(
                bits=bits_tx,
                tipo_mod=mod_nombre,
                tipo_canal="ruido_fase",
                params_canal={
                    "desviacion_grados": float(desv),
                    "ebn0_fondo_db": 12.0
                },
                seed=semilla_punto
            )
            mber = punto["metricas_ber"]
            res_fase[mod_nombre].append({
                "desviacion_grados": float(desv),
                "ebn0_fondo_db": 12.0,
                "ber_simulada": mber.ber,
                "errores": mber.errores,
                "total_bits": mber.total_bits,
                "wilson_upper": mber.wilson_upper,
                "cota_tres": mber.cota_tres,
                "es_cero_errores": mber.es_cero_errores,
                "evm_pct": punto["evm_pct"],
                "evm_db": punto["evm_db"],
            })

    # 4B. Interferencia ACI (las cuatro modulaciones)
    modulaciones_todas = ["BFSK", "BPSK", "QPSK", "16-QAM"]
    res_aci: dict[str, list[dict[str, Any]]] = {m: [] for m in modulaciones_todas}
    for m_idx, mod_nombre in enumerate(modulaciones_todas):
        bits_tx = fundamentos.generar_bits(n_bits_ajustado, semilla=seed_base + 1000 + m_idx * 250)
        for a_idx, sir_db in enumerate(sir_aci_valores):
            semilla_punto = seed_base + 5000 + m_idx * 1500 + a_idx * 31
            punto = ejecutar_transmision_calibrada(
                bits=bits_tx,
                tipo_mod=mod_nombre,
                tipo_canal="aci",
                params_canal={
                    "delta_f": 1500.0,
                    "sir_db": float(sir_db),
                    "fase_inicial": 0.0,
                    "ebn0_fondo_db": 15.0
                },
                seed=semilla_punto
            )
            mber = punto["metricas_ber"]
            res_aci[mod_nombre].append({
                "sir_aci_db": float(sir_db),
                "delta_f_hz": 1500.0,
                "ebn0_fondo_db": 15.0,
                "ber_simulada": mber.ber,
                "errores": mber.errores,
                "total_bits": mber.total_bits,
                "wilson_upper": mber.wilson_upper,
                "cota_tres": mber.cota_tres,
                "es_cero_errores": mber.es_cero_errores,
                "evm_pct": punto["evm_pct"],
                "evm_db": punto["evm_db"],
            })

    return {
        "ruido_fase": res_fase,
        "aci": res_aci
    }


# =====================================================================
# 6. EXPERIMENTO 5: MATRIZ DE COMPROMISO GLOBAL (TRADE-OFF)
# =====================================================================

def consolidar_matriz_tradeoff(
    res_exp1: dict[str, list[dict[str, Any]]],
    res_exp2: dict[str, list[dict[str, Any]]],
    res_exp3: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """
    Consolida la matriz global de trade-off comparando las cuatro modulaciones:
    - Eficiencia espectral (convención Carson y nulo a nulo).
    - Eb/N0 requerido para BER <= 1e-3.
    - Sensibilidad de recepción Prx [dBm] para BER <= 1e-3 (ruido térmico).
    - Tolerancia CCI (SIR mínima para BER <= 1e-2).
    - Complejidad computacional por símbolo (FLOPs/símbolo).
    - Calificación cualitativa de robustez.
    """
    modulaciones_lista = ["BFSK", "BPSK", "QPSK", "16-QAM"]
    matriz: list[dict[str, Any]] = []

    for mod_nombre in modulaciones_lista:
        info_eficiencia = metricas.calcular_eficiencia_espectral(mod_nombre)
        info_complejidad = metricas.estimar_complejidad_operacional(mod_nombre)

        # 1. Eb/N0 para BER <= 1e-3 (Exp 1)
        puntos_awgn = res_exp1[mod_nombre]
        ebn0_req = None
        for p in puntos_awgn:
            if p["ber_simulada"] <= 1e-3:
                ebn0_req = p["ebn0_db"]
                break
        if ebn0_req is None:
            ebn0_req = float("> 12.0")

        # 2. Sensibilidad Prx para BER <= 1e-3 (Exp 2)
        puntos_term = res_exp2[mod_nombre]
        prx_req = None
        for p in puntos_term:
            if p["ber_simulada"] <= 1e-3:
                prx_req = p["potencia_rx_dbm"]
                break
        if prx_req is None:
            prx_req = float("> -115.0")

        # 3. Tolerancia CCI: SIR mínima para BER <= 1e-2 (Exp 3)
        puntos_cci = res_exp3[mod_nombre]
        sir_req = None
        for p in puntos_cci:
            if p["ber_simulada"] <= 1e-2:
                sir_req = p["sir_db"]
                break
        if sir_req is None:
            sir_req = float("> 20.0")

        # Robustez cualitativa
        if mod_nombre in ("BPSK", "QPSK"):
            robustez = "Excelente inmunidad al ruido y coherencia de fase"
        elif mod_nombre == "BFSK":
            robustez = "Excelente robustez no coherente (inmune a ruido de fase)"
        else:
            robustez = "Sensible a ruido y no linealidad; alta tasa de datos"

        matriz.append({
            "modulacion": mod_nombre,
            "bits_por_simbolo": info_eficiencia["bits_por_simbolo"],
            "eficiencia_principal_bps_hz": info_eficiencia["eficiencia_principal_bps_hz"],
            "convencion_ancho_banda": info_eficiencia["convencion_principal"],
            "ancho_banda_hz": info_eficiencia["ancho_banda_principal_hz"],
            "ebn0_req_ber_1e3_db": ebn0_req,
            "sensibilidad_prx_dbm": prx_req,
            "tolerancia_cci_sir_db": sir_req,
            "complejidad_ops_simbolo": info_complejidad["ops_símbolo"],
            "complejidad_ops_bit": info_complejidad["ops_bit"],
            "dominio_receptor": info_complejidad["dominio_receptor"],
            "evaluacion_robustez": robustez
        })

    return matriz

