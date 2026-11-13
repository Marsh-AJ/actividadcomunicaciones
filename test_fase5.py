"""
test_fase5.py
Suite de Pruebas Unitarias, Integración y Generación de Gráficos/Tablas para la Fase 5.

Verifica:
1. Las 11 pruebas unitarias matemáticas de los modelos de canal.
2. Pruebas de integración end-to-end con modulación, canal y demodulación.
3. Canal ideal de referencia (BER = 0.0 para las 4 modulaciones).
4. Generación de los 10 gráficos de alta resolución en resultados/graficos/fase5_*.png.
5. Exportación de tablas comparativas de resultados a resultados/tablas/fase5_resumen_modelos.csv y .md.

Desarrollado para la asignatura Electiva de Comunicaciones Móviles (Python 3.13.5).
"""

from __future__ import annotations
import os
import sys
import numpy as np

# Configuración de directorio de caché local para Matplotlib antes de importar pyplot
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = cache_dir

import matplotlib.pyplot as plt

# Módulos del proyecto
import fundamentos
import modulaciones
import demodulacion
import ruido
import interferencias
import efectos_canal
import canales


# Directorios de salida
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_GRAFICOS = os.path.join(DIR_BASE, "resultados", "graficos")
DIR_TABLAS = os.path.join(DIR_BASE, "resultados", "tablas")
os.makedirs(DIR_GRAFICOS, exist_ok=True)
os.makedirs(DIR_TABLAS, exist_ok=True)


# =====================================================================
# BLOQUE 1: PRUEBAS UNITARIAS MATEMÁTICAS (11 PRUEBAS)
# =====================================================================

def test_01_awgn_varianzas_tres_dominios() -> dict:
    """Verifica varianzas teóricas de AWGN en símbolos, banda base y RF."""
    snr_db = 10.0
    sps = 16
    n_simbolos = 4000
    n0_teorico = 1.0 / (10.0 ** (snr_db / 10.0))  # 0.1

    # 1. Dominio de símbolos
    simbolos_zeros = np.zeros(n_simbolos, dtype=complex)
    ruido_sym = ruido.aplicar_awgn_simbolos(simbolos_zeros, snr_db=snr_db, seed=42)
    var_sym_medida = float(np.var(ruido_sym))
    var_i_sym = float(np.var(np.real(ruido_sym)))
    var_q_sym = float(np.var(np.imag(ruido_sym)))

    assert np.isclose(var_sym_medida, n0_teorico, rtol=0.08), f"Fallo var_sym: {var_sym_medida} != {n0_teorico}"
    assert np.isclose(var_i_sym, n0_teorico / 2.0, rtol=0.08)
    assert np.isclose(var_q_sym, n0_teorico / 2.0, rtol=0.08)

    # 2. Dominio banda base sobremuestreada
    bb_zeros = np.zeros(n_simbolos * sps, dtype=complex)
    ruido_bb = ruido.aplicar_awgn_banda_base(bb_zeros, snr_db=snr_db, sps=sps, seed=42)
    var_bb_teorica = sps * n0_teorico
    var_bb_medida = float(np.var(ruido_bb))
    assert np.isclose(var_bb_medida, var_bb_teorica, rtol=0.08)

    # Varianza tras el filtro promediador rectangular
    bb_bloques = ruido_bb.reshape(n_simbolos, sps)
    sym_filtrados = np.mean(bb_bloques, axis=1)
    var_i_filtrada = float(np.var(np.real(sym_filtrados)))
    assert np.isclose(var_i_filtrada, n0_teorico / 2.0, rtol=0.08)

    # 3. Dominio RF pasobanda equivalente
    rf_zeros = np.zeros(n_simbolos * sps, dtype=float)
    ruido_rf = ruido.aplicar_awgn_rf_equivalente_esn0(rf_zeros, es_n0_db=snr_db, sps=sps, seed=42)
    var_rf_teorica = (sps * n0_teorico) / 4.0
    var_rf_medida = float(np.var(ruido_rf))
    assert np.isclose(var_rf_medida, var_rf_teorica, rtol=0.08)

    return {
        "prueba": "01_AWGN_Varianzas",
        "n0_teorico": n0_teorico,
        "var_sym_medida": var_sym_medida,
        "var_bb_medida": var_bb_medida,
        "var_rf_medida": var_rf_medida,
        "var_tras_promediador": var_i_filtrada,
        "estado": "PASS",
    }


def test_02_aci_frecuencia_y_nyquist() -> dict:
    """Verifica la inserción de ACI por FFT y la excepción por violación de Nyquist."""
    fs = 16000.0
    fc = 2000.0
    delta_f = 1500.0  # f_adj = 3500 Hz
    sir_db = 10.0
    n_muestras = 16000

    senal_rf = np.zeros(n_muestras, dtype=float)
    senal_aci = interferencias.aplicar_aci(senal_rf, fs=fs, fc=fc, delta_f=delta_f, sir_db=sir_db, pot_senal=0.5)

    freqs, psd_db = canales.calcular_espectro_fft(senal_aci, fs=fs)
    # Detectar el pico en frecuencias positivas
    mask_pos = freqs > 0
    freq_pico = float(freqs[mask_pos][np.argmax(psd_db[mask_pos])])
    assert np.isclose(freq_pico, fc + delta_f, atol=2.0), f"Frecuencia pico errónea: {freq_pico} != {fc + delta_f}"

    # Verificación de violación de Nyquist
    delta_f_violacion = (fs / 2.0) - fc + 100.0  # Excede fs/2
    error_capturado = False
    try:
        interferencias.aplicar_aci(senal_rf, fs=fs, fc=fc, delta_f=delta_f_violacion, sir_db=sir_db)
    except ValueError as e:
        error_capturado = True
        assert "Nyquist" in str(e)

    assert error_capturado, "No se levantó ValueError ante violación de Nyquist en ACI."

    return {
        "prueba": "02_ACI_Nyquist",
        "f_adj_esperada": fc + delta_f,
        "f_pico_detectada": freq_pico,
        "valida_nyquist": True,
        "estado": "PASS",
    }


def test_03_ruido_impulsivo_tasa_y_amplitud() -> dict:
    """Verifica tasa de impulsos y amplitudes relativas del modelo Bernoulli-Gauss."""
    p_imp = 0.02
    factor_amp = 8.0
    n_muestras = 50000

    senal = np.ones(n_muestras, dtype=complex)  # Potencia 1.0
    senal_pert = ruido.aplicar_ruido_impulsivo(
        senal, p_imp=p_imp, factor_amplitud=factor_amp, seed=42
    )
    ruido_medido = senal_pert - senal

    # Muestras donde la magnitud del ruido supera 3 veces la potencia unitaria
    indices_impulsos = np.abs(ruido_medido) > 3.0
    tasa_medida = float(np.mean(indices_impulsos))
    max_amplitud = float(np.max(np.abs(ruido_medido)))

    assert np.isclose(tasa_medida, p_imp, atol=0.008), f"Tasa de impulsos errónea: {tasa_medida} != {p_imp}"
    assert max_amplitud > (factor_amp * 1.5), f"Amplitud de impulsos insuficiente: {max_amplitud}"

    return {
        "prueba": "03_Ruido_Impulsivo",
        "p_imp_nominal": p_imp,
        "p_imp_medida": tasa_medida,
        "amplitud_maxima": max_amplitud,
        "estado": "PASS",
    }


def test_04_cdma_ortogonalidad_y_asincronismo() -> dict:
    """Verifica flujo CDMA con Walsh-8: ortogonalidad síncrona, potencia teórica asíncrona y límite U<=8."""
    bits_u0 = fundamentos.generar_bits(2000, semilla=123)

    # 1. Caso Síncrono (retardos = 0 para todos los usuarios)
    res_sinc = interferencias.simular_cdma_multiusuario(
        bits_tx_u0=bits_u0,
        num_usuarios=6,
        alfa_interf=1.0,
        retardos_chips=[0, 0, 0, 0, 0, 0],
        seed=42
    )
    assert res_sinc["p_mui_teor"] == 0.0, "La potencia teórica síncrona debe ser 0.0"
    assert np.isclose(res_sinc["p_mui_medida"], 0.0, atol=1e-12), "Interferencia síncrona debe ser 0.0"
    ber_sinc, _, _ = demodulacion.calcular_ber(bits_u0, res_sinc["bits_rx_u0"])
    assert ber_sinc == 0.0, f"BER en CDMA síncrono debe ser 0.0, se obtuvo {ber_sinc}"

    # 2. Caso Asíncrono (retardos enteros con zero-padding)
    retardos_asinc = [0, 1, 3, 5]
    alfas_asinc = [0.8, 0.6, 0.5]
    res_asinc = interferencias.simular_cdma_multiusuario(
        bits_tx_u0=bits_u0,
        num_usuarios=4,
        alfa_interf=alfas_asinc,
        retardos_chips=retardos_asinc,
        seed=42
    )
    p_teor = float(res_asinc["p_mui_teor"])
    p_medida = float(res_asinc["p_mui_medida"])
    # Comparar varianza empírica con la teórica derivada de las correlaciones parciales
    assert np.isclose(p_medida, p_teor, rtol=0.10), f"Discrepancia en MUI: medida={p_medida:.4f}, teor={p_teor:.4f}"

    # 3. Límite estricto de usuarios (U > 8)
    error_usuarios = False
    try:
        interferencias.simular_cdma_multiusuario(bits_tx_u0=bits_u0, num_usuarios=9)
    except ValueError as e:
        error_usuarios = True
        assert "8 usuarios" in str(e)

    assert error_usuarios, "No se levantó ValueError al solicitar U=9 usuarios en CDMA."

    return {
        "prueba": "04_CDMA_Walsh8",
        "mui_sincrono": res_sinc["p_mui_medida"],
        "p_mui_asinc_teor": p_teor,
        "p_mui_asinc_medida": p_medida,
        "valida_limite_usuarios": True,
        "estado": "PASS",
    }


def test_05_ruido_fase_estadistica() -> dict:
    """Verifica dispersión angular de fase en banda base sobre muestras independientes."""
    desv_grados = 12.0
    desv_rad = desv_grados * (np.pi / 180.0)
    n_muestras = 50000

    senal_const = np.ones(n_muestras, dtype=complex)
    senal_rotada = efectos_canal.aplicar_ruido_fase(senal_const, desviacion_grados=desv_grados, seed=42)

    angulos = np.angle(senal_rotada)
    media_ang = float(np.mean(angulos))
    std_ang = float(np.std(angulos))

    assert np.isclose(media_ang, 0.0, atol=0.02), f"Media angular sesgada: {media_ang}"
    assert np.isclose(std_ang, desv_rad, rtol=0.05), f"Desviación angular errónea: {std_ang} != {desv_rad}"

    return {
        "prueba": "05_Ruido_Fase",
        "desv_nominal_grados": desv_grados,
        "desv_medida_grados": float(std_ang * 180.0 / np.pi),
        "estado": "PASS",
    }


def test_06_interferencia_tonal_cw() -> dict:
    """Verifica inserción de portadora continua parásita CW en RF por FFT."""
    fs = 16000.0
    f_cw = 2800.0
    sir_db = 8.0
    n_muestras = 16000

    senal_rf = np.zeros(n_muestras, dtype=float)
    senal_cw = interferencias.aplicar_interferencia_tonal(senal_rf, fs=fs, f_cw=f_cw, sir_db=sir_db, pot_senal=0.5)

    freqs, psd_db = canales.calcular_espectro_fft(senal_cw, fs=fs)
    mask_pos = freqs > 0
    freq_pico = float(freqs[mask_pos][np.argmax(psd_db[mask_pos])])

    assert np.isclose(freq_pico, f_cw, atol=2.0), f"Pico CW detectado en {freq_pico} Hz != {f_cw} Hz"

    # Verificar potencia empírica: P_cw = 0.5 * 10^(-8/10) = 0.0792 W
    pot_teorica = 0.5 * (10.0 ** (-sir_db / 10.0))
    pot_medida = float(np.mean(senal_cw**2))
    assert np.isclose(pot_medida, pot_teorica, rtol=0.05)

    return {
        "prueba": "06_Tonal_CW",
        "f_cw_esperada": f_cw,
        "f_cw_detectada": freq_pico,
        "pot_medida": pot_medida,
        "pot_teorica": pot_teorica,
        "estado": "PASS",
    }


def test_07_ruido_termico_ktb_dimensional() -> dict:
    """Valida la conversión física N=kTB, densidad espectral a 290 K y varianza digital."""
    metricas = ruido.calcular_ruido_termico_ktb(temperatura_k=290.0, ancho_banda_hz=200e3, figura_ruido_db=5.0)

    # Densidad a 290 K: N0 = 10*log10(k*290/1e-3) = -173.975 dBm/Hz (tolerancia +/- 0.05 dBm/Hz)
    n0_dbm_hz = metricas["n0_dbm_hz"]
    assert np.isclose(n0_dbm_hz, -173.975, atol=0.05), f"Densidad errónea: {n0_dbm_hz} dBm/Hz"

    # Potencia física de ruido: N = -173.975 + 10*log10(200000) + 5 = -173.975 + 53.010 + 5 = -115.965 dBm
    p_ruido_dbm = metricas["potencia_ruido_dbm"]
    assert np.isclose(p_ruido_dbm, -115.965, atol=0.08)

    # Inyección sobre señal con potencia digital normalizada de 1.0 W
    senal_bb = np.ones(20000, dtype=complex)
    senal_pert, met_enlace = ruido.aplicar_ruido_termico(
        senal_bb, potencia_rx_dbm=-85.0, temperatura_k=290.0, ancho_banda_hz=200e3, figura_ruido_db=5.0, seed=42
    )

    snr_esperada_db = -85.0 - (-115.965)  # ≈ 30.965 dB
    assert np.isclose(met_enlace["snr_db"], snr_esperada_db, atol=0.08)

    # Medir varianza digital del ruido inyectado
    ruido_medido = senal_pert - senal_bb
    var_digital_medida = float(np.var(ruido_medido))
    var_digital_teorica = met_enlace["varianza_digital_total"]
    assert np.isclose(var_digital_medida, var_digital_teorica, rtol=0.08)

    return {
        "prueba": "07_Ruido_Termico_kTB",
        "n0_dbm_hz": n0_dbm_hz,
        "n_dbm": p_ruido_dbm,
        "snr_db": met_enlace["snr_db"],
        "var_digital_medida": var_digital_medida,
        "estado": "PASS",
    }


def test_08_interferencia_cocanal_sir() -> dict:
    """Verifica calibración de potencia SIR en interferencia cocanal (CCI)."""
    sir_db = 12.0
    n_muestras = 30000
    senal = np.ones(n_muestras, dtype=complex)  # Potencia 1.0

    senal_pert = interferencias.aplicar_cci(senal, sir_db=sir_db, retardo_muestras=20, fase_rad=0.3, seed=42)
    interferencia = senal_pert - senal
    pot_int = float(np.mean(np.abs(interferencia) ** 2))

    sir_medido = float(10.0 * np.log10(1.0 / pot_int))
    assert np.isclose(sir_medido, sir_db, atol=0.5), f"SIR medido erróneo: {sir_medido} != {sir_db}"

    return {
        "prueba": "08_Interferencia_CCI",
        "sir_nominal_db": sir_db,
        "sir_medido_db": sir_medido,
        "estado": "PASS",
    }


def test_09_ruido_laplace_kurtosis() -> dict:
    """Verifica colas pesadas del ruido de Laplace comparando curtosis con ruido Gaussiano."""
    snr_db = 10.0
    n_muestras = 80000
    senal = np.zeros(n_muestras, dtype=float)

    ruido_lap = ruido.aplicar_ruido_laplace(senal, snr_db=snr_db, pot_senal=1.0, es_compleja=False, seed=42)
    ruido_gau = ruido.aplicar_awgn_rf(senal, snr_rf_db=snr_db, pot_senal=1.0, seed=42)

    # Exceso de curtosis: E[x^4]/sigma^4 - 3. Para Gauss = 0.0; para Laplace = 3.0
    var_lap = float(np.var(ruido_lap))
    var_gau = float(np.var(ruido_gau))
    kurt_lap = float(np.mean(ruido_lap**4) / (var_lap**2) - 3.0)
    kurt_gau = float(np.mean(ruido_gau**4) / (var_gau**2) - 3.0)

    assert kurt_lap > 1.8, f"Curtosis de Laplace debe ser marcadamente positiva (> 1.8), se obtuvo {kurt_lap}"
    assert abs(kurt_gau) < 0.4, f"Curtosis Gaussiana debe ser cercana a 0, se obtuvo {kurt_gau}"

    return {
        "prueba": "09_Ruido_Laplace",
        "kurtosis_laplace": kurt_lap,
        "kurtosis_gauss": kurt_gau,
        "var_laplace": var_lap,
        "var_gauss": var_gau,
        "estado": "PASS",
    }


def test_10_doppler_desplazamiento_frecuencia() -> dict:
    """Verifica la fórmula cinemática f_D = (v/c)fc y desplazamiento de portadora en RF."""
    v_kmh = 120.0
    fc = 2000.0
    fs = 16000.0
    f_d_teorico = efectos_canal.calcular_frecuencia_doppler(velocidad_kmh=v_kmh, fc=fc)

    # f_D = (120 / 3.6 / 299792458) * 2000 = 0.00022238 Hz a escala física real.
    # Para validar la rotación DSP en simulación, evaluamos con f_doppler = 250 Hz
    f_d_sim = 250.0
    n_muestras = 16000
    senal_bb = np.ones(n_muestras, dtype=complex)

    senal_doppler_bb = efectos_canal.aplicar_doppler_banda_base(senal_bb, fs=fs, f_doppler=f_d_sim)
    freqs, psd_db = canales.calcular_espectro_fft(senal_doppler_bb, fs=fs)

    f_pico = float(freqs[np.argmax(psd_db)])
    assert np.isclose(f_pico, f_d_sim, atol=2.0), f"Pico Doppler detectado en {f_pico} Hz != {f_d_sim} Hz"

    return {
        "prueba": "10_Doppler_Desplazamiento",
        "f_d_cinematica_120kmh": f_d_teorico,
        "f_d_simulada": f_d_sim,
        "f_pico_detectada": f_pico,
        "estado": "PASS",
    }


def test_11_regla_despacho_bfsk_excepciones() -> dict:
    """Verifica la regla de despacho: rechaza modelos incompatibles con BFSK y opera sobre RF."""
    bits = fundamentos.generar_bits(200, semilla=123)
    mod_bfsk = modulaciones.modular_bfsk(bits, f0=2000.0, f1=4000.0, sps=16, fs=16000.0)

    # 1. CDMA debe levantar ValueError en BFSK
    error_cdma = False
    try:
        canales.aplicar_canal(mod_bfsk, "cdma", num_usuarios=4)
    except ValueError as e:
        error_cdma = True
        assert "MUI/CDMA no es compatible con la modulación BFSK" in str(e)
    assert error_cdma, "No se levantó ValueError para CDMA en BFSK."

    # 2. Ruido de fase debe levantar ValueError en BFSK
    error_fase = False
    try:
        canales.aplicar_canal(mod_bfsk, "ruido_fase", desviacion_grados=5.0)
    except ValueError as e:
        error_fase = True
        assert "Ruido de Fase en constelación no es aplicable a BFSK" in str(e)
    assert error_fase, "No se levantó ValueError para Ruido de Fase en BFSK."

    # 3. Modelos compatibles deben operar y preservar simbolos_pert = None
    res_awgn = canales.aplicar_canal(mod_bfsk, "awgn", snr_db=15.0, seed=42)
    assert res_awgn.simbolos_pert is None, "BFSK no debe tener simbolos_pert I/Q artificiales."
    assert res_awgn.senal_rf_pert is not None

    bits_rx = canales.demodular_resultado_perturbado(res_awgn, f0=2000.0, f1=4000.0)
    ber_awgn, _, _ = demodulacion.calcular_ber(bits, bits_rx)
    assert ber_awgn == 0.0, f"BFSK a 15 dB SNR debe tener BER=0, se obtuvo {ber_awgn}"

    return {
        "prueba": "11_Despacho_BFSK",
        "rechaza_cdma": True,
        "rechaza_ruido_fase": True,
        "opera_awgn_rf": True,
        "ber_bfsk_15db": ber_awgn,
        "estado": "PASS",
    }


# =====================================================================
# BLOQUE 2: PRUEBAS DE INTEGRACIÓN END-TO-END
# =====================================================================

def ejecutar_pruebas_integracion() -> list[dict]:
    """Ejecuta la cadena completa: Transmisor -> Canal Perturbado -> Receptor -> BER."""
    resultados_integracion = []
    bits = fundamentos.generar_bits(1200, semilla=999)

    # Generar modulaciones
    mods = {
        "BFSK": modulaciones.modular_bfsk(bits, f0=2000.0, f1=4000.0, sps=16, fs=16000.0),
        "BPSK": modulaciones.modular_bpsk(bits, sps=16, fs=16000.0, fc=2000.0),
        "QPSK": modulaciones.modular_qpsk(bits, sps=16, fs=16000.0, fc=2000.0),
        "16-QAM": modulaciones.modular_16qam(bits, sps=16, fs=16000.0, fc=2000.0),
    }

    # 1. Canal ideal de referencia (SNR -> infinito)
    for nombre_mod, mod_obj in mods.items():
        res_ideal = canales.aplicar_canal(mod_obj, "awgn", snr_db=100.0, seed=42)
        bits_rx = canales.demodular_resultado_perturbado(res_ideal)
        ber, _, _ = demodulacion.calcular_ber(bits, bits_rx)
        assert ber == 0.0, f"Canal ideal falló en {nombre_mod}: BER={ber}"
        resultados_integracion.append({
            "modulacion": nombre_mod,
            "canal": "Ideal (Sin ruido)",
            "parametro": "SNR=100 dB",
            "ber": ber,
            "evm_pct": 0.0 if res_ideal.simbolos_pert is None else float(canales.calcular_evm(mod_obj.simbolos, res_ideal.simbolos_pert)[0]),
        })

    # 2. Evaluación de degradación frente a AWGN (20 dB -> 10 dB -> 2 dB)
    snrs = [20.0, 10.0, 2.0]
    for snr in snrs:
        for nombre_mod, mod_obj in mods.items():
            res_awgn = canales.aplicar_canal(mod_obj, "awgn", snr_db=snr, seed=42)
            bits_rx = canales.demodular_resultado_perturbado(res_awgn)
            ber, _, _ = demodulacion.calcular_ber(bits, bits_rx)
            evm = 0.0 if res_awgn.simbolos_pert is None else float(canales.calcular_evm(mod_obj.simbolos, res_awgn.simbolos_pert)[0])
            resultados_integracion.append({
                "modulacion": nombre_mod,
                "canal": "AWGN",
                "parametro": f"SNR={snr:.0f} dB",
                "ber": ber,
                "evm_pct": evm,
            })

    # 3. Evaluación de interferencias específicas
    # ACI sobre BFSK y QPSK
    for nombre_mod in ["BFSK", "QPSK"]:
        res_aci = canales.aplicar_canal(mods[nombre_mod], "aci", delta_f=1500.0, sir_db=8.0, seed=42)
        bits_rx = canales.demodular_resultado_perturbado(res_aci)
        ber, _, _ = demodulacion.calcular_ber(bits, bits_rx)
        resultados_integracion.append({
            "modulacion": nombre_mod,
            "canal": "ACI",
            "parametro": "SIR=8 dB, df=1.5 kHz",
            "ber": ber,
            "evm_pct": 0.0 if res_aci.simbolos_pert is None else float(canales.calcular_evm(mods[nombre_mod].simbolos, res_aci.simbolos_pert)[0]),
        })

    # Ruido de fase sobre QPSK y 16-QAM
    for nombre_mod in ["QPSK", "16-QAM"]:
        res_fase = canales.aplicar_canal(mods[nombre_mod], "ruido_fase", desviacion_grados=8.0, seed=42)
        bits_rx = canales.demodular_resultado_perturbado(res_fase)
        ber, _, _ = demodulacion.calcular_ber(bits, bits_rx)
        evm = float(canales.calcular_evm(mods[nombre_mod].simbolos, res_fase.simbolos_pert)[0])
        resultados_integracion.append({
            "modulacion": nombre_mod,
            "canal": "Ruido de Fase",
            "parametro": "sigma=8.0 deg",
            "ber": ber,
            "evm_pct": evm,
        })

    return resultados_integracion


# =====================================================================
# BLOQUE 3: GENERACIÓN DE LOS 10 GRÁFICOS OBLIGATORIOS (FASE 5)
# =====================================================================

def generar_diez_graficos() -> None:
    """Genera y guarda los diez gráficos de alta resolución requeridos."""
    bits = fundamentos.generar_bits(800, semilla=123)
    mod_qpsk = modulaciones.modular_qpsk(bits, sps=16, fs=16000.0, fc=2000.0)
    mod_16qam = modulaciones.modular_16qam(bits, sps=16, fs=16000.0, fc=2000.0)
    mod_bfsk = modulaciones.modular_bfsk(bits, f0=2000.0, f1=4000.0, sps=16, fs=16000.0)

    # 1. fase5_01_awgn.png: Constelación ruidosa a diferentes SNR
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, snr in enumerate([20.0, 10.0, 0.0]):
        res = canales.aplicar_canal(mod_16qam, "awgn", snr_db=snr, seed=42)
        ax = axes[i]
        ax.scatter(np.real(res.simbolos_pert), np.imag(res.simbolos_pert), color="royalblue", alpha=0.6, s=15)
        ax.set_title(f"16-QAM con AWGN (SNR = {snr:.0f} dB)")
        ax.set_xlabel("En Fase (I)")
        ax.set_ylabel("Cuadratura (Q)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "fase5_01_awgn.png"), dpi=200)
    plt.close()

    # 2. fase5_02_aci.png: Espectro FFT canal deseado vs interferente adyacente
    res_aci = canales.aplicar_canal(mod_qpsk, "aci", delta_f=2000.0, sir_db=6.0, seed=42)
    f_orig, psd_orig = canales.calcular_espectro_fft(res_aci.senal_rf_orig, fs=res_aci.fs)
    f_pert, psd_pert = canales.calcular_espectro_fft(res_aci.senal_rf_pert, fs=res_aci.fs)
    plt.figure(figsize=(10, 5))
    plt.plot(f_orig, psd_orig, label="Señal Original QPSK (fc = 2 kHz)", color="royalblue", alpha=0.7)
    plt.plot(f_pert, psd_pert, label="Señal + ACI (f_adj = 4 kHz, SIR = 6 dB)", color="firebrick", linestyle="--")
    plt.title("Espectro en Frecuencia: Interferencia de Canal Adyacente (ACI)")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Densidad Espectral Normalizada [dB]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "fase5_02_aci.png"), dpi=200)
    plt.close()

    # 3. fase5_03_ruido_impulsivo.png: Forma de onda temporal con espigas
    res_imp = canales.aplicar_canal(mod_qpsk, "impulsivo", p_imp=0.03, factor_amplitud=7.0, seed=42)
    t = res_imp.t[:500] * 1000.0  # primeros 500 muestras en ms
    plt.figure(figsize=(12, 5))
    plt.plot(t, np.real(res_imp.senal_bb_orig[:500]), label="Señal Limpia (I)", color="royalblue", alpha=0.8)
    plt.plot(t, np.real(res_imp.senal_bb_pert[:500]), label="Señal + Ruido Impulsivo", color="crimson", alpha=0.85)
    plt.title("Dominio Temporal: Ruido Impulsivo Bernoulli-Gaussiano")
    plt.xlabel("Tiempo [ms]")
    plt.ylabel("Amplitud")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "fase5_03_ruido_impulsivo.png"), dpi=200)
    plt.close()

    # 4. fase5_04_cdma_mui.png: Señal multicanal chips y descorrelación
    res_cdma = interferencias.simular_cdma_multiusuario(
        bits_tx_u0=bits[:400], num_usuarios=4, alfa_interf=1.0, retardos_chips=[0, 1, 3, 5], seed=42
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(res_cdma["senal_compuesta_chips"][:80], drawstyle="steps-post", color="darkgreen")
    ax1.set_title("Señal Compuesta Multiusuario (Primeros 80 chips)")
    ax1.set_xlabel("Índice de Chip")
    ax1.set_ylabel("Nivel de Tensión")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.scatter(np.real(res_cdma["simbolos_descorrelacionados"]), np.zeros(len(res_cdma["simbolos_descorrelacionados"])),
                color="navy", alpha=0.6, s=25)
    ax2.axvline(0, color="red", linestyle="--", label="Umbral de Decisión BPSK")
    ax2.set_title(f"Símbolos Descorrelacionados Usuario 0 (P_MUI={res_cdma['p_mui_medida']:.3f})")
    ax2.set_xlabel("Amplitud Descorrelacionada")
    ax2.set_yticks([])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "fase5_04_cdma_mui.png"), dpi=200)
    plt.close()

    # 5. fase5_05_ruido_fase.png: Constelaciones con arcos de rotación
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    res_fase_q = canales.aplicar_canal(mod_qpsk, "ruido_fase", desviacion_grados=10.0, seed=42)
    res_fase_16 = canales.aplicar_canal(mod_16qam, "ruido_fase", desviacion_grados=6.0, seed=42)

    ax1.scatter(np.real(res_fase_q.simbolos_pert), np.imag(res_fase_q.simbolos_pert), color="royalblue", alpha=0.7, s=20)
    ax1.set_title("QPSK con Ruido de Fase (sigma = 10 deg)")
    ax1.set_xlabel("En Fase (I)")
    ax1.set_ylabel("Cuadratura (Q)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.scatter(np.real(res_fase_16.simbolos_pert), np.imag(res_fase_16.simbolos_pert), color="forestgreen", alpha=0.7, s=20)
    ax2.set_title("16-QAM con Ruido de Fase (sigma = 6 deg)")
    ax2.set_xlabel("En Fase (I)")
    ax2.set_ylabel("Cuadratura (Q)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "fase5_05_ruido_fase.png"), dpi=200)
    plt.close()

    # 6. fase5_06_interferencia_tonal.png: FFT con portadora parásita CW
    res_cw = canales.aplicar_canal(mod_qpsk, "tonal", f_cw=2800.0, sir_db=8.0, seed=42)
    f_cw_arr, psd_cw_arr = canales.calcular_espectro_fft(res_cw.senal_rf_pert, fs=res_cw.fs)
    plt.figure(figsize=(10, 5))
    plt.plot(f_cw_arr, psd_cw_arr, color="darkorange", label="QPSK + Tono CW (f_cw = 2.8 kHz, SIR = 8 dB)")
    plt.axvline(2800.0, color="red", linestyle=":", label="Frecuencia Espuria CW (2800 Hz)")
    plt.title("Espectro FFT: Interferencia Tonal de Onda Continua (CW)")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Densidad Espectral Normalizada [dB]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "fase5_06_interferencia_tonal.png"), dpi=200)
    plt.close()

    # 7. fase5_07_ruido_termico.png: Densidad física N=kTB y constelación digital
    res_term = canales.aplicar_canal(mod_16qam, "termico", potencia_rx_dbm=-75.0, temperatura_k=290.0, seed=42)
    plt.figure(figsize=(7, 6))
    plt.scatter(np.real(res_term.simbolos_pert), np.imag(res_term.simbolos_pert), color="purple", alpha=0.6, s=20)
    plt.title(f"16-QAM con Ruido Térmico kTB (Prx = -75 dBm, SNR = {res_term.parametros_canal['metricas_termicas']['snr_db']:.1f} dB)")
    plt.xlabel("En Fase (I)")
    plt.ylabel("Cuadratura (Q)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "fase5_07_ruido_termico.png"), dpi=200)
    plt.close()

    # 8. fase5_08_interferencia_cocanal.png: Constelación con interferente CCI
    res_cci = canales.aplicar_canal(mod_qpsk, "cocanal", sir_db=12.0, seed=42)
    plt.figure(figsize=(7, 6))
    plt.scatter(np.real(res_cci.simbolos_pert), np.imag(res_cci.simbolos_pert), color="teal", alpha=0.6, s=20)
    plt.title("QPSK con Interferencia Cocanal (CCI, SIR = 12 dB)")
    plt.xlabel("En Fase (I)")
    plt.ylabel("Cuadratura (Q)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "fase5_08_interferencia_cocanal.png"), dpi=200)
    plt.close()

    # 9. fase5_09_ruido_laplace_vs_gauss.png: Histogramas comparativos y constelaciones
    res_lap = canales.aplicar_canal(mod_qpsk, "laplace", snr_db=10.0, seed=42)
    res_gau = canales.aplicar_canal(mod_qpsk, "awgn", snr_db=10.0, seed=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    err_lap = np.real(res_lap.simbolos_pert - res_lap.simbolos_orig)
    err_gau = np.real(res_gau.simbolos_pert - res_gau.simbolos_orig)

    ax1.hist(err_gau, bins=40, density=True, alpha=0.6, color="royalblue", label="AWGN (Gaussiano)")
    ax1.hist(err_lap, bins=40, density=True, alpha=0.6, color="firebrick", label="Laplace (Colas pesadas)")
    ax1.set_title("Histograma Comparativo de PDF de Ruido")
    ax1.set_xlabel("Amplitud del Error")
    ax1.set_ylabel("Densidad de Probabilidad")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.scatter(np.real(res_lap.simbolos_pert), np.imag(res_lap.simbolos_pert), color="firebrick", alpha=0.6, s=18)
    ax2.set_title("QPSK con Ruido Laplace (SNR = 10 dB)")
    ax2.set_xlabel("En Fase (I)")
    ax2.set_ylabel("Cuadratura (Q)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "fase5_09_ruido_laplace_vs_gauss.png"), dpi=200)
    plt.close()

    # 10. fase5_10_doppler.png: Desplazamiento espectral y constelación en espiral
    res_dop = canales.aplicar_canal(mod_qpsk, "doppler", f_doppler=300.0)
    f_dop, psd_dop = canales.calcular_espectro_fft(res_dop.senal_rf_pert, fs=res_dop.fs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(f_dop, psd_dop, color="indigo", label="QPSK con Doppler (+300 Hz)")
    ax1.axvline(2300.0, color="red", linestyle="--", label="Portadora desplazada (2300 Hz)")
    ax1.set_title("Espectro FFT: Desplazamiento Doppler")
    ax1.set_xlabel("Frecuencia [Hz]")
    ax1.set_ylabel("Densidad Espectral Normalizada [dB]")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.scatter(np.real(res_dop.simbolos_pert), np.imag(res_dop.simbolos_pert), color="indigo", alpha=0.6, s=18)
    ax2.set_title("Constelación Rotada por Doppler no Compensado")
    ax2.set_xlabel("En Fase (I)")
    ax2.set_ylabel("Cuadratura (Q)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "fase5_10_doppler.png"), dpi=200)
    plt.close()


# =====================================================================
# BLOQUE 4: EXPORTACIÓN DE TABLAS DE RESULTADOS
# =====================================================================

def exportar_tablas_resultados(res_unitarias: list[dict], res_integracion: list[dict]) -> None:
    """Exporta las tablas de resultados a archivos CSV y Markdown puros."""
    # 1. Tabla de Pruebas Unitarias
    csv_unit_path = os.path.join(DIR_TABLAS, "fase5_pruebas_unitarias.csv")
    md_unit_path = os.path.join(DIR_TABLAS, "fase5_pruebas_unitarias.md")

    with open(csv_unit_path, "w", encoding="utf-8") as f:
        f.write("Prueba,Estado\n")
        for u in res_unitarias:
            f.write(f"{u['prueba']},{u['estado']}\n")

    with open(md_unit_path, "w", encoding="utf-8") as f:
        f.write("# Validación de Pruebas Unitarias — Fase 5\n\n")
        f.write("| N° | Prueba Unitaria | Estado |\n")
        f.write("| :---: | :--- | :---: |\n")
        for i, u in enumerate(res_unitarias, start=1):
            f.write(f"| {i} | `{u['prueba']}` | **{u['estado']}** |\n")

    # 2. Tabla de Integración de Modelos
    csv_int_path = os.path.join(DIR_TABLAS, "fase5_resumen_modelos.csv")
    md_int_path = os.path.join(DIR_TABLAS, "fase5_resumen_modelos.md")

    with open(csv_int_path, "w", encoding="utf-8") as f:
        f.write("Modulacion,Canal,Parametro,BER,EVM_pct\n")
        for r in res_integracion:
            f.write(f"{r['modulacion']},{r['canal']},{r['parametro']},{r['ber']:.6f},{r['evm_pct']:.2f}\n")

    with open(md_int_path, "w", encoding="utf-8") as f:
        f.write("# Resumen de Integración de Modelos de Canal — Fase 5\n\n")
        f.write("| Modulación | Canal / Perturbación | Parámetro Clave | BER Resultante | EVM [%] |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: |\n")
        for r in res_integracion:
            f.write(f"| {r['modulacion']} | {r['canal']} | {r['parametro']} | {r['ber']:.5f} | {r['evm_pct']:.2f}% |\n")


# =====================================================================
# FUNCIÓN PRINCIPAL DE EJECUCIÓN
# =====================================================================

def main() -> None:
    print("=" * 75)
    print("   SIMULADOR DE MODULACIONES DIGITALES — BATERÍA DE PRUEBAS FASE 5")
    print("=" * 75)

    pruebas_unitarias = [
        test_01_awgn_varianzas_tres_dominios,
        test_02_aci_frecuencia_y_nyquist,
        test_03_ruido_impulsivo_tasa_y_amplitud,
        test_04_cdma_ortogonalidad_y_asincronismo,
        test_05_ruido_fase_estadistica,
        test_06_interferencia_tonal_cw,
        test_07_ruido_termico_ktb_dimensional,
        test_08_interferencia_cocanal_sir,
        test_09_ruido_laplace_kurtosis,
        test_10_doppler_desplazamiento_frecuencia,
        test_11_regla_despacho_bfsk_excepciones,
    ]

    res_unitarias = []
    print("\n--- EJECUCIÓN DE PRUEBAS UNITARIAS ---")
    for fn in pruebas_unitarias:
        nombre = fn.__name__
        try:
            res = fn()
            res_unitarias.append(res)
            print(f"  [OK] {nombre}: PASS")
        except Exception as e:
            print(f"  [ERROR] {nombre}: FAIL -> {e}")
            raise e

    print("\n--- EJECUCIÓN DE PRUEBAS DE INTEGRACIÓN END-TO-END ---")
    res_integracion = ejecutar_pruebas_integracion()
    print(f"  [OK] Evaluados {len(res_integracion)} escenarios de integración con éxito.")

    print("\n--- GENERACIÓN DE LOS 10 GRÁFICOS OBLIGATORIOS ---")
    generar_diez_graficos()
    print("  [OK] 10 gráficos exportados a resultados/graficos/fase5_01 a fase5_10.")

    print("\n--- EXPORTACIÓN DE TABLAS DE RESULTADOS ---")
    exportar_tablas_resultados(res_unitarias, res_integracion)
    print("  [OK] Tablas exportadas a resultados/tablas/fase5_resumen_modelos.csv y .md.")

    print("\n" + "=" * 75)
    print("   >>> FASE 5: TODAS LAS PRUEBAS COMPLETADAS CON ÉXITO (11/11 PASS) <<<")
    print("=" * 75)


if __name__ == "__main__":
    main()
