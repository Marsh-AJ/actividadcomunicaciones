"""
test_fase6.py
Suite de Pruebas Unitarias, Integración Experimental y No Regresión para la FASE 6.

Desarrollado para la asignatura Electiva de Comunicaciones Móviles (Python 3.13.5).
Fase 6: Métricas y Evaluación del Rendimiento de las Modulaciones.

Ejecución:
1. Bloque 1: 8 pruebas unitarias matemáticas de metricas.py y calibración.
2. Bloque 2: Ejecución Monte Carlo de los 5 experimentos completos (Nb = 60.000 bits).
3. Bloque 3: Generación de 6 gráficos experimentales de alta calidad en resultados/graficos/.
4. Bloque 4: Exportación de tablas en formato CSV y Markdown en resultados/tablas/.
5. Bloque 5: Verificación estricta de no regresión de las Fases 1, 2, 3, 4 y 5.
"""

from __future__ import annotations
import os
import sys
import math
import subprocess
import numpy as np

# Configuración del directorio de caché de matplotlib antes de importar pyplot
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_CACHE_MPL = os.path.join(DIR_BASE, ".cache", "matplotlib")
os.makedirs(DIR_CACHE_MPL, exist_ok=True)
os.environ["MPLCONFIGDIR"] = DIR_CACHE_MPL

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para guardado robusto
import matplotlib.pyplot as plt

# Módulos del proyecto
import fundamentos
import modulaciones
import demodulacion
import ruido
import interferencias
import efectos_canal
import canales
import metricas
import experimentos


# Carpetas de salida
DIR_GRAFICOS = os.path.join(DIR_BASE, "resultados", "graficos")
DIR_TABLAS = os.path.join(DIR_BASE, "resultados", "tablas")
os.makedirs(DIR_GRAFICOS, exist_ok=True)
os.makedirs(DIR_TABLAS, exist_ok=True)


# =====================================================================
# BLOQUE 1: PRUEBAS UNITARIAS MATEMÁTICAS (8 PRUEBAS)
# =====================================================================

def test_01_conversiones_ebn0_esn0() -> dict:
    """Verifica conversiones analíticas exactas Eb/N0 <-> Es/N0."""
    ebn0_db = 8.0

    # BFSK (k=1)
    esn0_bfsk = metricas.ebn0_a_esn0(ebn0_db, 1)
    assert np.isclose(esn0_bfsk, 8.0), f"Fallo BFSK: {esn0_bfsk} != 8.0"

    # BPSK (k=1)
    esn0_bpsk = metricas.ebn0_a_esn0(ebn0_db, 1)
    assert np.isclose(esn0_bpsk, 8.0), f"Fallo BPSK: {esn0_bpsk} != 8.0"

    # QPSK (k=2) -> +3.0103 dB
    esn0_qpsk = metricas.ebn0_a_esn0(ebn0_db, 2)
    assert np.isclose(esn0_qpsk, 8.0 + 10.0 * np.log10(2.0)), f"Fallo QPSK: {esn0_qpsk}"

    # 16-QAM (k=4) -> +6.0206 dB
    esn0_qam = metricas.ebn0_a_esn0(ebn0_db, 4)
    assert np.isclose(esn0_qam, 8.0 + 10.0 * np.log10(4.0)), f"Fallo 16-QAM: {esn0_qam}"

    # Conversión inversa
    ebn0_rec = metricas.esn0_a_ebn0(esn0_qam, 4)
    assert np.isclose(ebn0_rec, ebn0_db), f"Fallo inversa: {ebn0_rec} != {ebn0_db}"

    return {"prueba": "01_Conversiones_EbN0_EsN0", "estado": "PASS"}


def test_02_calibracion_varianzas_awgn() -> dict:
    """Verifica el cálculo de N0 y las varianzas en banda base, RF y símbolos."""
    ebn0_db = 10.0
    k = 2  # QPSK
    sps = 16

    n0 = metricas.calcular_n0_desde_ebn0(ebn0_db, k, es=1.0)
    # gamma_b = 10, Eb = 0.5, N0 = 0.5 / 10 = 0.05
    assert np.isclose(n0, 0.05), f"Fallo N0: {n0} != 0.05"

    var_bb_tot, var_bb_comp = metricas.calcular_varianza_bb(n0, sps)
    assert np.isclose(var_bb_tot, sps * n0)
    assert np.isclose(var_bb_comp, (sps * n0) / 2.0)

    var_rf = metricas.calcular_varianza_rf(n0, sps)
    assert np.isclose(var_rf, (sps * n0) / 4.0)

    var_sym_tot, var_sym_comp = metricas.calcular_varianza_simbolo(n0)
    assert np.isclose(var_sym_tot, n0)
    assert np.isclose(var_sym_comp, n0 / 2.0)

    return {"prueba": "02_Calibracion_Varianzas", "n0": n0, "estado": "PASS"}


def test_03_metricas_ber_intervalos() -> dict:
    """Verifica cálculo de BER, intervalo Wilson al 95% y regla de tres."""
    nb = 60000

    # Caso A: Con errores observados
    tx = np.zeros(nb, dtype=np.int8)
    rx = np.zeros(nb, dtype=np.int8)
    rx[:60] = 1  # 60 errores -> BER = 1e-3
    mber = metricas.calcular_ber_con_intervalo(tx, rx)

    assert mber.errores == 60
    assert np.isclose(mber.ber, 0.001)
    assert mber.es_cero_errores is False
    assert mber.cota_tres is None
    assert mber.wilson_lower < mber.ber < mber.wilson_upper

    # Caso B: Cero errores observados (Nb = 60.000)
    rx_cero = np.zeros(nb, dtype=np.int8)
    mber_cero = metricas.calcular_ber_con_intervalo(tx, rx_cero)

    assert mber_cero.errores == 0
    assert mber_cero.ber == 0.0
    assert mber_cero.es_cero_errores is True
    # Regla de tres: 3 / 60000 = 5e-5
    assert np.isclose(mber_cero.cota_tres, 5.0e-5), f"Fallo cota 3: {mber_cero.cota_tres}"
    # Wilson con p_hat = 0: z^2 / (Nb + z^2) ≈ 1.96^2 / 60000 ≈ 6.40e-5
    z = 1.959963984540054
    w_esp = (z**2) / (nb + z**2)
    assert np.isclose(mber_cero.wilson_upper, w_esp, rtol=1e-4)
    assert mber_cero.wilson_lower == 0.0

    return {"prueba": "03_Metricas_BER_Intervalos", "cota_tres": mber_cero.cota_tres, "estado": "PASS"}


def test_04_evm_simbolos() -> dict:
    """Verifica el cálculo de EVM a nivel de símbolo discreto y manejo de BFSK."""
    n_sym = 1000
    rng = np.random.default_rng(42)

    # Caso A: BFSK debe retornar (None, None)
    evm_pct_bfsk, evm_db_bfsk = metricas.calcular_evm_simbolos(None, None, "BFSK")
    assert evm_pct_bfsk is None and evm_db_bfsk is None, "BFSK debe retornar None para EVM."

    # Caso B: Constelación QPSK ideal (error = 0)
    simbolos_tx = (rng.choice([-1, 1], size=n_sym) + 1j * rng.choice([-1, 1], size=n_sym)) / np.sqrt(2.0)
    evm_pct_0, evm_db_0 = metricas.calcular_evm_simbolos(simbolos_tx, simbolos_tx, "QPSK")
    assert np.isclose(evm_pct_0, 0.0)
    assert evm_db_0 <= -200.0

    # Caso C: Error controlado del 10% RMS (-20 dB)
    ruido_error = (rng.normal(0, 0.1 / np.sqrt(2), size=n_sym) + 1j * rng.normal(0, 0.1 / np.sqrt(2), size=n_sym))
    simbolos_rx = simbolos_tx + ruido_error
    evm_pct_med, evm_db_med = metricas.calcular_evm_simbolos(simbolos_tx, simbolos_rx, "QPSK")
    assert np.isclose(evm_pct_med, 10.0, atol=1.5), f"EVM pct: {evm_pct_med}"
    assert np.isclose(evm_db_med, -20.0, atol=1.5), f"EVM dB: {evm_db_med}"

    # Caso D: Desalineación de longitud debe levantar ValueError
    error_len = False
    try:
        metricas.calcular_evm_simbolos(simbolos_tx, simbolos_rx[:500], "QPSK")
    except ValueError:
        error_len = True
    assert error_len, "No se levantó ValueError por longitudes dispares."

    return {"prueba": "04_EVM_Simbolos", "evm_pct_bfsk": evm_pct_bfsk, "estado": "PASS"}


def test_05_curvas_teoricas_coherencia() -> dict:
    """Verifica que las expresiones teóricas coincidan con los valores canónicos a Eb/N0 = 10 dB."""
    ebn0_10db = 10.0

    # BPSK / QPSK exacta: Q(sqrt(2 * 10)) = 0.5 * erfc(sqrt(10)) ≈ 3.872e-6
    ber_bpsk = metricas.ber_teorica_bpsk_qpsk(ebn0_10db)
    assert np.isclose(ber_bpsk, 3.8721e-6, rtol=1e-3), f"BER BPSK: {ber_bpsk}"

    # 16-QAM Gray aproximada: (3/8)*erfc(sqrt(0.4*10)) = (3/8)*erfc(2) ≈ 1.754e-3
    ber_qam = metricas.ber_teorica_16qam_aproximada(ebn0_10db)
    assert np.isclose(ber_qam, 1.7541e-3, rtol=1e-3), f"BER 16-QAM aprox: {ber_qam}"

    # BFSK no coherente: 0.5 * exp(-0.5 * 10) = 0.5 * exp(-5) ≈ 3.36897e-3
    ber_bfsk = metricas.ber_teorica_bfsk_no_coherente(ebn0_10db)
    assert np.isclose(ber_bfsk, 3.36897e-3, rtol=1e-3), f"BER BFSK no coherente: {ber_bfsk}"

    return {"prueba": "05_Curvas_Teoricas", "ber_bpsk_10db": ber_bpsk, "estado": "PASS"}


def test_06_eficiencia_espectral_convenciones() -> dict:
    """Verifica el cálculo de eficiencia espectral y documentación de convenciones."""
    # BFSK: Carson Delta_f + 2*Rs = 2000 + 2000 = 4000 Hz -> eta = 0.25 bps/Hz
    ef_bfsk = metricas.calcular_eficiencia_espectral("BFSK", rs=1000.0, delta_f=2000.0)
    assert ef_bfsk["eficiencia_principal_bps_hz"] == 0.25, f"Eficiencia BFSK errónea: {ef_bfsk}"
    assert ef_bfsk["ancho_banda_principal_hz"] == 4000.0

    # BPSK: 0.50 en lóbulo principal RF, 1.00 en Nyquist
    ef_bpsk = metricas.calcular_eficiencia_espectral("BPSK", rs=1000.0)
    assert ef_bpsk["eficiencia_principal_bps_hz"] == 0.50
    assert ef_bpsk["detalle_convenciones"]["nyquist_ideal"]["eficiencia_bps_hz"] == 1.00

    # QPSK: 1.00 en RF, 2.00 en Nyquist
    ef_qpsk = metricas.calcular_eficiencia_espectral("QPSK", rs=1000.0)
    assert ef_qpsk["eficiencia_principal_bps_hz"] == 1.00
    assert ef_qpsk["detalle_convenciones"]["nyquist_ideal"]["eficiencia_bps_hz"] == 2.00

    # 16-QAM: 2.00 en RF, 4.00 en Nyquist
    ef_qam = metricas.calcular_eficiencia_espectral("16-QAM", rs=1000.0)
    assert ef_qam["eficiencia_principal_bps_hz"] == 2.00
    assert ef_qam["detalle_convenciones"]["nyquist_ideal"]["eficiencia_bps_hz"] == 4.00

    return {"prueba": "06_Eficiencia_Espectral", "eta_bfsk_carson": ef_bfsk["eficiencia_principal_bps_hz"], "estado": "PASS"}


def test_07_complejidad_operacional() -> dict:
    """Verifica estimación formal de operaciones por símbolo."""
    c_bfsk = metricas.estimar_complejidad_operacional("BFSK", sps=16)
    c_bpsk = metricas.estimar_complejidad_operacional("BPSK")
    c_qpsk = metricas.estimar_complejidad_operacional("QPSK")
    c_qam = metricas.estimar_complejidad_operacional("16-QAM")

    assert c_bfsk["ops_símbolo"] == (4 * 16 + 2) + (4 * 15 + 2) + 1  # 66 + 62 + 1 = 129
    assert c_bpsk["ops_símbolo"] == 1
    assert c_qpsk["ops_símbolo"] == 2
    assert c_qam["ops_símbolo"] == 8

    return {"prueba": "07_Complejidad_Operacional", "ops_bfsk": c_bfsk["ops_símbolo"], "estado": "PASS"}


def test_08_integracion_mini_transmision() -> dict:
    """Verifica la cadena de simulación calibrada en miniatura (Nb = 4000 bits)."""
    bits = fundamentos.generar_bits(4000, semilla=777)
    res_awgn = experimentos.ejecutar_transmision_calibrada(
        bits=bits,
        tipo_mod="QPSK",
        tipo_canal="awgn",
        params_canal={"ebn0_db": 10.0},
        seed=123
    )
    mber = res_awgn["metricas_ber"]
    assert mber.ber <= 0.005, f"BER QPSK a 10 dB anormal: {mber.ber}"
    assert res_awgn["evm_pct"] is not None and res_awgn["evm_pct"] < 35.0

    return {"prueba": "08_Integracion_Mini", "ber": mber.ber, "evm_pct": res_awgn["evm_pct"], "estado": "PASS"}


def ejecutar_bloque_unitario() -> list[dict]:
    """Ejecuta todas las pruebas unitarias y reporta estado."""
    print("\n" + "=" * 76)
    print(" BLOQUE 1: PRUEBAS UNITARIAS MATEMÁTICAS Y DE CALIBRACIÓN")
    print("=" * 76)
    pruebas = [
        test_01_conversiones_ebn0_esn0,
        test_02_calibracion_varianzas_awgn,
        test_03_metricas_ber_intervalos,
        test_04_evm_simbolos,
        test_05_curvas_teoricas_coherencia,
        test_06_eficiencia_espectral_convenciones,
        test_07_complejidad_operacional,
        test_08_integracion_mini_transmision
    ]
    resultados = []
    for p in pruebas:
        res = p()
        resultados.append(res)
        print(f"  [OK] {res['prueba']:<36} -> ESTADO: {res['estado']}")
    return resultados


# =====================================================================
# BLOQUE 2: GENERACIÓN DE GRÁFICOS EXPERIMENTALES
# =====================================================================

def generar_grafico_1_ber_awgn(res_exp1: dict[str, list[dict[str, Any]]]):
    """Gráfico 1: BER vs Eb/N0 con curvas teóricas rotuladas formalmente."""
    plt.figure(figsize=(10, 7), dpi=300)
    colores = {"BFSK": "#D95F02", "BPSK": "#1B9E77", "QPSK": "#7570B3", "16-QAM": "#E7298A"}
    marcadores = {"BFSK": "o", "BPSK": "s", "QPSK": "^", "16-QAM": "d"}

    ebn0_fino = np.linspace(0.0, 12.0, 200)

    # 1. Curvas Teóricas de Referencia
    plt.plot(
        ebn0_fino,
        metricas.ber_teorica_bfsk_no_coherente(ebn0_fino),
        label="BFSK Teórica No Coherente (Ortogonal)",
        color=colores["BFSK"],
        linestyle="--",
        linewidth=1.8
    )
    plt.plot(
        ebn0_fino,
        metricas.ber_teorica_bpsk_qpsk(ebn0_fino),
        label="BPSK/QPSK Teórica (Exacta)",
        color=colores["BPSK"],
        linestyle="--",
        linewidth=1.8
    )
    plt.plot(
        ebn0_fino,
        metricas.ber_teorica_16qam_aproximada(ebn0_fino),
        label="16-QAM Teórica (BER_teorica_aproximada)",
        color=colores["16-QAM"],
        linestyle="--",
        linewidth=1.8
    )

    # 2. Puntos Simulados Monte Carlo
    cota_tres_graficada = False
    for mod_nombre, puntos in res_exp1.items():
        ebn0_vals = [p["ebn0_db"] for p in puntos]
        ber_vals = []
        for p in puntos:
            if p["es_cero_errores"]:
                # Se utiliza la cota de 3 / Nb para graficar el punto sin colapsar el eje semilog
                ber_vals.append(p["cota_tres"])
                cota_tres_graficada = True
            else:
                ber_vals.append(p["ber_simulada"])

        plt.semilogy(
            ebn0_vals,
            ber_vals,
            label=f"{mod_nombre} Simulado (Nb=60.000)",
            color=colores[mod_nombre],
            marker=marcadores[mod_nombre],
            linestyle="-",
            markersize=6,
            linewidth=1.2
        )

    # Línea de referencia de la regla de tres para cero errores (3/60000 = 5e-5)
    plt.axhline(
        5.0e-5,
        color="#7F7F7F",
        linestyle=":",
        linewidth=1.2,
        label="Cota regla de 3 con 0 errores (3/Nb = 5·10⁻⁵)"
    )

    plt.title("Experimento 1: Desempeño BER vs Eb/N0 en Canal AWGN (60.000 bits)", fontsize=13, fontweight="bold")
    plt.xlabel("Relación Energía por Bit a Densidad de Ruido Eb/N0 [dB]", fontsize=11)
    plt.ylabel("Tasa de Error de Bit (BER)", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.ylim([2e-5, 0.6])
    plt.xlim([-0.2, 12.2])
    plt.legend(loc="lower left", fontsize=9, framealpha=0.92)
    plt.tight_layout()

    ruta = os.path.join(DIR_GRAFICOS, "fase6_01_ber_awgn_vs_teoria.png")
    plt.savefig(ruta)
    plt.close()
    print(f"  [GUARDADO] {ruta}")


def generar_grafico_2_evm_vs_ebn0(res_exp1: dict[str, list[dict[str, Any]]]):
    """Gráfico 2: Curvas de EVM (%) vs Eb/N0 para modulaciones lineales (BFSK es N/A)."""
    plt.figure(figsize=(9, 6), dpi=300)
    colores = {"BPSK": "#1B9E77", "QPSK": "#7570B3", "16-QAM": "#E7298A"}
    marcadores = {"BPSK": "s", "QPSK": "^", "16-QAM": "d"}

    for mod_nombre in ["BPSK", "QPSK", "16-QAM"]:
        puntos = res_exp1[mod_nombre]
        ebn0_vals = [p["ebn0_db"] for p in puntos]
        evm_pct_vals = [p["evm_pct"] for p in puntos]

        plt.plot(
            ebn0_vals,
            evm_pct_vals,
            label=f"{mod_nombre} (Símbolos I/Q de decisión)",
            color=colores[mod_nombre],
            marker=marcadores[mod_nombre],
            linewidth=1.8,
            markersize=6
        )

    plt.title("Experimento 1: EVM RMS (%) vs Eb/N0 a Nivel de Símbolo en Decisión", fontsize=12, fontweight="bold")
    plt.xlabel("Eb/N0 [dB]", fontsize=11)
    plt.ylabel("Magnitud del Vector de Error (EVM RMS) [%]", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xlim([-0.2, 12.2])
    plt.ylim([0.0, 110.0])

    # Cuadro informativo explicitando la exclusión formal de BFSK
    plt.annotate(
        "NOTA: BFSK no opera sobre constelación I/Q en banda base\n"
        "(decisión pasobanda por energía en cuadratura).\n"
        "EVM reportado formalmente como N/A.",
        xy=(0.03, 0.08),
        xycoords="axes fraction",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF2CC", edgecolor="#D6B656", alpha=0.9)
    )

    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    ruta = os.path.join(DIR_GRAFICOS, "fase6_02_evm_vs_ebn0.png")
    plt.savefig(ruta)
    plt.close()
    print(f"  [GUARDADO] {ruta}")


def generar_grafico_3_ruido_termico(res_exp2: dict[str, list[dict[str, Any]]]):
    """Gráfico 3: BER vs Potencia de Recepción Prx [dBm] con calibración física kTB."""
    plt.figure(figsize=(10, 7), dpi=300)
    colores = {"BFSK": "#D95F02", "BPSK": "#1B9E77", "QPSK": "#7570B3", "16-QAM": "#E7298A"}
    marcadores = {"BFSK": "o", "BPSK": "s", "QPSK": "^", "16-QAM": "d"}

    for mod_nombre, puntos in res_exp2.items():
        prx_vals = [p["potencia_rx_dbm"] for p in puntos]
        ber_vals = []
        for p in puntos:
            if p["es_cero_errores"]:
                ber_vals.append(p["cota_tres"])
            else:
                ber_vals.append(p["ber_simulada"])

        plt.semilogy(
            prx_vals,
            ber_vals,
            label=f"{mod_nombre}",
            color=colores[mod_nombre],
            marker=marcadores[mod_nombre],
            linewidth=1.6,
            markersize=6
        )

    # Umbral de sensibilidad en telecomunicaciones móviles BER = 1e-3
    plt.axhline(1e-3, color="black", linestyle="--", linewidth=1.2, label="Umbral de Sensibilidad (BER = 10⁻³)")
    plt.axhline(5.0e-5, color="#7F7F7F", linestyle=":", linewidth=1.0, label="Cota 3/Nb con 0 errores (5·10⁻⁵)")

    plt.title(
        "Experimento 2: Sensibilidad de Recepción vs Potencia Prx [dBm]\n"
        "(Física: 50 Ω, T=290 K, B=2000 Hz, NF=4.0 dB, PN,fis ≈ -136.96 dBm)",
        fontsize=12,
        fontweight="bold"
    )
    plt.xlabel("Potencia de Recepción Prx [dBm]", fontsize=11)
    plt.ylabel("Tasa de Error de Bit (BER)", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.ylim([2e-5, 0.6])
    plt.xlim([-136.0, -114.0])
    plt.legend(loc="lower left", fontsize=9.5)
    plt.tight_layout()

    ruta = os.path.join(DIR_GRAFICOS, "fase6_03_ber_ruido_termico.png")
    plt.savefig(ruta)
    plt.close()
    print(f"  [GUARDADO] {ruta}")


def generar_grafico_4_cci(res_exp3: dict[str, list[dict[str, Any]]]):
    """Gráfico 4: Sensibilidad a Interferencia Cocanal (BER y EVM vs SIR)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    colores = {"BFSK": "#D95F02", "BPSK": "#1B9E77", "QPSK": "#7570B3", "16-QAM": "#E7298A"}
    marcadores = {"BFSK": "o", "BPSK": "s", "QPSK": "^", "16-QAM": "d"}

    # Subplot 1: BER vs SIR
    for mod_nombre, puntos in res_exp3.items():
        sir_vals = [p["sir_db"] for p in puntos]
        ber_vals = []
        for p in puntos:
            if p["es_cero_errores"]:
                ber_vals.append(p["cota_tres"])
            else:
                ber_vals.append(p["ber_simulada"])

        ax1.semilogy(
            sir_vals,
            ber_vals,
            label=f"{mod_nombre}",
            color=colores[mod_nombre],
            marker=marcadores[mod_nombre],
            linewidth=1.6,
            markersize=5
        )

    ax1.axhline(1e-2, color="gray", linestyle="--", linewidth=1.0, label="Umbral Crítico CCI (BER = 10⁻²)")
    ax1.set_title("BER vs Relación Señal a Interferencia (SIR)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("SIR Cocanal [dB]", fontsize=10)
    ax1.set_ylabel("Tasa de Error de Bit (BER)", fontsize=10)
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.set_ylim([2e-5, 0.6])
    ax1.set_xlim([-0.5, 20.5])
    ax1.legend(loc="lower left", fontsize=9)

    # Subplot 2: EVM (%) vs SIR (Lineales)
    for mod_nombre in ["BPSK", "QPSK", "16-QAM"]:
        puntos = res_exp3[mod_nombre]
        sir_vals = [p["sir_db"] for p in puntos]
        evm_vals = [p["evm_pct"] for p in puntos]

        ax2.plot(
            sir_vals,
            evm_vals,
            label=f"{mod_nombre}",
            color=colores[mod_nombre],
            marker=marcadores[mod_nombre],
            linewidth=1.6,
            markersize=5
        )

    ax2.set_title("EVM RMS (%) vs SIR Cocanal (Constelación)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("SIR Cocanal [dB]", fontsize=10)
    ax2.set_ylabel("EVM RMS [%]", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_xlim([-0.5, 20.5])
    ax2.set_ylim([0.0, 105.0])
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle("Experimento 3: Degradación por Interferencia Cocanal (CCI) (Retardo 7 muestras, Fase π/4, Eb/N0 fondo 15 dB)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    ruta = os.path.join(DIR_GRAFICOS, "fase6_04_ber_evm_cci.png")
    plt.savefig(ruta)
    plt.close()
    print(f"  [GUARDADO] {ruta}")


def generar_grafico_5_ruido_fase_y_aci(res_exp4: dict[str, Any]):
    """Gráfico 5: Ruido de Fase (4A) e Interferencia Canal Adyacente (4B)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    colores = {"BFSK": "#D95F02", "BPSK": "#1B9E77", "QPSK": "#7570B3", "16-QAM": "#E7298A"}
    marcadores = {"BFSK": "o", "BPSK": "s", "QPSK": "^", "16-QAM": "d"}

    # 4A: Ruido de Fase (Lineales)
    for mod_nombre in ["BPSK", "QPSK", "16-QAM"]:
        puntos = res_exp4["ruido_fase"][mod_nombre]
        desv_vals = [p["desviacion_grados"] for p in puntos]
        ber_vals = [p["cota_tres"] if p["es_cero_errores"] else p["ber_simulada"] for p in puntos]

        ax1.semilogy(
            desv_vals,
            ber_vals,
            label=f"{mod_nombre}",
            color=colores[mod_nombre],
            marker=marcadores[mod_nombre],
            linewidth=1.6,
            markersize=5
        )

    ax1.set_title("4A: Impacto del Ruido de Fase (Eb/N0 fondo = 12 dB)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Desviación Estándar de Fase σ_θ [grados]", fontsize=10)
    ax1.set_ylabel("BER", fontsize=10)
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.set_ylim([2e-5, 0.6])
    ax1.legend(loc="upper left", fontsize=9)
    ax1.annotate(
        "BFSK: N/A (demodulador no coherente\ninmune a fase de portadora)",
        xy=(0.05, 0.15),
        xycoords="axes fraction",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E6F2FF", edgecolor="#6BAED6", alpha=0.9)
    )

    # 4B: Interferencia ACI
    for mod_nombre, puntos in res_exp4["aci"].items():
        sir_vals = [p["sir_aci_db"] for p in puntos]
        ber_vals = [p["cota_tres"] if p["es_cero_errores"] else p["ber_simulada"] for p in puntos]

        ax2.semilogy(
            sir_vals,
            ber_vals,
            label=f"{mod_nombre}",
            color=colores[mod_nombre],
            marker=marcadores[mod_nombre],
            linewidth=1.6,
            markersize=5
        )

    ax2.set_title("4B: Interferencia de Canal Adyacente (Δf = 1500 Hz)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("SIR Canal Adyacente [dB]", fontsize=10)
    ax2.set_ylabel("BER", fontsize=10)
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.set_ylim([2e-5, 0.6])
    ax2.set_xlim([-0.5, 16.5])
    ax2.legend(loc="lower left", fontsize=9)

    fig.suptitle("Experimento 4: Degradación por Ruido de Fase e Interferencia ACI (Nb = 60.000 bits)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    ruta = os.path.join(DIR_GRAFICOS, "fase6_05_ruido_fase_y_aci.png")
    plt.savefig(ruta)
    plt.close()
    print(f"  [GUARDADO] {ruta}")


def generar_grafico_6_tradeoff(matriz: list[dict[str, Any]]):
    """Gráfico 6: Comparación multidimensional de trade-off (barras agrupadas normalizadas)."""
    plt.figure(figsize=(11, 6), dpi=300)

    modulaciones_nombres = [m["modulacion"] for m in matriz]
    x = np.arange(len(modulaciones_nombres))
    ancho = 0.22

    # Métricas normalizadas a escala 0-10 para comparación directa
    # 1. Eficiencia Espectral (Carson para BFSK 0.25, RF para lineales 0.5, 1.0, 2.0)
    eficiencias = [m["eficiencia_principal_bps_hz"] for m in matriz]
    ef_norm = [e * 5.0 for e in eficiencias]  # max 2.0 -> 10.0

    # 2. Inmunidad al Ruido: Inversa del Eb/N0 requerido para 1e-3 (menos dB es mejor -> mayor nota)
    ebn0_reqs = [m["ebn0_req_ber_1e3_db"] for m in matriz]
    inmunidad_ruido = [max(0.0, 14.0 - float(eb)) for eb in ebn0_reqs]

    # 3. Tolerancia a CCI: Inversa del SIR requerido para 1e-2 (menos SIR requerido es mejor)
    sir_reqs = [m["tolerancia_cci_sir_db"] for m in matriz]
    tolerancia_cci = [max(0.0, 20.0 - float(s)) for s in sir_reqs]

    plt.bar(x - ancho, ef_norm, width=ancho, label="Eficiencia Espectral (Normalizada x5)", color="#1F77B4")
    plt.bar(x, inmunidad_ruido, width=ancho, label="Inmunidad al Ruido (14 - Eb/N0_req [dB])", color="#2CA02C")
    plt.bar(x + ancho, tolerancia_cci, width=ancho, label="Tolerancia CCI (20 - SIR_req [dB])", color="#FF7F0E")

    plt.title("Experimento 5: Matriz de Compromiso Global (Trade-off) Multidimensional", fontsize=12, fontweight="bold")
    plt.xlabel("Esquema de Modulación", fontsize=11)
    plt.ylabel("Puntuación Relativa de Desempeño [0 - 10]", fontsize=11)
    plt.xticks(x, modulaciones_nombres, fontsize=11, fontweight="bold")
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.ylim([0, 12])

    for i in range(len(modulaciones_nombres)):
        plt.text(x[i] - ancho, ef_norm[i] + 0.2, f"{eficiencias[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold")
        plt.text(x[i], inmunidad_ruido[i] + 0.2, f"{ebn0_reqs[i]:.1f}dB", ha="center", fontsize=8.5, fontweight="bold")
        plt.text(x[i] + ancho, tolerancia_cci[i] + 0.2, f"{sir_reqs[i]:.1f}dB", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    ruta = os.path.join(DIR_GRAFICOS, "fase6_06_tradeoff_radar_barras.png")
    plt.savefig(ruta)
    plt.close()
    print(f"  [GUARDADO] {ruta}")


# =====================================================================
# BLOQUE 3: EXPORTACIÓN DE TABLAS (CSV Y MARKDOWN)
# =====================================================================

def exportar_tablas_fase6(
    res_exp1: dict[str, list[dict[str, Any]]],
    res_exp2: dict[str, list[dict[str, Any]]],
    res_exp3: dict[str, list[dict[str, Any]]],
    res_exp4: dict[str, Any],
    matriz_tradeoff: list[dict[str, Any]]
):
    """Genera todas las tablas de resultados en formato CSV y Markdown."""
    print("\n" + "=" * 76)
    print(" BLOQUE 4: EXPORTACIÓN DE TABLAS (CSV Y MARKDOWN)")
    print("=" * 76)

    # 1. Tabla Experimento 1 (AWGN)
    ruta_csv1 = os.path.join(DIR_TABLAS, "fase6_experimento1_awgn.csv")
    ruta_md1 = os.path.join(DIR_TABLAS, "fase6_experimento1_awgn.md")
    filas_exp1 = []
    for mod, puntos in res_exp1.items():
        for p in puntos:
            evm_str = f"{p['evm_pct']:.2f}%" if p["evm_pct"] is not None else "N/A"
            filas_exp1.append({
                "Modulacion": mod,
                "EbN0_dB": f"{p['ebn0_db']:.1f}",
                "EsN0_dB": f"{p['esn0_db']:.2f}",
                "BER_Simulada": f"{p['ber_simulada']:.6e}",
                "Errores": p["errores"],
                "Total_Bits": p["total_bits"],
                "Wilson_Upper_95": f"{p['wilson_upper']:.6e}",
                "Cota_Tres_95": f"{p['cota_tres']:.6e}" if p["cota_tres"] is not None else "N/A",
                "BER_Teorica": f"{p['ber_teorica']:.6e}",
                "Etiqueta_Teorica": p["etiqueta_teorica"],
                "EVM_RMS": evm_str
            })

    # Guardar CSV 1
    import csv
    with open(ruta_csv1, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_exp1[0].keys()))
        writer.writeheader()
        writer.writerows(filas_exp1)

    # Guardar MD 1
    with open(ruta_md1, "w", encoding="utf-8") as f:
        f.write("# FASE 6 — EXPERIMENTO 1: RESULTADOS AWGN VS MODELOS TEÓRICOS\n\n")
        f.write("| Modulación | Eb/N0 [dB] | Es/N0 [dB] | BER Simulada | Errores | Wilson Sup 95% | Cota 3/Nb | BER Teórica | Etiqueta Teórica | EVM RMS |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |\n")
        for r in filas_exp1:
            f.write(f"| {r['Modulacion']} | {r['EbN0_dB']} | {r['EsN0_dB']} | {r['BER_Simulada']} | {r['Errores']} | {r['Wilson_Upper_95']} | {r['Cota_Tres_95']} | {r['BER_Teorica']} | {r['Etiqueta_Teorica']} | {r['EVM_RMS']} |\n")

    print(f"  [GUARDADO] {ruta_csv1} y {ruta_md1}")

    # 2. Tabla Experimento 2 (Ruido Térmico)
    ruta_csv2 = os.path.join(DIR_TABLAS, "fase6_experimento2_termico.csv")
    ruta_md2 = os.path.join(DIR_TABLAS, "fase6_experimento2_termico.md")
    filas_exp2 = []
    for mod, puntos in res_exp2.items():
        for p in puntos:
            evm_str = f"{p['evm_pct']:.2f}%" if p["evm_pct"] is not None else "N/A"
            filas_exp2.append({
                "Modulacion": mod,
                "Prx_dBm": f"{p['potencia_rx_dbm']:.1f}",
                "BER_Simulada": f"{p['ber_simulada']:.6e}",
                "Errores": p["errores"],
                "Total_Bits": p["total_bits"],
                "Wilson_Upper_95": f"{p['wilson_upper']:.6e}",
                "Cota_Tres_95": f"{p['cota_tres']:.6e}" if p["cota_tres"] is not None else "N/A",
                "EVM_RMS": evm_str
            })

    with open(ruta_csv2, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_exp2[0].keys()))
        writer.writeheader()
        writer.writerows(filas_exp2)

    with open(ruta_md2, "w", encoding="utf-8") as f:
        f.write("# FASE 6 — EXPERIMENTO 2: SENSIBILIDAD DE RECEPCIÓN Y RUIDO TÉRMICO (kTB)\n\n")
        f.write("Parámetros: R0 = 50 Ω, T = 290 K, B = 2000 Hz, NF = 4.0 dB, PN,fis ≈ -136.96 dBm.\n\n")
        f.write("| Modulación | Prx [dBm] | BER Simulada | Errores | Wilson Sup 95% | Cota 3/Nb | EVM RMS |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in filas_exp2:
            f.write(f"| {r['Modulacion']} | {r['Prx_dBm']} | {r['BER_Simulada']} | {r['Errores']} | {r['Wilson_Upper_95']} | {r['Cota_Tres_95']} | {r['EVM_RMS']} |\n")

    print(f"  [GUARDADO] {ruta_csv2} y {ruta_md2}")

    # 3. Tabla Experimento 3 (CCI)
    ruta_csv3 = os.path.join(DIR_TABLAS, "fase6_experimento3_cci.csv")
    ruta_md3 = os.path.join(DIR_TABLAS, "fase6_experimento3_cci.md")
    filas_exp3 = []
    for mod, puntos in res_exp3.items():
        for p in puntos:
            evm_str = f"{p['evm_pct']:.2f}%" if p["evm_pct"] is not None else "N/A"
            filas_exp3.append({
                "Modulacion": mod,
                "SIR_dB": f"{p['sir_db']:.1f}",
                "EbN0_Fondo_dB": f"{p['ebn0_fondo_db']:.1f}",
                "BER_Simulada": f"{p['ber_simulada']:.6e}",
                "Errores": p["errores"],
                "Total_Bits": p["total_bits"],
                "Wilson_Upper_95": f"{p['wilson_upper']:.6e}",
                "Cota_Tres_95": f"{p['cota_tres']:.6e}" if p["cota_tres"] is not None else "N/A",
                "EVM_RMS": evm_str
            })

    with open(ruta_csv3, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_exp3[0].keys()))
        writer.writeheader()
        writer.writerows(filas_exp3)

    with open(ruta_md3, "w", encoding="utf-8") as f:
        f.write("# FASE 6 — EXPERIMENTO 3: SENSIBILIDAD A INTERFERENCIA COCANAL (CCI)\n\n")
        f.write("Parámetros: Retardo 7 muestras (0.35 Ts), Fase π/4, Eb/N0 fondo = 15.0 dB.\n\n")
        f.write("| Modulación | SIR [dB] | BER Simulada | Errores | Wilson Sup 95% | Cota 3/Nb | EVM RMS |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in filas_exp3:
            f.write(f"| {r['Modulacion']} | {r['SIR_dB']} | {r['BER_Simulada']} | {r['Errores']} | {r['Wilson_Upper_95']} | {r['Cota_Tres_95']} | {r['EVM_RMS']} |\n")

    print(f"  [GUARDADO] {ruta_csv3} y {ruta_md3}")

    # 4. Tabla Experimento 4 (Ruido de Fase y ACI)
    ruta_csv4 = os.path.join(DIR_TABLAS, "fase6_experimento4_aci_fase.csv")
    ruta_md4 = os.path.join(DIR_TABLAS, "fase6_experimento4_aci_fase.md")
    filas_exp4 = []
    # 4A. Fase
    for mod, puntos in res_exp4["ruido_fase"].items():
        for p in puntos:
            evm_str = f"{p['evm_pct']:.2f}%" if p["evm_pct"] is not None else "N/A"
            filas_exp4.append({
                "Subexperimento": "4A_Ruido_Fase",
                "Modulacion": mod,
                "Parametro_Perturbacion": f"sigma_theta={p['desviacion_grados']:.1f}deg",
                "EbN0_Fondo_dB": f"{p['ebn0_fondo_db']:.1f}",
                "BER_Simulada": f"{p['ber_simulada']:.6e}",
                "Errores": p["errores"],
                "Total_Bits": p["total_bits"],
                "Wilson_Upper_95": f"{p['wilson_upper']:.6e}",
                "Cota_Tres_95": f"{p['cota_tres']:.6e}" if p["cota_tres"] is not None else "N/A",
                "EVM_RMS": evm_str
            })
    # 4B. ACI
    for mod, puntos in res_exp4["aci"].items():
        for p in puntos:
            evm_str = f"{p['evm_pct']:.2f}%" if p["evm_pct"] is not None else "N/A"
            filas_exp4.append({
                "Subexperimento": "4B_ACI",
                "Modulacion": mod,
                "Parametro_Perturbacion": f"SIR={p['sir_aci_db']:.1f}dB, Df={p['delta_f_hz']:.0f}Hz",
                "EbN0_Fondo_dB": f"{p['ebn0_fondo_db']:.1f}",
                "BER_Simulada": f"{p['ber_simulada']:.6e}",
                "Errores": p["errores"],
                "Total_Bits": p["total_bits"],
                "Wilson_Upper_95": f"{p['wilson_upper']:.6e}",
                "Cota_Tres_95": f"{p['cota_tres']:.6e}" if p["cota_tres"] is not None else "N/A",
                "EVM_RMS": evm_str
            })

    with open(ruta_csv4, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_exp4[0].keys()))
        writer.writeheader()
        writer.writerows(filas_exp4)

    with open(ruta_md4, "w", encoding="utf-8") as f:
        f.write("# FASE 6 — EXPERIMENTO 4: RUIDO DE FASE Y CANAL ADYACENTE (ACI)\n\n")
        f.write("Parte 4A: Ruido de Fase (0° a 25°, Eb/N0 fondo = 12 dB). BFSK es N/A (demodulación no coherente pasobanda).\n")
        f.write("Parte 4B: Interferencia ACI (SIR 0 a 16 dB, Δf = 1500 Hz, Eb/N0 fondo = 15 dB).\n\n")
        f.write("| Subexperimento | Modulación | Parámetro Perturbación | BER Simulada | Errores | Wilson Sup 95% | Cota 3/Nb | EVM RMS |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in filas_exp4:
            f.write(f"| {r['Subexperimento']} | {r['Modulacion']} | {r['Parametro_Perturbacion']} | {r['BER_Simulada']} | {r['Errores']} | {r['Wilson_Upper_95']} | {r['Cota_Tres_95']} | {r['EVM_RMS']} |\n")

    print(f"  [GUARDADO] {ruta_csv4} y {ruta_md4}")

    # 5. Tabla Resumen Trade-off
    ruta_csv5 = os.path.join(DIR_TABLAS, "fase6_resumen_tradeoff.csv")
    ruta_md5 = os.path.join(DIR_TABLAS, "fase6_resumen_tradeoff.md")

    with open(ruta_csv5, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(matriz_tradeoff[0].keys()))
        writer.writeheader()
        writer.writerows(matriz_tradeoff)

    with open(ruta_md5, "w", encoding="utf-8") as f:
        f.write("# FASE 6 — RESUMEN GLOBAL DE COMPROMISO (TRADE-OFF)\n\n")
        f.write("| Modulación | Bits/Símbolo | Eficiencia [bps/Hz] | Convención Ancho Banda | Eb/N0 (BER≤10⁻³) [dB] | Sensibilidad Prx [dBm] | SIR CCI (BER≤10⁻²) [dB] | Complejidad [Ops/Símbolo] | Dominio Receptor | Evaluación Robustez |\n")
        f.write("| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |\n")
        for m in matriz_tradeoff:
            f.write(
                f"| {m['modulacion']} | {m['bits_por_simbolo']} | {m['eficiencia_principal_bps_hz']:.2f} | "
                f"{m['convencion_ancho_banda']} | {m['ebn0_req_ber_1e3_db']} | {m['sensibilidad_prx_dbm']} | "
                f"{m['tolerancia_cci_sir_db']} | {m['complejidad_ops_simbolo']} | {m['dominio_receptor']} | "
                f"{m['evaluacion_robustez']} |\n"
            )

    print(f"  [GUARDADO] {ruta_csv5} y {ruta_md5}")


# =====================================================================
# BLOQUE 4: PRUEBAS DE NO REGRESIÓN (FASES 1 A 5)
# =====================================================================

def ejecutar_pruebas_no_regresion():
    """Ejecuta los scripts de prueba de las fases 1 a 5 para certificar cero regresión."""
    print("\n" + "=" * 76)
    print(" BLOQUE 5: VERIFICACIÓN DE NO REGRESIÓN (FASES 1 A 5)")
    print("=" * 76)

    scripts = [
        "verificar_fase1.py",
        "test_fase2.py",
        "test_fase3.py",
        "test_fase4.py",
        "test_fase5.py"
    ]

    python_exe = sys.executable

    for s in scripts:
        ruta_script = os.path.join(DIR_BASE, s)
        if not os.path.exists(ruta_script):
            raise FileNotFoundError(f"Script no encontrado: {ruta_script}")

        print(f"  Ejecutando {s:<22} ...", end=" ", flush=True)
        res = subprocess.run([python_exe, ruta_script], capture_output=True, text=True)
        if res.returncode == 0:
            print("[PASS - 100% EXITOSO]")
        else:
            print(f"[FAIL - Código {res.returncode}]")
            print("--- SALIDA ERROR ---")
            print(res.stderr)
            raise RuntimeError(f"Fallo en prueba de no regresión: {s}")


# =====================================================================
# FUNCIÓN PRINCIPAL DE EJECUCIÓN
# =====================================================================

def main():
    print("*" * 76)
    print(" INICIANDO EJECUCIÓN INTEGRAL FASE 6: MÉTRICAS Y EVALUACIÓN EXPERIMENTAL")
    print("*" * 76)

    # 1. Bloque unitario
    res_unitario = ejecutar_bloque_unitario()

    # 2. Bloque experimental Monte Carlo (Nb = 60.000 bits)
    print("\n" + "=" * 76)
    print(" BLOQUE 2: EJECUCIÓN DE SIMULACIONES MONTE CARLO (Nb = 60.000 bits)")
    print("=" * 76)

    print("  -> Ejecutando Experimento 1: AWGN vs Modelos Teóricos (Eb/N0 = 0..12 dB) ...")
    res_exp1 = experimentos.ejecutar_experimento_1_awgn(n_bits=60000, seed_base=1000)

    print("  -> Ejecutando Experimento 2: Ruido Térmico Físico kTB (Prx = -135..-115 dBm) ...")
    res_exp2 = experimentos.ejecutar_experimento_2_ruido_termico(n_bits=60000, seed_base=2000)

    print("  -> Ejecutando Experimento 3: Interferencia Cocanal CCI (SIR = 0..20 dB) ...")
    res_exp3 = experimentos.ejecutar_experimento_3_interferencia_cci(n_bits=60000, seed_base=3000)

    print("  -> Ejecutando Experimento 4: Ruido de Fase (0..25°) y ACI (SIR = 0..16 dB) ...")
    res_exp4 = experimentos.ejecutar_experimento_4_aci_y_fase(n_bits=60000, seed_base=4000)

    print("  -> Consolidando Experimento 5: Matriz de Compromiso Global (Trade-off) ...")
    matriz_tradeoff = experimentos.consolidar_matriz_tradeoff(res_exp1, res_exp2, res_exp3)

    # 3. Bloque gráfico
    print("\n" + "=" * 76)
    print(" BLOQUE 3: GENERACIÓN Y GUARDADO DE GRÁFICOS EXPERIMENTALES (PNG)")
    print("=" * 76)
    generar_grafico_1_ber_awgn(res_exp1)
    generar_grafico_2_evm_vs_ebn0(res_exp1)
    generar_grafico_3_ruido_termico(res_exp2)
    generar_grafico_4_cci(res_exp3)
    generar_grafico_5_ruido_fase_y_aci(res_exp4)
    generar_grafico_6_tradeoff(matriz_tradeoff)

    # 4. Bloque tablas
    exportar_tablas_fase6(res_exp1, res_exp2, res_exp3, res_exp4, matriz_tradeoff)

    # 5. Bloque de no regresión
    ejecutar_pruebas_no_regresion()

    print("\n" + "*" * 76)
    print(" FASE 6 COMPLETADA AL 100% EXITOSAMENTE")
    print("*" * 76)


if __name__ == "__main__":
    main()

