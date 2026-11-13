"""
test_fase7.py
Suite de Pruebas Unitarias, Integración Experimental y No Regresión para la FASE 7.

Desarrollado para la asignatura Electiva de Comunicaciones Móviles (Python 3.13.5).
Fase 7: Técnicas de Mitigación y Mejora del Rendimiento.

Contenido:
1. Bloque 1: 8 pruebas unitarias matemáticas de mitigacion.py (Filtro FIR, Ecualizador ZF, Diversidad MRC/SC).
2. Bloque 2: Ejecución Monte Carlo de los 3 experimentos comparativos (Nb = 60.000 bits).
3. Bloque 3: Generación y guardado de 3 gráficos comparativos de alta resolución en resultados/graficos/.
4. Bloque 4: Exportación de tablas completas en CSV y Markdown en resultados/tablas/.
5. Bloque 5: Verificación estricta de no regresión de las Fases 1 a 6.
"""

from __future__ import annotations
import os
import sys
import math
import subprocess
import csv
import numpy as np

# Configuración del directorio de caché de matplotlib antes de importar pyplot
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_CACHE_MPL = os.path.join(DIR_BASE, ".cache", "matplotlib")
os.makedirs(DIR_CACHE_MPL, exist_ok=True)
os.environ["MPLCONFIGDIR"] = DIR_CACHE_MPL

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Módulos del proyecto
import fundamentos
import modulaciones
import demodulacion
import ruido
import interferencias
import canales
import metricas
import experimentos
import mitigacion
import experimentos_fase7


# Carpetas de salida
DIR_GRAFICOS = os.path.join(DIR_BASE, "resultados", "graficos")
DIR_TABLAS = os.path.join(DIR_BASE, "resultados", "tablas")
os.makedirs(DIR_GRAFICOS, exist_ok=True)
os.makedirs(DIR_TABLAS, exist_ok=True)


# =====================================================================
# BLOQUE 1: PRUEBAS UNITARIAS MATEMÁTICAS (8 PRUEBAS)
# =====================================================================

def test_01_diseno_filtro_fir_atenuacion() -> dict:
    """Verifica diseño de filtros FIR (paso bajo y paso banda), atenuación ACI y preservación útil."""
    fs = 16000.0
    fc = 1000.0
    orden = 48
    h_fir = mitigacion.disenar_filtro_fir_pasabajos(fs=fs, f_corte=fc, orden=orden)

    assert len(h_fir) == orden + 1, f"Longitud incorrecta: {len(h_fir)}"
    assert np.isclose(np.sum(h_fir), 1.0), "Ganancia DC no normalizada a 1.0"
    # Simetría de fase lineal: h[n] == h[N - 1 - n]
    assert np.allclose(h_fir, h_fir[::-1]), "El filtro FIR no posee fase lineal simétrica"

    info_at = mitigacion.medir_atenuacion_filtro(h_fir, fs=fs, f_interferencia=1500.0)
    at_db = info_at["atenuacion_relativa_db"]
    # Debe atenuar al menos 35 dB a 1500 Hz (frecuencia de la ACI)
    assert at_db > 35.0, f"Atenuación insuficiente a 1500 Hz: {at_db:.2f} dB"

    # Verificar filtro paso banda para BFSK RF (portadora central 3000 Hz, tonos 2000/4000 Hz, ACI en 4500 Hz)
    h_bp = mitigacion.disenar_filtro_fir_pasabanda(fs=fs, f_baja=1800.0, f_alta=4200.0, orden=72)
    info_bp = mitigacion.medir_atenuacion_filtro(
        h_bp, fs=fs, f_interferencia=4500.0, f_referencia=3000.0
    )
    assert info_bp["atenuacion_relativa_db"] > 25.0, (
        f"Atenuación paso banda insuficiente a 4500 Hz: {info_bp['atenuacion_relativa_db']:.2f} dB"
    )
    info_tono_f0 = mitigacion.medir_atenuacion_filtro(
        h_bp, fs=fs, f_interferencia=2000.0, f_referencia=3000.0
    )
    assert info_tono_f0["ganancia_interf_db"] > -2.0, "El paso banda atenúa en exceso los tonos útiles BFSK."

    # Verificar que en banda base limpia no se eliminen componentes útiles (ISI residual < 2% en ventana [2:14])
    bits_chk = fundamentos.generar_bits(2000, semilla=99)
    mod_chk = modulaciones.modular_16qam(bits_chk, fc=2000.0, sps=16, fs=fs, normalizar=True)
    bb_f = mitigacion.aplicar_filtro_fir(mod_chk.senal_bb, h_fir, compensar_retardo=True)
    sym_f = np.mean(bb_f.reshape(-1, 16)[:, 2:14], axis=1)
    evm_isi, _ = metricas.calcular_evm_simbolos(mod_chk.simbolos, sym_f, "16-QAM")
    assert evm_isi is not None and evm_isi < 2.0, f"Distorsión útil excesiva tras FIR: EVM={evm_isi:.2f}%"

    return {
        "prueba": "01_Diseno_FIR_Atenuacion",
        "orden": orden,
        "atenuacion_1500hz_db": at_db,
        "atenuacion_bp_4500hz_db": info_bp["atenuacion_relativa_db"],
        "isi_util_16qam_pct": evm_isi,
        "estado": "PASS"
    }


def test_02_compensacion_retardo_fir() -> dict:
    """Verifica que la compensación de retardo de grupo alinee la señal en fase y bordes."""
    fs = 16000.0
    fc = 1000.0
    orden = 48
    h_fir = mitigacion.disenar_filtro_fir_pasabajos(fs=fs, f_corte=fc, orden=orden)

    t = np.arange(1000) / fs
    f_tono = 200.0  # Tono dentro de la banda de paso
    s_in = np.cos(2 * np.pi * f_tono * t)

    s_out = mitigacion.aplicar_filtro_fir(s_in, h_fir, compensar_retardo=True)

    assert len(s_out) == len(s_in), "Longitud alterada tras filtrado FIR"
    # Verificar alineamiento en fase tanto en estado estacionario como hasta el final de la trama
    corr = np.corrcoef(s_in[50:950], s_out[50:950])[0, 1]
    assert corr > 0.99, f"Desfase residual excesivo tras compensación: corr={corr}"
    assert np.std(s_out[-24:]) > 0.01, "El borde final quedó congelado artificialmente"

    return {"prueba": "02_Compensacion_Retardo_FIR", "correlacion": corr, "estado": "PASS"}


def test_03_diseno_ecualizador_zf_convolucion() -> dict:
    """Verifica que el ecualizador ZF cancele la ISI del canal h=[1.0, 0.35]."""
    h_canal = np.array([1.0, 0.35])
    info_zf = mitigacion.disenar_ecualizador_zf(h_canal, longitud_w=7, retardo_decision=0)

    c = info_zf["respuesta_combinada_c"]
    tap0 = float(np.real(c[0]))
    isi_residual_db = info_zf["isi_residual_db"]

    assert np.isclose(tap0, 1.0, atol=1e-3), f"Tap principal no unitario: {tap0}"
    assert isi_residual_db < -30.0, f"ISI residual excesiva: {isi_residual_db:.2f} dB"

    return {
        "prueba": "03_Diseno_ZF_Convolucion",
        "tap_principal": tap0,
        "isi_residual_db": isi_residual_db,
        "estado": "PASS"
    }


def test_04_amplificacion_ruido_zf_nef() -> dict:
    """Verifica el cálculo analítico de amplificación de ruido (NEF = ||w||^2)."""
    h_canal = np.array([1.0, 0.35])
    info_zf = mitigacion.disenar_ecualizador_zf(h_canal, longitud_w=7, retardo_decision=0)

    nef_lin = info_zf["nef_lineal"]
    nef_db = info_zf["nef_db"]

    # Con h=[1, 0.35], w = [1, -0.35, 0.1225, -0.0429, ...], ||w||^2 ≈ 1.1396 (+0.568 dB)
    assert np.isclose(nef_lin, 1.139595, rtol=1e-4), f"NEF erróneo: {nef_lin}"
    assert np.isclose(nef_db, 0.5675, rtol=1e-3), f"NEF dB erróneo: {nef_db}"

    return {
        "prueba": "04_Amplificacion_Ruido_ZF_NEF",
        "nef_lineal": nef_lin,
        "nef_db": nef_db,
        "estado": "PASS"
    }


def test_05_independencia_ramas_rayleigh() -> dict:
    """Verifica que los canales Rayleigh en ramas distintas sean estadísticamente independientes."""
    n_sim = 40000
    h_ramas = mitigacion.generar_canales_rayleigh_independientes(n_simbolos=n_sim, n_ramas=2, seed=42)

    pot_r1 = float(np.mean(np.abs(h_ramas[0]) ** 2))
    pot_r2 = float(np.mean(np.abs(h_ramas[1]) ** 2))
    corr = float(np.abs(np.corrcoef(np.abs(h_ramas[0]), np.abs(h_ramas[1]))[0, 1]))

    assert np.isclose(pot_r1, 1.0, atol=0.03), f"Potencia rama 1 no unitaria: {pot_r1}"
    assert np.isclose(pot_r2, 1.0, atol=0.03), f"Potencia rama 2 no unitaria: {pot_r2}"
    assert corr < 0.03, f"Ramas correlacionadas indebidamente: corr={corr}"

    return {
        "prueba": "05_Independencia_Ramas_Rayleigh",
        "pot_r1": pot_r1,
        "pot_r2": pot_r2,
        "correlacion_cruzada": corr,
        "estado": "PASS"
    }


def test_06_combinacion_mrc_ponderacion() -> dict:
    """Verifica que la combinación MRC combine coherentemente las ramas."""
    n_sim = 100
    s_tx = np.ones(n_sim, dtype=complex)
    # Canales constantes arbitrarios para probar la ecuación analítica
    h = np.zeros((2, n_sim), dtype=complex)
    h[0] = 0.8 + 0.6j  # |h1| = 1.0
    h[1] = 0.6 - 0.8j  # |h2| = 1.0
    r = h * s_tx  # Sin ruido

    y_mrc = mitigacion.combinar_diversidad_mrc(r, h)
    assert np.allclose(y_mrc, s_tx), "MRC distorsionó la señal sin ruido."

    return {"prueba": "06_Combinacion_MRC_Ponderacion", "estado": "PASS"}


def test_07_combinacion_sc_seleccion() -> dict:
    """Verifica que SC seleccione estrictamente la rama de mayor ganancia instantánea."""
    n_sim = 4
    s_tx = np.array([1.0, 1.0, 1.0, 1.0], dtype=complex)
    h = np.array([
        [0.2, 0.9, 0.1, 0.8],
        [0.7, 0.3, 0.6, 0.4]
    ], dtype=complex)
    r = h * s_tx

    y_sc = mitigacion.combinar_diversidad_sc(r, h)
    assert np.allclose(y_sc, s_tx), "SC no recuperó la señal deseada."

    return {"prueba": "07_Combinacion_SC_Seleccion", "estado": "PASS"}


def test_08_integracion_mini_fase7() -> dict:
    """Verifica corrida en miniatura de los 3 experimentos de mitigación (Nb=4000 bits)."""
    r1 = experimentos_fase7.ejecutar_experimento_1_filtrado(sir_valores_db=[0.0, 10.0], n_bits=4000)
    r2 = experimentos_fase7.ejecutar_experimento_2_ecualizacion(ebn0_valores_db=[0.0, 10.0], n_bits=4000)
    r3 = experimentos_fase7.ejecutar_experimento_3_diversidad(ebn0_valores_db=[0.0, 10.0], n_bits=4000)

    assert len(r1["resultados"]["sin_filtro"]["QPSK"]) == 2
    assert len(r1["resultados"]["con_filtro"]["BFSK"]) == 2
    assert r1["resultados"]["con_filtro"]["BFSK"][0]["ber"] < r1["resultados"]["sin_filtro"]["BFSK"][0]["ber"]
    assert r1["resultados"]["con_filtro"]["16-QAM"][0]["evm_pct"] < r1["resultados"]["sin_filtro"]["16-QAM"][0]["evm_pct"]
    assert len(r2["resultados"]["con_ecualizador_zf"]["16-QAM"]) == 2
    assert len(r3["resultados"]["MRC"]["BPSK"]) == 2

    return {"prueba": "08_Integracion_Mini_Fase7", "estado": "PASS"}


def ejecutar_bloque_unitario() -> list[dict]:
    """Ejecuta el bloque de pruebas unitarias."""
    print("\n" + "=" * 76)
    print(" BLOQUE 1: PRUEBAS UNITARIAS DE TÉCNICAS DE MITIGACIÓN (FASE 7)")
    print("=" * 76)
    pruebas = [
        test_01_diseno_filtro_fir_atenuacion,
        test_02_compensacion_retardo_fir,
        test_03_diseno_ecualizador_zf_convolucion,
        test_04_amplificacion_ruido_zf_nef,
        test_05_independencia_ramas_rayleigh,
        test_06_combinacion_mrc_ponderacion,
        test_07_combinacion_sc_seleccion,
        test_08_integracion_mini_fase7
    ]
    resultados = []
    for p in pruebas:
        res = p()
        resultados.append(res)
        print(f"  [OK] {res['prueba']:<38} -> ESTADO: {res['estado']}")
    return resultados


# =====================================================================
# BLOQUE 2: GENERACIÓN DE GRÁFICOS COMPARATIVOS ANTES/DESPUÉS
# =====================================================================

def generar_grafico_1_filtrado(res_exp1: dict):
    """Gráfico 1: Desempeño BER y EVM vs SIR ACI (Sin Filtro vs Con Filtro FIR)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    colores = {
        "BFSK": "#D95F02",
        "BPSK": "#1B9E77",
        "QPSK": "#7570B3",
        "16-QAM": "#E7298A",
    }

    res = res_exp1["resultados"]
    info_f = res_exp1["info_filtro"]
    info_bp = res_exp1["info_filtro_bfsk"]

    for mod_nombre in ["BFSK", "BPSK", "QPSK", "16-QAM"]:
        p_sin = res["sin_filtro"][mod_nombre]
        p_con = res["con_filtro"][mod_nombre]

        sir_vals = [p["sir_db"] for p in p_sin]
        ber_sin = [p["cota_tres"] if p["es_cero_errores"] else p["ber"] for p in p_sin]
        ber_con = [p["cota_tres"] if p["es_cero_errores"] else p["ber"] for p in p_con]

        # Subplot 1: BER (todas las modulaciones)
        ax1.semilogy(sir_vals, ber_sin, label=f"{mod_nombre} Sin Filtro", color=colores[mod_nombre], linestyle="--", marker="x", markersize=6)
        ax1.semilogy(sir_vals, ber_con, label=f"{mod_nombre} Con Filtro FIR", color=colores[mod_nombre], linestyle="-", marker="o", markersize=5, linewidth=1.8)

        # Subplot 2: EVM (solo modulaciones con constelación I/Q)
        if mod_nombre != "BFSK":
            evm_sin = [p["evm_pct"] for p in p_sin]
            evm_con = [p["evm_pct"] for p in p_con]
            ax2.plot(sir_vals, evm_sin, label=f"{mod_nombre} Sin Filtro", color=colores[mod_nombre], linestyle="--", marker="x", markersize=6)
            ax2.plot(sir_vals, evm_con, label=f"{mod_nombre} Con Filtro FIR", color=colores[mod_nombre], linestyle="-", marker="o", markersize=5, linewidth=1.8)

    ax1.axhline(5e-5, color="#7F7F7F", linestyle=":", linewidth=1.0, label="Cota 3/Nb (0 errores = 5·10⁻⁵)")
    ax1.set_title("Tasa de Error de Bit (BER) vs SIR ACI", fontsize=11, fontweight="bold")
    ax1.set_xlabel("SIR de Canal Adyacente [dB]", fontsize=10)
    ax1.set_ylabel("BER", fontsize=10)
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.set_ylim([2e-5, 0.3])
    ax1.legend(loc="upper right", fontsize=8, ncol=2)

    ax2.set_title("Magnitud del Vector de Error (EVM RMS %) vs SIR ACI", fontsize=11, fontweight="bold")
    ax2.set_xlabel("SIR de Canal Adyacente [dB]", fontsize=10)
    ax2.set_ylabel("EVM RMS [%]", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.annotate(
        "BFSK: EVM N/A (detección no coherente RF;\nsu mejora BER se muestra en el panel izq.)",
        xy=(0.04, 0.05),
        xycoords="axes fraction",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF2CC", edgecolor="#D6B656", alpha=0.9)
    )

    fig.suptitle(
        f"Experimento 1: Mitigación de ACI (Δf={res_exp1['delta_f_hz']:.0f} Hz) mediante Filtros FIR\n"
        f"(BB: LPF Ord.{res_exp1['orden_filtro']}, fc={res_exp1['f_corte_hz']:.0f} Hz, Aten. {info_f['atenuacion_relativa_db']:.1f} dB | "
        f"BFSK: BPF Ord.{res_exp1['orden_bfsk']}, Aten. {info_bp['atenuacion_relativa_db']:.1f} dB)",
        fontsize=11.5,
        fontweight="bold"
    )
    plt.tight_layout()

    ruta = os.path.join(DIR_GRAFICOS, "fase7_01_filtrado_antes_despues.png")
    plt.savefig(ruta)
    plt.close()
    print(f"  [GUARDADO] {ruta}")


def generar_grafico_2_ecualizacion(res_exp2: dict):
    """Gráfico 2: Desempeño BER y respuesta combinada de canal (Sin Ecualizar vs Con ZF)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    colores = {"BPSK": "#1B9E77", "QPSK": "#7570B3", "16-QAM": "#E7298A"}

    res = res_exp2["resultados"]
    info_zf = res_exp2["info_zf"]

    for mod_nombre in ["BPSK", "QPSK", "16-QAM"]:
        p_sin = res["sin_ecualizar"][mod_nombre]
        p_con = res["con_ecualizador_zf"][mod_nombre]

        ebn0_vals = [p["ebn0_db"] for p in p_sin]
        ber_sin = [p["cota_tres"] if p["es_cero_errores"] else p["ber"] for p in p_sin]
        ber_con = [p["cota_tres"] if p["es_cero_errores"] else p["ber"] for p in p_con]

        ax1.semilogy(ebn0_vals, ber_sin, label=f"{mod_nombre} Sin Ecualizar (Piso ISI)", color=colores[mod_nombre], linestyle="--", marker="x", markersize=6)
        ax1.semilogy(ebn0_vals, ber_con, label=f"{mod_nombre} Con Ecualizador ZF", color=colores[mod_nombre], linestyle="-", marker="o", markersize=5, linewidth=1.8)

    ax1.set_title("BER vs Eb/N0 en Canal Dispersivo h=[1.0, 0.35]", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Eb/N0 [dB]", fontsize=10)
    ax1.set_ylabel("BER", fontsize=10)
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.set_ylim([2e-5, 0.6])
    ax1.legend(loc="lower left", fontsize=8.5)

    # Subplot 2: Respuesta Combinada c = h * w
    c_comb = np.real(info_zf["respuesta_combinada_c"])
    indices = np.arange(len(c_comb))
    ax2.stem(indices, c_comb, basefmt="black", linefmt="blue", markerfmt="bo")
    ax2.set_title("Respuesta Impulsional Combinada c = h * w", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Índice de Tap n", fontsize=10)
    ax2.set_ylabel("Amplitud del Tap", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_ylim([-0.2, 1.2])

    ax2.annotate(
        f"Tap principal (n=0): {c_comb[0]:.4f}\n"
        f"ISI residual: {info_zf['isi_residual_db']:.2f} dB\n"
        f"Amplificación de ruido (NEF): {info_zf['nef_db']:+.3f} dB\n"
        f"Longitud ecualizador: {info_zf['longitud_w']} taps",
        xy=(0.35, 0.65),
        xycoords="axes fraction",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF2CC", edgecolor="#D6B656", alpha=0.9)
    )

    fig.suptitle("Experimento 2: Ecualización Zero-Forcing (ZF) FIR frente a Distorsión e ISI (Nb = 60.000 bits)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    ruta = os.path.join(DIR_GRAFICOS, "fase7_02_ecualizacion_antes_despues.png")
    plt.savefig(ruta)
    plt.close()
    print(f"  [GUARDADO] {ruta}")


def generar_grafico_3_diversidad(res_exp3: dict):
    """Gráfico 3: Diversidad de Recepción en Canal Rayleigh (SISO vs SC vs MRC)."""
    plt.figure(figsize=(10, 7), dpi=300)
    colores = {"BPSK": "#1B9E77", "QPSK": "#7570B3"}

    res = res_exp3["resultados"]

    for mod_nombre in ["BPSK", "QPSK"]:
        p_siso = res["SISO"][mod_nombre]
        p_sc = res["SC"][mod_nombre]
        p_mrc = res["MRC"][mod_nombre]

        ebn0_vals = [p["ebn0_db"] for p in p_siso]
        ber_siso = [p["cota_tres"] if p["es_cero_errores"] else p["ber"] for p in p_siso]
        ber_sc = [p["cota_tres"] if p["es_cero_errores"] else p["ber"] for p in p_sc]
        ber_mrc = [p["cota_tres"] if p["es_cero_errores"] else p["ber"] for p in p_mrc]

        plt.semilogy(ebn0_vals, ber_siso, label=f"{mod_nombre} SISO (L=1, Sin Diversidad)", color=colores[mod_nombre], linestyle="--", marker="^", markersize=6)
        plt.semilogy(ebn0_vals, ber_sc, label=f"{mod_nombre} SC (L=2, Selección)", color=colores[mod_nombre], linestyle="-.", marker="s", markersize=5)
        plt.semilogy(ebn0_vals, ber_mrc, label=f"{mod_nombre} MRC (L=2, Máxima Relación)", color=colores[mod_nombre], linestyle="-", marker="o", markersize=6, linewidth=2.0)

    plt.axhline(1e-3, color="black", linestyle=":", linewidth=1.2, label="Umbral de Calidad (BER = 10⁻³)")
    plt.axhline(5e-5, color="#7F7F7F", linestyle=":", linewidth=1.0, label="Cota regla 3/Nb con 0 errores (5·10⁻⁵)")

    plt.title("Experimento 3: Ganancia de Diversidad de Recepción en Canal Rayleigh (60.000 bits)", fontsize=12, fontweight="bold")
    plt.xlabel("Eb/N0 Medio por Bit [dB]", fontsize=11)
    plt.ylabel("Tasa de Error de Bit (BER)", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.ylim([2e-5, 0.4])
    plt.xlim([-0.2, 18.2])
    plt.legend(loc="lower left", fontsize=9.5)
    plt.tight_layout()

    ruta = os.path.join(DIR_GRAFICOS, "fase7_03_diversidad_mrc_vs_siso.png")
    plt.savefig(ruta)
    plt.close()
    print(f"  [GUARDADO] {ruta}")


# =====================================================================
# BLOQUE 3: EXPORTACIÓN DE TABLAS (CSV Y MARKDOWN)
# =====================================================================

def exportar_tablas_fase7(
    res_exp1: dict,
    res_exp2: dict,
    res_exp3: dict
):
    """Exporta todas las tablas de la Fase 7 en formato CSV y Markdown."""
    print("\n" + "=" * 76)
    print(" BLOQUE 4: EXPORTACIÓN DE TABLAS DE MITIGACIÓN (CSV Y MARKDOWN)")
    print("=" * 76)

    # 1. Tabla Experimento 1: Filtrado FIR vs ACI
    ruta_csv1 = os.path.join(DIR_TABLAS, "fase7_experimento1_filtrado.csv")
    ruta_md1 = os.path.join(DIR_TABLAS, "fase7_experimento1_filtrado.md")

    filas1 = []
    res1 = res_exp1["resultados"]
    for mod in ["BFSK", "BPSK", "QPSK", "16-QAM"]:
        p_sin = res1["sin_filtro"][mod]
        p_con = res1["con_filtro"][mod]
        for i in range(len(p_sin)):
            evm_s = f"{p_sin[i]['evm_pct']:.2f}%" if p_sin[i]["evm_pct"] is not None else "N/A"
            evm_c = f"{p_con[i]['evm_pct']:.2f}%" if p_con[i]["evm_pct"] is not None else "N/A"
            filas1.append({
                "Modulacion": mod,
                "SIR_ACI_dB": f"{p_sin[i]['sir_db']:.1f}",
                "BER_Sin_Filtro": f"{p_sin[i]['ber']:.6e}",
                "BER_Con_Filtro": f"{p_con[i]['ber']:.6e}",
                "Errores_Sin": p_sin[i]["errores"],
                "Errores_Con": p_con[i]["errores"],
                "EVM_Sin_Filtro": evm_s,
                "EVM_Con_Filtro": evm_c,
                "Mejora_BER": "SÍ" if p_con[i]["ber"] < p_sin[i]["ber"] else ("IGUAL" if p_con[i]["ber"] == p_sin[i]["ber"] else "NO")
            })

    with open(ruta_csv1, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas1[0].keys()))
        writer.writeheader()
        writer.writerows(filas1)

    info_f = res_exp1["info_filtro"]
    info_bp = res_exp1["info_filtro_bfsk"]
    with open(ruta_md1, "w", encoding="utf-8") as f:
        f.write("# FASE 7 — EXPERIMENTO 1: FILTRADO DIGITAL FIR FRENTE A CANAL ADYACENTE (ACI)\n\n")
        f.write(
            f"Parámetros LPF (BPSK/QPSK/16-QAM): FIR Hamming Orden {res_exp1['orden_filtro']}, "
            f"fc = {res_exp1['f_corte_hz']:.0f} Hz, fs = {res_exp1['fs_hz']:.0f} Hz. "
            f"Atenuación analítica a Δf = {res_exp1['delta_f_hz']:.0f} Hz: **{info_f['atenuacion_relativa_db']:.2f} dB**. "
            f"ISI propia residual (16-QAM limpia): **{res_exp1['isi_residual_limpia_16qam_pct']:.2f}%**.\n\n"
        )
        f.write(
            f"Parámetros BPF (BFSK RF): FIR Hamming Orden {res_exp1['orden_bfsk']}, "
            f"banda {res_exp1['banda_bfsk_hz'][0]:.0f}–{res_exp1['banda_bfsk_hz'][1]:.0f} Hz (centro 3000 Hz). "
            f"Atenuación analítica a 4500 Hz: **{info_bp['atenuacion_relativa_db']:.2f} dB**.\n\n"
        )
        f.write("| Modulación | SIR ACI [dB] | BER Sin Filtro | BER Con Filtro FIR | Errores Sin | Errores Con | EVM Sin | EVM Con | ¿Mejora BER? |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in filas1:
            f.write(f"| {r['Modulacion']} | {r['SIR_ACI_dB']} | {r['BER_Sin_Filtro']} | {r['BER_Con_Filtro']} | {r['Errores_Sin']} | {r['Errores_Con']} | {r['EVM_Sin_Filtro']} | {r['EVM_Con_Filtro']} | {r['Mejora_BER']} |\n")

    print(f"  [GUARDADO] {ruta_csv1} y {ruta_md1}")

    # 2. Tabla Experimento 2: Ecualización ZF
    ruta_csv2 = os.path.join(DIR_TABLAS, "fase7_experimento2_ecualizacion.csv")
    ruta_md2 = os.path.join(DIR_TABLAS, "fase7_experimento2_ecualizacion.md")

    filas2 = []
    res2 = res_exp2["resultados"]
    info_zf = res_exp2["info_zf"]
    for mod in ["BPSK", "QPSK", "16-QAM"]:
        p_sin = res2["sin_ecualizar"][mod]
        p_con = res2["con_ecualizador_zf"][mod]
        for i in range(len(p_sin)):
            filas2.append({
                "Modulacion": mod,
                "EbN0_dB": f"{p_sin[i]['ebn0_db']:.1f}",
                "BER_Sin_Ecualizar": f"{p_sin[i]['ber']:.6e}",
                "BER_Con_ZF": f"{p_con[i]['ber']:.6e}",
                "Errores_Sin": p_sin[i]["errores"],
                "Errores_Con": p_con[i]["errores"],
                "EVM_Sin_Ecualizar": f"{p_sin[i]['evm_pct']:.2f}%",
                "EVM_Con_ZF": f"{p_con[i]['evm_pct']:.2f}%",
                "Mejora_BER": "SÍ" if p_con[i]["ber"] < p_sin[i]["ber"] else ("IGUAL" if p_con[i]["ber"] == p_sin[i]["ber"] else "NO")
            })

    with open(ruta_csv2, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas2[0].keys()))
        writer.writeheader()
        writer.writerows(filas2)

    with open(ruta_md2, "w", encoding="utf-8") as f:
        f.write("# FASE 7 — EXPERIMENTO 2: ECUALIZACIÓN ZERO-FORCING (ZF) EN CANAL CON ISI\n\n")
        f.write(f"Canal Dispersivo: h = {res_exp2['h_canal']}. Ecualizador: FIR Nw = {info_zf['longitud_w']}, Retardo Δ = {info_zf['retardo_decision']}.\n")
        f.write(f"Factor de Amplificación de Ruido medido (NEF): **{info_zf['nef_db']:+.3f} dB** ({info_zf['nef_lineal']:.4f}).\n")
        f.write(f"ISI Residual combinada: **{info_zf['isi_residual_db']:.2f} dB**.\n\n")
        f.write("| Modulación | Eb/N0 [dB] | BER Sin Ecualizar | BER Con ZF | Errores Sin | Errores Con | EVM Sin | EVM Con | ¿Mejora BER? |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in filas2:
            f.write(f"| {r['Modulacion']} | {r['EbN0_dB']} | {r['BER_Sin_Ecualizar']} | {r['BER_Con_ZF']} | {r['Errores_Sin']} | {r['Errores_Con']} | {r['EVM_Sin_Ecualizar']} | {r['EVM_Con_ZF']} | {r['Mejora_BER']} |\n")

    print(f"  [GUARDADO] {ruta_csv2} y {ruta_md2}")

    # 3. Tabla Experimento 3: Diversidad SIMO Rayleigh
    ruta_csv3 = os.path.join(DIR_TABLAS, "fase7_experimento3_diversidad.csv")
    ruta_md3 = os.path.join(DIR_TABLAS, "fase7_experimento3_diversidad.md")

    filas3 = []
    res3 = res_exp3["resultados"]
    for mod in ["BPSK", "QPSK"]:
        p_siso = res3["SISO"][mod]
        p_sc = res3["SC"][mod]
        p_mrc = res3["MRC"][mod]
        for i in range(len(p_siso)):
            filas3.append({
                "Modulacion": mod,
                "EbN0_Medio_dB": f"{p_siso[i]['ebn0_db']:.1f}",
                "BER_SISO_L1": f"{p_siso[i]['ber']:.6e}",
                "BER_SC_L2": f"{p_sc[i]['ber']:.6e}",
                "BER_MRC_L2": f"{p_mrc[i]['ber']:.6e}",
                "Errores_SISO": p_siso[i]["errores"],
                "Errores_SC": p_sc[i]["errores"],
                "Errores_MRC": p_mrc[i]["errores"],
                "Ganancia_MRC_vs_SISO": f"{p_siso[i]['ber'] / max(p_mrc[i]['ber'], 1e-12):.1f}x"
            })

    with open(ruta_csv3, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas3[0].keys()))
        writer.writeheader()
        writer.writerows(filas3)

    with open(ruta_md3, "w", encoding="utf-8") as f:
        f.write("# FASE 7 — EXPERIMENTO 3: DIVERSIDAD DE RECEPCIÓN EN CANAL RAYLEIGH (SISO VS SC VS MRC)\n\n")
        f.write("Condición: Potencia de transmisión idéntica (Es = 1.0), ramas Rayleigh independientes y ruidos incorrelados.\n\n")
        f.write("| Modulación | Eb/N0 [dB] | BER SISO (L=1) | BER SC (L=2) | BER MRC (L=2) | Errores SISO | Errores SC | Errores MRC | Reducción de Error MRC |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in filas3:
            f.write(f"| {r['Modulacion']} | {r['EbN0_Medio_dB']} | {r['BER_SISO_L1']} | {r['BER_SC_L2']} | {r['BER_MRC_L2']} | {r['Errores_SISO']} | {r['Errores_SC']} | {r['Errores_MRC']} | {r['Ganancia_MRC_vs_SISO']} |\n")

    print(f"  [GUARDADO] {ruta_csv3} y {ruta_md3}")

    # 4. Tabla Resumen Global de Mitigación (calculada dinámicamente desde resultados reales)
    ruta_csv4 = os.path.join(DIR_TABLAS, "fase7_resumen_mitigacion.csv")
    ruta_md4 = os.path.join(DIR_TABLAS, "fase7_resumen_mitigacion.md")

    ber_bfsk_sin_0 = res1["sin_filtro"]["BFSK"][0]["ber"]
    ber_bfsk_con_0 = res1["con_filtro"]["BFSK"][0]["ber"]
    ber_16q_sin_0 = res1["sin_filtro"]["16-QAM"][0]["ber"]
    ber_16q_con_0 = res1["con_filtro"]["16-QAM"][0]["ber"]
    evm_16q_sin_0 = res1["sin_filtro"]["16-QAM"][0]["evm_pct"]
    evm_16q_con_0 = res1["con_filtro"]["16-QAM"][0]["evm_pct"]

    ber_16q_isi_sin_14 = res2["sin_ecualizar"]["16-QAM"][7]["ber"]
    ber_16q_isi_zf_14 = res2["con_ecualizador_zf"]["16-QAM"][7]["ber"]

    ber_bpsk_siso_10 = res3["SISO"]["BPSK"][5]["ber"]
    ber_bpsk_mrc_10 = res3["MRC"]["BPSK"][5]["ber"]

    filas_resumen = [
        {
            "Tecnica": "1. Filtrado Digital FIR",
            "Escenario_Canal": "Interferencia Canal Adyacente (ACI Δf=1500 Hz)",
            "Configuracion_Tecnica": (
                f"LPF Ord.{res_exp1['orden_filtro']} (fc=1 kHz) y BPF BFSK Ord.{res_exp1['orden_bfsk']} (1.8-4.2 kHz)"
            ),
            "Desempeño_Sin_Mitigacion": (
                f"A SIR=0 dB: BFSK BER={ber_bfsk_sin_0:.4f}, 16-QAM BER={ber_16q_sin_0:.4e} (EVM={evm_16q_sin_0:.1f}%)"
            ),
            "Desempeño_Con_Mitigacion": (
                f"Aten. {info_f['atenuacion_relativa_db']:.1f} dB (LPF) / {info_bp['atenuacion_relativa_db']:.1f} dB (BPF); "
                f"BFSK BER={ber_bfsk_con_0:.2e}, 16-QAM BER={ber_16q_con_0:.2e} (EVM={evm_16q_con_0:.1f}%)"
            ),
            "Beneficio_Demostrado": "Supresión real de interferencia ACI sin destruir componentes útiles"
        },
        {
            "Tecnica": "2. Ecualización Zero-Forcing (ZF)",
            "Escenario_Canal": "Canal Dispersivo con ISI (h=[1.0, 0.35])",
            "Configuracion_Tecnica": f"FIR Nw={info_zf['longitud_w']}, Retardo Δ={info_zf['retardo_decision']}, NEF={info_zf['nef_db']:+.2f} dB",
            "Desempeño_Sin_Mitigacion": f"Piso de error severo (16-QAM BER={ber_16q_isi_sin_14:.4f} a 14 dB)",
            "Desempeño_Con_Mitigacion": f"Cancelación de ISI (16-QAM BER={ber_16q_isi_zf_14:.2e} a 14 dB)",
            "Beneficio_Demostrado": "Eliminación del piso de error por dispersión multitrayecto"
        },
        {
            "Tecnica": "3. Diversidad de Recepción (MRC L=2)",
            "Escenario_Canal": "Canal Móvil con Desvanecimiento Rayleigh Plano",
            "Configuracion_Tecnica": "SIMO 1x2, Ramas independientes, MRC coherente",
            "Desempeño_Sin_Mitigacion": f"Decaimiento orden 1 (BPSK SISO BER={ber_bpsk_siso_10:.4f} a 10 dB)",
            "Desempeño_Con_Mitigacion": f"Decaimiento orden 2 (BPSK MRC BER={ber_bpsk_mrc_10:.4e} a 10 dB)",
            "Beneficio_Demostrado": "Reducción drástica de errores por desvanecimiento Rayleigh"
        }
    ]

    with open(ruta_csv4, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_resumen[0].keys()))
        writer.writeheader()
        writer.writerows(filas_resumen)

    with open(ruta_md4, "w", encoding="utf-8") as f:
        f.write("# FASE 7 — RESUMEN GLOBAL DE TÉCNICAS DE MITIGACIÓN\n\n")
        f.write("| Técnica | Escenario de Canal | Configuración Técnica | Desempeño Sin Mitigación | Desempeño Con Mitigación | Beneficio Demostrado |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in filas_resumen:
            f.write(f"| {r['Tecnica']} | {r['Escenario_Canal']} | {r['Configuracion_Tecnica']} | {r['Desempeño_Sin_Mitigacion']} | {r['Desempeño_Con_Mitigacion']} | {r['Beneficio_Demostrado']} |\n")

    print(f"  [GUARDADO] {ruta_csv4} y {ruta_md4}")


# =====================================================================
# BLOQUE 4: PRUEBAS DE NO REGRESIÓN (FASES 1 A 6)
# =====================================================================

def ejecutar_pruebas_no_regresion():
    """Ejecuta los scripts de validación de las fases 1 a 6 para certificar cero regresión."""
    print("\n" + "=" * 76)
    print(" BLOQUE 5: VERIFICACIÓN DE NO REGRESIÓN (FASES 1 A 6)")
    print("=" * 76)

    scripts = [
        "verificar_fase1.py",
        "test_fase2.py",
        "test_fase3.py",
        "test_fase4.py",
        "test_fase5.py",
        "test_fase6.py"
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
    print(" INICIANDO EJECUCIÓN INTEGRAL FASE 7: TÉCNICAS DE MITIGACIÓN")
    print("*" * 76)

    # 1. Pruebas unitarias
    ejecutar_bloque_unitario()

    # 2. Simulaciones Monte Carlo (Nb = 60.000 bits)
    print("\n" + "=" * 76)
    print(" BLOQUE 2: SIMULACIONES EXPERIMENTALES MONTE CARLO (Nb = 60.000 bits)")
    print("=" * 76)

    print("  -> Ejecutando Experimento 1: Filtrado FIR vs Interferencia ACI ...")
    res_exp1 = experimentos_fase7.ejecutar_experimento_1_filtrado(n_bits=60000, seed_base=5000)

    print("  -> Ejecutando Experimento 2: Ecualización ZF vs Canal con ISI ...")
    res_exp2 = experimentos_fase7.ejecutar_experimento_2_ecualizacion(n_bits=60000, seed_base=6000)

    print("  -> Ejecutando Experimento 3: Diversidad de Recepción (SISO vs SC vs MRC) ...")
    res_exp3 = experimentos_fase7.ejecutar_experimento_3_diversidad(n_bits=60000, seed_base=7000)

    # 3. Gráficos comparativos
    print("\n" + "=" * 76)
    print(" BLOQUE 3: GENERACIÓN Y GUARDADO DE GRÁFICOS EXPERIMENTALES (PNG)")
    print("=" * 76)
    generar_grafico_1_filtrado(res_exp1)
    generar_grafico_2_ecualizacion(res_exp2)
    generar_grafico_3_diversidad(res_exp3)

    # 4. Tablas
    exportar_tablas_fase7(res_exp1, res_exp2, res_exp3)

    # 5. Pruebas de no regresión
    ejecutar_pruebas_no_regresion()

    print("\n" + "*" * 76)
    print(" FASE 7 COMPLETADA AL 100% EXITOSAMENTE")
    print("*" * 76)


if __name__ == "__main__":
    main()

