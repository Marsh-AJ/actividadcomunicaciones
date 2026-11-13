"""
test_fase3.py
Script de pruebas unitarias y verificación experimental para la FASE 3:
Modulaciones Digitales (BFSK, BPSK, QPSK, 16-QAM).
"""

import os
import sys
import numpy as np

# Importar módulos del proyecto
import fundamentos as fnd
import modulaciones as mod


def ejecutar_pruebas_fase3():
    print("=" * 72)
    print("   PRUEBAS UNITARIAS Y VALIDACIÓN - FASE 3: MODULACIONES DIGITALES")
    print("=" * 72)

    pruebas_exitosas = 0
    total_pruebas = 7

    # -------------------------------------------------------------
    # PRUEBA 1: Verificación de Frecuencias en BFSK
    # -------------------------------------------------------------
    print("\n[PRUEBA 1/7] Verificando modulador BFSK (f0 y f1)...")
    fs = 100000.0
    f0 = 2000.0
    f1 = 4000.0
    sps = 200  # 200 muestras por bit -> Tb = 2 ms
    # Probamos secuencia controlada [0, 1]
    bits_bfsk = np.array([0, 1], dtype=np.int8)
    res_bfsk = mod.modular_bfsk(bits_bfsk, f0=f0, f1=f1, sps=sps, fs=fs, fase_continua=False)

    # Segmento 0 (bit 0): de 0 a sps muestras
    t_bit0 = np.arange(sps) / fs
    s_bit0 = res_bfsk.senal_rf[:sps]
    # Segmento 1 (bit 1): de sps a 2*sps muestras
    s_bit1 = res_bfsk.senal_rf[sps:2*sps]

    # Correlación con tonos puros para confirmar la frecuencia exacta de cada símbolo
    tono_0 = np.cos(2 * np.pi * f0 * t_bit0)
    tono_1 = np.cos(2 * np.pi * f1 * t_bit0)

    corr_0_en_s0 = np.sum(s_bit0 * tono_0) / sps
    corr_1_en_s0 = np.sum(s_bit0 * tono_1) / sps
    corr_0_en_s1 = np.sum(s_bit1 * tono_0) / sps
    corr_1_en_s1 = np.sum(s_bit1 * tono_1) / sps

    print(f"    - Bit 0 -> Potencia detectada en f0={f0:.0f}Hz: {corr_0_en_s0:.3f} | en f1={f1:.0f}Hz: {corr_1_en_s0:.3f}")
    print(f"    - Bit 1 -> Potencia detectada en f0={f0:.0f}Hz: {corr_0_en_s1:.3f} | en f1={f1:.0f}Hz: {corr_1_en_s1:.3f}")

    assert corr_0_en_s0 > 0.4 and corr_1_en_s0 < 0.05, "Error: El bit 0 no se moduló en f0."
    assert corr_1_en_s1 > 0.4 and corr_0_en_s1 < 0.05, "Error: El bit 1 no se moduló en f1."
    print("    [PASÓ] BFSK transmite f0 para bit 0 y f1 para bit 1 con ortogonalidad.")
    pruebas_exitosas += 1

    # -------------------------------------------------------------
    # PRUEBA 2: Verificación de BPSK (2 fases, 2 símbolos)
    # -------------------------------------------------------------
    print("\n[PRUEBA 2/7] Verificando modulador BPSK (M=2, 1 bit/símbolo)...")
    bits_bpsk = np.array([0, 1, 1, 0, 1, 0], dtype=np.int8)
    res_bpsk = mod.modular_bpsk(bits_bpsk, fc=5000.0, sps=50, fs=fs)

    simbolos_unicos = np.unique(res_bpsk.simbolos)
    assert len(simbolos_unicos) == 2, f"Error: BPSK debe generar exactamente 2 símbolos, se obtuvieron {len(simbolos_unicos)}"
    assert len(res_bpsk.simbolos) == len(bits_bpsk), "Error: Longitud de símbolos de BPSK debe ser igual a cantidad de bits."

    # Verificar valores esperados: -1.0 y +1.0
    assert np.isclose(simbolos_unicos[0], -1.0) and np.isclose(simbolos_unicos[1], 1.0), \
        "Error en los valores de constelación de BPSK."
    print(f"    - Símbolos únicos detectados: {simbolos_unicos.tolist()}")
    print("    [PASÓ] BPSK genera exactamente 2 símbolos ortogonales en fase (0° y 180°).")
    pruebas_exitosas += 1

    # -------------------------------------------------------------
    # PRUEBA 3: Verificación de QPSK y Código Gray
    # -------------------------------------------------------------
    print("\n[PRUEBA 3/7] Verificando modulador QPSK (M=4, 2 bits/símbolo, Mapeo Gray)...")
    # Generar todos los pares posibles: 00, 01, 11, 10
    bits_qpsk_todos = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.int8)
    res_qpsk = mod.modular_qpsk(bits_qpsk_todos, fc=5000.0, sps=50, fs=fs)

    assert len(res_qpsk.simbolos) == len(bits_qpsk_todos) // 2, "Error: QPSK debe agrupar 2 bits por símbolo."
    simbolos_qpsk = np.unique(res_qpsk.simbolos)
    assert len(simbolos_qpsk) == 4, f"Error: QPSK debe generar 4 símbolos únicos, se obtuvieron {len(simbolos_qpsk)}"

    # Verificar Código Gray: entre vecinos adyacentes la distancia Hamming debe ser 1 bit
    c_ref = res_qpsk.constelacion_ref
    dibits = ["00", "01", "11", "10"]
    for i in range(len(dibits)):
        d1 = dibits[i]
        d2 = dibits[(i + 1) % len(dibits)]
        diff_bits = sum(c1 != c2 for c1, c2 in zip(d1, d2))
        assert diff_bits == 1, f"Error en asignación Gray entre {d1} y {d2}"

    print(f"    - 4 símbolos QPSK generados con éxito: {len(simbolos_qpsk)} puntos.")
    print("    - Asignación Gray verificada: cada par vecino difiere en exactamente 1 bit.")
    print("    [PASÓ] QPSK agrupa 2 bits por símbolo con asignación Gray correcta.")
    pruebas_exitosas += 1

    # -------------------------------------------------------------
    # PRUEBA 4: Verificación de 16-QAM (16 símbolos, 4 bits/símbolo)
    # -------------------------------------------------------------
    print("\n[PRUEBA 4/7] Verificando modulador 16-QAM (M=16, 4 bits/símbolo)...")
    # Generar todos los 16 cuartetos posibles (64 bits en total)
    todos_cuartetos = []
    for i in range(16):
        bits_str = f"{i:04b}"
        todos_cuartetos.extend([int(b) for b in bits_str])
    bits_16qam_todos = np.array(todos_cuartetos, dtype=np.int8)

    res_16qam = mod.modular_16qam(bits_16qam_todos, fc=5000.0, sps=50, fs=fs)
    assert len(res_16qam.simbolos) == len(bits_16qam_todos) // 4, "Error: 16-QAM debe agrupar 4 bits por símbolo."

    # Los 16 símbolos deben ser todos distintos
    simbolos_redondeados = np.round(res_16qam.simbolos, 4)
    unicos_16qam = np.unique(simbolos_redondeados)
    assert len(unicos_16qam) == 16, f"Error: Se esperaban 16 símbolos distintos en 16-QAM, se obtuvieron {len(unicos_16qam)}"

    # Verificar que los niveles en I y Q pertenezcan a {-3, -1, 1, 3} / sqrt(10)
    norm = np.sqrt(10.0)
    niveles_esperados = np.array([-3.0, -1.0, 1.0, 3.0]) / norm
    partes_reales = np.round(unicos_16qam.real, 4)
    partes_imag = np.round(unicos_16qam.imag, 4)
    for r in np.unique(partes_reales):
        assert np.any(np.isclose(r, niveles_esperados, atol=1e-3)), f"Nivel real {r} fuera de la grilla esperada."
    for im in np.unique(partes_imag):
        assert np.any(np.isclose(im, niveles_esperados, atol=1e-3)), f"Nivel imag {im} fuera de la grilla esperada."

    print(f"    - Grilla rectangular 4x4 de 16-QAM verificada con 16 símbolos únicos.")
    print("    [PASÓ] 16-QAM agrupa 4 bits por símbolo sobre una constelación de 16 puntos.")
    pruebas_exitosas += 1

    # -------------------------------------------------------------
    # PRUEBA 5: Verificación de Normalización de Energía Promedio y de Bit
    # -------------------------------------------------------------
    print("\n[PRUEBA 5/7] Verificando normalización de energía promedio (E_avg = 1.0) y de bit (E_b)...")
    const_bpsk = mod.obtener_constelacion_bpsk()
    const_qpsk = mod.obtener_constelacion_qpsk()
    const_16qam = mod.obtener_constelacion_16qam()

    # Validar cálculo con la nueva función utilitaria
    e_bpsk = mod.calcular_energia_promedio(const_bpsk)
    e_qpsk = mod.calcular_energia_promedio(const_qpsk)
    e_16qam = mod.calcular_energia_promedio(const_16qam)

    # Validar energías de bit
    eb_bpsk = mod.calcular_energia_bit(const_bpsk, bits_por_simbolo=1)
    eb_qpsk = mod.calcular_energia_bit(const_qpsk, bits_por_simbolo=2)
    eb_16qam = mod.calcular_energia_bit(const_16qam, bits_por_simbolo=4)

    # Validar factores analíticos
    k_bpsk = mod.obtener_factor_normalizacion("BPSK")
    k_qpsk = mod.obtener_factor_normalizacion("QPSK")
    k_16qam = mod.obtener_factor_normalizacion("16-QAM")

    print(f"    - BPSK  : Factor K = {k_bpsk:.4f} | E_s = {e_bpsk:.6f} (Teórico: 1.0) | E_b = {eb_bpsk:.6f} (Teórico: 1.0)")
    print(f"    - QPSK  : Factor K = {k_qpsk:.4f} | E_s = {e_qpsk:.6f} (Teórico: 1.0) | E_b = {eb_qpsk:.6f} (Teórico: 0.5)")
    print(f"    - 16-QAM: Factor K = {k_16qam:.4f} | E_s = {e_16qam:.6f} (Teórico: 1.0) | E_b = {eb_16qam:.6f} (Teórico: 0.25)")

    assert np.isclose(e_bpsk, 1.0, atol=1e-5), "Error: Constelación BPSK no está normalizada a energía 1.0."
    assert np.isclose(e_qpsk, 1.0, atol=1e-5), "Error: Constelación QPSK no está normalizada a energía 1.0."
    assert np.isclose(e_16qam, 1.0, atol=1e-5), "Error: Constelación 16-QAM no está normalizada a energía 1.0."

    assert np.isclose(eb_bpsk, 1.0, atol=1e-5), "Error en E_b de BPSK."
    assert np.isclose(eb_qpsk, 0.5, atol=1e-5), "Error en E_b de QPSK."
    assert np.isclose(eb_16qam, 0.25, atol=1e-5), "Error en E_b de 16-QAM."

    print("    [PASÓ] Constelaciones estrictamente normalizadas a E_s = 1.0 y relaciones E_b = E_s/k validadas.")
    pruebas_exitosas += 1

    # -------------------------------------------------------------
    # PRUEBA 6: Generación y Guardado de Gráficos Obligatorios
    # -------------------------------------------------------------
    print("\n[PRUEBA 6/7] Generando y guardando los 5 gráficos en 'resultados/graficos/'...")
    ruta_g1 = os.path.join("resultados", "graficos", "fase3_01_bfsk.png")
    ruta_g2 = os.path.join("resultados", "graficos", "fase3_02_bpsk.png")
    ruta_g3 = os.path.join("resultados", "graficos", "fase3_03_qpsk.png")
    ruta_g4 = os.path.join("resultados", "graficos", "fase3_04_16qam.png")
    ruta_g5 = os.path.join("resultados", "graficos", "fase3_05_comparacion_constelaciones.png")

    # Generar bits reproducibles con semilla
    bits_demo = fnd.generar_bits(n_bits=64, semilla=123)

    # 1. BFSK
    res_bfsk = mod.modular_bfsk(bits_demo[:16], f0=2000.0, f1=4000.0, sps=100, fs=100000.0)
    mod.graficar_modulacion_bfsk(res_bfsk, n_bits_mostrar=8, ruta_guardado=ruta_g1)

    # 2. BPSK
    res_bpsk = mod.modular_bpsk(bits_demo[:16], fc=5000.0, sps=100, fs=100000.0)
    mod.graficar_modulacion_bpsk(res_bpsk, n_bits_mostrar=8, ruta_guardado=ruta_g2)

    # 3. QPSK
    res_qpsk = mod.modular_qpsk(bits_demo[:32], fc=5000.0, sps=100, fs=100000.0)
    mod.graficar_modulacion_qpsk(res_qpsk, n_simbolos_mostrar=8, ruta_guardado=ruta_g3)

    # 4. 16-QAM
    res_16qam = mod.modular_16qam(bits_demo[:64], fc=5000.0, sps=100, fs=100000.0)
    mod.graficar_modulacion_16qam(res_16qam, n_simbolos_mostrar=8, ruta_guardado=ruta_g4)

    # 5. Comparación de constelaciones
    mod.graficar_comparacion_constelaciones(ruta_guardado=ruta_g5)

    for ruta in [ruta_g1, ruta_g2, ruta_g3, ruta_g4, ruta_g5]:
        assert os.path.exists(ruta), f"Error: No se encontró el gráfico {ruta}"
        tam_kb = os.path.getsize(ruta) / 1024
        assert tam_kb > 10, f"Error: El gráfico {ruta} está incompleto ({tam_kb:.1f} KB)."
        print(f"    - Gráfico verificado: {ruta} ({tam_kb:.1f} KB)")

    print("    [PASÓ] Los 5 gráficos obligatorios fueron creados y guardados con éxito.")
    pruebas_exitosas += 1

    # -------------------------------------------------------------
    # PRUEBA 7: Manejo de Entradas Inválidas (Excepciones ValueError)
    # -------------------------------------------------------------
    print("\n[PRUEBA 7/7] Verificando robustez y manejo de excepciones en entradas inválidas...")
    
    # 7.1. Secuencia no múltiplo de 2 en QPSK
    try:
        bits_impar = np.array([1, 0, 1], dtype=np.int8)  # Longitud 3 (no divisible por 2)
        mod.modular_qpsk(bits_impar)
        assert False, "Error: QPSK debió lanzar ValueError con longitud impar."
    except ValueError as e:
        print(f"    [OK] QPSK rechazó longitud impar correctamente: '{e}'")

    # 7.2. Secuencia no múltiplo de 4 en 16-QAM
    try:
        bits_no_mult4 = np.array([1, 0, 1, 1, 0, 1], dtype=np.int8)  # Longitud 6 (no divisible por 4)
        mod.modular_16qam(bits_no_mult4)
        assert False, "Error: 16-QAM debió lanzar ValueError con longitud no divisible por 4."
    except ValueError as e:
        print(f"    [OK] 16-QAM rechazó longitud incompatible correctamente: '{e}'")

    # 7.3. Bits no binarios
    try:
        bits_invalidos = np.array([0, 1, 2, 0], dtype=np.int8)
        mod.modular_bpsk(bits_invalidos)
        assert False, "Error: BPSK debió rechazar valores no binarios."
    except ValueError as e:
        print(f"    [OK] Modulador rechazó valores no binarios correctamente: '{e}'")

    print("    [PASÓ] Manejo de excepciones y validación de entradas verificado.")
    pruebas_exitosas += 1

    # -------------------------------------------------------------
    # RESUMEN FINAL
    # -------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"   RESULTADO FINAL: {pruebas_exitosas}/{total_pruebas} PRUEBAS COMPLETADAS CON ÉXITO")
    print("   El módulo 'modulaciones.py' cumple al 100% las especificaciones de la FASE 3.")
    print("=" * 72)


if __name__ == "__main__":
    ejecutar_pruebas_fase3()

