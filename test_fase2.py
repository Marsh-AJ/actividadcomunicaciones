"""
test_fase2.py
Script de ejecución y validación de pruebas unitarias para la FASE 2:
Fundamentos de Simulación en Comunicaciones Digitales.
"""

import os
import sys
import numpy as np

# Importar el módulo desarrollado en la Fase 2
import fundamentos as fnd

def ejecutar_pruebas_fase2():
    print("=" * 70)
    print("   PRUEBAS UNITARIAS Y VALIDACIÓN - FASE 2: FUNDAMENTOS DE SEÑALES")
    print("=" * 70)

    pruebas_exitosas = 0
    total_pruebas = 5

    # -------------------------------------------------------------
    # PRUEBA 1: Generador de Bits (reproducibilidad y estadísticas)
    # -------------------------------------------------------------
    print("\n[PRUEBA 1/5] Verificación del Generador de Bits...")
    n_bits = 10000
    semilla = 42
    bits_1 = fnd.generar_bits(n_bits=n_bits, semilla=semilla)
    bits_2 = fnd.generar_bits(n_bits=n_bits, semilla=semilla)

    assert len(bits_1) == n_bits, "Error: La longitud del vector de bits no coincide."
    assert np.array_equal(bits_1, bits_2), "Error: La semilla no garantiza reproducibilidad determinista."
    assert set(np.unique(bits_1)).issubset({0, 1}), "Error: Los valores de los bits no son binarios {0, 1}."

    # Proporción estadística: en una muestra aleatoria de 10.000 bits, la media debe estar cerca de 0.5 (~50%)
    media_bits = np.mean(bits_1)
    print(f"    - Bits generados: {n_bits}")
    print(f"    - Proporción de '1's: {media_bits * 100:.2f} % (Teórico: 50.0 %)")
    assert 0.48 <= media_bits <= 0.52, "Error: La secuencia binaria presenta sesgo estadístico notable."
    print("    [PASÓ] Generador de bits determinista y estadísticamente balanceado.")
    pruebas_exitosas += 1

    # -------------------------------------------------------------
    # PRUEBA 2: Conversión a Formas de Onda NRZ Unipolar y Polar
    # -------------------------------------------------------------
    print("\n[PRUEBA 2/5] Verificación de Formas de Onda NRZ...")
    bits_muestra = np.array([1, 0, 1, 1, 0])
    sps = 20  # muestras por símbolo/bit
    nrz_uni = fnd.bits_a_nrz(bits_muestra, muestras_por_bit=sps, polar=False)
    nrz_pol = fnd.bits_a_nrz(bits_muestra, muestras_por_bit=sps, polar=True)

    assert len(nrz_uni) == len(bits_muestra) * sps, "Error en longitud de NRZ unipolar."
    assert len(nrz_pol) == len(bits_muestra) * sps, "Error en longitud de NRZ polar."
    assert np.all(np.isin(nrz_uni, [0.0, 1.0])), "Error en niveles de tensión NRZ unipolar."
    assert np.all(np.isin(nrz_pol, [-1.0, 1.0])), "Error en niveles de tensión NRZ polar."

    # Verificar que el primer bloque de sps muestras corresponde al primer bit (1)
    assert np.all(nrz_uni[:sps] == 1.0) and np.all(nrz_pol[:sps] == 1.0)
    # Segundo bloque al bit 0 (unipolar 0, polar -1)
    assert np.all(nrz_uni[sps:2*sps] == 0.0) and np.all(nrz_pol[sps:2*sps] == -1.0)
    print(f"    - Longitud esperada ({len(bits_muestra)} bits * {sps} sps = {len(bits_muestra)*sps} muestras): Correcta.")
    print("    [PASÓ] Mapeo de bits a señales NRZ unipolar y polar validado.")
    pruebas_exitosas += 1

    # -------------------------------------------------------------
    # PRUEBA 3: Señal Sinusoidal Muestreada y Potencia Teórica
    # -------------------------------------------------------------
    print("\n[PRUEBA 3/5] Verificación de la Señal Sinusoidal y Teorema de Nyquist...")
    fs = 200e3  # 200 kHz
    f0 = 2e3    # 2 kHz
    amplitud = 2.5
    duracion = 10e-3  # 10 ms (20 ciclos completos)
    t, senal = fnd.generar_senoidal(frecuencia=f0, amplitud=amplitud, fase_rad=0.0, duracion=duracion, fs=fs)

    # Potencia teórica para A*cos(wt) es A^2 / 2
    potencia_teorica = (amplitud ** 2) / 2.0
    potencia_calculada = fnd.calcular_potencia(senal)
    error_potencia = abs(potencia_calculada - potencia_teorica) / potencia_teorica * 100

    print(f"    - Amplitud pico: {amplitud} V")
    print(f"    - Frecuencia: {f0/1e3} kHz | Frecuencia de muestreo: {fs/1e3} kHz")
    print(f"    - Potencia teórica: {potencia_teorica:.5f} W")
    print(f"    - Potencia numérica: {potencia_calculada:.5f} W (Error: {error_potencia:.3f} %)")
    assert error_potencia < 0.1, "Error: La potencia calculada discrepa del valor teórico."
    print("    [PASÓ] Señal sinusoidal y cálculo de potencia rigurosamente exactos.")
    pruebas_exitosas += 1

    # -------------------------------------------------------------
    # PRUEBA 4: Modulación y Síntesis I/Q en Cuadratura
    # -------------------------------------------------------------
    print("\n[PRUEBA 4/5] Verificación de Ortogonalidad y Síntesis I/Q...")
    # Probamos la propiedad matemática de ortogonalidad:
    # Si I=1 y Q=0 => señal es cos(wt), magnitud pico es 1
    # Si I=0 y Q=1 => señal es -sin(wt), magnitud pico es 1
    # Si I=1 y Q=1 => señal tiene magnitud pico sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.4142 V
    fc = 10e3
    num_muestras = int(10e-3 * fs)
    I_val = 1.0
    Q_val = 1.0
    I_bb = np.full(num_muestras, I_val)
    Q_bb = np.full(num_muestras, Q_val)

    t_iq, s_rf = fnd.modular_iq_pasobanda(I_bb, Q_bb, fc=fc, fs=fs)
    potencia_rf = fnd.calcular_potencia(s_rf)
    # Potencia teórica de I*cos - Q*sin cuando I y Q son constantes:
    # P = (I^2 + Q^2) / 2 = (1 + 1)/2 = 1.0 W
    potencia_rf_teorica = (I_val**2 + Q_val**2) / 2.0
    print(f"    - Potencia RF observada: {potencia_rf:.5f} W (Teórica: {potencia_rf_teorica:.5f} W)")
    assert abs(potencia_rf - potencia_rf_teorica) < 0.01, "Error en la potencia de la señal modulada I/Q."
    print("    [PASÓ] Ortogonalidad I/Q y modulación de banda base a pasobanda verificadas.")
    pruebas_exitosas += 1

    # -------------------------------------------------------------
    # PRUEBA 5: Generación y Guardado Físico de Gráficos Requeridos
    # -------------------------------------------------------------
    print("\n[PRUEBA 5/5] Generación y Exportación de Gráficos a 'resultados/graficos/'...")
    ruta_g1 = os.path.join("resultados", "graficos", "fase2_01_secuencia_bits.png")
    ruta_g2 = os.path.join("resultados", "graficos", "fase2_02_ondas_sinusoidales.png")
    ruta_g3 = os.path.join("resultados", "graficos", "fase2_03_representacion_iq.png")

    # Gráfico 1: Secuencia de bits de ejemplo representativa
    bits_grafico = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    fnd.graficar_secuencia_bits(bits_grafico, muestras_por_bit=100, fs=100e3, ruta_guardado=ruta_g1)

    # Gráfico 2: Variaciones sinusoidales
    fnd.graficar_ondas_sinusoidales(fs=100e3, duracion=2e-3, ruta_guardado=ruta_g2)

    # Gráfico 3: Demostración I/Q con plano complejo
    fnd.graficar_demostracion_iq(fc=5000, fs=200e3, ruta_guardado=ruta_g3)

    for ruta in [ruta_g1, ruta_g2, ruta_g3]:
        assert os.path.exists(ruta), f"Error: No se encontró el archivo generado: {ruta}"
        tamano_kb = os.path.getsize(ruta) / 1024
        assert tamano_kb > 10, f"Error: El archivo de imagen {ruta} está vacío o corrupto ({tamano_kb} KB)."
        print(f"    - Archivo verificado: {ruta} ({tamano_kb:.1f} KB)")

    print("    [PASÓ] Todos los gráficos generados y verificados con éxito.")
    pruebas_exitosas += 1

    # -------------------------------------------------------------
    # RESUMEN DE PRUEBAS
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"   RESULTADO FINAL: {pruebas_exitosas}/{total_pruebas} PRUEBAS COMPLETADAS CON ÉXITO")
    print("   El módulo 'fundamentos.py' está listo para su uso en las siguientes fases.")
    print("=" * 70)


if __name__ == "__main__":
    ejecutar_pruebas_fase2()

