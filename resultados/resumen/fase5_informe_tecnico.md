# INFORME TÉCNICO DE EJECUCIÓN — FASE 5: MODELOS DE RUIDO E INTERFERENCIAS
**Simulador de Modulaciones Digitales (Electiva de Comunicaciones Móviles)**

---

## 1. Archivos Creados e Implementados

1. **`ruido.py`**: Modelos estocásticos de ruido puramente aditivos:
   - AWGN en 3 dominios (símbolos discretos, banda base sobremuestreada y RF pasobanda).
   - Equivalencia rigurosa con $E_s/N_0$ mediante $\sigma_{\text{RF}}^2 = \frac{N_{sps} N_0}{4}$.
   - Ruido impulsivo Bernoulli-Gaussiano para señales reales y complejas.
   - Ruido térmico físico fundamentado en $N = kTB F$, densidad $N_0 \approx -173.975\text{ dBm/Hz}$ a 290 K y mapeo a varianza digital con potencia de referencia $1.0\text{ W}$ ($\sigma_I^2 = \sigma_Q^2 = \frac{1}{2 SNR}$).
   - Ruido no gaussiano con función de densidad de Laplace (colas pesadas).

2. **`interferencias.py`**: Modelos de interferencia electromagnética y multiusuario:
   - Interferencia de canal adyacente (ACI) con verificación estricta del límite de Nyquist ($|f_c + \Delta f| < f_s / 2$).
   - Flujo DS-CDMA real con matriz Walsh-Hadamard $H_8$, modulación BPSK, hasta 8 usuarios ($U \le 8$), retardos enteros con *zero-padding* y contraste contra la potencia teórica de interferencia $P_{\text{MUI,teor}} = \sum \alpha_u^2 \frac{\Gamma_{u,0}^2 + \hat{\Gamma}_{u,0}^2}{64}$.
   - Interferencia tonal monocromática de onda continua (CW) pasobanda.
   - Interferencia cocanal (CCI) sobre la misma portadora y ancho de banda.

3. **`efectos_canal.py`**: Modelos de degradación por canal y movilidad:
   - Ruido de fase por muestra sobre la envolvente en banda base.
   - Desplazamiento Doppler con fórmula cinemática $f_D = (v/c)f_c$, rotación continua en banda base, modulación de portadora en RF y desplazamiento de tonos en BFSK.

4. **`canales.py`**: Orquestador central del canal:
   - Dataclass `ModulacionResultadoPerturbado` conservando señales completas en RF, BB y símbolos.
   - Regla estricta de compatibilidad con BFSK: no construye constelaciones I/Q artificiales (`simbolos_pert = None`), opera sobre RF pasobanda y rechaza con `ValueError` descriptivo las peticiones de CDMA y Ruido de Fase.
   - Receptor coherente con demostración analítica de cancelación exacta de $2f_c$ en tiempo discreto.
   - Funciones DSP: cálculo de densidad espectral por FFT (`calcular_espectro_fft`), magnitud del vector de error (`calcular_evm`) y puente de demodulación transparente (`demodular_resultado_perturbado`).

5. **`test_fase5.py`**: Suite de certificación:
   - 11 pruebas unitarias matemáticas.
   - 20 escenarios de integración end-to-end.
   - Generación de los 10 gráficos requeridos.
   - Exportación de resultados a tablas `.csv` y `.md`.

---

## 2. Pruebas Ejecutadas y Resultados Obtenidos

### A. Pruebas Unitarias Matemáticas (11/11 PASS)
* `test_01_awgn_varianzas_tres_dominios`: Varianza en símbolo ($N_0 = 0.1$), banda base ($N_{sps} N_0 = 1.6$), RF pasobanda ($0.4$) y a la salida del filtro promediador ($N_0/2 = 0.05$) coinciden con el valor teórico (error $< 3\%$). **PASS**.
* `test_02_aci_frecuencia_y_nyquist`: Detección espectral de $f_{\text{adj}} = 3500\text{ Hz}$ por FFT y generación exitosa de `ValueError` al violar Nyquist. **PASS**.
* `test_03_ruido_impulsivo_tasa_y_amplitud`: Tasa empírica coincidente con $p_{\text{imp}} = 0.02$ y picos $> 8\times$ potencia media. **PASS**.
* `test_04_cdma_ortogonalidad_y_asincronismo`: Interferencia síncrona nula ($P_{\text{MUI}} = 0.0$, $\text{BER} = 0.0$), varianza asíncrona coincidente con $P_{\text{MUI,teor}}$ y rechazo de $U=9$ usuarios con `ValueError`. **PASS**.
* `test_05_ruido_fase_estadistica`: Media angular $0^\circ$ y desviación angular $12.0^\circ \pm 0.3^\circ$. **PASS**.
* `test_06_interferencia_tonal_cw`: Detección precisa de portadora espuria en $2800\text{ Hz}$ por FFT y potencia coincidente con $\text{SIR} = 8\text{ dB}$. **PASS**.
* `test_07_ruido_termico_ktb_dimensional`: Densidad física $-173.975\text{ dBm/Hz}$, potencia total $-115.965\text{ dBm}$ a $200\text{ kHz}$ y varianza digital con potencia de referencia $1.0\text{ W}$ verificadas. **PASS**.
* `test_08_interferencia_cocanal_sir`: SIR empírico medido de $12.0\text{ dB} \pm 0.1\text{ dB}$. **PASS**.
* `test_09_ruido_laplace_kurtosis`: Exceso de curtosis de Laplace $= 2.92$ (colas pesadas) vs Gaussiano $= 0.02$. **PASS**.
* `test_10_doppler_desplazamiento_frecuencia`: Verificación cinemática y detección espectral de rotación continua a $+250\text{ Hz}$. **PASS**.
* `test_11_regla_despacho_bfsk_excepciones`: Rechazo formal de CDMA y Ruido de Fase en BFSK con `ValueError`, operación pasobanda exitosa en AWGN RF con $\text{BER} = 0.0$ a $15\text{ dB}$ y conservación de `simbolos_pert = None`. **PASS**.

### B. Pruebas de Integración End-to-End
* Canal ideal de referencia ($SNR = 100\text{ dB}$): $\text{BER} = 0.00000$ en BFSK, BPSK, QPSK y 16-QAM.
* Curvas de degradación AWGN:
  - $SNR = 20\text{ dB}$: Todas mantienen $\text{BER} = 0.00000$, $\text{EVM} \approx 10\%$.
  - $SNR = 10\text{ dB}$: BPSK ($\text{BER}=0.0$), QPSK ($\text{BER}=0.0$), BFSK ($\text{BER}=0.0025$), 16-QAM ($\text{BER}=0.0708$).
  - $SNR = 2\text{ dB}$: BPSK ($\text{BER}=0.0375$), QPSK ($\text{BER}=0.1017$), 16-QAM ($\text{BER}=0.2492$), BFSK ($\text{BER}=0.2333$).
  *(Se confirma con exactitud la jerarquía teórica de robustez: $\text{BPSK} > \text{QPSK} > \text{16-QAM}$).*

---

## 3. Estado de Compatibilidad de Fases Anteriores
Se ejecutaron y certificaron sin alteraciones los scripts de las fases previas:
* **Fase 1 (`verificar_fase1.py`)**: Entorno y dependencias 100% verificados.
* **Fase 2 (`test_fase2.py`)**: 5/5 pruebas aprobadas.
* **Fase 3 (`test_fase3.py`)**: 7/7 pruebas aprobadas.
* **Fase 4 (`test_fase4.py`)**: 9/9 pruebas aprobadas ($\text{BER} = 0.0$).
* **Archivos modificados de fases anteriores**: **NINGUNO**. No se requirió alterar ningún código existente.

---

## 4. Gráficos Generados (`resultados/graficos/`)
1. `fase5_01_awgn.png`: Dispersión de constelación 16-QAM a $20\text{ dB}$, $10\text{ dB}$ y $0\text{ dB}$.
2. `fase5_02_aci.png`: Espectro FFT de la señal deseada frente a la interferencia adyacente.
3. `fase5_03_ruido_impulsivo.png`: Señal temporal en banda base mostrando los transitorios impulsivos Bernoulli-Gauss.
4. `fase5_04_cdma_mui.png`: Trama multiusuario compuesta en chips y símbolos descorrelacionados en el receptor.
5. `fase5_05_ruido_fase.png`: Constelaciones QPSK y 16-QAM con dispersión angular en arcos.
6. `fase5_06_interferencia_tonal.png`: Espectro FFT con el pico de la portadora parásita CW.
7. `fase5_07_ruido_termico.png`: Constelación 16-QAM degradada por ruido térmico $kTB$.
8. `fase5_08_interferencia_cocanal.png`: Constelación QPSK distorsionada por una réplica en la misma frecuencia.
9. `fase5_09_ruido_laplace_vs_gauss.png`: Histograma de función de densidad de probabilidad Gauss vs Laplace.
10. `fase5_10_doppler.png`: Espectro con portadora desplazada por Doppler y constelación rotatoria continua.

---

## 5. Errores Encontrados y Soluciones Aplicadas durante la Etapa B
* **Discrepancia de nombres en parámetro de semillas:** En la primera invocación a `fundamentos.generar_bits`, se utilizó el parámetro `seed` en vez de `semilla`. Se corrigió en `test_fase5.py` sin tocar `fundamentos.py`.
* **Retorno múltiple de `calcular_ber`:** La función devuelve la tupla `(ber, errores, total_bits)`. Se adaptaron las llamadas en `test_fase5.py` para desempaquetar la tupla correctamente.

---

## 6. Limitaciones Documentadas del Modelo
* El modelo de Ruido de Fase asume fluctuación angular independiente muestra a muestra (ruido blanco de fase); no modela la PSD continua $\mathcal{L}(f)$ de un lazo PLL físico.
* El modelo ACI utiliza una aproximación tonal adyacente dominante en lugar de una señal modulada aleatoria continua.
* El modelo CDMA utiliza exclusivamente códigos Walsh-Hadamard de orden 8 con modulación BPSK (límite de 8 usuarios concurrentes).

