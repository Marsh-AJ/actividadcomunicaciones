# DOCUMENTO MAESTRO DE CONTINUIDAD
## Proyecto de simulación de modulaciones digitales y técnicas de mitigación en comunicaciones móviles

**Instrucción para ChatGPT:** Lee todo este documento antes de responder. Contiene el contexto, las decisiones técnicas, los resultados y las restricciones de un proyecto académico. Tu objetivo es ayudar a continuar y reproducir el proyecto desde una computadora nueva, sin asumir que tienes acceso a los archivos, programas o entorno de desarrollo de la persona que elaboró este documento.

La información de este documento representa el estado de referencia del proyecto original. No debes afirmar que tienes los archivos originales ni que has ejecutado sus pruebas.

---

# 1. CONTEXTO GENERAL

Estoy desarrollando una actividad universitaria para la asignatura **Electiva de Comunicaciones Móviles**, correspondiente al tercer corte y con un valor del 15%.

El trabajo consiste en desarrollar mediante software un sistema de simulación de modulaciones digitales, estudiar su comportamiento ante distintos tipos de ruido e interferencias y evaluar técnicas de mitigación.

La actividad contempla:

- Modulación FSK.
- Modulación PSK.
- Modulación QAM.
- Diez modelos de ruido e interferencias.
- Evaluación cuantitativa de calidad de señal.
- Comparación de eficiencia espectral.
- Técnicas de mitigación.
- Conclusiones sobre la modulación más conveniente según cada escenario.

**Fecha de entrega original:** 12 de noviembre de 2026, antes de las 12:00 a. m.

# 2. ENTORNO DE DESARROLLO ORIGINAL

El proyecto original fue desarrollado utilizando:

- Windows 11.
- Visual Studio Code 1.142.0.
- Python 3.13.5.
- Entorno virtual `.venv`.
- NumPy 2.5.4.
- SciPy 1.18.1.
- Matplotlib 3.11.2.
- Pandas 3.1.1.

No se utilizó hardware SDR ni equipos físicos de radiofrecuencia. Todo se realizó mediante simulación.

**IMPORTANTE PARA LA NUEVA COMPUTADORA:**

No utilices rutas de la computadora original. Debes crear una carpeta de proyecto en la nueva máquina, por ejemplo:

`Actividad_Comunicaciones_Moviles`

Debes crear un entorno virtual propio e instalar las dependencias.

Archivo `requirements.txt`:

```text
numpy>=2.1.0
scipy>=1.14.0
matplotlib>=3.9.0
pandas>=2.2.0
```

Comandos orientativos para preparar un entorno nuevo:

```bash
python -m venv .venv
```

En Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Instalación:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Estructura general esperada:

```text
Actividad_Comunicaciones_Moviles/
│
├── fundamentos.py
├── modulaciones.py
├── demodulacion.py
├── ruido.py
├── interferencias.py
├── efectos_canal.py
├── canales.py
├── metricas.py
├── experimentos.py
│
├── resultados/
│   ├── graficos/
│   ├── tablas/
│   └── resumen/
│
└── requisitos.txt
```

Esta es una estructura de referencia. No significa que los archivos originales estén disponibles en la nueva máquina.

# 3. METODOLOGÍA Y ESTADO DEL PROYECTO

El desarrollo se organizó en ocho fases:

1. Preparación del entorno.
2. Fundamentos de simulación.
3. Implementación de modulaciones.
4. Demodulación y recuperación.
5. Modelos de ruido e interferencias.
6. Métricas y evaluación experimental.
7. Técnicas de mitigación y mejora del rendimiento.
8. Integración y entrega final.

Estado del proyecto original:

- Fases 1 a 6: completadas y aprobadas.
- Fase 7: plan técnico aprobado e implementación autorizada.
- Fase 8: pendiente.

Los resultados y aprobaciones descritos a continuación pertenecen al proyecto original. En una computadora nueva, si se requiere reproducir el sistema, será necesario reconstruir o recibir los archivos fuente correspondientes.

# 4. FASE 1 — PREPARACIÓN

**Estado original: COMPLETADA Y APROBADA.**

Se verificó:

- Python.
- Entorno virtual.
- Dependencias.
- Operaciones de NumPy y SciPy.
- Estructura de carpetas.

Se creó `verificar_fase1.py`.

# 5. FASE 2 — FUNDAMENTOS DE SIMULACIÓN

**Estado original: COMPLETADA Y APROBADA.**

Se implementaron:

- Generación de 10.000 bits.
- Señales NRZ.
- Señales sinusoidales.
- Representación I/Q.
- Síntesis RF.

Resultados de prueba:

- Porcentaje de unos: 49,03 %.
- Amplitud: 2,5 V.
- Frecuencia: 2 kHz.
- Muestreo: 200 kHz.
- Potencia: 3,125 W, suponiendo una carga de 1 Ω.

Representación compleja:

\[
s[n]=I[n]+jQ[n]
\]

Gráficos:

- `fase2_01_secuencia_bits.png`
- `fase2_02_ondas_sinusoidales.png`
- `fase2_03_representacion_iq.png`

# 6. FASE 3 — MODULACIONES DIGITALES

**Estado original: COMPLETADA Y APROBADA.**

Se implementaron:

- BFSK.
- BPSK.
- QPSK.
- 16-QAM.

Parámetros BFSK:

- \(f_0=2000\ Hz\)
- \(f_1=4000\ Hz\)

Mapeo BPSK:

- Bit 0 → -1.
- Bit 1 → +1.

Mapeo Gray QPSK:

- 00 → (+1+j)/√2
- 01 → (-1+j)/√2
- 11 → (-1-j)/√2
- 10 → (+1-j)/√2

Mapeo 16-QAM:

- 00 → -3.
- 01 → -1.
- 11 → +1.
- 10 → +3.

Normalización 16-QAM: \(1/\sqrt{10}\).

Energías normalizadas:

| Modulación | Es | Eb | Distancia mínima |
|---|---:|---:|---:|
| BPSK | 1 | 1 | 2 |
| QPSK | 1 | 0,5 | √2 |
| 16-QAM | 1 | 0,25 | 2/√10 |

Funciones auxiliares:

- `obtener_factor_normalizacion`
- `calcular_energia_promedio`
- `calcular_energia_bit`

Gráficos:

- `fase3_01_bfsk.png`
- `fase3_02_bpsk.png`
- `fase3_03_qpsk.png`
- `fase3_04_16qam.png`
- `fase3_05_comparacion_constelaciones.png`

# 7. FASE 4 — DEMODULACIÓN Y RECUPERACIÓN

**Estado original: COMPLETADA Y APROBADA.**

Se implementaron los demoduladores de las cuatro modulaciones.

Resultados:

- BFSK: BER = 0.
- BPSK: BER = 0 en 1.000 bits.
- QPSK: BER = 0 en 1.000 bits.
- 16-QAM: BER = 0 en 2.000 bits.

Se probaron secuencias aleatorias, secuencias de ceros y unos, patrones alternados, todos los símbolos y diferentes fases iniciales para BFSK.

También se validaron entradas incorrectas y condiciones límite.

Pruebas: 9 de 9 satisfactorias.

Gráficos:

- `fase4_01_recuperacion_bfsk.png`
- `fase4_02_recuperacion_bpsk.png`
- `fase4_03_recuperacion_qpsk.png`
- `fase4_04_recuperacion_16qam.png`
- `fase4_05_comparativa_recuperacion.png`

# 8. FASE 5 — RUIDO E INTERFERENCIAS

**Estado original: COMPLETADA Y APROBADA.**

Archivos originales:

- `ruido.py`
- `interferencias.py`
- `efectos_canal.py`
- `canales.py`
- `test_fase5.py`

Se implementaron diez fenómenos:

1. AWGN.
2. Interferencia de canal adyacente (ACI).
3. Ruido impulsivo Bernoulli-Gaussiano.
4. Interferencia multiusuario (MUI) mediante CDMA simplificado.
5. Ruido de fase.
6. Interferencia tonal CW.
7. Ruido térmico.
8. Interferencia cocanal.
9. Ruido impulsivo Laplace.
10. Efectos Doppler.

Consideraciones:

- Se diferenciaron SNR RF y equivalencias Es/N0.
- El ruido térmico utiliza \(kTB\).
- CDMA utiliza códigos Walsh-Hadamard de orden 8 y BPSK.
- El ruido de fase es una aproximación blanca independiente por muestra, no un modelo completo de PLL.
- Doppler se modeló mediante rotación en banda base y desplazamiento de portadora RF.

Validación original:

- 11 pruebas unitarias satisfactorias.
- 20 escenarios de integración.

Gráficos:

`fase5_01_awgn.png` hasta `fase5_10_doppler.png`.

Informes:

- `fase5_resumen_modelos.csv`
- `fase5_resumen_modelos.md`
- `fase5_informe_tecnico.md`

No se debe afirmar que los diez fenómenos se evaluaron exhaustivamente con todas las modulaciones. Se realizaron escenarios representativos.

# 9. FASE 6 — MÉTRICAS Y EVALUACIÓN EXPERIMENTAL

**Estado original: COMPLETADA Y APROBADA.**

Archivos:

- `metricas.py`
- `experimentos.py`
- `test_fase6.py`

Se utilizaron 60.000 bits por punto experimental.

Métricas:

- BER.
- EVM.
- SNR.
- Eb/N0.
- Es/N0.
- Eficiencia espectral.
- Intervalos de confianza Wilson del 95 %.
- Regla de tres para cero errores.

BER teórica:

BPSK y QPSK:

\[
BER=0.5\,erfc(\sqrt{Eb/N0})
\]

16-QAM, aproximación Gray:

\[
BER\approx0.75Q(\sqrt{0.8Eb/N0})
\]

BFSK ortogonal no coherente:

\[
BER=0.5e^{-Eb/(2N0)}
\]

### Eficiencia espectral

| Modulación | Ideal Nyquist | Lóbulo principal RF |
|---|---:|---:|
| BFSK | 0,25 bps/Hz según Carson | 0,25 bps/Hz |
| BPSK | 1 bps/Hz | 0,5 bps/Hz |
| QPSK | 2 bps/Hz | 1 bps/Hz |
| 16-QAM | 4 bps/Hz | 2 bps/Hz |

### Resultados AWGN a 10 dB

- BFSK: BER 0,003417; teórica 0,003369.
- BPSK: cero errores; límite aproximado 0,00005.
- QPSK: BER 0,00001667.
- 16-QAM: BER 0,00170; teórica aproximada 0,001754.

### Ruido térmico a -135 dBm

| Modulación | Eb/N0 aproximado | BER |
|---|---:|---:|
| BPSK | 4,97 dB | 0,006517 |
| QPSK | 1,96 dB | 0,0385 |
| BFSK | — | 0,1029 |
| 16-QAM | — | 0,1640 |

Se corrigió una calibración inicial que producía una ganancia artificial de procesamiento de 12,04 dB.

### Interferencia cocanal a SIR = 0 dB

- BPSK: BER 0,004917.
- QPSK: BER 0,2514.
- 16-QAM: BER 0,3209.
- BFSK: BER 0,02518.

Se corrigió un error inicial que utilizaba una réplica retrasada de la señal deseada como interferente. La versión final usa bits interferentes independientes.

### Umbrales aproximados de potencia para BER ≤ 10⁻³

- BFSK: -129 dBm.
- BPSK: -133 dBm.
- QPSK: -129 dBm.
- 16-QAM: -123 dBm.

### Umbrales aproximados de SIR para BER ≤ 10⁻²

- BFSK: 2 dB.
- BPSK: 0 dB.
- QPSK: 6 dB.
- 16-QAM: 16 dB.

Los umbrales de potencia proceden de un barrido discreto de 2 dB, no son sensibilidades mínimas exactas.

Pruebas originales: 8 de 8 satisfactorias. También pasaron las pruebas de regresión de Fases 1 a 5.

# 10. FASE 7 — MITIGACIÓN

**Estado original: PLAN APROBADO. IMPLEMENTACIÓN AUTORIZADA.**

Esta fase debe implementar tres técnicas:

1. Filtrado digital FIR.
2. Ecualización Zero-Forcing.
3. Diversidad de recepción MRC y SC.

El plan ya fue revisado y autorizado. No se debe volver a diseñar desde cero.

### 10.1. Filtrado FIR

Objetivo: reducir ruido fuera de banda e interferencia ACI.

Diseño:

- FIR paso bajo.
- Ventana Hamming.
- `scipy.signal.firwin`.
- Frecuencia de corte inicial: 1.000 Hz en banda base.
- Para BFSK RF, filtro paso banda centrado en la portadora.

Escenario:

- ACI con separación de 1.500 Hz.
- SIR de 0 a 16 dB.
- Comparación antes y después.
- BER y EVM cuando corresponda.

Condiciones:

- Documentar frecuencia de muestreo, corte y orden.
- Medir la atenuación real de la interferencia.
- Verificar que no se eliminen componentes útiles.
- No garantizar anticipadamente una atenuación superior a 25 dB ni la eliminación de errores.
- Mantener iguales condiciones de señal, potencia y ruido.

### 10.2. Ecualización ZF

Canal:

\[
r[n]=s[n]+\alpha s[n-1]+w[n]
\]

Con:

\[
h=[1.0,\ 0.35]
\]

Se utilizará un ecualizador FIR de longitud finita para BPSK, QPSK y 16-QAM.

Barrido de Eb/N0 entre 0 y 16 dB.

Condiciones:

- Definir longitud del ecualizador y retardo de decisión.
- Verificar la respuesta combinada canal-ecualizador.
- Medir amplificación de ruido.
- No afirmar que ZF siempre mejora.
- BFSK queda fuera de este experimento si se trabaja con símbolos complejos.

### 10.3. Diversidad MRC y SC

Canal SIMO 1×2:

\[
r_1=h_1s+w_1
\]

\[
r_2=h_2s+w_2
\]

Combinación MRC:

\[
y=h_1^*r_1+h_2^*r_2
\]

Comparar:

- SISO: una rama.
- SC: dos ramas con selección.
- MRC: dos ramas con combinación de máxima relación.

Barrido de SNR media de 0 a 18 dB.

Condiciones:

- Canales Rayleigh independientes.
- Ruido independiente e igual varianza.
- Potencia transmitida comparable entre SISO y SIMO.
- No presentar ganancias teóricas como resultados medidos.
- Una ganancia superior a 10 dB es expectativa, no garantía.

### Archivos previstos

```text
mitigacion.py
experimentos_fase7.py
test_fase7.py
```

Gráficos:

- `fase7_01_filtrado_antes_despues.png`
- `fase7_02_ecualizacion_antes_despues.png`
- `fase7_03_diversidad_mrc_vs_siso.png`

Tablas CSV y Markdown:

- `fase7_experimento1_filtrado`
- `fase7_experimento2_ecualizacion`
- `fase7_experimento3_diversidad`
- `fase7_resumen_mitigacion`

También debe generarse un informe técnico.

**IMPORTANTE:** El plan está autorizado, pero la implementación y sus resultados todavía deben comprobarse. No inventar resultados ni asumir que los archivos ya existen.

# 11. FASE 8 — ENTREGA FINAL

**Estado: PENDIENTE.**

Se debe elaborar una presentación PowerPoint llamada:

`Nombre_Apellido_Electiva_SCM4.pptx`

Contenido mínimo:

1. Portada.
2. Objetivo.
3. Metodología y entorno.
4. Fundamentos de modulación.
5. BFSK, BPSK, QPSK y 16-QAM.
6. Constelaciones y formas de onda.
7. Diez modelos de ruido e interferencias.
8. Resultados experimentales.
9. Comparación BER.
10. Comparación EVM.
11. Eficiencia espectral.
12. Complejidad y compromisos.
13. Filtrado FIR.
14. Ecualización ZF.
15. Diversidad MRC y SC.
16. Resultados antes y después de mitigación.
17. Conclusiones.
18. Referencias técnicas.

La presentación debe utilizar resultados reales y gráficos legibles. No es necesario colocar absolutamente todos los gráficos.

# 12. REGLAS OBLIGATORIAS

1. No inventar archivos, resultados ni pruebas.
2. No afirmar que se tiene acceso a la computadora original.
3. No utilizar rutas locales de la computadora original.
4. No asumir que los módulos Python están disponibles en la nueva máquina.
5. No modificar decisiones técnicas aprobadas sin una razón importante.
6. No afirmar que una técnica mejora el rendimiento si los resultados no lo demuestran.
7. Diferenciar resultados teóricos, aproximaciones y resultados simulados.
8. Mantener comparaciones en condiciones equivalentes.
9. Usar semillas aleatorias para reproducibilidad cuando corresponda.
10. Priorizar una entrega funcional y defendible frente a la sobreingeniería.

# 13. INSTRUCCIONES PARA EL NUEVO CHATGPT

Actúa como asesor técnico y académico del proyecto.

Mi nivel de programación es intermedio. Necesito explicaciones claras y directas, sin perder tiempo en explicaciones innecesarias.

Debes comprender que el proyecto original fue desarrollado en otra computadora. Hay dos posibilidades:

- Si recibes los archivos originales, debes inspeccionarlos y continuar desde ellos.
- Si no recibes los archivos, debes ayudar a reconstruir el proyecto en la nueva máquina utilizando las especificaciones y resultados documentados aquí. No debes afirmar que estás continuando código que no has recibido.

No reinicies fases aprobadas ni vuelvas a diseñar la Fase 7. Su plan ya fue autorizado.

Si hay que reconstruir el código, hazlo de forma ordenada, manteniendo la arquitectura y los resultados de referencia, y deja claro qué resultados se han reproducido realmente y cuáles proceden del registro del proyecto original.

La prioridad es terminar la Fase 7 y preparar la presentación de la Fase 8.

# 14. PUNTO DE PARTIDA

El proyecto original terminó la Fase 6 y autorizó la implementación de la Fase 7.

La siguiente tarea es continuar con la implementación o reconstrucción de:

- `mitigacion.py`
- `experimentos_fase7.py`
- `test_fase7.py`

Después se deben preparar los resultados y la presentación PowerPoint.

**Primero confirma que comprendiste el proyecto y pregunta si la persona dispone de los archivos fuente originales o si necesita reconstruirlos desde cero en su computadora. No asumas que tiene acceso a la máquina original.**

