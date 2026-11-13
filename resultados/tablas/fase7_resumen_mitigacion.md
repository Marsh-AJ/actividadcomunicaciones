# FASE 7 — RESUMEN GLOBAL DE TÉCNICAS DE MITIGACIÓN

| Técnica | Escenario de Canal | Configuración Técnica | Desempeño Sin Mitigación | Desempeño Con Mitigación | Beneficio Demostrado |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1. Filtrado Digital FIR | Interferencia Canal Adyacente (ACI Δf=1500 Hz) | FIR Orden 48, Hamming, fc=1 kHz | Degradación en ACI (BER hasta 0.06) | Atenuación ACI de 40.6 dB; BER reducida a 0.0 | Supresión de interferencia fuera de banda |
| 2. Ecualización Zero-Forcing (ZF) | Canal Dispersivo con ISI (h=[1.0, 0.35]) | FIR Nw=7, Retardo Δ=0, NEF=+0.57 dB | Piso de error severo (BER > 10% en 16-QAM) | Cancelación de ISI (BER < 1e-4 a 14 dB) | Eliminación del piso de error por dispersión multitrayecto |
| 3. Diversidad de Recepción (MRC L=2) | Canal Móvil con Desvanecimiento Rayleigh Plano | SIMO 1x2, Ramas independientes, MRC coherente | Decaimiento lento orden 1 (BER ≈ 2.2% a 10 dB) | Decaimiento rápido orden 2 (BER ≈ 0.15% a 10 dB) | Ganancia de diversidad > 10 dB a BER = 10^-3 |
