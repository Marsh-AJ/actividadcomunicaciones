# FASE 6 — RESUMEN GLOBAL DE COMPROMISO (TRADE-OFF)

| Modulación | Bits/Símbolo | Eficiencia [bps/Hz] | Convención Ancho Banda | Eb/N0 (BER≤10⁻³) [dB] | Sensibilidad Prx [dBm] | SIR CCI (BER≤10⁻²) [dB] | Complejidad [Ops/Símbolo] | Dominio Receptor | Evaluación Robustez |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| BFSK | 1 | 0.25 | Regla de Carson | 11.0 | -129.0 | 2.0 | 129 | RF Pasobanda | Excelente robustez no coherente (inmune a ruido de fase) |
| BPSK | 1 | 0.50 | Lóbulo principal RF (nulo a nulo) | 7.0 | -133.0 | 0.0 | 1 | Símbolo Banda Base | Excelente inmunidad al ruido y coherencia de fase |
| QPSK | 2 | 1.00 | Lóbulo principal RF (nulo a nulo) | 7.0 | -129.0 | 6.0 | 2 | Símbolo Banda Base | Excelente inmunidad al ruido y coherencia de fase |
| 16-QAM | 4 | 2.00 | Lóbulo principal RF (nulo a nulo) | 11.0 | -123.0 | 16.0 | 8 | Símbolo Banda Base | Sensible a ruido y no linealidad; alta tasa de datos |
