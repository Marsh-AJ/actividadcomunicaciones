# FASE 6 — EXPERIMENTO 4: RUIDO DE FASE Y CANAL ADYACENTE (ACI)

Parte 4A: Ruido de Fase (0° a 25°, Eb/N0 fondo = 12 dB). BFSK es N/A (demodulación no coherente pasobanda).
Parte 4B: Interferencia ACI (SIR 0 a 16 dB, Δf = 1500 Hz, Eb/N0 fondo = 15 dB).

| Subexperimento | Modulación | Parámetro Perturbación | BER Simulada | Errores | Wilson Sup 95% | Cota 3/Nb | EVM RMS |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 4A_Ruido_Fase | BPSK | sigma_theta=0.0deg | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 25.09% |
| 4A_Ruido_Fase | BPSK | sigma_theta=5.0deg | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 25.17% |
| 4A_Ruido_Fase | BPSK | sigma_theta=10.0deg | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 25.54% |
| 4A_Ruido_Fase | BPSK | sigma_theta=15.0deg | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 26.17% |
| 4A_Ruido_Fase | BPSK | sigma_theta=20.0deg | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 27.13% |
| 4A_Ruido_Fase | BPSK | sigma_theta=25.0deg | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 28.63% |
| 4A_Ruido_Fase | QPSK | sigma_theta=0.0deg | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 17.62% |
| 4A_Ruido_Fase | QPSK | sigma_theta=5.0deg | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 17.94% |
| 4A_Ruido_Fase | QPSK | sigma_theta=10.0deg | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 18.35% |
| 4A_Ruido_Fase | QPSK | sigma_theta=15.0deg | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 19.25% |
| 4A_Ruido_Fase | QPSK | sigma_theta=20.0deg | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 20.61% |
| 4A_Ruido_Fase | QPSK | sigma_theta=25.0deg | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 22.35% |
| 4A_Ruido_Fase | 16-QAM | sigma_theta=0.0deg | 1.166667e-04 | 7 | 2.408231e-04 | N/A | 12.48% |
| 4A_Ruido_Fase | 16-QAM | sigma_theta=5.0deg | 8.333333e-05 | 5 | 1.950807e-04 | N/A | 12.74% |
| 4A_Ruido_Fase | 16-QAM | sigma_theta=10.0deg | 3.000000e-04 | 18 | 4.742010e-04 | N/A | 13.35% |
| 4A_Ruido_Fase | 16-QAM | sigma_theta=15.0deg | 1.200000e-03 | 72 | 1.510773e-03 | N/A | 14.48% |
| 4A_Ruido_Fase | 16-QAM | sigma_theta=20.0deg | 3.816667e-03 | 229 | 4.342821e-03 | N/A | 16.30% |
| 4A_Ruido_Fase | 16-QAM | sigma_theta=25.0deg | 9.333333e-03 | 560 | 1.013477e-02 | N/A | 18.63% |
| 4B_ACI | BFSK | SIR=0.0dB, Df=1500Hz | 6.743333e-02 | 4046 | 6.946770e-02 | N/A | N/A |
| 4B_ACI | BFSK | SIR=2.0dB, Df=1500Hz | 1.166667e-02 | 700 | 1.255768e-02 | N/A | N/A |
| 4B_ACI | BFSK | SIR=4.0dB, Df=1500Hz | 2.100000e-03 | 126 | 2.499539e-03 | N/A | N/A |
| 4B_ACI | BFSK | SIR=6.0dB, Df=1500Hz | 3.166667e-04 | 19 | 4.945674e-04 | N/A | N/A |
| 4B_ACI | BFSK | SIR=8.0dB, Df=1500Hz | 6.666667e-05 | 4 | 1.714194e-04 | N/A | N/A |
| 4B_ACI | BFSK | SIR=10.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | N/A |
| 4B_ACI | BFSK | SIR=12.0dB, Df=1500Hz | 1.666667e-05 | 1 | 9.440933e-05 | N/A | N/A |
| 4B_ACI | BFSK | SIR=14.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | N/A |
| 4B_ACI | BFSK | SIR=16.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | N/A |
| 4B_ACI | BPSK | SIR=0.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 27.75% |
| 4B_ACI | BPSK | SIR=2.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 24.53% |
| 4B_ACI | BPSK | SIR=4.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 22.28% |
| 4B_ACI | BPSK | SIR=6.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 20.70% |
| 4B_ACI | BPSK | SIR=8.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 19.74% |
| 4B_ACI | BPSK | SIR=10.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 19.04% |
| 4B_ACI | BPSK | SIR=12.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 18.52% |
| 4B_ACI | BPSK | SIR=14.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 18.30% |
| 4B_ACI | BPSK | SIR=16.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 18.15% |
| 4B_ACI | QPSK | SIR=0.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 24.65% |
| 4B_ACI | QPSK | SIR=2.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 21.07% |
| 4B_ACI | QPSK | SIR=4.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 18.39% |
| 4B_ACI | QPSK | SIR=6.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 16.46% |
| 4B_ACI | QPSK | SIR=8.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 15.20% |
| 4B_ACI | QPSK | SIR=10.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 14.25% |
| 4B_ACI | QPSK | SIR=12.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 13.64% |
| 4B_ACI | QPSK | SIR=14.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 13.32% |
| 4B_ACI | QPSK | SIR=16.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 13.04% |
| 4B_ACI | 16-QAM | SIR=0.0dB, Df=1500Hz | 2.050000e-03 | 123 | 2.445181e-03 | N/A | 23.08% |
| 4B_ACI | 16-QAM | SIR=2.0dB, Df=1500Hz | 3.833333e-04 | 23 | 5.751774e-04 | N/A | 19.03% |
| 4B_ACI | 16-QAM | SIR=4.0dB, Df=1500Hz | 2.000000e-04 | 12 | 3.495784e-04 | N/A | 16.14% |
| 4B_ACI | 16-QAM | SIR=6.0dB, Df=1500Hz | 5.000000e-05 | 3 | 1.470092e-04 | N/A | 13.84% |
| 4B_ACI | 16-QAM | SIR=8.0dB, Df=1500Hz | 3.333333e-05 | 2 | 1.215414e-04 | N/A | 12.28% |
| 4B_ACI | 16-QAM | SIR=10.0dB, Df=1500Hz | 1.666667e-05 | 1 | 9.440933e-05 | N/A | 11.18% |
| 4B_ACI | 16-QAM | SIR=12.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 10.39% |
| 4B_ACI | 16-QAM | SIR=14.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 9.86% |
| 4B_ACI | 16-QAM | SIR=16.0dB, Df=1500Hz | 0.000000e+00 | 0 | 6.402021e-05 | 5.000000e-05 | 9.55% |
