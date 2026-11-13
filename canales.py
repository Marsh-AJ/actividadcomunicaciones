"""
canales.py
Módulo Orquestador de Canales Perturbados, Reconstrucción de Dominios y DSP.

Integra y despacha los diez fenómenos de ruido e interferencia sobre los objetos
ModulacionResultado, encapsulando las señales resultantes en la estructura formal
ModulacionResultadoPerturbado.

Desarrollado para la asignatura Electiva de Comunicaciones Móviles (Python 3.13.5).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Union, Any

# Módulos del proyecto
from modulaciones import ModulacionResultado
import demodulacion
from ruido import (
    aplicar_awgn_simbolos,
    aplicar_awgn_banda_base,
    aplicar_awgn_rf,
    aplicar_awgn_rf_equivalente_esn0,
    aplicar_ruido_impulsivo,
    aplicar_ruido_termico,
    aplicar_ruido_laplace,
)
from interferencias import (
    aplicar_aci,
    simular_cdma_multiusuario,
    aplicar_interferencia_tonal,
    aplicar_cci,
)
from efectos_canal import (
    aplicar_ruido_fase,
    calcular_frecuencia_doppler,
    aplicar_doppler_banda_base,
    aplicar_doppler_rf,
    aplicar_doppler_bfsk,
)


@dataclass
class ModulacionResultadoPerturbado:
    """
    Estructura de datos que encapsula el resultado completo de la transmisión a través
    de un canal con perturbaciones (ruido, interferencias o efectos de propagación).

    Conserva las señales completas en los tres dominios, referencias no distorsionadas
    y metadatos para trazabilidad analítica y visualización.
    """
    # 1. Señales perturbadas completas
    senal_rf_pert: np.ndarray             # Señal pasobanda real perturbada (1D, fs)
    senal_bb_pert: np.ndarray             # Envolvente compleja en banda base perturbada (1D, fs)
    simbolos_pert: np.ndarray | None      # Símbolos complejos perturbados (1 muestra/símbolo, Rs)
                                          # NOTA: Para BFSK es None para no construir I/Q artificial.

    # 2. Metadatos de modulación heredados
    tipo: str                             # 'BFSK', 'BPSK', 'QPSK', '16-QAM'
    bits_tx: np.ndarray                   # Bits originales transmitidos
    fs: float                             # Frecuencia de muestreo [Hz]
    fc: float                             # Frecuencia de portadora [Hz]
    sps: int                              # Muestras por símbolo (Nsps)
    bits_por_simbolo: int                 # Bits por símbolo (k = log2(M))
    constelacion_ref: dict[str, complex] | None  # Constelación de referencia (None en BFSK)

    # 3. Señales originales sin distorsión de referencia
    senal_rf_orig: np.ndarray             # Señal RF limpia original
    senal_bb_orig: np.ndarray             # Banda base limpia original
    simbolos_orig: np.ndarray | None      # Símbolos limpios originales (None en BFSK)

    # 4. Metadatos del canal aplicado
    canal_aplicado: str                   # Identificador del modelo (ej. 'AWGN', 'ACI', 'Doppler')
    parametros_canal: dict[str, Any]      # Parámetros utilizados (SNR, SIR, velocidad, etc.)

    # =================================================================
    # Propiedades para interoperabilidad con ModulacionResultado
    # =================================================================
    @property
    def simbolos(self) -> np.ndarray | None:
        """Símbolos perturbados (compatibilidad con demoduladores)."""
        return self.simbolos_pert

    @property
    def senal_rf(self) -> np.ndarray:
        """Señal RF pasobanda perturbada."""
        return self.senal_rf_pert

    @property
    def senal_bb(self) -> np.ndarray:
        """Señal en banda base sobremuestreada perturbada."""
        return self.senal_bb_pert

    @property
    def bits(self) -> np.ndarray:
        """Bits transmitidos originales."""
        return self.bits_tx

    @property
    def t(self) -> np.ndarray:
        """Vector temporal muestreado correspondiente a la señal RF."""
        return np.arange(len(self.senal_rf_pert)) / float(self.fs)


# =====================================================================
# RECEPTOR COHERENTE Y RECONSTRUCCIÓN ENTRE DOMINIOS
# =====================================================================

def demodular_envolvente_coherente(
    senal_rf: np.ndarray,
    fc: float,
    fs: float,
    sps: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Demodula coherentemente una señal pasobanda real multiplicando por 2·cos(2π fc t)
    y -2·sin(2π fc t) seguido de un filtro promediador por bloques de Nsps muestras.

    Condición analítica demostrada:
        Cancela de forma exacta e idéntica los armónicos a 2fc si y sólo si
        2fc · Ts = 2fc · Nsps / fs = m_cyc ∈ Z+ y fs > 4fc.

    Retorna:
        tuple[np.ndarray, np.ndarray]: (simbolos_recuperados, senal_bb_recuperada).
    """
    n_muestras = len(senal_rf)
    t = np.arange(n_muestras) / float(fs)

    # Producto coherente por los osciladores locales recuperados con factor 2
    rama_i_doble = 2.0 * senal_rf * np.cos(2.0 * np.pi * float(fc) * t)
    rama_q_doble = -2.0 * senal_rf * np.sin(2.0 * np.pi * float(fc) * t)

    n_simbolos = n_muestras // int(sps)
    i_bloques = rama_i_doble[: n_simbolos * sps].reshape(n_simbolos, sps)
    q_bloques = rama_q_doble[: n_simbolos * sps].reshape(n_simbolos, sps)

    # Promediador de integración y muestreo sobre el intervalo de símbolo
    i_sym = np.mean(i_bloques, axis=1)
    q_sym = np.mean(q_bloques, axis=1)
    simbolos = i_sym + 1j * q_sym

    # Envolvente en banda base sobremuestreada a tasa fs
    senal_bb = np.repeat(simbolos, sps)
    if len(senal_bb) < n_muestras:
        # Completar si la longitud no era múltiplo exacto
        sobrante = n_muestras - len(senal_bb)
        senal_bb = np.pad(senal_bb, (0, sobrante), mode="edge")

    return simbolos, senal_bb


def reconstruir_senal_rf_desde_bb(
    senal_bb: np.ndarray,
    fc: float,
    fs: float
) -> np.ndarray:
    """
    Sintetiza la señal real pasobanda (RF) a partir de la envolvente compleja en banda base:
        s_RF[n] = Re{s_BB[n]} · cos(2π fc t_n) - Im{s_BB[n]} · sin(2π fc t_n)
    """
    t = np.arange(len(senal_bb)) / float(fs)
    i = np.real(senal_bb)
    q = np.imag(senal_bb)
    return i * np.cos(2.0 * np.pi * float(fc) * t) - q * np.sin(2.0 * np.pi * float(fc) * t)


# =====================================================================
# DESPACHADOR CENTRAL DE CANAL (APLICAR_CANAL)
# =====================================================================

def aplicar_canal(
    mod_resultado: ModulacionResultado,
    tipo_canal: str,
    **kwargs: Any
) -> ModulacionResultadoPerturbado:
    """
    Aplica uno de los diez modelos de perturbación sobre el resultado de una modulación,
    respetando rigurosamente las reglas de despacho por modulación y dominios de señal.

    Modelos soportados (tipo_canal):
        - 'awgn': Ruido Blanco Gaussiano Aditivo (parámetro: snr_db o es_n0_db, dominio='bb'|'sym'|'rf').
        - 'aci': Interferencia de Canal Adyacente (parámetros: delta_f, sir_db).
        - 'impulsivo': Ruido impulsivo Bernoulli-Gauss (parámetros: p_imp, factor_amplitud).
        - 'cdma' / 'mui': Interferencia multiusuario CDMA (parámetros: num_usuarios, alfa_interf, retardos_chips).
        - 'ruido_fase': Ruido de fase por muestra en banda base (parámetro: desviacion_grados).
        - 'tonal': Interferencia tonal CW pasobanda (parámetros: f_cw, sir_db).
        - 'termico': Ruido térmico kTB (parámetros: potencia_rx_dbm, temperatura_k, ancho_banda_hz, figura_ruido_db).
        - 'cocanal' / 'cci': Interferencia cocanal (parámetros: sir_db, retardo_muestras, fase_rad).
        - 'laplace': Ruido de Laplace (parámetro: snr_db).
        - 'doppler': Efecto Doppler (parámetros: velocidad_kmh o f_doppler, angulo_rad).

    Regla estricta de despacho para BFSK:
        - BFSK opera en RF real (senal_rf).
        - Si se solicita 'cdma'/'mui' o 'ruido_fase' sobre BFSK, levanta ValueError descriptivo.
        - En BFSK, simbolos_pert se establece en None (no crea constelación I/Q artificial).

    Retorna:
        ModulacionResultadoPerturbado con las señales completas perturbadas y metadatos.
    """
    canal_norm = str(tipo_canal).strip().lower()
    tipo_mod = mod_resultado.tipo.upper()
    fs = float(mod_resultado.fs)
    fc = float(mod_resultado.fc)
    sps = int(mod_resultado.sps)
    seed = kwargs.get("seed", None)

    # =================================================================
    # REGLAS DE COMPATIBILIDAD BFSK
    # =================================================================
    if tipo_mod == "BFSK":
        if canal_norm in ("cdma", "mui"):
            raise ValueError(
                "El modelo MUI/CDMA no es compatible con la modulación BFSK: la técnica DSSS con "
                "códigos Walsh opera sobre constelaciones en fase (BPSK/QPSK) y BFSK se modula y "
                "demodula por frecuencias ortogonales en el canal de radiofrecuencia (RF)."
            )
        if canal_norm in ("ruido_fase", "fase"):
            raise ValueError(
                "El modelo de Ruido de Fase en constelación no es aplicable a BFSK, debido a que "
                "BFSK no posee constelación en banda base y su demodulación se realiza directamente "
                "en el dominio pasobanda RF mediante banco de correlación en cuadratura."
            )

    # Copias de referencia
    s_rf_orig = np.copy(mod_resultado.senal_rf)
    s_bb_orig = np.copy(mod_resultado.senal_bb)
    s_sym_orig = np.copy(mod_resultado.simbolos) if mod_resultado.simbolos is not None else None

    # Inicialización de salidas perturbadas
    s_rf_pert: np.ndarray
    s_bb_pert: np.ndarray
    s_sym_pert: np.ndarray | None = None
    params_usados = dict(kwargs)

    # =================================================================
    # DESPACHO POR MODELO
    # =================================================================

    if canal_norm == "awgn":
        snr_db = float(kwargs.get("snr_db", kwargs.get("es_n0_db", 10.0)))
        dominio = str(kwargs.get("dominio", "bb" if tipo_mod != "BFSK" else "rf")).lower()

        if tipo_mod == "BFSK" or dominio == "rf":
            # Caso RF: Si se solicita equivalencia estricta Es/N0
            modo_equiv = kwargs.get("modo_equivalencia", True)
            if modo_equiv:
                s_rf_pert = aplicar_awgn_rf_equivalente_esn0(s_rf_orig, es_n0_db=snr_db, sps=sps, seed=seed)
            else:
                s_rf_pert = aplicar_awgn_rf(s_rf_orig, snr_rf_db=snr_db, pot_senal=0.5, seed=seed)

            if tipo_mod != "BFSK":
                s_sym_pert, s_bb_pert = demodular_envolvente_coherente(s_rf_pert, fc, fs, sps)
            else:
                s_bb_pert = np.zeros_like(s_bb_orig)
                s_sym_pert = None

        elif dominio == "sym":
            assert s_sym_orig is not None
            s_sym_pert = aplicar_awgn_simbolos(s_sym_orig, snr_db=snr_db, seed=seed)
            s_bb_pert = np.repeat(s_sym_pert, sps)
            s_rf_pert = reconstruir_senal_rf_desde_bb(s_bb_pert, fc, fs)

        else:  # 'bb' por defecto para lineales
            s_bb_pert = aplicar_awgn_banda_base(s_bb_orig, snr_db=snr_db, sps=sps, seed=seed)
            # Recuperar símbolos pasando por el promediador rectangular
            n_sym = len(s_bb_pert) // sps
            s_sym_pert = np.mean(s_bb_pert[: n_sym * sps].reshape(n_sym, sps), axis=1)
            s_rf_pert = reconstruir_senal_rf_desde_bb(s_bb_pert, fc, fs)

    elif canal_norm == "aci":
        delta_f = float(kwargs.get("delta_f", 1500.0))
        sir_db = float(kwargs.get("sir_db", 10.0))
        fase_ini = float(kwargs.get("fase_inicial", 0.0))

        s_rf_pert = aplicar_aci(
            s_rf_orig, fs=fs, fc=fc, delta_f=delta_f, sir_db=sir_db, pot_senal=0.5, fase_inicial=fase_ini
        )
        if tipo_mod != "BFSK":
            s_sym_pert, s_bb_pert = demodular_envolvente_coherente(s_rf_pert, fc, fs, sps)
        else:
            s_bb_pert = np.zeros_like(s_bb_orig)
            s_sym_pert = None

    elif canal_norm == "impulsivo":
        p_imp = float(kwargs.get("p_imp", 0.01))
        factor_amp = float(kwargs.get("factor_amplitud", 5.0))

        if tipo_mod == "BFSK":
            s_rf_pert = aplicar_ruido_impulsivo(
                s_rf_orig, p_imp=p_imp, factor_amplitud=factor_amp, es_compleja=False, seed=seed
            )
            s_bb_pert = np.zeros_like(s_bb_orig)
            s_sym_pert = None
        else:
            s_bb_pert = aplicar_ruido_impulsivo(
                s_bb_orig, p_imp=p_imp, factor_amplitud=factor_amp, es_compleja=True, seed=seed
            )
            n_sym = len(s_bb_pert) // sps
            s_sym_pert = np.mean(s_bb_pert[: n_sym * sps].reshape(n_sym, sps), axis=1)
            s_rf_pert = reconstruir_senal_rf_desde_bb(s_bb_pert, fc, fs)

    elif canal_norm in ("cdma", "mui"):
        # Solo alcanzable para modulaciones lineales compatibles (BPSK)
        num_u = int(kwargs.get("num_usuarios", 4))
        alfa_u = kwargs.get("alfa_interf", 1.0)
        retardos = kwargs.get("retardos_chips", None)

        res_cdma = simular_cdma_multiusuario(
            bits_tx_u0=mod_resultado.bits,
            num_usuarios=num_u,
            alfa_interf=alfa_u,
            retardos_chips=retardos,
            seed=seed
        )
        params_usados["detalles_cdma"] = res_cdma

        # Símbolos descorrelacionados equivalentes
        s_sym_pert = np.asarray(res_cdma["simbolos_descorrelacionados"], dtype=complex)
        s_bb_pert = np.repeat(s_sym_pert, sps)
        s_rf_pert = reconstruir_senal_rf_desde_bb(s_bb_pert, fc, fs)

    elif canal_norm in ("ruido_fase", "fase"):
        desv_grados = float(kwargs.get("desviacion_grados", 5.0))
        s_bb_pert = aplicar_ruido_fase(s_bb_orig, desviacion_grados=desv_grados, seed=seed)
        n_sym = len(s_bb_pert) // sps
        s_sym_pert = np.mean(s_bb_pert[: n_sym * sps].reshape(n_sym, sps), axis=1)
        s_rf_pert = reconstruir_senal_rf_desde_bb(s_bb_pert, fc, fs)

    elif canal_norm == "tonal":
        f_cw = float(kwargs.get("f_cw", fc + 500.0))
        sir_db = float(kwargs.get("sir_db", 10.0))
        fase_ini = float(kwargs.get("fase_inicial", 0.0))

        s_rf_pert = aplicar_interferencia_tonal(
            s_rf_orig, fs=fs, f_cw=f_cw, sir_db=sir_db, pot_senal=0.5, fase_inicial=fase_ini
        )
        if tipo_mod != "BFSK":
            s_sym_pert, s_bb_pert = demodular_envolvente_coherente(s_rf_pert, fc, fs, sps)
        else:
            s_bb_pert = np.zeros_like(s_bb_orig)
            s_sym_pert = None

    elif canal_norm == "termico":
        p_rx_dbm = float(kwargs.get("potencia_rx_dbm", -80.0))
        temp_k = float(kwargs.get("temperatura_k", 290.0))
        bw_hz = float(kwargs.get("ancho_banda_hz", 200e3))
        nf_db = float(kwargs.get("figura_ruido_db", 5.0))

        if tipo_mod == "BFSK":
            s_rf_pert, metricas = aplicar_ruido_termico(
                s_rf_orig,
                potencia_rx_dbm=p_rx_dbm,
                temperatura_k=temp_k,
                ancho_banda_hz=bw_hz,
                figura_ruido_db=nf_db,
                pot_ref_digital=0.5,  # Potencia RF
                es_compleja=False,
                seed=seed
            )
            s_bb_pert = np.zeros_like(s_bb_orig)
            s_sym_pert = None
        else:
            s_bb_pert, metricas = aplicar_ruido_termico(
                s_bb_orig,
                potencia_rx_dbm=p_rx_dbm,
                temperatura_k=temp_k,
                ancho_banda_hz=bw_hz,
                figura_ruido_db=nf_db,
                pot_ref_digital=1.0,  # Potencia BB
                es_compleja=True,
                seed=seed
            )
            n_sym = len(s_bb_pert) // sps
            s_sym_pert = np.mean(s_bb_pert[: n_sym * sps].reshape(n_sym, sps), axis=1)
            s_rf_pert = reconstruir_senal_rf_desde_bb(s_bb_pert, fc, fs)

        params_usados["metricas_termicas"] = metricas

    elif canal_norm in ("cocanal", "cci"):
        sir_db = float(kwargs.get("sir_db", 10.0))
        retardo = int(kwargs.get("retardo_muestras", 15))
        fase_rad = float(kwargs.get("fase_rad", 0.5))
        s_interf = kwargs.get("senal_interferente", None)

        if tipo_mod == "BFSK":
            s_rf_pert = aplicar_cci(
                s_rf_orig, sir_db=sir_db, senal_interferente=s_interf, retardo_muestras=retardo,
                fase_rad=fase_rad, es_compleja=False, seed=seed
            )
            s_bb_pert = np.zeros_like(s_bb_orig)
            s_sym_pert = None
        else:
            s_bb_pert = aplicar_cci(
                s_bb_orig, sir_db=sir_db, senal_interferente=s_interf, retardo_muestras=retardo,
                fase_rad=fase_rad, es_compleja=True, seed=seed
            )
            n_sym = len(s_bb_pert) // sps
            s_sym_pert = np.mean(s_bb_pert[: n_sym * sps].reshape(n_sym, sps), axis=1)
            s_rf_pert = reconstruir_senal_rf_desde_bb(s_bb_pert, fc, fs)

    elif canal_norm == "laplace":
        snr_db = float(kwargs.get("snr_db", 10.0))

        if tipo_mod == "BFSK":
            s_rf_pert = aplicar_ruido_laplace(s_rf_orig, snr_db=snr_db, pot_senal=0.5, es_compleja=False, seed=seed)
            s_bb_pert = np.zeros_like(s_bb_orig)
            s_sym_pert = None
        else:
            s_bb_pert = aplicar_ruido_laplace(s_bb_orig, snr_db=snr_db, pot_senal=1.0, es_compleja=True, seed=seed)
            n_sym = len(s_bb_pert) // sps
            s_sym_pert = np.mean(s_bb_pert[: n_sym * sps].reshape(n_sym, sps), axis=1)
            s_rf_pert = reconstruir_senal_rf_desde_bb(s_bb_pert, fc, fs)

    elif canal_norm == "doppler":
        if "f_doppler" in kwargs:
            f_d = float(kwargs["f_doppler"])
        else:
            v_kmh = float(kwargs.get("velocidad_kmh", 120.0))
            ang_rad = float(kwargs.get("angulo_rad", 0.0))
            f_d = calcular_frecuencia_doppler(v_kmh, fc, ang_rad)
        params_usados["f_doppler_calculada"] = f_d

        if tipo_mod == "BFSK":
            f0 = float(kwargs.get("f0", fc - 1000.0))
            f1 = float(kwargs.get("f1", fc + 1000.0))
            s_rf_pert = aplicar_doppler_bfsk(
                bits=mod_resultado.bits, fs=fs, fc=fc, f0=f0, f1=f1, sps=sps, f_doppler=f_d
            )
            s_bb_pert = np.zeros_like(s_bb_orig)
            s_sym_pert = None
        else:
            # Rotación continua en banda base
            s_bb_pert = aplicar_doppler_banda_base(s_bb_orig, fs=fs, f_doppler=f_d)
            n_sym = len(s_bb_pert) // sps
            s_sym_pert = np.mean(s_bb_pert[: n_sym * sps].reshape(n_sym, sps), axis=1)
            # Pasobanda RF con portadora desplazada
            i_bb = np.real(s_bb_orig)
            q_bb = np.imag(s_bb_orig)
            s_rf_pert = aplicar_doppler_rf(i_bb=i_bb, q_bb=q_bb, fs=fs, fc=fc, f_doppler=f_d)

    else:
        raise ValueError(
            f"Tipo de canal '{tipo_canal}' no reconocido. Opciones válidas: 'awgn', 'aci', "
            f"'impulsivo', 'cdma', 'ruido_fase', 'tonal', 'termico', 'cocanal', 'laplace', 'doppler'."
        )

    return ModulacionResultadoPerturbado(
        senal_rf_pert=s_rf_pert,
        senal_bb_pert=s_bb_pert,
        simbolos_pert=s_sym_pert,
        tipo=tipo_mod,
        bits_tx=mod_resultado.bits,
        fs=fs,
        fc=fc,
        sps=sps,
        bits_por_simbolo=mod_resultado.bits_por_simbolo,
        constelacion_ref=mod_resultado.constelacion_ref,
        senal_rf_orig=s_rf_orig,
        senal_bb_orig=s_bb_orig,
        simbolos_orig=s_sym_orig,
        canal_aplicado=canal_norm,
        parametros_canal=params_usados,
    )


# =====================================================================
# HERRAMIENTAS DE ANÁLISIS DSP (FFT Y EVM) Y DEMODULACIÓN
# =====================================================================

def calcular_espectro_fft(
    senal: np.ndarray,
    fs: float,
    ventana: str = "hann"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula el espectro de densidad de potencia normalizado en dB mediante FFT.

    Parámetros:
        senal: Arreglo 1D (real o complejo).
        fs: Frecuencia de muestreo en Hz.
        ventana: Tipo de ventana ('hann' o 'rect').

    Retorna:
        tuple[np.ndarray, np.ndarray]: (frecuencias_hz, magnitud_db).
    """
    s = np.asarray(senal)
    n = len(s)
    if n == 0:
        raise ValueError("La señal no puede estar vacía.")

    if ventana.lower() == "hann":
        win = np.hanning(n)
    else:
        win = np.ones(n)

    s_win = s * win
    fft_val = np.fft.fftshift(np.fft.fft(s_win))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / float(fs)))

    # Potencia espectral en dB normalizada al pico máximo
    psd = np.abs(fft_val) ** 2
    psd_max = np.max(psd)
    if psd_max == 0.0:
        psd_max = 1.0
    psd_db = 10.0 * np.log10(np.maximum(psd / psd_max, 1e-12))

    return freqs, psd_db


def calcular_evm(
    simbolos_tx: np.ndarray,
    simbolos_rx: np.ndarray
) -> tuple[float, float]:
    """
    Calcula la magnitud del vector de error (EVM) porcentual (%) y en dB:
        EVM_rms = sqrt( sum(|s_rx - s_tx|²) / sum(|s_tx|²) ) · 100 %
        EVM_dB = 20 · log10(EVM_rms / 100)

    Retorna:
        tuple[float, float]: (evm_porcentaje, evm_db).
    """
    tx = np.asarray(simbolos_tx, dtype=complex)
    rx = np.asarray(simbolos_rx, dtype=complex)
    if len(tx) != len(rx):
        raise ValueError(f"Las longitudes difieren: tx={len(tx)}, rx={len(rx)}.")

    pot_error = np.sum(np.abs(rx - tx) ** 2)
    pot_ref = np.sum(np.abs(tx) ** 2)
    if pot_ref == 0.0:
        pot_ref = 1.0

    evm_rms = np.sqrt(pot_error / pot_ref)
    evm_pct = float(evm_rms * 100.0)
    evm_db = float(20.0 * np.log10(max(evm_rms, 1e-12)))

    return evm_pct, evm_db


def demodular_resultado_perturbado(
    res_pert: ModulacionResultadoPerturbado,
    **kwargs: Any
) -> np.ndarray:
    """
    Demodula un objeto ModulacionResultadoPerturbado conectando de forma transparente
    con las funciones probadas y aprobadas de demodulacion.py.

    - Para BFSK: invoca demodular_bfsk con la señal RF pasobanda.
    - Para BPSK: invoca demodular_bpsk con simbolos_pert.
    - Para QPSK: invoca demodular_qpsk con simbolos_pert.
    - Para 16-QAM: invoca demodular_16qam con simbolos_pert.

    Retorna:
        np.ndarray: Secuencia binaria recuperada (0 y 1) de tipo np.int8.
    """
    tipo = res_pert.tipo.upper()

    if tipo == "BFSK":
        f0 = float(kwargs.get("f0", res_pert.fc - 1000.0))
        f1 = float(kwargs.get("f1", res_pert.fc + 1000.0))
        return demodulacion.demodular_bfsk(
            entrada=res_pert.senal_rf_pert,
            f0=f0,
            f1=f1,
            sps=res_pert.sps,
            fs=res_pert.fs
        )
    elif tipo == "BPSK":
        assert res_pert.simbolos_pert is not None
        return demodulacion.demodular_bpsk(res_pert.simbolos_pert)
    elif tipo == "QPSK":
        assert res_pert.simbolos_pert is not None
        metodo = str(kwargs.get("metodo", "umbral"))
        return demodulacion.demodular_qpsk(res_pert.simbolos_pert, metodo=metodo)
    elif tipo == "16-QAM":
        assert res_pert.simbolos_pert is not None
        metodo = str(kwargs.get("metodo", "umbral"))
        return demodulacion.demodular_16qam(res_pert.simbolos_pert, metodo=metodo)
    else:
        raise ValueError(f"Modulación '{tipo}' no soportada.")

