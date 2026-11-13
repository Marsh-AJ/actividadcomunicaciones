"""
generar_presentacion_fase8.py
Generador automatizado de la Presentación Final (FASE 8) en formato PowerPoint (.pptx).

Universidad Alejandro de Humboldt — Electiva de Comunicaciones Móviles (SCM4)
Autores: Jesús Balza y Andrea Blazque

Lee directamente las tablas CSV reales y gráficos PNG generados en Fases 2 a 7
desde resultados/tablas/ y resultados/graficos/ para producir:
    Jesus_Balza_Andrea_Blazque_Electiva_SCM4.pptx
"""

from __future__ import annotations
import csv
import os
from PIL import Image

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_GRAFICOS = os.path.join(DIR_BASE, "resultados", "graficos")
DIR_TABLAS = os.path.join(DIR_BASE, "resultados", "tablas")
ARCHIVO_SALIDA = os.path.join(DIR_BASE, "Jesus_Balza_Andrea_Blazque_Electiva_SCM4.pptx")

ANCHO_SLIDE = Inches(13.333)
ALTO_SLIDE = Inches(7.5)
TOTAL_DIAPOSITIVAS = 19

# Paleta institucional y de ingeniería
AZUL_OSCURO = RGBColor(15, 41, 66)      # #0F2942
AZUL_MEDIO = RGBColor(27, 108, 168)     # #1B6CA8
VERDE_TEAL = RGBColor(13, 148, 136)     # #0D9488
AMBAR = RGBColor(217, 119, 6)           # #D97706
FONDO_CLARO = RGBColor(248, 250, 252)   # #F8FAFC
FONDO_TARJETA = RGBColor(255, 255, 255) # #FFFFFF
FONDO_ZEBRA = RGBColor(241, 245, 249)   # #F1F5F9
BORDE_SUAVE = RGBColor(203, 213, 225)   # #CBD5E1
TEXTO_OSCURO = RGBColor(30, 41, 59)     # #1E293B
TEXTO_SUAVE = RGBColor(71, 85, 105)     # #475569
BLANCO = RGBColor(255, 255, 255)
VERDE_SUAVE = RGBColor(220, 252, 231)   # #DCFCE7
VERDE_TEXTO = RGBColor(22, 101, 52)     # #166534
AZUL_SUAVE = RGBColor(224, 242, 254)    # #E0F2FE
AMBAR_SUAVE = RGBColor(254, 243, 199)   # #FEF3C7


# =====================================================================
# UTILIDADES DE LECTURA DE DATOS REALES
# =====================================================================

def leer_csv(nombre_archivo: str) -> list[dict]:
    """Lee un archivo CSV de resultados/tablas/ y devuelve sus filas."""
    ruta = os.path.join(DIR_TABLAS, nombre_archivo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Tabla no encontrada: {ruta}")
    with open(ruta, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ruta_grafico(nombre_archivo: str) -> str:
    """Valida y retorna la ruta absoluta de un gráfico PNG en resultados/graficos/."""
    ruta = os.path.join(DIR_GRAFICOS, nombre_archivo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Gráfico no encontrado: {ruta}")
    return ruta


def buscar_fila(filas: list[dict], **criterios: str) -> dict:
    """Devuelve la primera fila del CSV que coincide con los criterios exactos."""
    for fila in filas:
        if all(fila.get(k) == v for k, v in criterios.items()):
            return fila
    raise KeyError(f"No se encontró fila con criterios {criterios}")


# =====================================================================
# HELPERS DE DISEÑO Y MAQUETACIÓN POWERPOINT
# =====================================================================

def nueva_diapositiva_base(
    prs: Presentation,
    etiqueta_fase: str,
    titulo: str,
    numero: int,
    subtitulo: str = "",
):
    """Crea una diapositiva 16:9 limpia con encabezado y pie institucional."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Fondo suave
    fondo = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), ANCHO_SLIDE, ALTO_SLIDE
    )
    fondo.fill.solid()
    fondo.fill.fore_color.rgb = FONDO_CLARO
    fondo.line.fill.background()

    # Banda superior de encabezado
    barra = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), ANCHO_SLIDE, Inches(0.86)
    )
    barra.fill.solid()
    barra.fill.fore_color.rgb = AZUL_OSCURO
    barra.line.fill.background()

    # Línea de acento inferior del encabezado
    acento = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0.86), ANCHO_SLIDE, Inches(0.05)
    )
    acento.fill.solid()
    acento.fill.fore_color.rgb = VERDE_TEAL
    acento.line.fill.background()

    # Píldora de fase / sección
    pill = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(0.18), Inches(1.55), Inches(0.46)
    )
    pill.fill.solid()
    pill.fill.fore_color.rgb = VERDE_TEAL
    pill.line.fill.background()
    tf_pill = pill.text_frame
    tf_pill.word_wrap = False
    tf_pill.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_pill = tf_pill.paragraphs[0]
    p_pill.alignment = PP_ALIGN.CENTER
    p_pill.text = etiqueta_fase
    p_pill.font.name = "Calibri"
    p_pill.font.size = Pt(11)
    p_pill.font.bold = True
    p_pill.font.color.rgb = BLANCO

    # Título de la diapositiva
    tb_tit = slide.shapes.add_textbox(
        Inches(2.15), Inches(0.10), Inches(10.75), Inches(0.68)
    )
    tf_tit = tb_tit.text_frame
    tf_tit.word_wrap = True
    tf_tit.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_tit = tf_tit.paragraphs[0]
    p_tit.text = titulo
    p_tit.font.name = "Calibri"
    p_tit.font.size = Pt(21)
    p_tit.font.bold = True
    p_tit.font.color.rgb = BLANCO

    if subtitulo:
        p_sub = tf_tit.add_paragraph()
        p_sub.text = subtitulo
        p_sub.font.name = "Calibri"
        p_sub.font.size = Pt(11)
        p_sub.font.color.rgb = RGBColor(186, 230, 253)

    # Línea separadora del pie
    sep = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(7.10), Inches(12.43), Inches(0.015)
    )
    sep.fill.solid()
    sep.fill.fore_color.rgb = BORDE_SUAVE
    sep.line.fill.background()

    # Pie institucional
    tb_pie = slide.shapes.add_textbox(
        Inches(0.45), Inches(7.13), Inches(10.2), Inches(0.28)
    )
    tf_pie = tb_pie.text_frame
    p_pie = tf_pie.paragraphs[0]
    p_pie.text = (
        "Universidad Alejandro de Humboldt  ·  Electiva de Comunicaciones Móviles (SCM4)"
        "  ·  Jesús Balza & Andrea Blazque"
    )
    p_pie.font.name = "Calibri"
    p_pie.font.size = Pt(9.5)
    p_pie.font.color.rgb = TEXTO_SUAVE

    tb_num = slide.shapes.add_textbox(
        Inches(11.3), Inches(7.13), Inches(1.58), Inches(0.28)
    )
    tf_num = tb_num.text_frame
    p_num = tf_num.paragraphs[0]
    p_num.alignment = PP_ALIGN.RIGHT
    p_num.text = f"{numero} / {TOTAL_DIAPOSITIVAS}"
    p_num.font.name = "Calibri"
    p_num.font.size = Pt(9.5)
    p_num.font.bold = True
    p_num.font.color.rgb = AZUL_OSCURO

    return slide


def agregar_tarjeta(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    titulo: str,
    lineas: list[str],
    color_acento: RGBColor = AZUL_MEDIO,
    fondo: RGBColor = FONDO_TARJETA,
    tam_titulo: int = 12,
    tam_texto: int = 10,
):
    """Dibuja una tarjeta con barra lateral de color, título breve y viñetas concisas."""
    caja = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    caja.adjustments[0] = 0.03
    caja.fill.solid()
    caja.fill.fore_color.rgb = fondo
    caja.line.color.rgb = BORDE_SUAVE
    caja.line.width = Pt(1.0)

    # Franja superior de color
    franja = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.08)
    )
    franja.fill.solid()
    franja.fill.fore_color.rgb = color_acento
    franja.line.fill.background()

    tb = slide.shapes.add_textbox(
        Inches(x + 0.14), Inches(y + 0.12), Inches(w - 0.28), Inches(h - 0.18)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP

    p0 = tf.paragraphs[0]
    p0.text = titulo
    p0.font.name = "Calibri"
    p0.font.size = Pt(tam_titulo)
    p0.font.bold = True
    p0.font.color.rgb = AZUL_OSCURO
    p0.space_after = Pt(4)

    for linea in lineas:
        p = tf.add_paragraph()
        p.text = linea
        p.font.name = "Calibri"
        p.font.size = Pt(tam_texto)
        p.font.color.rgb = TEXTO_OSCURO
        p.space_after = Pt(3)


def agregar_imagen_en_marco(
    slide,
    nombre_grafico: str,
    x: float,
    y: float,
    w: float,
    h: float,
    pie_texto: str = "",
):
    """Inserta una imagen PNG preservando relación de aspecto dentro de un marco blanco."""
    marco = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    marco.adjustments[0] = 0.02
    marco.fill.solid()
    marco.fill.fore_color.rgb = BLANCO
    marco.line.color.rgb = BORDE_SUAVE
    marco.line.width = Pt(1.0)

    ruta = ruta_grafico(nombre_grafico)
    with Image.open(ruta) as img:
        ancho_px, alto_px = img.size

    espacio_pie = 0.30 if pie_texto else 0.08
    pad_x = 0.10
    pad_top = 0.08
    max_w = w - 2 * pad_x
    max_h = h - pad_top - espacio_pie

    escala = min(max_w / ancho_px, max_h / alto_px)
    draw_w = ancho_px * escala
    draw_h = alto_px * escala
    pos_x = x + (w - draw_w) / 2.0
    pos_y = y + pad_top + (max_h - draw_h) / 2.0

    slide.shapes.add_picture(
        ruta, Inches(pos_x), Inches(pos_y), width=Inches(draw_w), height=Inches(draw_h)
    )

    if pie_texto:
        tb = slide.shapes.add_textbox(
            Inches(x + 0.12), Inches(y + h - 0.30), Inches(w - 0.24), Inches(0.26)
        )
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = pie_texto
        p.font.name = "Calibri"
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = TEXTO_SUAVE


def agregar_tabla_estilizada(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    encabezados: list[str],
    filas: list[list[str]],
    anchos_relativos: list[float] | None = None,
    tam_header: int = 10,
    tam_celda: int = 9,
    col_izquierda: int = 0,
    resaltar_si_col: int | None = None,
):
    """Crea una tabla estilizada con encabezado oscuro, cebra suave y ajuste limpio."""
    n_filas = len(filas) + 1
    n_cols = len(encabezados)
    shape = slide.shapes.add_table(n_filas, n_cols, Inches(x), Inches(y), Inches(w), Inches(h))
    tabla = shape.table
    tabla.first_row = True
    tabla.horz_banding = False

    if anchos_relativos is None:
        anchos_relativos = [1.0] * n_cols
    suma = sum(anchos_relativos)
    for j, frac in enumerate(anchos_relativos):
        tabla.columns[j].width = Inches(w * frac / suma)

    # Encabezados
    for j, texto in enumerate(encabezados):
        celda = tabla.cell(0, j)
        celda.fill.solid()
        celda.fill.fore_color.rgb = AZUL_OSCURO
        celda.vertical_anchor = MSO_ANCHOR.MIDDLE
        celda.margin_left = Inches(0.06)
        celda.margin_right = Inches(0.06)
        celda.margin_top = Inches(0.04)
        celda.margin_bottom = Inches(0.04)
        p = celda.text_frame.paragraphs[0]
        p.text = texto
        p.alignment = PP_ALIGN.LEFT if j == col_izquierda else PP_ALIGN.CENTER
        p.font.name = "Calibri"
        p.font.size = Pt(tam_header)
        p.font.bold = True
        p.font.color.rgb = BLANCO

    # Filas de datos
    for i, fila in enumerate(filas):
        fondo_fila = BLANCO if i % 2 == 0 else FONDO_ZEBRA
        for j, valor in enumerate(fila):
            celda = tabla.cell(i + 1, j)
            celda.vertical_anchor = MSO_ANCHOR.MIDDLE
            celda.margin_left = Inches(0.06)
            celda.margin_right = Inches(0.06)
            celda.margin_top = Inches(0.03)
            celda.margin_bottom = Inches(0.03)

            es_destacado = (
                resaltar_si_col is not None
                and j == resaltar_si_col
                and ("0 errores" in valor.lower() or valor.strip() in {"SÍ", "0"} or "124.5x" in valor or "86.7x" in valor)
            )
            celda.fill.solid()
            celda.fill.fore_color.rgb = VERDE_SUAVE if es_destacado else fondo_fila

            p = celda.text_frame.paragraphs[0]
            p.text = valor
            p.alignment = PP_ALIGN.LEFT if j == col_izquierda else PP_ALIGN.CENTER
            p.font.name = "Calibri"
            p.font.size = Pt(tam_celda)
            p.font.bold = (j == 0) or es_destacado
            p.font.color.rgb = VERDE_TEXTO if es_destacado else TEXTO_OSCURO

    return tabla


# =====================================================================
# CONSTRUCCIÓN DE LAS 19 DIAPOSITIVAS (18 PUNTOS DEL DOCUMENTO MAESTRO)
# =====================================================================

def slide_01_portada(prs: Presentation):
    """1. Portada institucional."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    fondo = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), ANCHO_SLIDE, ALTO_SLIDE
    )
    fondo.fill.solid()
    fondo.fill.fore_color.rgb = AZUL_OSCURO
    fondo.line.fill.background()

    # Franja vertical de acento izquierda
    barra_izq = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.28), ALTO_SLIDE
    )
    barra_izq.fill.solid()
    barra_izq.fill.fore_color.rgb = VERDE_TEAL
    barra_izq.line.fill.background()

    # Acento decorativo superior derecho
    barra_sup = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.28), Inches(0), Inches(13.053), Inches(0.10)
    )
    barra_sup.fill.solid()
    barra_sup.fill.fore_color.rgb = AZUL_MEDIO
    barra_sup.line.fill.background()

    # Etiqueta institucional
    pill = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(0.75), Inches(8.1), Inches(0.46)
    )
    pill.fill.solid()
    pill.fill.fore_color.rgb = AZUL_MEDIO
    pill.line.fill.background()
    p_pill = pill.text_frame.paragraphs[0]
    p_pill.alignment = PP_ALIGN.CENTER
    pill.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_pill.text = "UNIVERSIDAD ALEJANDRO DE HUMBOLDT  ·  ELECTIVA DE COMUNICACIONES MÓVILES (SCM4)"
    p_pill.font.name = "Calibri"
    p_pill.font.size = Pt(10.5)
    p_pill.font.bold = True
    p_pill.font.color.rgb = BLANCO

    # Título principal
    tb = slide.shapes.add_textbox(Inches(0.85), Inches(1.50), Inches(11.6), Inches(2.55))
    tf = tb.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "Evaluación de Modulaciones Digitales, Modelos de Ruido e Interferencias y Técnicas de Mitigación"
    p1.font.name = "Calibri"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = BLANCO
    p1.space_after = Pt(12)

    p2 = tf.add_paragraph()
    p2.text = "Simulación integral en Python de BFSK, BPSK, QPSK y 16-QAM (Nb = 60.000 bits por punto)"
    p2.font.name = "Calibri"
    p2.font.size = Pt(17)
    p2.font.color.rgb = RGBColor(167, 243, 208)

    # Panel de autores y contexto
    panel = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(4.45), Inches(11.6), Inches(1.35)
    )
    panel.adjustments[0] = 0.04
    panel.fill.solid()
    panel.fill.fore_color.rgb = RGBColor(22, 58, 92)
    panel.line.color.rgb = VERDE_TEAL
    panel.line.width = Pt(1.5)

    tf_pan = panel.text_frame
    tf_pan.word_wrap = True
    tf_pan.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf_pan.margin_left = Inches(0.30)

    pa = tf_pan.paragraphs[0]
    pa.text = "Autores:   Jesús Balza  y  Andrea Blazque"
    pa.font.name = "Calibri"
    pa.font.size = Pt(18)
    pa.font.bold = True
    pa.font.color.rgb = BLANCO
    pa.space_after = Pt(6)

    pb = tf_pan.add_paragraph()
    pb.text = "Institución:   Universidad Alejandro de Humboldt   |   Asignatura: Electiva de Comunicaciones Móviles   |   Noviembre 2026"
    pb.font.name = "Calibri"
    pb.font.size = Pt(13)
    pb.font.color.rgb = RGBColor(203, 213, 225)

    # 4 chips resumen al pie
    chips = [
        ("4 Modulaciones", "BFSK · BPSK · QPSK · 16-QAM"),
        ("10 Perturbaciones", "AWGN, ACI, Impulsivo, CDMA, Fase, CW, kTB, CCI, Laplace, Doppler"),
        ("3 Mitigaciones", "Filtrado FIR · Ecualización ZF · Diversidad MRC/SC"),
        ("Validación Monte Carlo", "60.000 bits/punto · IC Wilson 95% · Cota 3/Nb"),
    ]
    for idx, (tit_c, desc_c) in enumerate(chips):
        cx = 0.85 + idx * 2.95
        c_box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx), Inches(5.98), Inches(2.75), Inches(1.05)
        )
        c_box.adjustments[0] = 0.06
        c_box.fill.solid()
        c_box.fill.fore_color.rgb = RGBColor(19, 50, 80)
        c_box.line.color.rgb = AZUL_MEDIO
        c_box.line.width = Pt(1.0)
        tf_c = c_box.text_frame
        tf_c.word_wrap = True
        tf_c.vertical_anchor = MSO_ANCHOR.MIDDLE
        pc1 = tf_c.paragraphs[0]
        pc1.alignment = PP_ALIGN.CENTER
        pc1.text = tit_c
        pc1.font.name = "Calibri"
        pc1.font.size = Pt(12)
        pc1.font.bold = True
        pc1.font.color.rgb = RGBColor(94, 234, 212)
        pc2 = tf_c.add_paragraph()
        pc2.alignment = PP_ALIGN.CENTER
        pc2.text = desc_c
        pc2.font.name = "Calibri"
        pc2.font.size = Pt(8.8)
        pc2.font.color.rgb = RGBColor(226, 232, 240)


def slide_02_objetivo(prs: Presentation):
    """2. Objetivo general y específicos."""
    slide = nueva_diapositiva_base(
        prs, "FASE 1 · CONTEXTO", "Objetivo General y Alcance del Proyecto", 2
    )

    agregar_tarjeta(
        slide, 0.45, 1.05, 12.43, 1.28,
        "Objetivo General",
        [
            "Desarrollar y validar un sistema de simulación digital en Python para analizar el comportamiento, calidad (BER/EVM), "
            "eficiencia espectral y técnicas de mitigación de modulaciones móviles (BFSK, BPSK, QPSK y 16-QAM) bajo diez modelos "
            "realistas de ruido e interferencias."
        ],
        color_acento=AZUL_OSCURO,
        fondo=AZUL_SUAVE,
        tam_titulo=14,
        tam_texto=12,
    )

    objetivos = [
        (
            "1. Cadena Tx/Rx Calibrada",
            [
                "• Síntesis en banda base I/Q y RF pasobanda (fs = 16 kHz, sps = 16).",
                "• Normalización de energía promedio Es = 1.0 para comparación justa.",
            ],
            AZUL_MEDIO,
        ),
        (
            "2. Diez Fenómenos del Canal",
            [
                "• Modelar AWGN, ACI, impulsivo, MUI CDMA, ruido de fase y tono CW.",
                "• Incorporar ruido térmico kTB, cocanal CCI, Laplace y Doppler.",
            ],
            VERDE_TEAL,
        ),
        (
            "3. Evaluación Cuantitativa",
            [
                "• Contrastar curvas BER simuladas (Nb = 60.000) contra expresiones teóricas.",
                "• Medir EVM RMS, eficiencia espectral y complejidad por símbolo.",
            ],
            AMBAR,
        ),
        (
            "4. Mitigación y Recomendación",
            [
                "• Aplicar filtrado FIR (LPF/BPF), ecualización Zero-Forcing y diversidad MRC/SC.",
                "• Definir en qué escenario conviene cada esquema de modulación.",
            ],
            AZUL_OSCURO,
        ),
    ]

    for idx, (tit, bullets, col) in enumerate(objetivos):
        col_i = idx % 2
        row_i = idx // 2
        agregar_tarjeta(
            slide,
            0.45 + col_i * 6.32,
            2.55 + row_i * 2.15,
            6.11,
            1.95,
            tit,
            bullets,
            color_acento=col,
            tam_titulo=13,
            tam_texto=11,
        )


def slide_03_metodologia(prs: Presentation):
    """3. Metodología y entorno."""
    slide = nueva_diapositiva_base(
        prs,
        "METODOLOGÍA",
        "Metodología de Simulación por Fases y Entorno Computacional",
        3,
    )

    # Flujo modular en 4 bloques
    etapas = [
        ("Fases 1–2\nFundamentos", "Entorno Python, generación binaria NRZ y plano complejo I/Q (fundamentos.py).", AZUL_OSCURO),
        ("Fases 3–4\nModulación/Demod.", "BFSK RF no coherente y BPSK/QPSK/16-QAM I/Q con BER=0 ideal (modulaciones.py, demodulacion.py).", AZUL_MEDIO),
        ("Fases 5–6\nCanales y Métricas", "10 modelos de canal, calibración Eb/N0, kTB, Wilson 95% y Nb=60.000 (canales.py, metricas.py).", VERDE_TEAL),
        ("Fases 7–8\nMitigación y Entrega", "FIR LPF/BPF vs ACI, ZF vs ISI y MRC/SC vs Rayleigh con regresión 100% (mitigacion.py).", AMBAR),
    ]

    for i, (tit, desc, col) in enumerate(etapas):
        ex = 0.45 + i * 3.16
        agregar_tarjeta(
            slide, ex, 1.10, 2.95, 2.55, tit, [desc], color_acento=col, tam_titulo=13, tam_texto=10.5
        )
        if i < 3:
            flecha = slide.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW, Inches(ex + 2.97), Inches(2.20), Inches(0.17), Inches(0.32)
            )
            flecha.fill.solid()
            flecha.fill.fore_color.rgb = TEXTO_SUAVE
            flecha.line.fill.background()

    # Tabla inferior izquierda: Entorno y parámetros
    agregar_tabla_estilizada(
        slide,
        0.45, 3.90, 7.35, 2.95,
        ["Componente / Parámetro", "Valor Implementado en el Simulador"],
        [
            ["Lenguaje y Librerías", "Python 3.12 · NumPy · SciPy · Matplotlib · Pandas · python-pptx"],
            ["Frecuencia de Muestreo / Tasa", "fs = 16.000 Hz  |  Rs = 1.000 símbolos/s  |  sps = 16 muestras/símbolo"],
            ["Portadora y Tonos BFSK", "fc = 2.000 Hz (lineales I/Q)  |  BFSK: f0 = 2.000 Hz, f1 = 4.000 Hz"],
            ["Muestreo Monte Carlo", "Nb = 60.000 bits por punto experimental (semillas reproducibles)"],
            ["Estadística de Error", "Intervalo Wilson 95% y cota regla de tres 3/Nb = 5·10⁻⁵ (0 errores)"],
        ],
        anchos_relativos=[0.38, 0.62],
        tam_header=10,
        tam_celda=9.5,
    )

    # Tarjeta derecha: Verificación acumulada
    agregar_tarjeta(
        slide,
        8.00, 3.90, 4.88, 2.95,
        "Rigor y Verificación (Fases 1–7)",
        [
            "• Fases 1–4: BER = 0.0 comprobado en canal sin ruido (9/9 pruebas demodulación).",
            "• Fase 5: 11 pruebas unitarias y 20 escenarios de canal integrados.",
            "• Fase 6: 8 pruebas matemáticas + calibración exacta kTB y CCI independiente.",
            "• Fase 7: 8 pruebas unitarias (FIR, ZF, MRC/SC) + no regresión Fases 1–6 PASS.",
        ],
        color_acento=VERDE_TEAL,
        fondo=VERDE_SUAVE,
        tam_titulo=12.5,
        tam_texto=10,
    )


def slide_04_fundamentos(prs: Presentation):
    """4. Fundamentos de modulación."""
    slide = nueva_diapositiva_base(
        prs,
        "FASE 2 · FUNDAMENTOS",
        "Fundamentos de Señalización I/Q y Síntesis en Radiofrecuencia",
        4,
    )

    agregar_tarjeta(
        slide, 0.45, 1.08, 4.15, 1.80,
        "Representación Compleja I/Q",
        [
            "• Cada símbolo se representa como s[n] = I[n] + j·Q[n].",
            "• Permite modular simultáneamente amplitud |s| y fase θ = atan2(Q, I).",
        ],
        color_acento=AZUL_OSCURO,
        tam_titulo=12.5,
        tam_texto=10.5,
    )
    agregar_tarjeta(
        slide, 0.45, 3.05, 4.15, 1.85,
        "Modulación en Cuadratura RF",
        [
            "• s_RF(t) = I(t)·cos(2πfc·t) − Q(t)·sin(2πfc·t).",
            "• Las portadoras ortogonales separan I y Q en el receptor coherente.",
        ],
        color_acento=AZUL_MEDIO,
        tam_titulo=12.5,
        tam_texto=10.5,
    )
    agregar_tarjeta(
        slide, 0.45, 5.07, 4.15, 1.82,
        "Normalización Energética Justa",
        [
            "• Energía promedio normalizada Es = 1.0 en todas las modulaciones.",
            "• Así Eb = Es / log₂(M), aislando eficiencia de ruido frente a potencia.",
        ],
        color_acento=VERDE_TEAL,
        tam_titulo=12.5,
        tam_texto=10.5,
    )

    agregar_imagen_en_marco(
        slide,
        "fase2_03_representacion_iq.png",
        4.78, 1.08, 8.10, 5.81,
        "Figura 1: Descomposición En Fase (I), Cuadratura (Q), señal RF y plano complejo (Fase 2).",
    )


def slide_05_modulaciones_comparativa(prs: Presentation, filas_tradeoff: list[dict]):
    """5. BFSK, BPSK, QPSK y 16-QAM."""
    slide = nueva_diapositiva_base(
        prs,
        "FASE 3 · ESQUEMAS",
        "Modulaciones Digitales Evaluadas: BFSK, BPSK, QPSK y 16-QAM",
        5,
    )

    f_bfsk = buscar_fila(filas_tradeoff, modulacion="BFSK")
    f_bpsk = buscar_fila(filas_tradeoff, modulacion="BPSK")
    f_qpsk = buscar_fila(filas_tradeoff, modulacion="QPSK")
    f_qam = buscar_fila(filas_tradeoff, modulacion="16-QAM")

    filas_tabla = [
        ["BFSK", f_bfsk["bits_por_simbolo"], "Tonos f0=2 kHz / f1=4 kHz", "1.00", "Ortogonal (Δf=2Rs)", f_bfsk["dominio_receptor"], f_bfsk["complejidad_ops_simbolo"]],
        ["BPSK", f_bpsk["bits_por_simbolo"], "Bit 0→−1, Bit 1→+1 (0°/180°)", "1.00", "2.000", f_bpsk["dominio_receptor"], f_bpsk["complejidad_ops_simbolo"]],
        ["QPSK", f_qpsk["bits_por_simbolo"], "4 fases Gray (±1±j)/√2", "0.50", "√2 ≈ 1.414", f_qpsk["dominio_receptor"], f_qpsk["complejidad_ops_simbolo"]],
        ["16-QAM", f_qam["bits_por_simbolo"], "Grilla 4×4 Gray (norm. 1/√10)", "0.25", "2/√10 ≈ 0.632", f_qam["dominio_receptor"], f_qam["complejidad_ops_simbolo"]],
    ]

    agregar_tabla_estilizada(
        slide,
        0.45, 1.08, 12.43, 2.75,
        ["Modulación", "k [bits/símb.]", "Mapeo / Parámetros Clave", "Eb (Es=1)", "Dist. Mínima d_min", "Dominio Detección", "Ops / Símbolo"],
        filas_tabla,
        anchos_relativos=[0.10, 0.11, 0.25, 0.09, 0.15, 0.19, 0.11],
        tam_header=10.5,
        tam_celda=10,
    )

    tarjetas = [
        ("BFSK (M = 2)", ["Envolvente constante.", "Demodulador de envolvente no coherente en RF: inmune a fase."], AMBAR),
        ("BPSK (M = 2)", ["Máxima d_min = 2.0.", "Referencia de robustez frente a ruido AWGN e interferencia."], VERDE_TEAL),
        ("QPSK (M = 4)", ["Duplica tasa binaria (2 bits/símb.).", "Igual BER por bit que BPSK con mapeo Gray."], AZUL_MEDIO),
        ("16-QAM (M = 16)", ["4 bits/símbolo en el mismo BW.", "Menor d_min (0.632) y PAPR variable: exige alta SNR."], AZUL_OSCURO),
    ]
    for i, (tit, lin, col) in enumerate(tarjetas):
        agregar_tarjeta(
            slide, 0.45 + i * 3.15, 4.10, 2.98, 2.72, tit, lin, color_acento=col, tam_titulo=13, tam_texto=10.5
        )


def slide_06_constelaciones_ondas(prs: Presentation):
    """6. Constelaciones y formas de onda."""
    slide = nueva_diapositiva_base(
        prs,
        "FASE 3 · SEÑALES",
        "Formas de Onda en Tiempo RF y Diagramas de Constelación",
        6,
        subtitulo="Comparación visual directa de frecuencia (BFSK), fase (BPSK/QPSK) y fase+amplitud (16-QAM)",
    )

    imagenes = [
        ("fase3_01_bfsk.png", 0.45, 1.04, "BFSK: conmutación entre tonos ortogonales 2 kHz y 4 kHz"),
        ("fase3_02_bpsk.png", 6.76, 1.04, "BPSK: inversiones de fase de 180° y constelación antipodal"),
        ("fase3_03_qpsk.png", 0.45, 4.02, "QPSK: ramas I/Q independientes y 4 puntos con mapeo Gray"),
        ("fase3_04_16qam.png", 6.76, 4.02, "16-QAM: 4 niveles en I/Q y variación simultánea de amplitud y fase"),
    ]
    for arch, ix, iy, pie in imagenes:
        agregar_imagen_en_marco(slide, arch, ix, iy, 6.12, 2.86, pie)


def slide_07_modelos_ruido_parte1(prs: Presentation):
    """7A. Modelos 1 al 5 de ruido e interferencias."""
    slide = nueva_diapositiva_base(
        prs,
        "FASE 5 · PERTURBACIONES (1/2)",
        "Diez Modelos de Ruido e Interferencias (Modelos 1 al 5)",
        7,
    )

    agregar_tabla_estilizada(
        slide,
        0.45, 1.05, 12.43, 2.40,
        ["#", "Fenómeno Modelado", "Implementación en Simulador", "Efecto Principal Observado"],
        [
            ["1", "Ruido AWGN", "Equivalencia exacta σ_RF² = Nsps·N0/4 en RF y BB", "Dispersión gaussiana radial uniforme en constelación"],
            ["2", "Canal Adyacente (ACI)", "Portadora vecina Δf = 1.500 Hz dentro de fs/2", "Degrada severamente BFSK (RF) y 16-QAM sin filtro"],
            ["3", "Ruido Impulsivo", "Modelo Bernoulli-Gaussiano (p_imp, σ_imp ≫ σ_fondo)", "Picos temporales intensos en ráfagas urbanas/industriales"],
            ["4", "Multiusuario (MUI CDMA)", "DS-CDMA con códigos Walsh-Hadamard H8 (hasta 8 usuarios)", "Pérdida de ortogonalidad ante retardos entre usuarios"],
            ["5", "Ruido de Fase", "Rotación aleatoria φ[n] ~ N(0, σ_θ²) sobre envolvente BB", "Rotación de constelación (crítico en 16-QAM; BFSK inmune)"],
        ],
        anchos_relativos=[0.05, 0.24, 0.36, 0.35],
        tam_header=10,
        tam_celda=9.5,
        col_izquierda=1,
    )

    agregar_imagen_en_marco(
        slide,
        "fase5_01_awgn.png",
        0.45, 3.60, 6.12, 3.32,
        "Modelo #1: Dispersión de constelación 16-QAM bajo AWGN (SNR 20, 10 y 0 dB).",
    )
    agregar_imagen_en_marco(
        slide,
        "fase5_03_ruido_impulsivo.png",
        6.76, 3.60, 6.12, 3.32,
        "Modelo #3: Picos impulsivos de alta energía en el dominio temporal (Bernoulli-Gauss).",
    )


def slide_08_modelos_ruido_parte2(prs: Presentation):
    """7B. Modelos 6 al 10 de ruido e interferencias."""
    slide = nueva_diapositiva_base(
        prs,
        "FASE 5 · PERTURBACIONES (2/2)",
        "Diez Modelos de Ruido e Interferencias (Modelos 6 al 10)",
        8,
    )

    agregar_tabla_estilizada(
        slide,
        0.45, 1.05, 12.43, 2.40,
        ["#", "Fenómeno Modelado", "Implementación en Simulador", "Efecto Principal Observado"],
        [
            ["6", "Interferencia Tonal (CW)", "Sinusoide monocromática acoplada en banda de señal", "Pico espectral discreto y desplazamiento sistemático I/Q"],
            ["7", "Ruido Térmico (kTB)", "Física del receptor: T=290 K, B=2 kHz, NF=4 dB (PN≈−136.96 dBm)", "Define el umbral mínimo de sensibilidad de recepción"],
            ["8", "Interferencia Cocanal (CCI)", "Señal independiente en misma portadora (retardo 7 muestras)", "Colapsa 16-QAM/QPSK si SIR es bajo (BER=0.32 a 0 dB)"],
            ["9", "Ruido Laplace (No Gauss.)", "Distribución exponencial doble con colas pesadas", "Mayor probabilidad de errores extremos frente a gaussiano"],
            ["10", "Efecto Doppler", "Desplazamiento f_D = (v/c)·fc (+300 Hz en pruebas RF/BB)", "Corrimiento espectral y giro continuo de fase en constelación"],
        ],
        anchos_relativos=[0.05, 0.24, 0.36, 0.35],
        tam_header=10,
        tam_celda=9.5,
        col_izquierda=1,
    )

    agregar_imagen_en_marco(
        slide,
        "fase5_09_ruido_laplace_vs_gauss.png",
        0.45, 3.60, 6.12, 3.32,
        "Modelo #9: Histograma de colas pesadas Laplace vs. Gaussiano y constelación QPSK.",
    )
    agregar_imagen_en_marco(
        slide,
        "fase5_10_doppler.png",
        6.76, 3.60, 6.12, 3.32,
        "Modelo #10: Corrimiento espectral FFT y rotación circular en constelación por Doppler.",
    )


def slide_09_resultados_experimentales(prs: Presentation, filas_tradeoff: list[dict]):
    """8. Resultados experimentales globales (Térmico y CCI)."""
    slide = nueva_diapositiva_base(
        prs,
        "FASE 6 · EXPERIMENTOS",
        "Resultados Experimentales: Sensibilidad Térmica (kTB) y Cocanal (CCI)",
        9,
    )

    agregar_imagen_en_marco(
        slide,
        "fase6_03_ber_ruido_termico.png",
        0.45, 1.04, 6.12, 3.92,
        "Experimento 2: Sensibilidad Prx [dBm] frente al piso térmico PN,fis ≈ −136.96 dBm.",
    )
    agregar_imagen_en_marco(
        slide,
        "fase6_04_ber_evm_cci.png",
        6.76, 1.04, 6.12, 3.92,
        "Experimento 3: Degradación BER y EVM RMS frente a interferencia cocanal independiente.",
    )

    filas_res = [
        [
            f["modulacion"],
            f"{f['sensibilidad_prx_dbm']} dBm",
            f"{f['tolerancia_cci_sir_db']} dB",
            f"{f['ebn0_req_ber_1e3_db']} dB",
            f["evaluacion_robustez"],
        ]
        for f in filas_tradeoff
    ]

    agregar_tabla_estilizada(
        slide,
        0.45, 5.10, 12.43, 1.82,
        [
            "Modulación",
            "Sensibilidad Prx (BER ≤ 10⁻³)",
            "SIR CCI Requerido (BER ≤ 10⁻²)",
            "Eb/N0 AWGN (BER ≤ 10⁻³)",
            "Comportamiento Experimental Observado",
        ],
        filas_res,
        anchos_relativos=[0.11, 0.21, 0.21, 0.19, 0.28],
        tam_header=9.5,
        tam_celda=9.5,
    )


def slide_10_comparacion_ber(prs: Presentation, filas_awgn: list[dict]):
    """9. Comparación BER (AWGN vs Teoría + Ruido de Fase y ACI)."""
    slide = nueva_diapositiva_base(
        prs,
        "FASE 6 · MÉTRICA BER",
        "Comparación de Tasa de Error de Bit (BER): Teoría vs. Simulación",
        10,
    )

    agregar_imagen_en_marco(
        slide,
        "fase6_01_ber_awgn_vs_teoria.png",
        0.45, 1.04, 6.12, 4.25,
        "Curvas BER vs Eb/N0 simuladas (Nb = 60.000 bits) superpuestas a cotas analíticas.",
    )
    agregar_imagen_en_marco(
        slide,
        "fase6_05_ruido_fase_y_aci.png",
        6.76, 1.04, 6.12, 4.25,
        "Sensibilidad BER ante ruido de fase (σ_θ = 0°..25°) y canal adyacente ACI (Δf = 1500 Hz).",
    )

    b_bfsk = buscar_fila(filas_awgn, Modulacion="BFSK", EbN0_dB="10.0")
    b_bpsk = buscar_fila(filas_awgn, Modulacion="BPSK", EbN0_dB="10.0")
    b_qpsk = buscar_fila(filas_awgn, Modulacion="QPSK", EbN0_dB="10.0")
    b_qam = buscar_fila(filas_awgn, Modulacion="16-QAM", EbN0_dB="10.0")

    err_bpsk = int(b_bpsk["Errores"])
    err_qpsk = int(b_qpsk["Errores"])
    etiq_qpsk = "1 error" if err_qpsk == 1 else f"{err_qpsk} errores"

    tarjetas = [
        (
            "Validación AWGN (Eb/N0 = 10 dB)",
            [
                f"BPSK: {err_bpsk} errores (cota {b_bpsk['Cota_Tres_95']})  |  QPSK: {etiq_qpsk}",
                f"16-QAM: BER={float(b_qam['BER_Simulada']):.4e} (Teór. {float(b_qam['BER_Teorica']):.4e})",
                f"BFSK: BER={float(b_bfsk['BER_Simulada']):.4e} (Teór. {float(b_bfsk['BER_Teorica']):.4e})",
            ],
            AZUL_OSCURO,
        ),
        (
            "Impacto del Ruido de Fase",
            [
                "• BFSK es inmune (detección no coherente por envolvente RF).",
                "• BPSK/QPSK mantienen BER ≤ 10⁻⁴ hasta σ_θ = 25° a 12 dB.",
                "• 16-QAM supera BER = 10⁻³ desde σ_θ ≈ 15° por menor margen de fase.",
            ],
            VERDE_TEAL,
        ),
        (
            "Impacto ACI sin Filtrado",
            [
                "• BFSK sufre mayor degradación espectral en RF (BER ≈ 0.068 a SIR=0 dB).",
                "• 16-QAM alcanza BER ≈ 2.28e-3 a SIR=0 dB por proximidad entre símbolos.",
                "• Justifica el filtrado digital FIR (LPF/BPF) en Fase 7.",
            ],
            AMBAR,
        ),
    ]
    for i, (tit, lin, col) in enumerate(tarjetas):
        agregar_tarjeta(
            slide, 0.45 + i * 4.18, 5.42, 4.06, 1.54, tit, lin, color_acento=col, tam_titulo=11, tam_texto=9.2
        )


def slide_11_comparacion_evm(prs: Presentation, filas_awgn: list[dict]):
    """10. Comparación EVM."""
    slide = nueva_diapositiva_base(
        prs,
        "FASE 6 · MÉTRICA EVM",
        "Evaluación de Calidad de Constelación mediante EVM RMS (%)",
        11,
    )

    agregar_imagen_en_marco(
        slide,
        "fase6_02_evm_vs_ebn0.png",
        0.45, 1.06, 7.25, 5.82,
        "EVM RMS (%) medido a nivel de símbolo de decisión frente a Eb/N0 (BFSK reportado como N/A).",
    )

    puntos_ebn0 = ["0.0", "4.0", "8.0", "10.0", "12.0"]
    filas_evm = []
    for eb in puntos_ebn0:
        f_bpsk = buscar_fila(filas_awgn, Modulacion="BPSK", EbN0_dB=eb)
        f_qpsk = buscar_fila(filas_awgn, Modulacion="QPSK", EbN0_dB=eb)
        f_qam = buscar_fila(filas_awgn, Modulacion="16-QAM", EbN0_dB=eb)
        filas_evm.append([f"{eb} dB", "N/A", f_bpsk["EVM_RMS"], f_qpsk["EVM_RMS"], f_qam["EVM_RMS"]])

    agregar_tabla_estilizada(
        slide,
        7.90, 1.06, 4.98, 2.95,
        ["Eb/N0", "BFSK", "BPSK", "QPSK", "16-QAM"],
        filas_evm,
        anchos_relativos=[0.20, 0.14, 0.22, 0.22, 0.22],
        tam_header=10,
        tam_celda=9.5,
    )

    agregar_tarjeta(
        slide,
        7.90, 4.20, 4.98, 2.68,
        "Interpretación Técnica del EVM",
        [
            "• A igual Eb/N0, 16-QAM tiene Es/N0 = 4·Eb/N0 (+6.02 dB sobre BPSK), "
            "por ello su EVM porcentual respecto a √Es es menor (~15.8% vs ~31.6% a 10 dB).",
            "• Sin embargo, al tener 16 puntos más juntos (d_min = 0.632), su tolerancia máxima "
            "de EVM antes de fallar es mucho menor que en BPSK.",
            "• BFSK se excluye formalmente (N/A) al no decidir sobre plano complejo I/Q.",
        ],
        color_acento=AMBAR,
        fondo=AMBAR_SUAVE,
        tam_titulo=12,
        tam_texto=9.8,
    )


def slide_12_eficiencia_espectral(prs: Presentation, filas_tradeoff: list[dict]):
    """11. Eficiencia espectral."""
    slide = nueva_diapositiva_base(
        prs,
        "FASE 6 · ESPECTRO",
        "Eficiencia Espectral: Convención de Lóbulo Principal RF vs. Nyquist",
        12,
    )

    # Gráfico nativo editable en PowerPoint
    marco_chart = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(1.06), Inches(7.15), Inches(5.82)
    )
    marco_chart.adjustments[0] = 0.02
    marco_chart.fill.solid()
    marco_chart.fill.fore_color.rgb = BLANCO
    marco_chart.line.color.rgb = BORDE_SUAVE
    marco_chart.line.width = Pt(1.0)

    mods = ["BFSK", "BPSK", "QPSK", "16-QAM"]
    chart_data = CategoryChartData()
    chart_data.categories = mods
    eta_rf = [float(buscar_fila(filas_tradeoff, modulacion=m)["eficiencia_principal_bps_hz"]) for m in mods]
    eta_nyquist = [0.25, 1.00, 2.00, 4.00]  # Carson BFSK = 0.25; Nyquist ideal lineales = log2(M)
    chart_data.add_series("Lóbulo Principal RF / Carson [bps/Hz]", eta_rf)
    chart_data.add_series("Límite Ideal Nyquist [bps/Hz]", eta_nyquist)

    chart_shape = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.65), Inches(1.22), Inches(6.75), Inches(5.45), chart_data
    )
    chart = chart_shape.chart
    chart.has_title = True
    chart.chart_title.text_frame.text = "Eficiencia Espectral Comparativa por Modulación [bps/Hz]"
    chart.chart_title.text_frame.paragraphs[0].font.size = Pt(13)
    chart.chart_title.text_frame.paragraphs[0].font.bold = True
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.font.size = Pt(10)

    colores_serie = [AZUL_MEDIO, VERDE_TEAL]
    plot = chart.plots[0]
    plot.has_data_labels = True
    plot.data_labels.font.size = Pt(9.5)
    plot.data_labels.font.bold = True
    plot.data_labels.number_format = "0.00"
    for idx, serie in enumerate(plot.series):
        serie.format.fill.solid()
        serie.format.fill.fore_color.rgb = colores_serie[idx]

    chart.value_axis.maximum_scale = 4.5
    chart.value_axis.minimum_scale = 0.0
    chart.value_axis.tick_labels.font.size = Pt(9.5)
    chart.category_axis.tick_labels.font.size = Pt(10.5)
    chart.category_axis.tick_labels.font.bold = True

    # Tabla derecha + síntesis
    filas_ef = [
        ["BFSK", "0.25", "0.25 (Carson)", f"{float(buscar_fila(filas_tradeoff, modulacion='BFSK')['ancho_banda_hz']):.0f} Hz"],
        ["BPSK", "0.50", "1.00", f"{float(buscar_fila(filas_tradeoff, modulacion='BPSK')['ancho_banda_hz']):.0f} Hz"],
        ["QPSK", "1.00", "2.00", f"{float(buscar_fila(filas_tradeoff, modulacion='QPSK')['ancho_banda_hz']):.0f} Hz"],
        ["16-QAM", "2.00", "4.00", f"{float(buscar_fila(filas_tradeoff, modulacion='16-QAM')['ancho_banda_hz']):.0f} Hz"],
    ]
    agregar_tabla_estilizada(
        slide,
        7.80, 1.06, 5.08, 2.65,
        ["Esquema", "η RF [bps/Hz]", "η Nyquist", "BW (Rs=1 kHz)"],
        filas_ef,
        anchos_relativos=[0.22, 0.26, 0.24, 0.28],
        tam_header=10,
        tam_celda=9.5,
    )

    agregar_tarjeta(
        slide,
        7.80, 3.95, 5.08, 2.93,
        "Hallazgos Clave de Espectro",
        [
            "• 16-QAM alcanza 2.00 bps/Hz en lóbulo principal RF (8× más eficiente que BFSK y 4× que BPSK).",
            "• QPSK duplica la eficiencia frente a BPSK (1.00 vs 0.50 bps/Hz) sin degradar la BER teórica por bit.",
            "• BFSK exige mayor ancho de banda (B ≈ Δf + 2Rs = 4.000 Hz) para garantizar ortogonalidad no coherente.",
        ],
        color_acento=AZUL_OSCURO,
        tam_titulo=12.5,
        tam_texto=10,
    )


def slide_13_complejidad_tradeoff(prs: Presentation, filas_tradeoff: list[dict]):
    """12. Complejidad y compromisos (Trade-off)."""
    slide = nueva_diapositiva_base(
        prs,
        "FASE 6 · TRADE-OFF",
        "Complejidad Operacional y Matriz Global de Compromisos",
        13,
    )

    filas_t = [
        [
            f["modulacion"],
            f["eficiencia_principal_bps_hz"],
            f["ebn0_req_ber_1e3_db"],
            f["sensibilidad_prx_dbm"],
            f["tolerancia_cci_sir_db"],
            f["complejidad_ops_simbolo"],
            f["complejidad_ops_bit"],
        ]
        for f in filas_tradeoff
    ]

    agregar_tabla_estilizada(
        slide,
        0.45, 1.04, 12.43, 2.05,
        [
            "Modulación",
            "η RF [bps/Hz]",
            "Eb/N0 (10⁻³) [dB]",
            "Sensib. Prx [dBm]",
            "SIR CCI (10⁻²) [dB]",
            "Ops / Símbolo",
            "Ops / Bit",
        ],
        filas_t,
        anchos_relativos=[0.13, 0.14, 0.16, 0.16, 0.16, 0.13, 0.12],
        tam_header=9.5,
        tam_celda=9.5,
    )

    # Gráfico nativo normalizado 0-10 (sin recortes de barras)
    marco = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(3.26), Inches(7.45), Inches(3.65)
    )
    marco.adjustments[0] = 0.02
    marco.fill.solid()
    marco.fill.fore_color.rgb = BLANCO
    marco.line.color.rgb = BORDE_SUAVE
    marco.line.width = Pt(1.0)

    chart_data = CategoryChartData()
    mods = ["BFSK", "BPSK", "QPSK", "16-QAM"]
    chart_data.categories = mods
    ef_norm = [round(float(buscar_fila(filas_tradeoff, modulacion=m)["eficiencia_principal_bps_hz"]) * 5.0, 2) for m in mods]
    inm_ruido = [round(max(0.0, 14.0 - float(buscar_fila(filas_tradeoff, modulacion=m)["ebn0_req_ber_1e3_db"])), 2) for m in mods]
    tol_cci = [round(max(0.0, (20.0 - float(buscar_fila(filas_tradeoff, modulacion=m)["tolerancia_cci_sir_db"])) / 2.0), 2) for m in mods]

    chart_data.add_series("Eficiencia Espectral (η × 5)", ef_norm)
    chart_data.add_series("Inmunidad AWGN (14 − Eb/N0)", inm_ruido)
    chart_data.add_series("Tolerancia CCI ((20 − SIR) / 2)", tol_cci)

    chart_shape = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.62), Inches(3.36), Inches(7.10), Inches(3.45), chart_data
    )
    chart = chart_shape.chart
    chart.has_title = True
    chart.chart_title.text_frame.text = "Perfil Relativo Normalizado [Escala 0 – 10, mayor es mejor]"
    chart.chart_title.text_frame.paragraphs[0].font.size = Pt(11.5)
    chart.chart_title.text_frame.paragraphs[0].font.bold = True
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.font.size = Pt(9)

    colores = [AZUL_MEDIO, VERDE_TEAL, AMBAR]
    plot = chart.plots[0]
    plot.has_data_labels = True
    plot.data_labels.font.size = Pt(8.5)
    plot.data_labels.number_format = "0.0"
    for idx, serie in enumerate(plot.series):
        serie.format.fill.solid()
        serie.format.fill.fore_color.rgb = colores[idx]

    chart.value_axis.maximum_scale = 11.5
    chart.value_axis.minimum_scale = 0.0
    chart.value_axis.tick_labels.font.size = Pt(8.5)
    chart.category_axis.tick_labels.font.size = Pt(10)
    chart.category_axis.tick_labels.font.bold = True

    agregar_tarjeta(
        slide,
        8.10, 3.26, 4.78, 3.65,
        "Lectura del Compromiso",
        [
            "• Complejidad en decisión: BFSK ejecuta 129 ops/símbolo (correlación de envolvente sobre 16 muestras RF), frente a 1–8 ops en I/Q.",
            "• QPSK logra el mejor balance global: eficiencia 1.0 bps/Hz con misma inmunidad AWGN que BPSK (7 dB).",
            "• 16-QAM maximiza espectro (10/10), pero sacrifica inmunidad a ruido e interferencia cocanal (SIR ≥ 16 dB).",
        ],
        color_acento=AZUL_OSCURO,
        tam_titulo=12.5,
        tam_texto=9.8,
    )


def slide_14_filtrado_fir(prs: Presentation, filas_fir: list[dict]):
    """13. Mitigación mediante Filtrado Digital FIR."""
    slide = nueva_diapositiva_base(
        prs,
        "FASE 7 · MITIGACIÓN 1",
        "Filtrado Digital FIR frente a Interferencia de Canal Adyacente (ACI)",
        14,
    )

    agregar_imagen_en_marco(
        slide,
        "fase7_01_filtrado_antes_despues.png",
        0.45, 1.05, 7.65, 5.85,
        "Experimento 1: BER (izq.) y EVM RMS (der.) antes y después de filtros FIR LPF/BPF.",
    )

    agregar_tarjeta(
        slide,
        8.28, 1.05, 4.60, 2.35,
        "Diseño FIR Implementado (Hamming)",
        [
            "• BPSK/QPSK/16-QAM: LPF Orden 48, fc = 1000 Hz, fs = 16 kHz (Aten. 40.63 dB a Δf = 1500 Hz; ISI propia 0.84% en [2:14]).",
            "• BFSK RF: BPF Orden 72, banda 1800–4200 Hz, centro 3000 Hz (Aten. 33.65 dB a 4500 Hz).",
            "• Compensación exacta de retardo de grupo.",
        ],
        color_acento=AZUL_MEDIO,
        tam_titulo=12,
        tam_texto=9.5,
    )

    r_bfsk_0 = buscar_fila(filas_fir, Modulacion="BFSK", SIR_ACI_dB="0.0")
    r_bfsk_4 = buscar_fila(filas_fir, Modulacion="BFSK", SIR_ACI_dB="4.0")
    r_qam_0 = buscar_fila(filas_fir, Modulacion="16-QAM", SIR_ACI_dB="0.0")
    r_qpsk_0 = buscar_fila(filas_fir, Modulacion="QPSK", SIR_ACI_dB="0.0")

    filas_t = [
        ["BFSK (0 dB)", f"{float(r_bfsk_0['BER_Sin_Filtro']):.2e}", f"{r_bfsk_0['Errores_Con']} errores", "N/A → N/A"],
        ["BFSK (4 dB)", f"{float(r_bfsk_4['BER_Sin_Filtro']):.2e}", f"{r_bfsk_4['Errores_Con']} errores", "N/A → N/A"],
        ["QPSK (0 dB)", "0 errores", "0 errores", f"{r_qpsk_0['EVM_Sin_Filtro']} → {r_qpsk_0['EVM_Con_Filtro']}"],
        ["16-QAM (0 dB)", f"{float(r_qam_0['BER_Sin_Filtro']):.2e}", f"{r_qam_0['Errores_Con']} errores", f"{r_qam_0['EVM_Sin_Filtro']} → {r_qam_0['EVM_Con_Filtro']}"],
    ]

    agregar_tabla_estilizada(
        slide,
        8.28, 3.55, 4.60, 2.30,
        ["Escenario (SIR)", "Sin Filtro", "Con FIR", "EVM Sin → Con"],
        filas_t,
        anchos_relativos=[0.27, 0.23, 0.22, 0.28],
        tam_header=9.5,
        tam_celda=8.8,
        resaltar_si_col=2,
    )

    agregar_tarjeta(
        slide,
        8.28, 5.98, 4.60, 0.92,
        "Resultado Clave",
        ["Supresión total de errores ACI en todo el barrido 0–16 dB para las 4 modulaciones."],
        color_acento=VERDE_TEAL,
        fondo=VERDE_SUAVE,
        tam_titulo=10.5,
        tam_texto=9.2,
    )


def slide_15_ecualizacion_zf(prs: Presentation, filas_zf: list[dict]):
    """14. Mitigación mediante Ecualización Zero-Forcing (ZF)."""
    slide = nueva_diapositiva_base(
        prs,
        "FASE 7 · MITIGACIÓN 2",
        "Ecualización Zero-Forcing (ZF) frente a Canal Dispersivo con ISI",
        15,
    )

    agregar_imagen_en_marco(
        slide,
        "fase7_02_ecualizacion_antes_despues.png",
        0.45, 1.05, 7.65, 5.85,
        "Experimento 2: Eliminación del piso de error ISI (izq.) y respuesta combinada c = h * w (der.).",
    )

    agregar_tarjeta(
        slide,
        8.28, 1.05, 4.60, 2.30,
        "Parámetros y Costo del Ecualizador",
        [
            "• Canal dispersivo: r[n] = s[n] + 0.35·s[n−1] + w[n].",
            "• Ecualizador FIR: Nw = 7 taps, retardo Δ = 0.",
            "• ISI residual medida: −64.40 dB (tap principal c[0] ≈ 1.00).",
            "• Amplificación de ruido (NEF = ||w||²): +0.568 dB (1.1396×).",
        ],
        color_acento=AZUL_OSCURO,
        tam_titulo=12,
        tam_texto=9.5,
    )

    q_10 = buscar_fila(filas_zf, Modulacion="16-QAM", EbN0_dB="10.0")
    q_12 = buscar_fila(filas_zf, Modulacion="16-QAM", EbN0_dB="12.0")
    q_14 = buscar_fila(filas_zf, Modulacion="16-QAM", EbN0_dB="14.0")
    qp_10 = buscar_fila(filas_zf, Modulacion="QPSK", EbN0_dB="10.0")

    filas_t = [
        ["QPSK (10 dB)", f"{float(qp_10['BER_Sin_Ecualizar']):.2e}", f"{qp_10['Errores_Con']} errores", f"{qp_10['EVM_Sin_Ecualizar']} → {qp_10['EVM_Con_ZF']}"],
        ["16-QAM (10 dB)", f"{float(q_10['BER_Sin_Ecualizar']):.2e}", f"{float(q_10['BER_Con_ZF']):.2e}", f"{q_10['EVM_Sin_Ecualizar']} → {q_10['EVM_Con_ZF']}"],
        ["16-QAM (12 dB)", f"{float(q_12['BER_Sin_Ecualizar']):.2e}", f"{float(q_12['BER_Con_ZF']):.2e}", f"{q_12['EVM_Sin_Ecualizar']} → {q_12['EVM_Con_ZF']}"],
        ["16-QAM (14 dB)", f"{float(q_14['BER_Sin_Ecualizar']):.2e}", f"{q_14['Errores_Con']} errores", f"{q_14['EVM_Sin_Ecualizar']} → {q_14['EVM_Con_ZF']}"],
    ]

    agregar_tabla_estilizada(
        slide,
        8.28, 3.50, 4.60, 2.30,
        ["Mod. (Eb/N0)", "Sin ZF (ISI)", "Con ZF", "EVM Sin → Con"],
        filas_t,
        anchos_relativos=[0.27, 0.23, 0.22, 0.28],
        tam_header=9.5,
        tam_celda=8.8,
        resaltar_si_col=2,
    )

    agregar_tarjeta(
        slide,
        8.28, 5.95, 4.60, 0.95,
        "Compromiso del ZF",
        ["Cancela el piso ISI (16-QAM cae de 6751 a 0 errores a 14 dB), con leve penalización NEF (+0.57 dB)."],
        color_acento=AMBAR,
        fondo=AMBAR_SUAVE,
        tam_titulo=10.5,
        tam_texto=9.0,
    )


def slide_16_diversidad_mrc_sc(prs: Presentation, filas_div: list[dict]):
    """15. Mitigación mediante Diversidad MRC y SC."""
    slide = nueva_diapositiva_base(
        prs,
        "FASE 7 · MITIGACIÓN 3",
        "Diversidad de Recepción en Canal Rayleigh Plano (SISO vs. SC vs. MRC)",
        16,
    )

    agregar_imagen_en_marco(
        slide,
        "fase7_03_diversidad_mrc_vs_siso.png",
        0.45, 1.05, 7.50, 5.85,
        "Experimento 3: Ganancia de diversidad L=2 frente a SISO L=1 bajo igual potencia transmitida (Es=1).",
    )

    agregar_tarjeta(
        slide,
        8.13, 1.05, 4.75, 2.15,
        "Arquitectura SIMO 1×2 Equivalente",
        [
            "• Dos ramas Rayleigh independientes y ruidos incorrelados.",
            "• SC (L=2): selecciona instantáneamente la rama de mayor |h_i|².",
            "• MRC (L=2): combina coherentemente y = (h₁*r₁ + h₂*r₂) / (|h₁|² + |h₂|²).",
        ],
        color_acento=VERDE_TEAL,
        tam_titulo=12,
        tam_texto=9.5,
    )

    puntos = [("BPSK", "10.0"), ("BPSK", "14.0"), ("BPSK", "18.0"), ("QPSK", "10.0"), ("QPSK", "18.0")]
    filas_t = []
    for mod, eb in puntos:
        r = buscar_fila(filas_div, Modulacion=mod, EbN0_Medio_dB=eb)
        filas_t.append([
            f"{mod} ({eb} dB)",
            f"{float(r['BER_SISO_L1']):.2e}",
            f"{float(r['BER_SC_L2']):.2e}",
            f"{float(r['BER_MRC_L2']):.2e}",
            r["Ganancia_MRC_vs_SISO"],
        ])

    agregar_tabla_estilizada(
        slide,
        8.13, 3.35, 4.75, 2.48,
        ["Mod. (SNR)", "SISO", "SC (L=2)", "MRC (L=2)", "Reducción"],
        filas_t,
        anchos_relativos=[0.24, 0.20, 0.20, 0.20, 0.16],
        tam_header=9.2,
        tam_celda=8.8,
        resaltar_si_col=4,
    )

    agregar_tarjeta(
        slide,
        8.13, 5.95, 4.75, 0.95,
        "Ganancia Demostrada",
        ["La diversidad de orden 2 reduce errores 14.5× a 10 dB y hasta 124.5× en QPSK a 18 dB."],
        color_acento=AZUL_OSCURO,
        fondo=AZUL_SUAVE,
        tam_titulo=10.5,
        tam_texto=9.2,
    )


def slide_17_resumen_antes_despues(prs: Presentation, filas_resumen: list[dict]):
    """16. Resultados consolidados antes y después de mitigación."""
    slide = nueva_diapositiva_base(
        prs,
        "FASE 7 · IMPACTO GLOBAL",
        "Consolidado de Desempeño: Antes vs. Después de la Mitigación",
        17,
    )

    filas_t = [
        [
            f["Tecnica"],
            f["Escenario_Canal"],
            f["Configuracion_Tecnica"],
            f["Desempeño_Sin_Mitigacion"],
            f["Desempeño_Con_Mitigacion"],
            f["Beneficio_Demostrado"],
        ]
        for f in filas_resumen
    ]

    agregar_tabla_estilizada(
        slide,
        0.45, 1.10, 12.43, 3.45,
        [
            "Técnica",
            "Escenario de Canal",
            "Configuración",
            "Sin Mitigación (Antes)",
            "Con Mitigación (Después)",
            "Beneficio Demostrado",
        ],
        filas_t,
        anchos_relativos=[0.15, 0.17, 0.17, 0.18, 0.19, 0.14],
        tam_header=10,
        tam_celda=9.2,
    )

    tarjetas_kpi = [
        (
            "1. Filtrado FIR (LPF + BPF)",
            [
                "• Atenuación analítica: 40.6 dB (LPF) y 33.7 dB (BPF).",
                "• BFSK (SIR=0 dB): 4.062 → 0 errores.",
                "• 16-QAM (SIR=0 dB): 137 → 0 errores; EVM 23.1% → 9.7%.",
            ],
            AZUL_MEDIO,
        ),
        (
            "2. Ecualización Zero-Forcing",
            [
                "• ISI residual combinada: −64.40 dB (7 taps).",
                "• 16-QAM (14 dB): 6.751 → 0 errores (elimina piso).",
                "• Penalización controlada de ruido: +0.57 dB.",
            ],
            AMBAR,
        ),
        (
            "3. Diversidad MRC (SIMO 1×2)",
            [
                "• Cambia pendiente de BER de orden 1 a orden 2.",
                "• BPSK (10 dB): BER 0.0237 → 1.63e-3 (14.5× menos).",
                "• A 18 dB: mejora de 86.7× (BPSK) y 124.5× (QPSK).",
            ],
            VERDE_TEAL,
        ),
    ]
    for i, (tit, lin, col) in enumerate(tarjetas_kpi):
        agregar_tarjeta(
            slide, 0.45 + i * 4.18, 4.80, 4.06, 2.08, tit, lin, color_acento=col, tam_titulo=12, tam_texto=10
        )


def slide_18_conclusiones(prs: Presentation):
    """17. Conclusiones (escenarios recomendados y trade-off calidad-eficiencia-complejidad)."""
    slide = nueva_diapositiva_base(
        prs,
        "CONCLUSIONES",
        "Escenarios Recomendados y Relación Calidad – Eficiencia – Complejidad",
        18,
    )

    agregar_tabla_estilizada(
        slide,
        0.45, 1.08, 12.43, 3.25,
        ["Modulación", "Escenario Móvil / Canal Recomendado", "Ventaja Decisiva", "Limitación Principal a Vigilar"],
        [
            [
                "BFSK",
                "Telemetría IoT, sensores remotos y osciladores de bajo costo con deriva o ruido de fase.",
                "Detección no coherente inmune a ruido de fase; envolvente constante.",
                "Menor eficiencia espectral (0.25 bps/Hz) y sensible a ACI sin filtro BPF.",
            ],
            [
                "BPSK",
                "Enlace satelital/móvil de largo alcance, baja potencia o fuerte interferencia cocanal.",
                "Máxima inmunidad AWGN (7 dB para 10⁻³), sensibilidad −133 dBm y SIR CCI 0 dB.",
                "Tasa binaria baja (0.50 bps/Hz en RF); requiere sincronización coherente.",
            ],
            [
                "QPSK",
                "Estándar general en comunicaciones móviles celulares y enlaces troncales equilibrados.",
                "Duplica caudal vs BPSK (1.00 bps/Hz) con idéntica BER teórica en AWGN.",
                "Requiere SIR ≥ 6 dB en cocanal y recuperación de fase coherente.",
            ],
            [
                "16-QAM",
                "Zonas urbanas con alta SNR, corta distancia y receptor equipado con filtros FIR y ecualización.",
                "Máxima eficiencia espectral evaluada (2.00 bps/Hz en RF, 4 bits/símbolo).",
                "Alta vulnerabilidad a CCI (SIR ≥ 16 dB), ISI y ruido de fase (σ_θ ≥ 15°).",
            ],
        ],
        anchos_relativos=[0.11, 0.35, 0.27, 0.27],
        tam_header=10,
        tam_celda=9.5,
    )

    reflexiones = [
        (
            "Calidad vs. Eficiencia Espectral",
            [
                "Incrementar orden M eleva la densidad de datos de 0.25 a 2.00 bps/Hz RF, pero reduce d_min (2.0 → 0.632) "
                "y exige +4 dB extra de Eb/N0 y +16 dB de margen CCI.",
            ],
            AZUL_OSCURO,
        ),
        (
            "Complejidad vs. Desempeño",
            [
                "La ganancia real en canales hostiles se consigue trasladando complejidad al DSP receptor: filtros FIR suprimen ACI "
                "y ecualizadores ZF rompen pisos de error ISI a costo de pocos taps.",
            ],
            VERDE_TEAL,
        ),
        (
            "Mitigación Física y Adaptación",
            [
                "En desvanecimiento Rayleigh, duplicar antenas receptoras (MRC 1×2) logra ganancias de 14.5× a 124.5× sin elevar "
                "potencia TX, complementando la modulación adaptativa.",
            ],
            AMBAR,
        ),
    ]
    for i, (tit, lin, col) in enumerate(reflexiones):
        agregar_tarjeta(
            slide, 0.45 + i * 4.18, 4.55, 4.06, 2.32, tit, lin, color_acento=col, tam_titulo=12, tam_texto=10
        )


def slide_19_referencias(prs: Presentation):
    """18. Referencias técnicas y cierre."""
    slide = nueva_diapositiva_base(
        prs,
        "REFERENCIAS",
        "Referencias Técnicas y Trazabilidad del Sistema",
        19,
    )

    agregar_tarjeta(
        slide,
        0.45, 1.12, 7.60, 5.72,
        "Bibliografía y Estándares de Referencia",
        [
            "• Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th ed.). McGraw-Hill. (Cotas BER, canal ISI, ecualización ZF y diversidad MRC).",
            "• Goldsmith, A. (2005). Wireless Communications. Cambridge University Press. (Desvanecimiento Rayleigh, capacidad y compromiso tasa-potencia).",
            "• Sklar, B. (2001). Digital Communications: Fundamentals and Applications (2nd ed.). Prentice Hall. (Constelaciones Gray, FSK no coherente y regla de Carson).",
            "• Rappaport, T. S. (2002). Wireless Communications: Principles and Practice (2nd ed.). Prentice Hall. (Interferencia cocanal, canal adyacente y ruido térmico kTB).",
            "• Documentación oficial del ecosistema científico Python: NumPy, SciPy (scipy.signal.firwin, linalg.toeplitz), Matplotlib, Pandas y python-pptx.",
        ],
        color_acento=AZUL_OSCURO,
        tam_titulo=13.5,
        tam_texto=10.8,
    )

    agregar_tarjeta(
        slide,
        8.25, 1.12, 4.63, 3.35,
        "Trazabilidad de Resultados",
        [
            "• Todos los valores numéricos y curvas proceden de simulaciones Monte Carlo reproducibles (Nb = 60.000 bits).",
            "• Tablas fuente en resultados/tablas/ (.csv y .md) y gráficos en resultados/graficos/ (.png).",
            "• Suites de validación y no regresión Fases 1–7 ejecutadas con resultado 100% PASS.",
        ],
        color_acento=VERDE_TEAL,
        fondo=VERDE_SUAVE,
        tam_titulo=13,
        tam_texto=10.5,
    )

    agregar_tarjeta(
        slide,
        8.25, 4.68, 4.63, 2.16,
        "Créditos del Proyecto",
        [
            "Universidad Alejandro de Humboldt",
            "Electiva de Comunicaciones Móviles (SCM4)",
            "Jesús Balza  ·  Andrea Blazque",
            "¡Gracias por su atención!",
        ],
        color_acento=AZUL_MEDIO,
        fondo=AZUL_SUAVE,
        tam_titulo=13,
        tam_texto=11,
    )


# =====================================================================
# FUNCIÓN PRINCIPAL
# =====================================================================

def generar_presentacion() -> str:
    """Construye la presentación PowerPoint completa de la Fase 8."""
    filas_tradeoff = leer_csv("fase6_resumen_tradeoff.csv")
    filas_awgn = leer_csv("fase6_experimento1_awgn.csv")
    filas_fir = leer_csv("fase7_experimento1_filtrado.csv")
    filas_zf = leer_csv("fase7_experimento2_ecualizacion.csv")
    filas_div = leer_csv("fase7_experimento3_diversidad.csv")
    filas_resumen_mit = leer_csv("fase7_resumen_mitigacion.csv")

    prs = Presentation()
    prs.slide_width = ANCHO_SLIDE
    prs.slide_height = ALTO_SLIDE

    slide_01_portada(prs)
    slide_02_objetivo(prs)
    slide_03_metodologia(prs)
    slide_04_fundamentos(prs)
    slide_05_modulaciones_comparativa(prs, filas_tradeoff)
    slide_06_constelaciones_ondas(prs)
    slide_07_modelos_ruido_parte1(prs)
    slide_08_modelos_ruido_parte2(prs)
    slide_09_resultados_experimentales(prs, filas_tradeoff)
    slide_10_comparacion_ber(prs, filas_awgn)
    slide_11_comparacion_evm(prs, filas_awgn)
    slide_12_eficiencia_espectral(prs, filas_tradeoff)
    slide_13_complejidad_tradeoff(prs, filas_tradeoff)
    slide_14_filtrado_fir(prs, filas_fir)
    slide_15_ecualizacion_zf(prs, filas_zf)
    slide_16_diversidad_mrc_sc(prs, filas_div)
    slide_17_resumen_antes_despues(prs, filas_resumen_mit)
    slide_18_conclusiones(prs)
    slide_19_referencias(prs)

    prs.save(ARCHIVO_SALIDA)
    return ARCHIVO_SALIDA


def main():
    ruta = generar_presentacion()
    tam_mb = os.path.getsize(ruta) / (1024.0 * 1024.0)
    print(f"[OK] Presentación generada: {ruta}")
    print(f"     Diapositivas: {TOTAL_DIAPOSITIVAS} | Tamaño: {tam_mb:.2f} MB")


if __name__ == "__main__":
    main()
