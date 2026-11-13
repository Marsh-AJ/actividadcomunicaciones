"""
verificar_fase1.py
Script de verificación para la FASE 1: Preparación del Entorno.
Comprueba versión de Python, entorno virtual, importación de librerías y estructura de directorios.
"""

import os
import sys
import platform

# Configurar directorio de caché local para Matplotlib si no tiene permisos en AppData
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", cache_dir)

def verificar_entorno():
    print("=" * 65)
    print("   SIMULADOR DE MODULACIONES DIGITALES - VERIFICACIÓN FASE 1")
    print("=" * 65)

    errores = []

    # 1. Versión de Python y SO
    print("\n[1] Sistema Operativo y Versión de Python:")
    print(f"    - Sistema Operativo: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"    - Versión de Python: {platform.python_version()}")
    print(f"    - Ejecutable: {sys.executable}")

    if sys.version_info.major != 3 or sys.version_info.minor != 13:
        print(f"    [ALERTA] Se esperaba Python 3.13.x, pero se detecta {platform.python_version()}")
    else:
        print("    [OK] Versión de Python 3.13 verificada correctamente.")

    # 2. Comprobar Entorno Virtual
    print("\n[2] Verificación del Entorno Virtual (venv):")
    en_venv = (sys.prefix != sys.base_prefix) or hasattr(sys, 'real_prefix')
    if en_venv:
        print(f"    [OK] Entorno virtual activo: {sys.prefix}")
    else:
        print("    [ADVERTENCIA] No parece estar ejecutándose dentro de un venv activo.")
        errores.append("Entorno virtual no detectado como activo.")

    # 3. Comprobar Librerías requeridas
    print("\n[3] Comprobación de Dependencias Principales:")
    librerias = ["numpy", "scipy", "matplotlib", "pandas"]
    for lib in librerias:
        try:
            modulo = __import__(lib)
            version = getattr(modulo, "__version__", "Desconocida")
            print(f"    [OK] {lib.capitalize()} instalada - Versión: {version}")
        except ImportError as e:
            print(f"    [ERROR] No se pudo importar '{lib}': {e}")
            errores.append(f"Falta instalar la librería: {lib}")

    # 4. Prueba rápida de cálculo matemático y DSP (NumPy + SciPy)
    if "numpy" not in [err.split()[-1] for err in errores] and "scipy" not in [err.split()[-1] for err in errores]:
        try:
            import numpy as np
            from scipy import signal
            # Generar señal de prueba
            t = np.linspace(0, 1e-3, 1000)
            onda = np.sin(2 * np.pi * 1000 * t)
            potencia = np.mean(onda**2)
            print("\n[4] Prueba de cálculo vectorial y DSP:")
            print(f"    [OK] NumPy & SciPy cálculo vectorial exitoso (Potencia de prueba: {potencia:.4f})")
        except Exception as e:
            print("\n[4] Prueba de cálculo vectorial:")
            print(f"    [ERROR] Falló la prueba matemática: {e}")
            errores.append("Error en cálculo básico de NumPy/SciPy")

    # 5. Comprobar estructura de carpetas de resultados
    print("\n[5] Comprobación de Estructura de Carpetas:")
    carpetas = [
        os.path.join("resultados", "graficos"),
        os.path.join("resultados", "tablas"),
        os.path.join("resultados", "resumen")
    ]
    for c in carpetas:
        if os.path.exists(c) and os.path.isdir(c):
            print(f"    [OK] Carpeta existe: {c}")
        else:
            print(f"    [CREANDO] Creando carpeta: {c}")
            os.makedirs(c, exist_ok=True)

    # Resumen final
    print("\n" + "=" * 65)
    if not errores:
        print("   >>> RESULTADO: ¡FASE 1 COMPLETADA CON ÉXITO! <<<")
        print("   El entorno está 100% listo para iniciar la FASE 2.")
    else:
        print("   >>> RESULTADO: FASE 1 INCOMPLETA <<<")
        print("   Por favor soluciona los siguientes puntos:")
        for err in errores:
            print(f"    - {err}")
    print("=" * 65)

if __name__ == "__main__":
    verificar_entorno()
