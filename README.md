# Panorama Stitching con OpenCV

Implementación de un sistema de **image stitching panorámico** utilizando OpenCV y la librería `stitching`.

# Objetivo

Diseñar un sistema capaz de generar panoramas coherentes a partir de múltiples imágenes con solapamiento, minimizando errores de correspondencia y mejorando la consistencia visual mediante técnicas de visión por computador.

---

El proyecto incluye:

* Detección y emparejamiento de características (ORB, SIFT y SURF cuando está disponible)
* Estimación de homografías
* Generación automática de panoramas
* Análisis cuantitativo de iluminación y brillo
* Corrección automática de diferencias de exposición entre imágenes
* Visualización de keypoints, matrices de confianza y métricas geométricas
* Comparación entre stitching original y stitching con normalización de brillo

También se incluye un **pipeline completo listo para ejecutar**, desde la carga de imágenes hasta la generación final del panorama y sus análisis visuales.

---

# Características principales

* Stitching panorámico automático
* Corrección de brillo basada en promedio global
* Comparación entre detectores de características
* Heatmaps de confianza entre imágenes
* Evaluación de dimensiones y deformación geométrica
* Visualizaciones explicativas para análisis académico y experimental

---

# Librerías utilizadas

* OpenCV
* stitching
* NumPy
* Matplotlib

---

# Pipeline incluido

El notebook incluye ejemplos completos y un pipeline preparado para:

1. Cargar imágenes con solapamiento
2. Detectar características
3. Emparejar puntos clave
4. Estimar homografías
5. Realizar warping y blending
6. Corregir diferencias de brillo
7. Generar y comparar panoramas finales

---

# Detectores evaluados

* ORB
* SIFT
* SURF *(cuando está disponible en OpenCV contrib)*

---

# Análisis realizados

El proyecto realiza distintos análisis cuantitativos y visuales:

* Histogramas de brillo
* Comparación de iluminación
* Matrices de confianza entre imágenes
* Visualización de keypoints
* Comparación de dimensiones de panoramas
* Evaluación de deformación geométrica
* Comparación entre stitching original y corregido

---
