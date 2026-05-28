import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os

from stitching.images import Images
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.subsetter import Subsetter
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.cropper import Cropper
from stitching.seam_finder import SeamFinder
from stitching.blender import Blender

# ============================================================
# Utilidades
# ============================================================

def brillo_medio(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
    return hsv[:, :, 2].mean()


def corregir_iluminacion(imagenes):

    brillos = [brillo_medio(img) for img in imagenes]
    brillo_global = np.mean(brillos)

    corregidas = []

    print("\nCorrección de iluminación:")

    for i, (img, b) in enumerate(zip(imagenes, brillos)):

        alpha = brillo_global / b
        alpha = float(np.clip(alpha, 0.7, 1.5))

        corregida = cv.convertScaleAbs(img, alpha=alpha, beta=0)

        corregidas.append(corregida)

        print(
            f"Imagen {i+1}: "
            f"{b:.1f} → "
            f"{brillo_medio(corregida):.1f}"
        )

    return corregidas


# ============================================================
# Pipeline
# ============================================================

def pipeline_desde_carpeta(
    carpeta='sample_stitching',
    detector='orb',
    nfeatures=2000,
    confidence_threshold=0.5
):

    carpeta = Path(carpeta)

    extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

    rutas = []

    for ext in extensiones:
        rutas.extend(carpeta.glob(ext))

    rutas = sorted(rutas)

    if len(rutas) < 2:
        print("Se necesitan mínimo 2 imágenes")
        return None

    rutas = [str(r.resolve()) for r in rutas]

    print("\nImágenes encontradas:")
    for r in rutas:
        print(" -", Path(r).name)

    # ========================================================
    # Carga imágenes
    # ========================================================

    imagenes = [cv.imread(r) for r in rutas]

    if any(img is None for img in imagenes):
        print("Error cargando imágenes")
        return None

    # ========================================================
    # Corrección iluminación
    # ========================================================

    imagenes_corr = corregir_iluminacion(imagenes)

    rutas_temp = []

    temp_dir = tempfile.gettempdir()

    for i, img in enumerate(imagenes_corr):

        ruta_tmp = os.path.join(
            temp_dir,
            f"stitch_corr_{i}.jpg"
        )

        cv.imwrite(ruta_tmp, img)

        rutas_temp.append(ruta_tmp)

    print("\nImágenes temporales:")
    for r in rutas_temp:
        print(r)

    # ========================================================
    # Visualización imágenes corregidas
    # ========================================================

    fig, axes = plt.subplots(2, len(imagenes), figsize=(5*len(imagenes), 7))

    if len(imagenes) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i in range(len(imagenes)):

        axes[0, i].imshow(cv.cvtColor(imagenes[i], cv.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')

        axes[1, i].imshow(cv.cvtColor(imagenes_corr[i], cv.COLOR_BGR2RGB))
        axes[1, i].set_title(f"Corregida {i+1}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    # ========================================================
    # Pipeline stitching
    # ========================================================

    try:

        t_total = time.perf_counter()

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        images = Images.of(rutas_temp)

        medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
        low_imgs    = list(images.resize(Images.Resolution.LOW))
        final_imgs  = list(images.resize(Images.Resolution.FINAL))

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        finder = FeatureDetector(
            detector=detector,
            nfeatures=nfeatures
        )

        features = [
            finder.detect_features(img)
            for img in medium_imgs
        ]

        print("\nKeypoints detectados:")

        for i, f in enumerate(features):
            print(f"Imagen {i+1}: {len(f.keypoints)}")

        # ----------------------------------------------------
        # Matching
        # ----------------------------------------------------

        matcher = FeatureMatcher()

        matches = matcher.match_features(features)

        conf = matcher.get_confidence_matrix(matches)

        print("\nMatriz de confianza:")
        print(conf)

        # ----------------------------------------------------
        # Subset
        # ----------------------------------------------------

        subsetter = Subsetter(confidence_threshold=confidence_threshold)

        indices = subsetter.get_indices_to_keep(
            features,
            matches
        )

        print(f"\nImagenes conservadas (umbral={confidence_threshold}): {list(indices)}")
        descartadas = [i for i in range(len(rutas)) if i not in list(indices)]
        if descartadas:
            print(f"Descartadas por baja confianza: {[Path(rutas[i]).name for i in descartadas]}")

        medium_imgs = subsetter.subset_list(medium_imgs, indices)
        low_imgs    = subsetter.subset_list(low_imgs, indices)
        final_imgs  = subsetter.subset_list(final_imgs, indices)

        features = subsetter.subset_list(features, indices)

        matches = subsetter.subset_matches(matches, indices)

        images.subset(indices)

        # ----------------------------------------------------
        # Cámaras
        # ----------------------------------------------------

        estimator = CameraEstimator()

        cameras = estimator.estimate(features, matches)

        adjuster = CameraAdjuster()

        cameras = adjuster.adjust(
            features,
            matches,
            cameras
        )

        wave_corrector = WaveCorrector()

        cameras = wave_corrector.correct(cameras)

        # ----------------------------------------------------
        # Warping
        # ----------------------------------------------------

        warper = Warper()

        warper.set_scale(cameras)

        final_sizes = images.get_scaled_img_sizes(
            Images.Resolution.FINAL
        )

        aspect = images.get_ratio(
            Images.Resolution.MEDIUM,
            Images.Resolution.FINAL
        )

        warped_imgs = list(
            warper.warp_images(
                final_imgs,
                cameras,
                aspect
            )
        )

        warped_masks = list(
            warper.create_and_warp_masks(
                final_sizes,
                cameras,
                aspect
            )
        )

        corners, sizes = warper.warp_rois(
            final_sizes,
            cameras,
            aspect
        )

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        cropper = Cropper()

        cropper.prepare(
            warped_imgs,
            warped_masks,
            corners,
            sizes
        )

        cropped_imgs = list(
            cropper.crop_images(warped_imgs)
        )

        cropped_masks = list(
            cropper.crop_images(warped_masks)
        )

        corners, sizes = cropper.crop_rois(
            corners,
            sizes
        )

        # ----------------------------------------------------
        # Seam finder
        # ----------------------------------------------------

        seam_finder = SeamFinder()

        seam_masks = seam_finder.find(
            cropped_imgs,
            corners,
            cropped_masks
        )

        seam_masks = [
            seam_finder.resize(seam_mask, cropped_mask)
            for seam_mask, cropped_mask in zip(seam_masks, cropped_masks)
        ]

        # ----------------------------------------------------
        # Blender
        # ----------------------------------------------------

        blender = Blender()

        blender.prepare(corners, sizes)

        for img, mask, corner in zip(
            cropped_imgs,
            seam_masks,
            corners
        ):

            blender.feed(img, mask, corner)

        panorama, _ = blender.blend()

        # ====================================================
        # Conversión final
        # ====================================================

        panorama = cv.convertScaleAbs(panorama)

        total = time.perf_counter() - t_total

        print(f"\nPanorama generado en {total:.2f}s")

        # ====================================================
        # Mostrar panorama
        # ====================================================

        plt.figure(figsize=(20, 8))

        plt.imshow(
            cv.cvtColor(panorama, cv.COLOR_BGR2RGB)
        )

        plt.title(
            f"Panorama final — {detector.upper()}",
            fontsize=15,
            fontweight='bold'
        )

        plt.axis('off')

        plt.tight_layout()

        plt.show()

        return panorama

    except Exception as e:

        print("\nERROR EN PIPELINE")
        print(e)

        import traceback
        traceback.print_exc()

        return None


# ============================================================
# Ejecutar
# ============================================================

panorama = pipeline_desde_carpeta(
    carpeta='sample_stitching',
    detector='orb',
    nfeatures=2000,
    confidence_threshold=0.5
)