#!/usr/bin/env python3
"""
API REST utilisant QGIS en mode headless (FastAPI)
Version améliorée : initialisation sûre, sessions thread-safe, endpoints fonctionnels,
startup/shutdown, gestion propre des fichiers temporaires et journalisation.
"""

import os
import sys
import uuid
import shutil
import tempfile
import threading
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# ---------- Configuration logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("qgis_headless_api")

# ---------- Globals (initialement vides, gérés via fonctions) ----------
qgis_manager = None
project_sessions: Dict[str, "ProjectSession"] = {}
project_sessions_lock = threading.Lock()


# ---------- QgisManager ----------
class QgisManager:
    """
    Gère l'initialisation "headless" de QGIS et fournit accès aux classes QGIS importées.
    L'initialisation est idempotente et journalisée.
    """
    def __init__(self):
        self._initialized = False
        self._initialization_attempted = False
        self.qgs_app = None
        self.classes: Dict[str, Any] = {}
        self.init_errors: List[str] = []

    def _setup_qgis_environment(self):
        """Configurer l'environnement pour QGIS headless avant import."""
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        os.environ.setdefault('QT_DEBUG_PLUGINS', '0')
        # Définitions optionnelles :
        # os.environ.setdefault('QGIS_PREFIX_PATH', '/usr')
        logger.debug("Variables d'environnement QGIS configurées")

    def initialize(self) -> Tuple[bool, Optional[str]]:
        """Initialiser QGIS de manière sûre (idempotente)."""
        if self._initialized:
            return True, None
        if self._initialization_attempted:
            return False, "Initialization already attempted"

        self._initialization_attempted = True
        logger.info("=== DÉBUT DE L'INITIALISATION QGIS ===")

        try:
            self._setup_qgis_environment()

            # Importer PyQt5 puis QGIS - après configuration d'environnement
            from PyQt5.QtCore import QCoreApplication  # noqa: F401

            from qgis.core import (
                Qgis,
                QgsApplication,
                QgsProject,
                QgsVectorLayer,
                QgsRasterLayer,
                QgsMapSettings,
                QgsMapRendererParallelJob,
                QgsProcessingFeedback,
                QgsProcessingContext,
                QgsRectangle,
                QgsPalLayerSettings,
                QgsTextFormat,
                QgsVectorLayerSimpleLabeling,
                QgsLayoutExporter,
            )

            from qgis.analysis import QgsNativeAlgorithms

            logger.info("Initialisation de l'application QGIS...")
            if not QgsApplication.instance():
                self.qgs_app = QgsApplication([], False)
                self.qgs_app.initQgis()
                logger.info("Application QGIS initialisée")
            else:
                self.qgs_app = QgsApplication.instance()
                logger.info("Instance QGIS existante utilisée")

            # Initialiser processing et enregistrer providers natifs
            try:
                # preferred import
                import processing  # type: ignore
                logger.info("Module processing importé directement")
            except Exception:
                # fallback
                try:
                    from qgis import processing  # type: ignore
                    logger.info("Module qgis.processing importé")
                except Exception as e2:
                    logger.warning(f"Import de processing a échoué: {e2}")
                    # mock minimal
                    class MockProcessing:
                        @staticmethod
                        def run(*args, **kwargs):
                            raise NotImplementedError("Processing module not available")
                    processing = MockProcessing()
                    logger.warning("Utilisation d'un mock 'processing'")

            # Register native algorithms (safe even if already registered)
            try:
                QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
                logger.info("QgsNativeAlgorithms ajoutés au processing registry")
            except Exception as e:
                logger.warning(f"Impossible d'ajouter QgsNativeAlgorithms: {e}")

            # Stockage des classes utiles pour usage ultérieur
            self.classes = {
                'Qgis': Qgis,
                'QgsApplication': QgsApplication,
                'QgsProject': QgsProject,
                'QgsVectorLayer': QgsVectorLayer,
                'QgsRasterLayer': QgsRasterLayer,
                'QgsMapSettings': QgsMapSettings,
                'QgsMapRendererParallelJob': QgsMapRendererParallelJob,
                'QgsProcessingFeedback': QgsProcessingFeedback,
                'QgsProcessingContext': QgsProcessingContext,
                'QgsNativeAlgorithms': QgsNativeAlgorithms,
                'QgsRectangle': QgsRectangle,
                'processing': processing,
                'QgsPalLayerSettings': QgsPalLayerSettings,
                'QgsTextFormat': QgsTextFormat,
                'QgsVectorLayerSimpleLabeling': QgsVectorLayerSimpleLabeling,
                'QgsLayoutExporter': QgsLayoutExporter,
            }

            self._initialized = True
            logger.info("=== QGIS INITIALISÉ AVEC SUCCÈS ===")
            return True, None

        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Erreur d'initialisation QGIS: {e}"
            self.init_errors.append(error_msg + "\n" + tb)
            logger.error(error_msg)
            logger.debug(tb)
            return False, error_msg

    def is_initialized(self) -> bool:
        return self._initialized

    def get_classes(self) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("QGIS not initialized")
        return self.classes

    def get_errors(self) -> List[str]:
        return self.init_errors

    def cleanup(self):
        """Nettoyage à l'arrêt (libérer QGIS)."""
        try:
            if self.qgs_app:
                try:
                    # exitQgis si disponible
                    self.qgs_app.exitQgis()
                    logger.info("QGIS arrêté (exitQgis).")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'arrêt de QGIS: {e}")
                self.qgs_app = None
            self._initialized = False
        except Exception as e:
            logger.error(f"Erreur cleanup QgisManager: {e}")


# ---------- ProjectSession ----------
class ProjectSession:
    """Gère un projet QGIS isolé pour une session donnée (création, accès, nettoyage)."""
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.project = None
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.temporary_files: List[str] = []

    def get_project(self, qgs_project_class):
        """Retourne une instance QgsProject pour la session (lazy)."""
        if self.project is None:
            # créer une instance indépendante (QgsProject())
            self.project = qgs_project_class()
            try:
                # si l'instance a setTitle
                self.project.setTitle(f"Session Project - {self.session_id}")
            except Exception:
                pass
        self.last_accessed = datetime.utcnow()
        return self.project

    def add_temp_file(self, path: str):
        self.temporary_files.append(path)

    def cleanup(self):
        """Nettoyage des ressources associées à la session."""
        try:
            if self.project:
                try:
                    # clear ou autre méthode pour libérer le projet
                    self.project.clear()
                except Exception:
                    pass
                self.project = None

            for p in list(self.temporary_files):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                        logger.debug(f"Supprimé fichier temporaire: {p}")
                except Exception as e:
                    logger.warning(f"Impossible de supprimer {p}: {e}")
            self.temporary_files = []
        except Exception as e:
            logger.error(f"Erreur nettoyage session {self.session_id}: {e}")


# ---------- Utilitaires pour gestion globale ----------
def get_qgis_manager() -> QgisManager:
    global qgis_manager
    if qgis_manager is None:
        qgis_manager = QgisManager()
    return qgis_manager


def initialize_qgis_if_needed() -> Tuple[bool, Optional[str]]:
    manager = get_qgis_manager()
    if not manager.is_initialized():
        return manager.initialize()
    return True, None


def get_project_session(session_id: Optional[str] = None) -> Tuple[ProjectSession, bool]:
    """
    Retourne (session, created_flag)
    created_flag=True si une nouvelle session a été créée.
    """
    global project_sessions, project_sessions_lock
    if session_id is None:
        session = ProjectSession()
        with project_sessions_lock:
            project_sessions[session.session_id] = session
        return session, True

    with project_sessions_lock:
        existing = project_sessions.get(session_id)
        if existing is None:
            session = ProjectSession(session_id)
            project_sessions[session.session_id] = session
            return session, True
        else:
            return existing, False


def cleanup_expired_sessions(max_age_hours: int = 24):
    global project_sessions, project_sessions_lock
    now = datetime.utcnow()
    expired = []
    with project_sessions_lock:
        for sid, s in list(project_sessions.items()):
            age_hours = (now - s.last_accessed).total_seconds() / 3600.0
            if age_hours > max_age_hours:
                expired.append(sid)
                try:
                    s.cleanup()
                except Exception as e:
                    logger.warning(f"Erreur cleanup session {sid}: {e}")
        for sid in expired:
            project_sessions.pop(sid, None)

    if expired:
        logger.info(f"Nettoyé {len(expired)} sessions expirées: {expired}")


# ---------- Pydantic models ----------
class MapRequest(BaseModel):
    layers: List[str]
    extent: Optional[List[float]] = None  # [xmin, ymin, xmax, ymax]
    width: int = 800
    height: int = 600
    crs: str = "EPSG:4326"
    format: str = "PNG"
    session_id: Optional[str] = None


class LayerInfo(BaseModel):
    name: str
    type: str
    crs: str
    extent: List[float]
    feature_count: Optional[int] = None


class ProcessingRequest(BaseModel):
    algorithm: str
    parameters: Dict[str, Any]
    session_id: Optional[str] = None


# ---------- FastAPI app ----------
app = FastAPI(
    title="API QGIS Headless",
    description="API REST pour traitement géospatial avec QGIS (headless)",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    """Initialisation QGIS au démarrage du serveur."""
    logger.info("Démarrage API - initialisation QGIS...")
    success, err = initialize_qgis_if_needed()
    if not success:
        logger.error("Échec de l'initialisation QGIS au démarrage: %s", err)
    else:
        logger.info("QGIS prêt")


@app.on_event("shutdown")
def shutdown_event():
    """Nettoyage à l'arrêt du serveur."""
    logger.info("Arrêt API - nettoyage QGIS et sessions...")
    try:
        cleanup_expired_sessions(max_age_hours=0)  # tout nettoyer maintenant
    except Exception:
        logger.exception("Erreur nettoyage sessions au shutdown")
    try:
        mgr = get_qgis_manager()
        mgr.cleanup()
    except Exception:
        logger.exception("Erreur cleanup qgis_manager au shutdown")


# ---------- Endpoints ----------
@app.get("/")
def root():
    manager = get_qgis_manager()
    qgis_ok = manager.is_initialized()
    return {"message": "API QGIS Headless", "version": "1.0.0", "qgis_initialized": qgis_ok}


@app.get("/health")
def health():
    manager = get_qgis_manager()
    return {
        "status": "healthy" if manager.is_initialized() else "degraded",
        "qgis_initialized": manager.is_initialized(),
        "init_errors": manager.get_errors()[:5]
    }


@app.post("/sessions", status_code=201)
def create_session():
    session, created = get_project_session()
    return {"session_id": session.session_id, "created": created}


@app.post("/layers/upload")
async def upload_layer(file: UploadFile = File(...), session_id: Optional[str] = None):
    """
    Téléverser un fichier et tenter de le charger comme couche (vectorielle ou raster).
    Retourne des métadonnées sur la couche ou une erreur.
    """
    manager = get_qgis_manager()
    if not manager.is_initialized():
        raise HTTPException(status_code=503, detail="QGIS not initialized")

    classes = manager.get_classes()
    QgsVectorLayer = classes['QgsVectorLayer']
    QgsRasterLayer = classes['QgsRasterLayer']
    QgsProject = classes['QgsProject']

    # récupérer ou créer une session
    session, _ = get_project_session(session_id)

    # sauvegarder fichier temporaire
    suffix = Path(file.filename).suffix or ""
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
        session.add_temp_file(tmp_path)

        # tenter comme couche vecteur
        layer = QgsVectorLayer(tmp_path, file.filename, "ogr")
        layer_type = "vector"
        if not layer.isValid():
            # essayer raster
            layer = QgsRasterLayer(tmp_path, file.filename)
            layer_type = "raster" if layer.isValid() else "unknown"

        if not layer.isValid():
            # supprimer le fichier temporaire
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise HTTPException(status_code=400, detail="Format de fichier non supporté ou couche invalide")

        # ajouter au projet de la session (si possible)
        try:
            pr = session.get_project(QgsProject)
            # certains projets QGIS utilisent instance(), mais on garde local
            try:
                pr.addMapLayer(layer)
            except Exception:
                # fallback: tenter QgsProject.instance()
                try:
                    QgsProject.instance().addMapLayer(layer)
                except Exception:
                    logger.debug("Impossible d'ajouter la couche au projet de session")
        except Exception:
            logger.debug("Impossible de récupérer ou utiliser QgsProject pour la session")

        info = {
            "message": "Couche chargée avec succès",
            "layer_name": layer.name(),
            "type": layer_type
        }
        # si vecteur, compter features (peut être coûteux)
        try:
            if layer_type == "vector":
                info["feature_count"] = layer.featureCount()
        except Exception:
            logger.debug("Impossible de compter les features (peut dépendre du provider)")

        return info

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur upload_layer")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/layers", response_model=List[LayerInfo])
def get_layers(session_id: Optional[str] = None):
    """
    Lister les couches du projet de la session (ou de l'instance globale si aucun projet de session).
    """
    manager = get_qgis_manager()
    if not manager.is_initialized():
        raise HTTPException(status_code=503, detail="QGIS not initialized")
    classes = manager.get_classes()
    QgsProject = classes['QgsProject']
    QgsVectorLayer = classes['QgsVectorLayer']

    session, _ = get_project_session(session_id)
    try:
        proj = session.get_project(QgsProject)
        layers_dict = {}
        # essayer mapLayers si disponible
        try:
            layers_map = proj.mapLayers()
            if layers_map:
                layers_dict = layers_map
        except Exception:
            try:
                layers_dict = QgsProject.instance().mapLayers()
            except Exception:
                layers_dict = {}

        layers_info = []
        for layer_id, layer in layers_dict.items():
            try:
                extent = layer.extent()
                info = LayerInfo(
                    name=layer.name(),
                    type="vector" if isinstance(layer, QgsVectorLayer) else "raster",
                    crs=layer.crs().authid() if hasattr(layer, "crs") else "UNKNOWN",
                    extent=[
                        extent.xMinimum(), extent.yMinimum(),
                        extent.xMaximum(), extent.yMaximum()
                    ],
                    feature_count=layer.featureCount() if isinstance(layer, QgsVectorLayer) else None
                )
                layers_info.append(info)
            except Exception:
                logger.debug(f"Impossible d'extraire les informations layer {layer_id}")

        return layers_info
    except Exception as e:
        logger.exception("Erreur get_layers")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/map/render")
def render_map(request: MapRequest):
    """
    Rendu de carte minimal basé sur les couches demandées.
    Retourne une image (PNG par défaut).
    """
    manager = get_qgis_manager()
    if not manager.is_initialized():
        raise HTTPException(status_code=503, detail="QGIS not initialized")
    classes = manager.get_classes()

    QgsMapSettings = classes['QgsMapSettings']
    QgsRectangle = classes['QgsRectangle']
    QgsMapRendererParallelJob = classes['QgsMapRendererParallelJob']
    QgsCoordinateReferenceSystem = None
    QgsSize = None

    # Certaines classes peuvent être absentes selon la version : on utilise des alternatives minimales
    try:
        # import CRS/Size si présents
        from qgis.core import QgsCoordinateReferenceSystem, QgsUnitTypes, QgsSize  # type: ignore
    except Exception:
        pass

    # Récupérer couches
    try:
        session, _ = get_project_session(request.session_id)
        proj = session.get_project(classes['QgsProject'])
        # assembler layers par nom
        layers = []
        try:
            layers_map = proj.mapLayers()
            for ln in request.layers:
                for lid, layer in layers_map.items():
                    if layer.name() == ln:
                        layers.append(layer)
        except Exception:
            # fallback: parcourir project instance
            try:
                layers_map = classes['QgsProject'].instance().mapLayers()
                for ln in request.layers:
                    for lid, layer in layers_map.items():
                        if layer.name() == ln:
                            layers.append(layer)
            except Exception:
                pass

        if not layers:
            raise HTTPException(status_code=400, detail="Aucune couche trouvée pour les noms fournis")

        settings = QgsMapSettings()
        settings.setOutputSize([request.width, request.height])
        settings.setLayers(layers)

        if request.extent:
            extent = QgsRectangle(*request.extent)
        else:
            extent = QgsRectangle()
            for layer in layers:
                try:
                    extent.combineExtentWith(layer.extent())
                except Exception:
                    pass
        settings.setExtent(extent)

        # Output file
        fd, out_path = tempfile.mkstemp(suffix=f".{request.format.lower()}")
        os.close(fd)

        # Créer et lancer job de rendu
        render = QgsMapRendererParallelJob(settings)
        render.start()
        render.waitForFinished()
        img = render.renderedImage()

        # Sauvegarder
        saved = img.save(out_path)
        if not saved:
            logger.warning("Enregistrement de l'image a échoué via QImage.save, essaie fallback PIL")
            try:
                # fallback: convertir via bytes si possible (optionnel)
                from PIL import Image  # type: ignore
                # attempt to export via layout exporter could be implemented if needed
            except Exception:
                logger.debug("PIL non disponible pour fallback d'enregistrement")

        return FileResponse(out_path, media_type=f"image/{request.format.lower()}", filename=f"map.{request.format.lower()}")

    except HTTPException:
        raise
    except Exception:
        logger.exception("Erreur render_map")
        raise HTTPException(status_code=500, detail="Erreur interne lors du rendu")


@app.post("/processing/run")
def run_processing(req: ProcessingRequest):
    """
    Exécuter un algorithme de processing par son id.
    """
    manager = get_qgis_manager()
    if not manager.is_initialized():
        raise HTTPException(status_code=503, detail="QGIS not initialized")

    classes = manager.get_classes()
    processing = classes.get('processing')
    QgsProcessingFeedback = classes.get('QgsProcessingFeedback')

    if processing is None:
        raise HTTPException(status_code=501, detail="Processing module is not available in this environment")

    # Vérifier existence algorithm (si registry disponible)
    try:
        from qgis.core import QgsApplication  # type: ignore
        registry = QgsApplication.processingRegistry()
        if not registry.algorithmById(req.algorithm):
            raise HTTPException(status_code=400, detail=f"Algorithme non trouvé: {req.algorithm}")
    except HTTPException:
        raise
    except Exception:
        logger.debug("Impossible d'interroger processing registry ; on essaye quand même d'exécuter")

    try:
        feedback = QgsProcessingFeedback()
    except Exception:
        feedback = None

    try:
        result = processing.run(req.algorithm, req.parameters, feedback=feedback)
        return {"result": result}
    except Exception as e:
        logger.exception("Erreur lors de l'exécution processing")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/processing/algorithms")
def get_algorithms():
    manager = get_qgis_manager()
    if not manager.is_initialized():
        raise HTTPException(status_code=503, detail="QGIS not initialized")

    try:
        from qgis.core import QgsApplication  # type: ignore
        registry = QgsApplication.processingRegistry()
        algorithms = []
        for provider in registry.providers():
            for alg in provider.algorithms():
                algorithms.append({
                    "id": alg.id(),
                    "name": alg.displayName(),
                    "group": alg.group(),
                    "provider": provider.name()
                })
        return algorithms
    except Exception:
        logger.exception("Erreur get_algorithms")
        raise HTTPException(status_code=500, detail="Impossible de lister les algorithmes")


@app.post("/sessions/cleanup")
def cleanup_sessions(max_age_hours: int = 24):
    try:
        cleanup_expired_sessions(max_age_hours=max_age_hours)
        return {"message": "Cleanup executed"}
    except Exception:
        logger.exception("Erreur cleanup_sessions endpoint")
        raise HTTPException(status_code=500, detail="Erreur lors du nettoyage")


# ---------- Lancer l'app ----------
if __name__ == "__main__":
    # Exemple: uvicorn.run("this_file_name:app", host="0.0.0.0", port=8000, reload=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
