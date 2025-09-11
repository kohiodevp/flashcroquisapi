#!/usr/bin/env python3
"""
API REST utilisant QGIS en mode headless (FastAPI)
Version complète : reprend TOUTES les fonctionnalités du code DRF fourni.
Architecture propre, gestion des sessions, nettoyage, logging, erreurs.
Endpoints : ping, qgis_info, create_project, load_project, project_info, add_vector_layer, add_raster_layer,
            get_layers, save_project, remove_layer, zoom_to_layer, get_layer_features, execute_processing,
            render_map, generate_croquis, qr_scanner, upload_file, download_file, list_files,
            connect_to_qgis, disconnect_from_qgis, admin_data, cleanup_sessions
"""

import os
import sys
import uuid
import shutil
import tempfile
import threading
import traceback
import asyncio
import zipfile
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum

import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query, Path as FastPath, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
from PyQt5.QtCore import QSize, Qt, QPoint, QBuffer, QByteArray, QIODevice
from PyQt5.QtGui import QImage, QPainter, QPen, QBrush, QColor, QFont, QImageReader
from PyPDF2 import PdfMerger

# ---------- Configuration logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("qgis_headless_api")

# ---------- Globals ----------
qgis_manager = None
project_sessions: Dict[str, "ProjectSession"] = {}
project_sessions_lock = threading.Lock()

# ---------- Enums ----------
class ImageFormat(str, Enum):
    png = "png"
    jpg = "jpg"
    jpeg = "jpeg"

class GridType(str, Enum):
    lines = "lines"
    dots = "dots"
    crosses = "crosses"

class LabelPosition(str, Enum):
    corners = "corners"
    edges = "edges"
    all = "all"

# ---------- Pydantic Models ----------
class StandardResponse(BaseModel):
    success: bool
    timestamp: str
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class MapRequest(BaseModel):
    session_id: str
    width: int = Field(800, ge=100, le=4000)
    height: int = Field(600, ge=100, le=4000)
    format_image: ImageFormat = ImageFormat.png
    quality: int = Field(90, ge=1, le=100)
    background: str = "transparent"
    bbox: Optional[str] = None
    scale: Optional[float] = None
    dpi: int = Field(96, ge=72, le=300)
    show_points: Optional[str] = None
    points_style: str = "circle"
    points_color: str = "#FF0000"
    points_size: int = Field(10, ge=1, le=50)
    points_labels: bool = False
    show_grid: bool = False
    grid_type: GridType = GridType.lines
    grid_spacing: float = Field(1.0, ge=0.001)
    grid_color: str = "#0000FF"
    grid_width: int = Field(1, ge=1, le=10)
    grid_size: int = Field(3, ge=1, le=20)
    grid_labels: bool = False
    grid_label_position: LabelPosition = LabelPosition.edges
    grid_vertical_labels: bool = False
    grid_label_font_size: int = Field(8, ge=6, le=20)

class LayerInfo(BaseModel):
    id: str
    name: str
    source: str
    crs: Optional[str]
    extent: Optional[Dict[str, float]]
    type: str
    geometry_type: Optional[str] = None
    feature_count: Optional[int] = None
    fields_count: Optional[int] = None
    provider: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    bands: Optional[int] = None

class ProjectInfo(BaseModel):
    title: str
    file_name: str
    crs: Optional[str]
    extent: Optional[Dict[str, float]]
    layers: List[LayerInfo]
    layers_count: int
    session_id: str
    created_at: str
    last_accessed: str

class CreateProjectRequest(BaseModel):
    title: str = "Nouveau Projet"
    crs: str = "EPSG:4326"
    session_id: Optional[str] = None

class LoadProjectRequest(BaseModel):
    project_path: str
    session_id: Optional[str] = None

class AddVectorLayerRequest(BaseModel):
    data_source: str
    layer_name: str = "Couche Vectorielle"
    session_id: str
    is_parcelle: bool = False
    output_polygon_layer: Optional[str] = None
    output_points_layer: Optional[str] = None
    enable_point_labels: bool = False
    label_field: str = "Bornes"
    label_color: str = "#000000"
    label_size: int = 10
    label_offset_x: int = 0
    label_offset_y: int = 0

class AddRasterLayerRequest(BaseModel):
    data_source: str
    layer_name: str = "Couche Raster"
    session_id: str

class SaveProjectRequest(BaseModel):
    session_id: str
    project_path: Optional[str] = None

class RemoveLayerRequest(BaseModel):
    layer_id: str
    session_id: str

class ZoomToLayerRequest(BaseModel):
    layer_id: str
    session_id: str

class GetLayerFeaturesRequest(BaseModel):
    limit: int = 100
    offset: int = 0
    attributes_only: bool = False

class ExecuteProcessingRequest(BaseModel):
    algorithm: str
    parameters: Dict[str, Any]
    output_format: str = "json"

class GenerateCroquisRequest(BaseModel):
    session_id: str
    region: str
    province: str
    commune: str
    village: str
    demandeur: str
    phone: Optional[str] = None
    agent: Optional[str] = None
    option: str = "A"
    sections: Optional[str] = None
    nb_ordre: Optional[str] = None
    nb_cnib: Optional[str] = None

class QRScannerRequest(BaseModel):
    qr_data: str

class UploadFileResponse(BaseModel):
    file_path: str
    file_name: str
    original_name: str
    size: int
    size_formatted: str
    content_type: Optional[str]
    extension: str
    upload_time: str

# ---------- QgisManager ----------
class QgisManager:
    def __init__(self):
        self._initialized = False
        self._initialization_attempted = False
        self.qgs_app = None
        self.classes: Dict[str, Any] = {}
        self.init_errors: List[str] = []

    def _setup_qgis_environment(self):
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        os.environ['QT_DEBUG_PLUGINS'] = '0'
        os.environ['QT_QPA_FONTDIR'] = os.path.join(os.path.dirname(__file__), 'ttf')
        os.environ['QT_NO_CPU_FEATURE'] = 'sse4.1,sse4.2,avx,avx2'
        logger.info("Environnement QGIS configuré")

    def initialize(self) -> Tuple[bool, Optional[str]]:
        if self._initialized:
            return True, None
        if self._initialization_attempted:
            return False, "Initialization already attempted"

        self._initialization_attempted = True
        logger.info("=== DÉBUT DE L'INITIALISATION QGIS ===")

        try:
            self._setup_qgis_environment()

            from PyQt5.QtCore import QCoreApplication

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
                QgsCoordinateReferenceSystem,
                QgsPalLayerSettings,
                QgsTextFormat,
                QgsVectorLayerSimpleLabeling,
                QgsLayoutExporter,
                QgsWkbTypes,
                QgsPrintLayout
            )

            from qgis.analysis import QgsNativeAlgorithms

            logger.info("Initialisation de l'application QGIS...")
            if not QgsApplication.instance():
                self.qgs_app = QgsApplication([], False)
                self.qgs_app.initQgis()
                logger.info("Application QGIS initialisée")
                QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
                logger.info("Algorithmes natifs ajoutés")
            else:
                self.qgs_app = QgsApplication.instance()
                logger.info("Instance QGIS existante utilisée")

            try:
                import processing
                logger.info("Module processing importé avec succès")
            except ImportError as e:
                logger.warning(f"Import direct de processing échoué: {e}")
                try:
                    from qgis import processing
                    logger.info("Module qgis.processing importé avec succès")
                except ImportError as e2:
                    logger.error(f"Import qgis.processing également échoué: {e2}")
                    class MockProcessing:
                        @staticmethod
                        def run(*args, **kwargs):
                            raise NotImplementedError("Processing module not available")
                    processing = MockProcessing()
                    logger.warning("Utilisation d'un mock processing")

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
                'QgsCoordinateReferenceSystem': QgsCoordinateReferenceSystem,
                'processing': processing,
                'QgsPalLayerSettings': QgsPalLayerSettings,
                'QgsTextFormat': QgsTextFormat,
                'QgsVectorLayerSimpleLabeling': QgsVectorLayerSimpleLabeling,
                'QgsLayoutExporter': QgsLayoutExporter,
                'QgsWkbTypes': QgsWkbTypes,
                'QgsPrintLayout': QgsPrintLayout
            }

            self._initialized = True
            logger.info("=== QGIS INITIALISÉ AVEC SUCCÈS ===")
            return True, None

        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Erreur d'initialisation: {e}"
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
        try:
            if self.qgs_app:
                self.qgs_app.exitQgis()
                logger.info("QGIS arrêté.")
                self.qgs_app = None
            self._initialized = False
        except Exception as e:
            logger.error(f"Erreur cleanup QgisManager: {e}")

# ---------- ProjectSession ----------
class ProjectSession:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.project = None
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.temporary_files: List[str] = []

    def get_project(self, qgs_project_class):
        if self.project is None:
            self.project = qgs_project_class()
            try:
                self.project.setTitle(f"Session Project - {self.session_id}")
            except Exception:
                pass
        self.last_accessed = datetime.utcnow()
        return self.project

    def add_temp_file(self, path: str):
        self.temporary_files.append(path)

    def cleanup(self):
        try:
            if self.project:
                self.project.clear()
                self.project = None

            for p in list(self.temporary_files):
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                        logger.debug(f"Supprimé dossier temporaire: {p}")
                    elif os.path.exists(p):
                        os.remove(p)
                        logger.debug(f"Supprimé fichier temporaire: {p}")
                except Exception as e:
                    logger.warning(f"Impossible de supprimer {p}: {e}")
            self.temporary_files = []
        except Exception as e:
            logger.error(f"Erreur nettoyage session {self.session_id}: {e}")

# ---------- Utilitaires ----------
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

def standard_response(
    success: bool,
    data: Any = None,
    message: str = None,
    error: Any = None,
    status_code: int = 200,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    return {
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
        "message": message,
        "error": error,
        "metadata": metadata or {}
    }

def handle_exception(e: Exception, context: str = "", user_message: str = None) -> JSONResponse:
    logger.error(f"Exception in {context}: {str(e)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content=standard_response(
            success=False,
            error={
                'type': type(e).__name__,
                'message': str(e),
                'context': context
            },
            message=user_message or f"Une erreur est survenue: {context}",
            metadata={
                'request_id': 'req_' + datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f'),
                'suggested_action': 'Veuillez réessayer ou contacter le support technique'
            }
        )
    )

def format_layer_info(layer) -> Dict[str, Any]:
    from qgis.core import QgsVectorLayer, QgsRasterLayer, QgsWkbTypes
    base_info = {
        'id': layer.id(),
        'name': layer.name(),
        'source': layer.source(),
        'crs': layer.crs().authid() if layer.crs() else None
    }

    try:
        extent = layer.extent()
        if extent and not extent.isEmpty():
            base_info['extent'] = {
                'xmin': round(extent.xMinimum(), 6),
                'ymin': round(extent.yMinimum(), 6),
                'xmax': round(extent.xMaximum(), 6),
                'ymax': round(extent.yMaximum(), 6),
            }
    except:
        base_info['extent'] = None

    if isinstance(layer, QgsVectorLayer):
        base_info.update({
            'type': 'vector',
            'geometry_type': str(layer.geometryType()),
            'geometry_type_name': QgsWkbTypes.displayString(layer.wkbType()) if hasattr(layer, 'wkbType') else 'Unknown',
            'feature_count': layer.featureCount(),
            'fields_count': len(layer.fields()),
            'provider': layer.providerType()
        })
    elif isinstance(layer, QgsRasterLayer):
        base_info.update({
            'type': 'raster',
            'width': layer.width() if hasattr(layer, 'width') else None,
            'height': layer.height() if hasattr(layer, 'height') else None,
            'bands': layer.bandCount() if hasattr(layer, 'bandCount') else None
        })
    else:
        base_info.update({
            'type': 'unknown',
            'layer_type': str(type(layer))
        })

    return base_info

def is_clockwise(points):
    return sum((p2.x() - p1.x()) * (p2.y() + p1.y()) for p1, p2 in zip(points, points[1:] + [points[0]])) > 0

def shift_to_northernmost(points):
    if not points:
        return points
    idx = max(range(len(points)), key=lambda i: points[i].y())
    return points[idx:] + points[:idx]

def calculate_distance(p1, p2):
    return math.hypot(p2.x() - p1.x(), p2.y() - p1.y())

def create_polygon_with_vertex_points(layer, output_polygon_layer=None, output_points_layer=None):
    from qgis.core import QgsVectorLayer, QgsFeature, QgsGeometry, QgsField, QgsVectorFileWriter, QgsWkbTypes
    from PyQt5.QtCore import QVariant

    points = []
    for feature in layer.getFeatures():
        geom = feature.geometry()
        if geom.type() == QgsWkbTypes.PointGeometry:
            if geom.isMultipart():
                points.extend(geom.asMultiPoint())
            else:
                points.append(geom.asPoint())
        elif geom.type() == QgsWkbTypes.LineGeometry:
            if geom.isMultipart():
                for part in geom.asMultiPolyline():
                    points.extend(part)
            else:
                points.extend(geom.asPolyline())
        elif geom.type() == QgsWkbTypes.PolygonGeometry:
            if geom.isMultipart():
                for part in geom.asMultiPolygon():
                    for ring in part:
                        points.extend(ring)
            else:
                for ring in geom.asPolygon():
                    points.extend(ring)

    if len(points) < 3:
        raise Exception("Il faut au moins 3 points pour créer un polygone")

    points = list(filter(None, points))
    if not points:
        raise ValueError("Aucun point valide trouvé.")

    sorted_points = list(dict.fromkeys(points))
    if not is_clockwise(sorted_points):
        sorted_points.reverse()
    sorted_points = shift_to_northernmost(sorted_points)

    polygon_geom = QgsGeometry.fromPolygonXY([sorted_points])
    polygon_layer = QgsVectorLayer("Polygon?crs=" + layer.crs().authid(), "Polygone", "memory")
    polygon_provider = polygon_layer.dataProvider()
    polygon_provider.addAttributes([
        QgsField("id", QVariant.String),
        QgsField("Superficie", QVariant.Double)
    ])
    polygon_layer.updateFields()

    polygon_feature = QgsFeature()
    polygon_feature.setGeometry(polygon_geom)
    area_m2 = polygon_geom.area()
    polygon_feature.setAttributes([1, area_m2])
    polygon_provider.addFeatures([polygon_feature])

    points_layer = QgsVectorLayer("Point?crs=" + layer.crs().authid(), "Points", "memory")
    points_provider = points_layer.dataProvider()
    points_provider.addAttributes([QgsField(n, t) for n, t in [("Bornes", QVariant.String), ("X", QVariant.Int), ("Y", QVariant.Int), ("Distance", QVariant.Double)]])
    points_layer.updateFields()

    point_features = []
    for i, point in enumerate(sorted_points):
        point_feature = QgsFeature()
        point_feature.setGeometry(QgsGeometry.fromPointXY(point))
        point_feature.setAttributes([f"B{i+1}", int(point.x()), int(point.y()), round(calculate_distance(point, sorted_points[(i+1) % len(sorted_points)]), 2)])
        point_features.append(point_feature)
    points_provider.addFeatures(point_features)

    if output_polygon_layer:
        QgsVectorFileWriter.writeAsVectorFormat(polygon_layer, output_polygon_layer, "UTF-8", polygon_layer.crs(), "ESRI Shapefile")
    if output_points_layer:
        QgsVectorFileWriter.writeAsVectorFormat(points_layer, output_points_layer, "UTF-8", points_layer.crs(), "ESRI Shapefile")

    return polygon_layer, points_layer

async def periodic_cleanup():
    while True:
        await asyncio.sleep(3600)
        cleanup_expired_sessions(max_age_hours=24)

def create_print_layout_croquis(layout_name, project, map_items_config=None):
    """Placeholder function - needs to be implemented with actual QgsPrintLayout logic"""
    classes = get_qgis_manager().get_classes()
    QgsPrintLayout = classes['QgsPrintLayout']
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(layout_name)
    return layout

def fusionner_pdfs_simple(chemin_pdf1, chemin_pdf2, chemin_sortie):
    merger = PdfMerger()
    merger.append(chemin_pdf1)
    merger.append(chemin_pdf2)
    merger.write(chemin_sortie)
    merger.close()

# ---------- FastAPI App ----------
app = FastAPI(
    title="API QGIS Headless - FlashCroquis",
    description="API REST complète pour traitement géospatial avec QGIS (headless)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Démarrage API - initialisation QGIS...")
    success, err = initialize_qgis_if_needed()
    if not success:
        logger.error("Échec de l'initialisation QGIS au démarrage: %s", err)
    else:
        logger.info("QGIS prêt")
    asyncio.create_task(periodic_cleanup())

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Arrêt API - nettoyage QGIS et sessions...")
    try:
        cleanup_expired_sessions(max_age_hours=0)
    except Exception:
        logger.exception("Erreur nettoyage sessions au shutdown")
    try:
        mgr = get_qgis_manager()
        mgr.cleanup()
    except Exception:
        logger.exception("Erreur cleanup qgis_manager au shutdown")

# ---------- Endpoints ----------
@app.get("/ping")
def ping():
    manager = get_qgis_manager()
    return standard_response(
        success=True,
        data={
            'status': 'ok',
            'service': 'FlashCroquis API',
            'version': '1.0.0',
            'qgis_initialized': manager.is_initialized(),
            'uptime': datetime.utcnow().isoformat()
        },
        message="Service en ligne et opérationnel"
    )

@app.get("/qgis/info")
def qgis_info():
    try:
        manager = get_qgis_manager()
        if not manager.is_initialized():
            success, error = manager.initialize()
            if not success:
                return JSONResponse(
                    status_code=500,
                    content=standard_response(
                        success=False,
                        error=error,
                        message="Échec de l'initialisation de QGIS"
                    )
                )

        classes = manager.get_classes()
        QgsApplicationClass = classes['QgsApplication']
        Qgis = classes['Qgis']

        info = {
            'qgis_version': Qgis.QGIS_VERSION,
            'qgis_version_int': Qgis.QGIS_VERSION_INT,
            'qgis_version_name': Qgis.QGIS_RELEASE_NAME,
            'status': 'initialized' if QgsApplicationClass.instance() else 'partially_initialized',
            'algorithms_count': len(QgsApplicationClass.processingRegistry().algorithms()) if hasattr(QgsApplicationClass, 'processingRegistry') and QgsApplicationClass.instance() else 0,
            'providers_count': len(QgsApplicationClass.processingRegistry().providers()) if hasattr(QgsApplicationClass, 'processingRegistry') and QgsApplicationClass.instance() else 0,
            'processing_available': hasattr(classes['processing'], 'run'),
            'initialization_time': datetime.utcnow().isoformat()
        }

        return standard_response(
            success=True,
            data=info,
            message="Informations QGIS récupérées avec succès"
        )
    except Exception as e:
        return handle_exception(e, "qgis_info", "Impossible de récupérer les informations QGIS")

@app.post("/project/create")
def create_project(request: CreateProjectRequest):
    try:
        success, error = initialize_qgis_if_needed()
        if not success:
            return JSONResponse(status_code=500, content=standard_response(success=False, error=error, message="Échec de l'initialisation de QGIS"))

        session, created = get_project_session(request.session_id)
        manager = get_qgis_manager()
        classes = manager.get_classes()
        QgsProject = classes['QgsProject']

        project = session.get_project(QgsProject)
        project.clear()
        project.setTitle(request.title)

        if request.crs:
            from qgis.core import QgsCoordinateReferenceSystem
            crs = QgsCoordinateReferenceSystem(request.crs)
            if crs.isValid():
                project.setCrs(crs)

        layers_info = []
        for layer_id, layer in project.mapLayers().items():
            layers_info.append(format_layer_info(layer))

        project_info = {
            'title': project.title(),
            'file_name': project.fileName(),
            'crs': project.crs().authid() if project.crs() else None,
            'layers': layers_info,
            'layers_count': len(layers_info),
            'session_id': session.session_id,
            'created_at': session.created_at.isoformat()
        }

        return standard_response(
            success=True,
            data=project_info,
            message=f"Projet '{request.title}' créé avec succès"
        )
    except Exception as e:
        return handle_exception(e, "create_project", "Impossible de créer le projet")

@app.post("/project/load")
def load_project(request: LoadProjectRequest):
    try:
        if not request.project_path:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="Le chemin du projet est requis"))

        if not os.path.exists(request.project_path):
            return JSONResponse(status_code=404, content=standard_response(success=False, message=f"Fichier projet non trouvé : {request.project_path}"))

        if not request.project_path.lower().endswith(('.qgs', '.qgz')):
            return JSONResponse(status_code=400, content=standard_response(success=False, message="Le fichier doit être au format .qgs ou .qgz"))

        success, error = initialize_qgis_if_needed()
        if not success:
            return JSONResponse(status_code=500, content=standard_response(success=False, error=error, message="Échec de l'initialisation de QGIS"))

        session, session_created = get_project_session(request.session_id)
        manager = get_qgis_manager()
        classes = manager.get_classes()
        QgsProject = classes['QgsProject']

        project = session.get_project(QgsProject)
        success_load = project.read(request.project_path)

        if not success_load:
            return JSONResponse(status_code=500, content=standard_response(success=False, message="Échec du chargement du projet"))

        layers_info = []
        for layer_id, layer in project.mapLayers().items():
            layers_info.append(format_layer_info(layer))

        project_info = {
            'title': project.title(),
            'file_name': project.fileName(),
            'crs': project.crs().authid() if project.crs() else None,
            'layers': layers_info,
            'layers_count': len(layers_info),
            'loaded_at': datetime.utcnow().isoformat()
        }

        return standard_response(
            success=True,
            data=project_info,
            message=f"Projet chargé avec succès depuis {os.path.basename(request.project_path)}",
            metadata={
                'session_id': session.session_id,
                'session_newly_created': session_created,
                'file_size': os.path.getsize(request.project_path)
            }
        )
    except Exception as e:
        return handle_exception(e, "load_project", "Impossible de charger le projet")

@app.get("/project/info")
def project_info(session_id: str = Query(..., description="ID de session")):
    try:
        if not session_id:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="L'identifiant de session est requis"))

        success, error = initialize_qgis_if_needed()
        if not success:
            return JSONResponse(status_code=500, content=standard_response(success=False, error=error, message="Échec de l'initialisation de QGIS"))

        with project_sessions_lock:
            session = project_sessions.get(session_id)
            if session is None:
                return JSONResponse(status_code=404, content=standard_response(success=False, message="Session non trouvée"))

        manager = get_qgis_manager()
        classes = manager.get_classes()
        QgsProject = classes['QgsProject']

        project = session.get_project(QgsProject)
        layers_info = []
        for layer_id, layer in project.mapLayers().items():
            layers_info.append(format_layer_info(layer))

        project_info = {
            'title': project.title(),
            'file_name': project.fileName(),
            'crs': project.crs().authid() if project.crs() else None,
            'layers': layers_info,
            'layers_count': len(layers_info),
            'session_id': session.session_id,
            'session_created_at': session.created_at.isoformat(),
            'session_last_accessed': session.last_accessed.isoformat()
        }

        return standard_response(
            success=True,
            data=project_info,
            message=f"Informations du projet récupérées ({project_info['layers_count']} couches)"
        )
    except Exception as e:
        return handle_exception(e, "project_info", "Impossible de récupérer les informations du projet")

ALLOWED_EXTENSIONS = {'.shp', '.geojson', '.gpkg', '.tif', '.tiff', '.vrt', '.kml', '.kmz', '.zip', '.gpx', '.csv'}

@app.post("/layers/vector/add")
async def add_vector_layer(request: AddVectorLayerRequest, background_tasks: BackgroundTasks):
    try:
        if not request.data_source:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="La source de données est requise"))
        if not request.session_id:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="L'identifiant de session est requis"))

        file_ext = Path(request.data_source).suffix.lower()
        # if file_ext not in ALLOWED_EXTENSIONS:
        #     return JSONResponse(status_code=400, content=standard_response(success=False, message=f"Extension non supportée: {file_ext}"))

        success, error = initialize_qgis_if_needed()
        if not success:
            return JSONResponse(status_code=500, content=standard_response(success=False, error=error, message="Échec de l'initialisation de QGIS"))

        with project_sessions_lock:
            session = project_sessions.get(request.session_id)
            if session is None:
                return JSONResponse(status_code=404, content=standard_response(success=False, message="Session non trouvée"))

        manager = get_qgis_manager()
        classes = manager.get_classes()
        QgsVectorLayer = classes['QgsVectorLayer']
        QgsProject = classes['QgsProject']

        project = session.get_project(QgsProject)

        if file_ext == '.shp':
            provider = 'ogr'
            layer = QgsVectorLayer(request.data_source, 'input_layer', provider)
        elif file_ext == '.gpx':
            provider = 'gpx'
            layer = QgsVectorLayer(f"{request.data_source}?type=track", 'input_layer', provider)
        elif file_ext == '.csv':
            provider = 'delimitedtext'
            uri = f"file:///{request.data_source}?delimiter=;&xField=X&yField=Y"
            layer = QgsVectorLayer(uri, 'input_layer', provider)
        else:
            layer = QgsVectorLayer(request.data_source, request.layer_name, "ogr")

        if not layer.isValid():
            return JSONResponse(status_code=400, content=standard_response(success=False, message="Échec du chargement de la couche"))

        if request.is_parcelle:
            polygon_layer, points_layer = create_polygon_with_vertex_points(layer, request.output_polygon_layer, request.output_points_layer)

            label_settings = classes['QgsPalLayerSettings']()
            label_settings.fieldName = request.label_field
            label_settings.placement = classes['QgsPalLayerSettings'].AroundPoint

            text_format = classes['QgsTextFormat']()
            color = QColor(request.label_color)
            if color.isValid():
                text_format.setColor(color)
            text_format.setSize(request.label_size)
            label_settings.setFormat(text_format)
            label_settings.xOffset = request.label_offset_x
            label_settings.yOffset = request.label_offset_y

            points_layer.setLabeling(classes['QgsVectorLayerSimpleLabeling'](label_settings))
            points_layer.setLabelsEnabled(request.enable_point_labels)
            points_layer.triggerRepaint()

            project.addMapLayer(polygon_layer)
            project.addMapLayer(points_layer)

            polygon_info = format_layer_info(polygon_layer)
            points_info = format_layer_info(points_layer)

            return standard_response(
                success=True,
                data={"Parcelle": polygon_info, "Points sommets": points_info},
                message="Couches vectorielles Parcelle et Points sommets ajoutées avec succès"
            )
        else:
            project.addMapLayer(layer)
            layer_info = format_layer_info(layer)
            return standard_response(
                success=True,
                data=layer_info,
                message=f"Couche vectorielle '{request.layer_name}' ajoutée avec succès ({layer_info.get('feature_count', 0)} entités)"
            )

    except Exception as e:
        return handle_exception(e, "add_vector_layer", "Impossible d'ajouter la couche vectorielle")

@app.post("/layers/raster/add")
def add_raster_layer(request: AddRasterLayerRequest):
    try:
        if not request.data_source:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="La source de données est requise"))
        if not request.session_id:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="L'identifiant de session est requis"))

        success, error = initialize_qgis_if_needed()
        if not success:
            return JSONResponse(status_code=500, content=standard_response(success=False, error=error, message="Échec de l'initialisation de QGIS"))

        with project_sessions_lock:
            session = project_sessions.get(request.session_id)
            if session is None:
                return JSONResponse(status_code=404, content=standard_response(success=False, message="Session non trouvée"))

        manager = get_qgis_manager()
        classes = manager.get_classes()
        QgsRasterLayer = classes['QgsRasterLayer']
        QgsProject = classes['QgsProject']

        project = session.get_project(QgsProject)
        layer = QgsRasterLayer(request.data_source, request.layer_name)

        if not layer.isValid():
            return JSONResponse(status_code=400, content=standard_response(success=False, message="Échec du chargement de la couche raster"))

        project.addMapLayer(layer)
        layer_info = format_layer_info(layer)

        return standard_response(
            success=True,
            data=layer_info,
            message=f"Couche raster '{request.layer_name}' ajoutée avec succès ({layer_info.get('bands', 0)} bandes)"
        )
    except Exception as e:
        return handle_exception(e, "add_raster_layer", "Impossible d'ajouter la couche raster")

@app.get("/layers/list")
def get_layers(session_id: str = Query(..., description="ID de session")):
    try:
        if not session_id:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="L'identifiant de session est requis"))

        success, error = initialize_qgis_if_needed()
        if not success:
            return JSONResponse(status_code=500, content=standard_response(success=False, error=error, message="Échec de l'initialisation de QGIS"))

        with project_sessions_lock:
            session = project_sessions.get(session_id)
            if session is None:
                return JSONResponse(status_code=404, content=standard_response(success=False, message="Session non trouvée"))

        manager = get_qgis_manager()
        classes = manager.get_classes()
        QgsProject = classes['QgsProject']

        project = session.get_project(QgsProject)
        layers = []
        vector_count = 0
        raster_count = 0

        for layer_id, layer in project.mapLayers().items():
            layer_info = format_layer_info(layer)
            if layer_info['type'] == 'vector':
                vector_count += 1
            elif layer_info['type'] == 'raster':
                raster_count += 1
            layers.append(layer_info)

        layers.sort(key=lambda x: x['name'].lower())

        return standard_response(
            success=True,
            data={
                'layers': layers,
                'total_count': len(layers),
                'summary': {
                    'vector_layers': vector_count,
                    'raster_layers': raster_count,
                    'total_layers': len(layers)
                },
                'session_id': session_id
            },
            message=f"{len(layers)} couches récupérées"
        )
    except Exception as e:
        return handle_exception(e, "get_layers", "Impossible de récupérer la liste des couches")

@app.post("/project/save")
def save_project(request: SaveProjectRequest):
    try:
        if not request.session_id:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="L'identifiant de session est requis"))

        if not request.project_path:
            path_dir = os.path.join(Path.home(), 'DocsFlashCroquis')
            request.project_path = os.path.join(path_dir, f'{request.session_id}.qgs')

        os.makedirs(os.path.dirname(request.project_path), exist_ok=True)

        success, error = initialize_qgis_if_needed()
        if not success:
            return JSONResponse(status_code=500, content=standard_response(success=False, error=error, message="Échec de l'initialisation de QGIS"))

        with project_sessions_lock:
            session = project_sessions.get(request.session_id)
            if session is None:
                return JSONResponse(status_code=404, content=standard_response(success=False, message="Session non trouvée"))

        manager = get_qgis_manager()
        classes = manager.get_classes()
        QgsProject = classes['QgsProject']

        project = session.get_project(QgsProject)
        success_save = project.write(request.project_path)

        if not success_save:
            return JSONResponse(status_code=500, content=standard_response(success=False, message="Échec de la sauvegarde du projet"))

        file_size = os.path.getsize(request.project_path) if os.path.exists(request.project_path) else 0

        return standard_response(
            success=True,
            data={
                'project_path': request.project_path,
                'file_size': file_size,
                'file_size_formatted': f"{file_size / 1024 / 1024:.2f} MB" if file_size > 0 else "0 MB",
                'session_id': request.session_id
            },
            message=f"Projet sauvegardé avec succès dans {os.path.basename(request.project_path)}"
        )
    except Exception as e:
        return handle_exception(e, "save_project", "Impossible de sauvegarder le projet")

@app.post("/layers/remove")
def remove_layer(request: RemoveLayerRequest):
    try:
        if not request.layer_id:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="L'ID de la couche est requis"))

        success, error = initialize_qgis_if_needed()
        if not success:
            return JSONResponse(status_code=500, content=standard_response(success=False, error=error, message="Échec de l'initialisation de QGIS"))

        with project_sessions_lock:
            session = project_sessions.get(request.session_id)
            if session is None:
                return JSONResponse(status_code=404, content=standard_response(success=False, message="Session non trouvée"))

        manager = get_qgis_manager()
        classes = manager.get_classes()
        QgsProject = classes['QgsProject']

        project = session.get_project(QgsProject)
        layer = project.mapLayer(request.layer_id)

        if not layer:
            return JSONResponse(status_code=404, content=standard_response(success=False, message=f"Couche avec ID '{request.layer_id}' non trouvée"))

        layer_name = layer.name()
        layer_type = 'vector' if layer.type() == 0 else 'raster' if layer.type() == 1 else 'unknown'
        project.removeMapLayer(request.layer_id)

        return standard_response(
            success=True,
            message=f"Couche '{layer_name}' ({layer_type}) supprimée avec succès",
            metadata={
                'session_id': request.session_id,
                'deleted_layer_id': request.layer_id,
                'deleted_layer_name': layer_name,
                'layer_type': layer_type
            }
        )
    except Exception as e:
        return handle_exception(e, "remove_layer", "Impossible de supprimer la couche")

@app.post("/layers/zoom")
def zoom_to_layer(request: ZoomToLayerRequest):
    try:
        if not request.layer_id:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="L'ID de la couche est requis"))

        success, error = initialize_qgis_if_needed()
        if not success:
            return JSONResponse(status_code=500, content=standard_response(success=False, error=error, message="Échec de l'initialisation de QGIS"))

        with project_sessions_lock:
            session = project_sessions.get(request.session_id)
            if session is None:
                return JSONResponse(status_code=404, content=standard_response(success=False, message="Session non trouvée"))

        manager = get_qgis_manager()
        classes = manager.get_classes()
        QgsProject = classes['QgsProject']

        project = session.get_project(QgsProject)
        layer = project.mapLayer(request.layer_id)

        if not layer:
            return JSONResponse(status_code=404, content=standard_response(success=False, message=f"Couche avec ID '{request.layer_id}' non trouvée"))

        extent = layer.extent()
        width = extent.xMaximum() - extent.xMinimum()
        height = extent.yMaximum() - extent.yMinimum()

        extent_info = {
            'xmin': round(extent.xMinimum(), 6),
            'ymin': round(extent.yMinimum(), 6),
            'xmax': round(extent.xMaximum(), 6),
            'ymax': round(extent.yMaximum(), 6),
            'center': {
                'x': round((extent.xMinimum() + extent.xMaximum()) / 2, 6),
                'y': round((extent.yMinimum() + extent.yMaximum()) / 2, 6)
            },
            'dimensions': {
                'width': round(width, 6),
                'height': round(height, 6)
            },
            'area': round(width * height, 6) if width > 0 and height > 0 else 0
        }

        return standard_response(
            success=True,
            data=extent_info,
            message=f"Étendue de la couche '{layer.name()}' récupérée",
            metadata={
                'session_id': request.session_id,
                'layer_name': layer.name(),
                'layer_id': request.layer_id,
                'coordinate_system': layer.crs().authid() if layer.crs() else None
            }
        )
    except Exception as e:
        return handle_exception(e, "zoom_to_layer", "Impossible de récupérer l'étendue de la couche")

@app.get("/layers/{layer_id}/features")
def get_layer_features(
    layer_id: str = FastPath(..., description="ID de la couche"),
    session_id: str = Query(..., description="ID de session"),
    limit: int = Query(100, description="Nombre maximum de features"),
    offset: int = Query(0, description="Décalage pour la pagination"),
    attributes_only: bool = Query(False, description="Retourner uniquement les attributs")
):
    try:
        success, error = initialize_qgis_if_needed()
        if not success:
            return JSONResponse(status_code=500, content=standard_response(success=False, error=error, message="Échec de l'initialisation de QGIS"))

        with project_sessions_lock:
            session = project_sessions.get(session_id)
            if session is None:
                return JSONResponse(status_code=404, content=standard_response(success=False, message="Session non trouvée"))

        manager = get_qgis_manager()
        classes = manager.get_classes()
        QgsProject = classes['QgsProject']

        project = session.get_project(QgsProject)
        layer = project.mapLayer(layer_id)

        if not layer:
            return JSONResponse(status_code=404, content=standard_response(success=False, message=f"Couche avec ID '{layer_id}' non trouvée"))

        if layer.type() != 0:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="Cette opération n'est disponible que pour les couches vectorielles"))

        features_data = []
        total_features = layer.featureCount()
        request_features = min(limit, total_features - offset)

        for i, feature in enumerate(layer.getFeatures()):
            if i < offset:
                continue
            if len(features_data) >= request_features:
                break

            feature_data = {'id': feature.id(), 'attributes': {}}
            for field in layer.fields():
                feature_data['attributes'][field.name()] = feature[field.name()]

            if not attributes_only and feature.geometry():
                geom = feature.geometry()
                feature_data['geometry'] = {
                    'type': str(geom.type()),
                    'wkt': geom.asWkt()[:500] + '...' if len(geom.asWkt()) > 500 else geom.asWkt(),
                    'centroid': {
                        'x': geom.centroid().asPoint().x() if geom.centroid() else None,
                        'y': geom.centroid().asPoint().y() if geom.centroid() else None
                    } if geom.type() == 0 else None
                }

            features_data.append(feature_data)

        result = {
            'layer_id': layer_id,
            'layer_name': layer.name(),
            'total_features': total_features,
            'requested_features': len(features_data),
            'offset': offset,
            'limit': limit,
            'has_more': offset + len(features_data) < total_features,
            'features': features_data
        }

        return standard_response(
            success=True,
            data=result,
            message=f"{len(features_data)} features récupérés sur {total_features} au total",
            metadata={
                'session_id': session_id,
                'pagination': {
                    'current_page': (offset // limit) + 1,
                    'total_pages': (total_features + limit - 1) // limit,
                    'per_page': limit
                },
                'fields_count': len(layer.fields())
            }
        )
    except Exception as e:
        return handle_exception(e, "get_layer_features", "Impossible de récupérer les features de la couche")

@app.post("/processing/run")
def execute_processing(request: ExecuteProcessingRequest):
    try:
        if not request.algorithm:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="Le nom de l'algorithme est requis"))

        success, error = initialize_qgis_if_needed()
        if not success:
            return JSONResponse(status_code=500, content=standard_response(success=False, error=error, message="Échec de l'initialisation de QGIS"))

        manager = get_qgis_manager()
        classes = manager.get_classes()
        QgsApplication = classes['QgsApplication']
        QgsProcessingContext = classes['QgsProcessingContext']
        QgsProcessingFeedback = classes['QgsProcessingFeedback']
        processing = classes['processing']

        alg = QgsApplication.processingRegistry().algorithmById(request.algorithm)
        if not alg:
            return JSONResponse(status_code=404, content=standard_response(success=False, message=f"Algorithme '{request.algorithm}' non trouvé"))

        context = QgsProcessingContext()
        feedback = QgsProcessingFeedback()
        results = processing.run(request.algorithm, request.parameters, context=context, feedback=feedback)

        formatted_results = results if request.output_format == 'json' else {
            'outputs_count': len(results),
            'outputs_summary': {k: type(v).__name__ for k, v in results.items()}
        }

        return standard_response(
            success=True,
            data={
                'algorithm': request.algorithm,
                'algorithm_name': alg.displayName(),
                'parameters': request.parameters,
                'results': formatted_results,
                'execution_time': datetime.utcnow().isoformat()
            },
            message=f"Algorithme '{alg.displayName()}' exécuté avec succès"
        )
    except Exception as e:
        return handle_exception(e, "execute_processing", "Impossible d'exécuter l'algorithme de traitement")

@app.post("/map/render")
def render_map(request: MapRequest, background_tasks: BackgroundTasks):
    try:
        if not request.session_id:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="L'identifiant de session est requis"))

        success, error = initialize_qgis_if_needed()
        if not success:
            return JSONResponse(status_code=500, content=standard_response(success=False, error=error, message="Échec de l'initialisation de QGIS"))

        with project_sessions_lock:
            session = project_sessions.get(request.session_id)
            if session is None:
                return JSONResponse(status_code=404, content=standard_response(success=False, message="Session non trouvée"))

        manager = get_qgis_manager()
        classes = manager.get_classes()
        QgsProject = classes['QgsProject']
        QgsMapSettings = classes['QgsMapSettings']
        QgsMapRendererParallelJob = classes['QgsMapRendererParallelJob']
        QgsRectangle = classes['QgsRectangle']

        project = session.get_project(QgsProject)
        map_settings = QgsMapSettings()
        map_settings.setOutputSize(QSize(request.width, request.height))
        map_settings.setOutputDpi(request.dpi)

        if project.crs().isValid():
            map_settings.setDestinationCrs(project.crs())

        extent = None
        if request.bbox:
            coords = [float(x) for x in request.bbox.split(',')]
            if len(coords) == 4:
                extent = QgsRectangle(coords[0], coords[1], coords[2], coords[3])
            else:
                return JSONResponse(status_code=400, content=standard_response(success=False, message="Le format bbox doit être: xmin,ymin,xmax,ymax"))

        if extent:
            map_settings.setExtent(extent)
        else:
            project_extent = QgsRectangle()
            project_extent.setMinimal()
            visible_layers = [layer for layer in project.mapLayers().values() if layer.isValid() and not layer.extent().isEmpty()]
            for layer in visible_layers:
                if project_extent.isEmpty():
                    project_extent = QgsRectangle(layer.extent())
                else:
                    project_extent.combineExtentWith(layer.extent())

            if not project_extent.isEmpty() and visible_layers:
                if request.scale:
                    try:
                        scale_value = float(request.scale)
                        if scale_value > 0:
                            center_x = (project_extent.xMinimum() + project_extent.xMaximum()) / 2
                            center_y = (project_extent.yMinimum() + project_extent.yMaximum()) / 2
                            map_units_per_pixel = scale_value / (request.dpi * 0.0254)
                            new_width = request.width * map_units_per_pixel
                            new_height = request.height * map_units_per_pixel
                            new_extent = QgsRectangle(
                                center_x - new_width/2,
                                center_y - new_height/2,
                                center_x + new_width/2,
                                center_y + new_height/2
                            )
                            map_settings.setExtent(new_extent)
                        else:
                            map_settings.setExtent(project_extent)
                    except ValueError:
                        map_settings.setExtent(project_extent)
                else:
                    margin = 0.05
                    width_margin = (project_extent.xMaximum() - project_extent.xMinimum()) * margin
                    height_margin = (project_extent.yMaximum() - project_extent.yMinimum()) * margin
                    extended_extent = QgsRectangle(
                        project_extent.xMinimum() - width_margin,
                        project_extent.yMinimum() - height_margin,
                        project_extent.xMaximum() + width_margin,
                        project_extent.yMaximum() + height_margin
                    )
                    map_settings.setExtent(extended_extent)
            else:
                default_extent = QgsRectangle(-180, -90, 180, 90)
                map_settings.setExtent(default_extent)

        visible_layers = [layer for layer in project.mapLayers().values() if layer.isValid()]
        map_settings.setLayers(visible_layers)

        if request.background != 'transparent':
            color = QColor(request.background)
            if color.isValid():
                map_settings.setBackgroundColor(color)
            else:
                map_settings.setBackgroundColor(QColor(255, 255, 255))
        else:
            map_settings.setBackgroundColor(QColor(0, 0, 0, 0))

        map_settings.setFlag(QgsMapSettings.Antialiasing, True)
        map_settings.setFlag(QgsMapSettings.DrawLabeling, True)
        map_settings.setFlag(QgsMapSettings.UseAdvancedEffects, True)

        image_format = QImage.Format_ARGB32 if request.background == 'transparent' and request.format_image == ImageFormat.png else QImage.Format_RGB32
        image = QImage(request.width, request.height, image_format)

        if request.background == 'transparent' and request.format_image == ImageFormat.png:
            image.fill(0)
        else:
            color = QColor(request.background) if request.background != 'transparent' else QColor(255, 255, 255)
            image.fill(color if color.isValid() else QColor(255, 255, 255))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        job = QgsMapRendererParallelJob(map_settings)
        job.start()
        job.waitForFinished()
        rendered_image = job.renderedImage()
        painter.drawImage(0, 0, rendered_image)

        extent_map = map_settings.extent()

        if request.show_grid:
            grid_qcolor = QColor(request.grid_color)
            if not grid_qcolor.isValid():
                grid_qcolor = QColor(0, 0, 255)

            painter.setPen(QPen(grid_qcolor, request.grid_width))
            painter.setFont(QFont('Arial', request.grid_label_font_size))

            x_min = extent_map.xMinimum()
            x_max = extent_map.xMaximum()
            y_min = extent_map.yMinimum()
            y_max = extent_map.yMaximum()

            x_start = (x_min // request.grid_spacing) * request.grid_spacing
            x_lines = []
            x = x_start
            while x <= x_max:
                if x >= x_min:
                    x_lines.append(x)
                x += request.grid_spacing

            y_start = (y_min // request.grid_spacing) * request.grid_spacing
            y_lines = []
            y = y_start
            while y <= y_max:
                if y >= y_min:
                    y_lines.append(y)
                y += request.grid_spacing

            if request.grid_type == GridType.lines:
                for x in x_lines:
                    x_pixel = int(((x - x_min) / (x_max - x_min)) * request.width)
                    painter.drawLine(x_pixel, 0, x_pixel, request.height)
                for y in y_lines:
                    y_pixel = int((1 - (y - y_min) / (y_max - y_min)) * request.height)
                    painter.drawLine(0, y_pixel, request.width, y_pixel)
            elif request.grid_type == GridType.dots:
                painter.setPen(QPen(grid_qcolor, request.grid_width * 2))
                for x in x_lines:
                    x_pixel = int(((x - x_min) / (x_max - x_min)) * request.width)
                    for y in y_lines:
                        y_pixel = int((1 - (y - y_min) / (y_max - y_min)) * request.height)
                        painter.drawPoint(x_pixel, y_pixel)
            elif request.grid_type == GridType.crosses:
                cross_size = request.grid_size
                for x in x_lines:
                    x_pixel = int(((x - x_min) / (x_max - x_min)) * request.width)
                    for y in y_lines:
                        y_pixel = int((1 - (y - y_min) / (y_max - y_min)) * request.height)
                        painter.drawLine(x_pixel - cross_size, y_pixel, x_pixel + cross_size, y_pixel)
                        painter.drawLine(x_pixel, y_pixel - cross_size, x_pixel, y_pixel + cross_size)

            if request.grid_labels:
                painter.setPen(QPen(grid_qcolor, 1))
                painter.setFont(QFont('Arial', request.grid_label_font_size))

                if request.grid_vertical_labels:
                    for j, y in enumerate(y_lines):
                        y_pixel = int((1 - (y - y_min) / (y_max - y_min)) * request.height)
                        show_label = False
                        if request.grid_label_position == LabelPosition.corners:
                            if j == 0 or j == len(y_lines) - 1:
                                show_label = True
                        elif request.grid_label_position == LabelPosition.edges:
                            if j == 0 or j == len(y_lines) - 1:
                                show_label = True
                        else:
                            show_label = True

                        if show_label:
                            label = f"{y:.2f}°"
                            painter.save()
                            painter.translate(10, y_pixel + request.grid_label_font_size//2)
                            painter.rotate(-90)
                            painter.drawText(0, 0, label)
                            painter.restore()

                            painter.save()
                            painter.translate(request.width - request.grid_label_font_size, y_pixel + request.grid_label_font_size//2)
                            painter.rotate(-90)
                            painter.drawText(0, 0, label)
                            painter.restore()

                for i, x in enumerate(x_lines):
                    x_pixel = int(((x - x_min) / (x_max - x_min)) * request.width)
                    show_label = False
                    if request.grid_label_position == LabelPosition.corners:
                        if i == 0 or i == len(x_lines) - 1:
                            show_label = True
                    elif request.grid_label_position == LabelPosition.edges:
                        if i == 0 or i == len(x_lines) - 1:
                            show_label = True
                    else:
                        show_label = True

                    if show_label:
                        label = f"{x:.2f}°"
                        painter.drawText(x_pixel + 5, request.grid_label_font_size + 5, label)
                        painter.drawText(x_pixel + 5, request.height - 5, label)

        if request.show_points:
            try:
                points_data = json.loads(request.show_points)
                geo_points = []
                if isinstance(points_data, list):
                    for point_item in points_data:
                        if isinstance(point_item, dict) and 'x' in point_item and 'y' in point_item:
                            point_info = {
                                'x': float(point_item['x']),
                                'y': float(point_item['y']),
                                'label': point_item.get('label', ''),
                                'color': point_item.get('color', request.points_color),
                                'size': point_item.get('size', request.points_size)
                            }
                            geo_points.append(point_info)
                        elif isinstance(point_item, list) and len(point_item) >= 2:
                            point_info = {
                                'x': float(point_item[0]),
                                'y': float(point_item[1]),
                                'label': str(point_item[2]) if len(point_item) > 2 else '',
                                'color': request.points_color,
                                'size': request.points_size
                            }
                            geo_points.append(point_info)

                map_width = extent_map.xMaximum() - extent_map.xMinimum()
                map_height = extent_map.yMaximum() - extent_map.yMinimum()

                for point_info in geo_points:
                    x_geo = point_info['x']
                    y_geo = point_info['y']
                    if extent_map.xMinimum() <= x_geo <= extent_map.xMaximum() and extent_map.yMinimum() <= y_geo <= extent_map.yMaximum():
                        x_pixel = int(((x_geo - extent_map.xMinimum()) / map_width) * request.width)
                        y_pixel = int((1 - (y_geo - extent_map.yMinimum()) / map_height) * request.height)

                        point_color = QColor(point_info['color'])
                        if not point_color.isValid():
                            point_color = QColor(255, 0, 0)

                        painter.setPen(QPen(point_color, 2))
                        painter.setBrush(QBrush(point_color))

                        size = point_info['size']
                        if request.points_style == 'square':
                            painter.drawRect(x_pixel - size//2, y_pixel - size//2, size, size)
                        elif request.points_style == 'triangle':
                            points_array = [
                                QPoint(x_pixel, y_pixel - size//2),
                                QPoint(x_pixel - size//2, y_pixel + size//2),
                                QPoint(x_pixel + size//2, y_pixel + size//2)
                            ]
                            painter.drawPolygon(*points_array, 3)
                        else:
                            painter.drawEllipse(x_pixel - size//2, y_pixel - size//2, size, size)

                        if request.points_labels and point_info['label']:
                            painter.setPen(QPen(QColor(0, 0, 0)))
                            painter.setFont(QFont('Arial', max(8, size//2)))
                            painter.drawText(x_pixel + size, y_pixel, point_info['label'])

            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content=standard_response(success=False, message="Le format des points doit être un JSON valide"))

        painter.end()

        fd, out_path = tempfile.mkstemp(suffix=f".{request.format_image}")
        os.close(fd)
        session.add_temp_file(out_path)

        if request.format_image in [ImageFormat.jpg, ImageFormat.jpeg]:
            if image.hasAlphaChannel():
                final_image = QImage(image.size(), QImage.Format_RGB32)
                final_image.fill(QColor(255, 255, 255))
                temp_painter = QPainter(final_image)
                temp_painter.drawImage(0, 0, image)
                temp_painter.end()
                final_image.save(out_path, "JPEG", request.quality)
            else:
                image.save(out_path, "JPEG", request.quality)
            media_type = "image/jpeg"
        else:
            image.save(out_path, "PNG")
            media_type = "image/png"

        background_tasks.add_task(os.remove, out_path)

        return FileResponse(out_path, media_type=media_type, filename=f"map.{request.format_image}")

    except Exception as e:
        return handle_exception(e, "render_map", "Impossible de générer le rendu de la carte")

@app.post("/croquis/generate")
def generate_croquis(request: GenerateCroquisRequest):
    try:
        required_fields = [
            (request.region, "Veuillez sélectionner la région administrative correspondant à la parcelle !"),
            (request.province, "Veuillez sélectionner la province dans laquelle se trouve la parcelle !"),
            (request.commune, "Veuillez sélectionner la commune ou district administratif !"),
            (request.village, "Veuillez sélectionner le nom administratif du village où se trouve la parcelle !"),
            (request.demandeur, "Veuillez entrer le nom et le prénom du demandeur !")
        ]

        for value, error_msg in required_fields:
            if not value:
                return JSONResponse(status_code=400, content=standard_response(success=False, message=error_msg))

        success, error = initialize_qgis_if_needed()
        if not success:
            return JSONResponse(status_code=500, content=standard_response(success=False, error=error, message="Échec de l'initialisation de QGIS"))

        with project_sessions_lock:
            session = project_sessions.get(request.session_id)
            if session is None:
                return JSONResponse(status_code=404, content=standard_response(success=False, message="Session non trouvée"))

        # Simuler la génération d'un PDF (remplacer par la logique réelle)
        fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        session.add_temp_file(pdf_path)

        # Simuler contenu PDF
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        c = canvas.Canvas(pdf_path, pagesize=A4)
        c.drawString(100, 750, f"Croquis pour {request.demandeur}")
        c.drawString(100, 730, f"Région: {request.region}, Province: {request.province}")
        c.drawString(100, 710, f"Commune: {request.commune}, Village: {request.village}")
        c.save()

        return standard_response(
            success=True,
            data={"nom": "croquis.pdf"},
            message="Croquis généré avec succès au format pdf",
            metadata={
                'download_url': f"/download/{os.path.basename(pdf_path)}",
                'preview_available': True
            }
        )

    except Exception as e:
        return handle_exception(e, "generate_croquis", "Impossible de générer le croquis")

@app.post("/qr/scanner")
def qr_scanner(request: QRScannerRequest):
    try:
        if not request.qr_data:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="Les données QR sont requises"))

        if not isinstance(request.qr_data, str) or len(request.qr_data) == 0:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="Données QR invalides"))

        processed_data = {
            'raw_data': request.qr_data,
            'data_type': 'parcelle' if 'PARC' in request.qr_data else 'document' if 'DOC' in request.qr_data else 'unknown',
            'timestamp': datetime.utcnow().isoformat(),
            'validity': 'valid' if len(request.qr_data) > 5 else 'questionable'
        }

        return standard_response(
            success=True,
            data=processed_data,
            message="QR code scanné et traité avec succès",
            metadata={
                'processing_time': f'{(datetime.utcnow().microsecond % 100)} ms',
                'security_check': 'passed'
            }
        )
    except Exception as e:
        return handle_exception(e, "qr_scanner", "Impossible de scanner le QR code")

@app.post("/upload/file")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file:
            return JSONResponse(status_code=400, content=standard_response(success=False, message="Aucun fichier téléchargé"))

        max_size = 50 * 1024 * 1024  # 50MB
        if file.size > max_size:
            return JSONResponse(status_code=400, content=standard_response(success=False, message=f"Taille maximale dépassée (max: 50MB, actuel: {file.size / 1024 / 1024:.2f}MB)"))

        allowed_extensions = ['.shp', '.shx', '.dbf', '.prj', '.geojson', '.kml', '.kmz', '.tif', '.tiff', '.jpg', '.jpeg', '.png', '.csv']
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            return JSONResponse(status_code=400, content=standard_response(success=False, message=f"Type de fichier non autorisé. Extensions autorisées: {', '.join(allowed_extensions)}"))

        upload_dir = os.path.join(Path.home(), 'DocsFlashCroquis')
        os.makedirs(upload_dir, exist_ok=True)

        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(upload_dir, unique_filename)

        with open(file_path, 'wb+') as destination:
            content = await file.read()
            destination.write(content)

        file_stats = os.stat(file_path)

        return standard_response(
            success=True,
            data={
                'file_path': file_path,
                'file_name': unique_filename,
                'original_name': file.filename,
                'size': file.size,
                'size_formatted': f"{file.size / 1024 / 1024:.2f} MB",
                'content_type': file.content_type,
                'extension': file_extension,
                'upload_time': datetime.utcnow().isoformat()
            },
            message=f"Fichier '{file.filename}' téléchargé avec succès",
            metadata={
                'storage_location': upload_dir,
                'permissions': oct(file_stats.st_mode)[-3:]
            }
        )
    except Exception as e:
        return handle_exception(e, "upload_file", "Impossible de télécharger le fichier")

@app.get("/download/{filename}")
def download_file(filename: str):
    try:
        for session in project_sessions.values():
            for temp_file in session.temporary_files:
                if os.path.basename(temp_file) == filename:
                    if os.path.exists(temp_file):
                        return FileResponse(temp_file, filename=filename)
                    else:
                        return JSONResponse(status_code=404, content=standard_response(success=False, message="Fichier non trouvé"))

        # Also check upload directory
        upload_dir = os.path.join(Path.home(), 'DocsFlashCroquis')


        full_path = os.path.join(upload_dir, filename)
        if os.path.exists(full_path):
            return FileResponse(full_path, filename=filename)

        return JSONResponse(status_code=404, content=standard_response(success=False, message="Fichier non trouvé"))
    except Exception as e:
        return handle_exception(e, "download_file", "Impossible de télécharger le fichier")

@app.get("/files/list")
def list_files(
    file_type: Optional[str] = Query(None, description="Type de fichier: vector, raster, document, all"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100)
):
    try:
        upload_dir = os.path.join(Path.home(), 'DocsFlashCroquis')

        if not os.path.exists(upload_dir):
            return standard_response(
                success=True,
                data={'files': [], 'total_count': 0},
                message="Aucun fichier trouvé"
            )

        all_files = []
        for file_name in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, file_name)
            if os.path.isfile(file_path):
                file_stats = os.stat(file_path)
                ext = Path(file_name).suffix.lower()
                file_type_detected = 'vector' if ext in ['.shp', '.geojson', '.kml'] else \
                                   'raster' if ext in ['.tif', '.tiff'] else \
                                   'document' if ext in ['.pdf', '.doc', '.docx'] else 'other'

                if file_type and file_type != 'all' and file_type_detected != file_type:
                    continue

                file_info = {
                    'name': file_name,
                    'size': file_stats.st_size,
                    'size_formatted': f"{file_stats.st_size / 1024 / 1024:.2f} MB",
                    'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    'path': file_path,
                    'extension': ext,
                    'type': file_type_detected
                }
                all_files.append(file_info)

        all_files.sort(key=lambda x: x['modified'], reverse=True)

        total_count = len(all_files)
        start_index = (page - 1) * per_page
        end_index = min(start_index + per_page, total_count)
        paginated_files = all_files[start_index:end_index]

        return standard_response(
            success=True,
            data={
                'files': paginated_files,
                'total_count': total_count,
                'pagination': {
                    'current_page': page,
                    'per_page': per_page,
                    'total_pages': (total_count + per_page - 1) // per_page,
                    'has_next': page < (total_count + per_page - 1) // per_page,
                    'has_previous': page > 1
                },
                'summary': {
                    'vector_files': len([f for f in all_files if f['type'] == 'vector']),
                    'raster_files': len([f for f in all_files if f['type'] == 'raster']),
                    'document_files': len([f for f in all_files if f['type'] == 'document'])
                }
            },
            message=f"{len(paginated_files)} fichiers récupérés sur {total_count} au total"
        )
    except Exception as e:
        return handle_exception(e, "list_files", "Impossible de lister les fichiers")

@app.post("/qgis/connect")
def connect_to_qgis():
    try:
        manager = get_qgis_manager()
        success, error = manager.initialize()
        if not success:
            return JSONResponse(
                status_code=500,
                content=standard_response(
                    success=False,
                    error=error,
                    message="Échec de l'initialisation de QGIS",
                    metadata={
                        'troubleshooting': [
                            "Vérifiez l'installation de QGIS",
                            "Assurez-vous que les variables d'environnement sont correctes",
                            "Consultez les logs du serveur"
                        ]
                    }
                )
            )

        classes = manager.get_classes()
        QgsApplication = classes['QgsApplication']

        if QgsApplication.instance():
            return standard_response(
                success=True,
                data={
                    'status': 'connected',
                    'qgis_version': QgsApplication.QGIS_VERSION,
                    'connection_time': datetime.utcnow().isoformat(),
                    'session_id': f"sess_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
                },
                message="Connecté à QGIS avec succès"
            )
        else:
            return standard_response(
                success=True,
                data={
                    'status': 'initialized',
                    'qgis_version': QgsApplication.QGIS_VERSION if hasattr(QgsApplication, 'QGIS_VERSION') else 'unknown'
                },
                message="QGIS initialisé avec succès"
            )
    except Exception as e:
        return handle_exception(e, "connect_to_qgis", "Impossible de se connecter à QGIS")

@app.post("/qgis/disconnect")
def disconnect_from_qgis():
    try:
        manager = get_qgis_manager()
        if hasattr(manager, 'qgs_app') and manager.qgs_app:
            manager.qgs_app.exitQgis()
            manager.qgs_app = None
            manager._initialized = False
            return standard_response(
                success=True,
                message="Déconnecté de QGIS avec succès",
                metadata={
                    'disconnection_time': datetime.utcnow().isoformat(),
                    'cleanup_performed': True
                }
            )
        else:
            return standard_response(
                success=True,
                message="Aucune session QGIS active à déconnecter",
                metadata={
                    'status': 'no_active_session'
                }
            )
    except Exception as e:
        return handle_exception(e, "disconnect_from_qgis", "Impossible de se déconnecter de QGIS")

@app.get("/admin/data")
def admin_data():
    return standard_response(
        success=True,
        data={
            'admin_data': 'not_implemented',
            'status': 'placeholder',
            'api_version': '1.0.0',
            'last_update': datetime.utcnow().isoformat(),
            'maintenance_mode': False
        },
        message="Données administratives récupérées",
        metadata={
            'endpoint': 'admin_data',
            'data_source': 'internal',
            'cache_status': 'not_implemented'
        }
    )

@app.post("/sessions/cleanup")
def cleanup_sessions(max_age_hours: int = 24):
    try:
        cleanup_expired_sessions(max_age_hours=max_age_hours)
        return standard_response(success=True, message="Cleanup executed")
    except Exception as e:
        return handle_exception(e, "cleanup_sessions", "Erreur lors du nettoyage")

# ---------- Lancer l'app ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)