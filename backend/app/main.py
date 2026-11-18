from fastapi import FastAPI

from app.api.errors import install_handlers
from app.api.routes import router
from app.core.settings import Settings, load_settings
from app.services.registry import DetectorRegistry, default_factory


def create_app(settings: Settings | None = None, detector_factory=default_factory) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Visual Assembly Verification", version="0.1")
    app.state.settings = settings
    app.state.registry = DetectorRegistry(settings, detector_factory)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    install_handlers(app)
    app.include_router(router)
    return app


app = create_app()