
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from src.treeleaf.api.routes.predictions import router as predictions_router
from src.treeleaf.model.predictor import LeafPredictor
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = LeafPredictor()

    yield

    del app.state.predictor


app = FastAPI(
    title="TreeLeaf Classifier API",
    description="API for tree leaf image classification using ResNet50",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Используем OpenAPI 3.0 для совместимости Swagger UI с загрузкой файлов.
app.openapi_version = "3.0.3"


@app.get("/api/v1/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "service": "TreeLeaf Classifier API",
    }


app.include_router(
    predictions_router,
    prefix="/api/v1",
    tags=["Predictions"],
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        openapi_version=app.openapi_version,
    )

    request_schema = (
        schema["paths"]["/api/v1/predict"]["post"]
        ["requestBody"]["content"]["multipart/form-data"]["schema"]
    )

    schema_name = request_schema["$ref"].split("/")[-1]

    files_schema = (
        schema["components"]["schemas"][schema_name]
        ["properties"]["files"]
    )

    files_schema["items"]["format"] = "binary"

    app.openapi_schema = schema

    return app.openapi_schema


app.openapi = custom_openapi
