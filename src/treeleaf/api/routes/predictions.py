
from io import BytesIO

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_FILES = 20

ALLOWED_FORMATS = {
    "JPEG",
    "PNG",
    "WEBP",
    "BMP",
    "TIFF",
}


class PredictionItem(BaseModel):
    class_name: str
    probability: float = Field(ge=0, le=1)


class PredictionResult(BaseModel):
    filename: str
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    top_predictions: list[PredictionItem]


class PredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResult]


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify leaf images",
)
def predict(
    request: Request,
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one image is required",
        )

    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum number of files is {MAX_FILES}",
        )

    predictor = request.app.state.predictor

    results = []

    for uploaded_file in files:
        filename = uploaded_file.filename or "unknown"

        try:
            content = uploaded_file.file.read(MAX_FILE_SIZE + 1)

            if not content:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{filename}' is empty",
                )

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File '{filename}' exceeds 10 MB",
                )

            try:
                image = Image.open(BytesIO(content))

                if image.format not in ALLOWED_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail=f"Unsupported image format: {image.format}",
                    )

                image.verify()

                image = Image.open(BytesIO(content))
                image.load()

            except HTTPException:
                raise

            except (
                UnidentifiedImageError,
                OSError,
                Image.DecompressionBombError,
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{filename}' is not a valid image",
                )

            prediction = predictor.predict(image)

            results.append(
                PredictionResult(
                    filename=filename,
                    predicted_class=prediction["predicted_class"],
                    confidence=prediction["confidence"],
                    top_predictions=[
                        PredictionItem(**item)
                        for item in prediction["top_predictions"]
                    ],
                )
            )

        finally:
            uploaded_file.file.close()

    return PredictionResponse(
        count=len(results),
        predictions=results,
    )
