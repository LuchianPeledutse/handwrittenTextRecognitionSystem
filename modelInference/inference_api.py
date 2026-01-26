from PIL import Image

import fastapi
from pydantic import BaseModel

app = fastapi.FastAPI()

class InputData(BaseModel):
    """Image will come in 64 base format"""
    image_64_base: str




@app.post('/prediction')
def predict(image_data: InputData):
    image = Image.open(io.BytesIO(base64.b64decode(data.image_64_base))).convert("L")
    return {"Image_size": image.size, "Image_mode": image.mode}

