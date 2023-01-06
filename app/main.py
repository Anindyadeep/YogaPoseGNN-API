import os 
import sys 
import json 
import uvicorn
from fastapi.logger import logger
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

import torch
from config import CONFIG
from inference import ModelInference
from schemas import FrameInferenceInput, FrameInferenceResponse
from exceptions import validation_exception_handler, python_exception_handler

# TODO: Add an auth feature before connecting to the API, so that we can count the API usage per user
# TODO: Mounting a API simple HTML landing page for developers with documentation, this will include the authentication too for that use firebase for the authetication

app = FastAPI(
    title="YogaPoseGNN Inference API-v0",
    description="An REST API for YogaPoseGNN that provides inference on request pose keypoints in real time",
    version="0.0.1",
    terms_of_service=None,
    contact="proanindyadeep@gmail.com",
    license_info={
        "license": "MIT",
    },
)

# allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)

# On Startup event
@app.on_event("startup")
async def startup_event():
    """
    All the initialization of variables and models are to be done here
    """
    logger.info("=> Running environment: {}".format(CONFIG["ENV"]))
    logger.info("=> PyTorch using device: {}".format(CONFIG["DEVICE"]))
    model_inference = ModelInference(CONFIG["DEVICE"])

    app.package = {"model_inference": model_inference}
    logger.info("=> Server listening on PORT")

app.mount("/", StaticFiles(directory="static/"), name="static")

# @app.get("/")
# def start():
#     return {"status": 200, "version": "0.0.1", "contact": "proanindyadeep@gmail.com"}

@app.get("/api/ping")
def ping():
    return {"status": 200, "info" : "Server is working and active"}


@app.post("/api/v1/predict", response_model=FrameInferenceResponse)
def predict(request: Request, body: FrameInferenceInput):
    """Perform prediction on incoming requests 

    Args:
        request (Request): An HTTP request
        body (FrameInferenceInput): The BaseModel schema that will contain the body of the request
    """

    logger.info(f"Input: {body}")
    info_dict = {
        'frame_number' : body.frame_number,
        'batch_size' : body.batch_size,
        'image_height' : body.image_height,
        'image_width' : body.image_width,
    }

    x = torch.tensor(json.loads(body.results_body), dtype=torch.float32).view(14, 3)
    results = app.package['model_inference'].get_results(x)

    response_dict = {**info_dict, **results}
    logger.info(f"Resuts: {json.dumps(response_dict, indent=4)}")
    return response_dict


@app.get("/api/about")
def show_about():
    """
    Provide deployment information for debugging
    """
    
    def bash(command):
        output = os.popen(command).read()
        return output
    
    system_dict = {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }

    config_dict = CONFIG
    response_dict = {
        'system_info' : system_dict, 
        'config_info' : config_dict
    }

    return response_dict


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9001, reload=True, log_config='app/log.ini')