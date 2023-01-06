import json 
import traceback

from fastapi.logger import logger 
from fastapi import Request, status 
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from config import CONFIG

def get_error_response(request, exc) -> dict:
    """
    Genereic error handelling function 
    """
    error_response = {
        "error" : True, 
        "message" : str(exc)
    }

    if CONFIG['DEBUG']:
        error_response['traceback'] = "".join(
            traceback.format_exception(
                etype=type(exc), value=exc, tb=exc.__traceback__
            )
        )

    return error_response


# for handelling errors during the validation of incoming requests
async def validation_exception_handler(request : Request, exc: RequestValidationError):
    """
    Handelling error in validating requests
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=get_error_response(request, exc)
    )

# for handelling errors which might occur interanlly in the code 
async def python_exception_handler(request : Request, exc: Exception):
    """
    For handelling any internal python errors/exceptions
    """
    logger.error('Request info:\n' + json.dumps({
        "host": request.client.host,
        "method": request.method,
        "url": str(request.url),
        "headers": str(request.headers),
        "path_params": str(request.path_params),
        "query_params": str(request.query_params),
        "cookies": str(request.cookies)
    }, indent=4))

    return JSONResponse(
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=get_error_response(request, exc)
    )