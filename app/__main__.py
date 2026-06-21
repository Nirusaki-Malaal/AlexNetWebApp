from app import app, templates, LOGS
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi import File, UploadFile, Form
from typing import Annotated
from .plugins.image_helper import process_directory

@app.get("/")
async def _help_message(request: Request , response_class=HTMLResponse): ## used to configure response type to html instead of json  response_class=... btw
    return templates.TemplateResponse(request=request, name="index.html", ) ## context={"id": id} pass for templating jinja2

## BOTH METHODS WORK

# @app.post("/process")
# async def help_message(file : Annotated[UploadFile, File()]): ## the variable name here must match the one in form
#     contents = await file.read()
#     file_name = file.filename
#     with open(f"temp/{file_name}", "wb") as f:
#         f.write(contents)
        
#     return {"file_size" : f"temp/{file_name}"}

@app.post("/process")
async def help_message(request : Request):
    async with request.form() as form:
        filename_in_html = "file"
        file = form.get(filename_in_html) ## you should know the variable name there
        contents = await file.read()
        file_name = file.filename
    directory = f"temp/{file_name}"
    with open(directory, "wb") as f:
        f.write(contents)
    max_class = process_directory(directory)
    return {"maximum_class" : max_class}