import os, logging, torch
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI
from starlette.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .plugins.model import Model, DEVICE

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE_NAME = "FastAPI@Log.txt"
RELATIVE_ROOT = "./app"


app = FastAPI()
app.mount("/static", StaticFiles(directory=f"{RELATIVE_ROOT}/static"), name="static")
model = Model(alpha=1e-4).to(DEVICE)
templates = Jinja2Templates(directory=str(Path(BASE_DIR, 'pages')))

if os.path.exists(str(Path(BASE_DIR, 'static', 'models', 'model.pth'))):
    model.load_state_dict(torch.load(str(Path(BASE_DIR, 'static', 'models', 'model.pth')), weights_only=True))
else:
    print(str(Path(BASE_DIR, 'static', 'models', 'model.pth')))
    raise Exception("Train the model first")

if os.path.exists(LOG_FILE_NAME):
    with open(LOG_FILE_NAME, "w") as file:
        pass

if not os.path.exists("./temp"):
    os.makedirs("temp")


logging.basicConfig( ## settings basic config for logger
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    handlers=[
        RotatingFileHandler(
            LOG_FILE_NAME,
            maxBytes=2097152000,
            backupCount=10
        ),
        logging.StreamHandler()
    ]
)

logging.getLogger("fastapi").setLevel(logging.INFO) ## setting logger to report fastAPI
LOGS = logging.getLogger(__name__) 
