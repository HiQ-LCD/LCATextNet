import os
from loguru import logger

# log storage path
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

LOG_LEVEL = "INFO"

logger.add(os.path.join(LOG_PATH, "eip_network.log"), level=LOG_LEVEL, rotation="10 MB", encoding="utf-8")

WORK_DIR = os.path.dirname(os.path.dirname(__file__))
logger.info(f"工作目录：{WORK_DIR}")

DATA_PATH = "/data"
logger.info(f"数据目录：{DATA_PATH}")

MODEL_ROOT_PATH = os.path.join(DATA_PATH, "model")
logger.info(f"模型目录：{MODEL_ROOT_PATH}")

# read environment variables from .env file
from dotenv import load_dotenv

env_file = os.path.join(WORK_DIR, '.env')
load_dotenv(env_file)

env_log_info = "Environment Variables:\n"
env_log_lines = []
for key, value in os.environ.items():
    env_log_line = f"{key}: {value}"
    env_log_lines.append(env_log_line)

env_log_lines.sort(key=lambda x: x.split(":")[0])
env_log_info += "\n---".join(env_log_lines)

print(env_log_info)

EMBEDDING_URL = os.getenv("EMBEDDING_URL")
