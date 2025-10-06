# vllm_launcher.py
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import uvicorn
from vllm.entrypoints.openai.api_server import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
