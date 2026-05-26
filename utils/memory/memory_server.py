import argparse
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils.memory.db import MemorySystem


memory_system = None
server_args = None


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def probability(value):
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Run the shared MIND memory server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=positive_int, default=6001)
    parser.add_argument("--memory-dir", default="artifacts/memory/shared")
    parser.add_argument("--M", type=positive_int, default=300, help="Active memory size used by MR/MM.")
    parser.add_argument("--N", type=positive_int, default=100, help="Number of newly collected examples that trigger one MM update.")
    parser.add_argument("--embedding-model", default="siliconflow/BAAI/bge-m3", help="Embedding model used by MM.")
    parser.add_argument("--mm-alpha", type=nonnegative_float, default=1.0, help="Quality weight in MM.")
    parser.add_argument("--mm-beta", type=nonnegative_float, default=1.0, help="Diversity weight in MM.")
    parser.add_argument("--lambda-val", dest="lambda_val", type=probability, default=0.5)
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory_system
    try:
        memory_system = MemorySystem(
            persist_directory=server_args.memory_dir,
            active_cap=server_args.M,
            update_batch_size=server_args.N,
            embedding_model=server_args.embedding_model,
            mm_alpha=server_args.mm_alpha,
            mm_beta=server_args.mm_beta,
            lambda_val=server_args.lambda_val,
        )
        logging.info("Memory server initialized at %s", server_args.memory_dir)
    except Exception as error:
        logging.error("Failed to initialize memory system: %s", error)
        raise
    yield


app = FastAPI(title="MIND Memory Server", lifespan=lifespan)


class MemoryRecords(BaseModel):
    records: List[Dict[str, Any]]


@app.post("/history/add")
async def add_to_history(data: MemoryRecords):
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system is not initialized.")
    memory_system.add_to_history(data.records)
    return {"status": "success", "count": len(data.records)}


@app.post("/active/update")
async def update_active_memory(data: MemoryRecords):
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system is not initialized.")
    memory_system.update_active_memory(data.records)
    return {"status": "success", "count": len(data.records)}


@app.get("/active/all")
async def get_all_active():
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system is not initialized.")
    return memory_system.get_all_active()


def main():
    global server_args
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    server_args = parse_args()
    uvicorn.run(app, host=server_args.host, port=server_args.port)


if __name__ == "__main__":
    main()
