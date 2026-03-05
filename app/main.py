from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.routers.events import router as events_router
from app.routers.surveillance import router as surveillance_router
from app.routers.history import router as history_router
from app.manager import manager
from app.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()         
    manager.load_events_from_db()    
    yield
    # Shutdown
    manager.stop()


app = FastAPI(
    title="Nova AEI — Nova Video Event Detection",
    description="Real-time webcam video recording, S3 upload, Amazon Nova analysis, and SQLite persistence",
    version="0.4.0",
    lifespan=lifespan,
)

# ── CORS Middleware ───────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Routers ───────────────────────────────────────────────────────────
app.include_router(events_router)
app.include_router(surveillance_router)
app.include_router(history_router)


# ── Core endpoints ────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ── WebSocket ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.register(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.unregister(ws)
