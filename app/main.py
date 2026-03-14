from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.routers.events import router as events_router
from app.routers.surveillance import router as surveillance_router
from app.routers.history import router as history_router
from app.routers.employees import router as employees_router
from app.routers.products import router as products_router
from app.routers.userstories import router as userstories_router
from app.routers.conversations import router as conversations_router
from app.routers.agent import router as agent_router
from app.routers.leads import router as leads_router
from app.routers.market import router as market_router
from app.routers.tickets import router as tickets_router
from app.routers.functions import router as functions_router
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
    version="0.5.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# ── CORS Middleware ───────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────
app.include_router(events_router, prefix="/api")
app.include_router(surveillance_router, prefix="/api")
app.include_router(history_router, prefix="/api")
app.include_router(employees_router, prefix="/api")
app.include_router(products_router, prefix="/api")
app.include_router(userstories_router, prefix="/api")
app.include_router(conversations_router, prefix="/api")
app.include_router(agent_router, prefix="/api")
app.include_router(leads_router, prefix="/api")
app.include_router(market_router, prefix="/api")
app.include_router(tickets_router, prefix="/api")
app.include_router(functions_router, prefix="/api")
# ── Core endpoints ────────────────────────────────────────────────────

@app.get("/api")
async def root():
    return {"message": "Nova AEI Backend API is running. Visit /api/docs for documentation."}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


# ── WebSocket ────────────────────────────────────────────────────────

@app.websocket("/api/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.register(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.unregister(ws)
