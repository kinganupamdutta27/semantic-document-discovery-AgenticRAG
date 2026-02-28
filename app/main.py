"""Main FastAPI Application."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.admin.database import init_admin_db
from app.admin.router import router as admin_api_router
from app.admin.runtime_config import seed_runtime_settings
from app.admin.service import seed_superadmin
from app.api.v1.router import api_router
from app.chatbot.agent.memory import init_memory_store, shutdown_memory_store
from app.chatbot.agent.rag import shutdown_checkpointer
from app.chatbot.exceptions import ChatbotError, LLMConnectionError
from app.core.config import settings
from app.core.hash_database import init_hash_db
from app.core.logging import logger
from app.core.scheduler import start_scheduler, stop_scheduler
from app.filesource.router import router as filesource_router
from app.filesource.scanner import register_filesource_scan_job
from app.sync_manager.router import router as sync_manager_router
from app.utils.hash_registry import sync_all_folders, load_all_files_to_vectorstore
from app.vectorstore.vectorstore import vsm, save_vectorstore

_STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting up...")

    # Initialize admin database, seed superadmin, and runtime settings
    logger.info("Initializing admin database...")
    init_admin_db()
    seed_superadmin()
    seed_runtime_settings()

    # Initialize in-memory hash database
    logger.info("Initializing hash registry database...")
    init_hash_db()

    # Restore the model mode from the admin panel setting
    try:
        from app.admin.runtime_config import rc
        use_remote = rc.get("use_remote_models", "false").lower() == "true"
        if use_remote:
            logger.info("Admin setting use_remote_models=true — switching to remote mode")
            vsm.switch_mode("remote")
        else:
            logger.info("Admin setting use_remote_models=false — staying in local mode")
    except Exception as exc:
        logger.warning("Could not read use_remote_models setting: %s — defaulting to local", exc)

    # Sync existing files with hash registry
    logger.info("Syncing existing files with hash registry...")
    sync_results = sync_all_folders(settings.BASE_DATA_FOLDER)
    for folder, count in sync_results.items():
        if count > 0:
            logger.info(f"Registered {count} files from '{folder}'")
    logger.info("Hash registry ready.")

    # Check if active vectorstore was loaded from disk or needs population
    active_vs = vsm.active
    if active_vs is not None and active_vs.index.ntotal == 0:
        logger.info("Active vector store is empty. Loading all files...")
        load_results = await load_all_files_to_vectorstore(settings.BASE_DATA_FOLDER)
        total_loaded = 0
        total_chunks = 0
        total_newly_processed = 0
        total_errors = 0
        for folder, result in load_results.items():
            loaded = result["loaded"]
            chunks = result["chunks"]
            newly_processed = result["newly_processed"]
            errors = len(result["errors"])
            total_loaded += loaded
            total_chunks += chunks
            total_newly_processed += newly_processed
            total_errors += errors
            if loaded > 0 or errors > 0:
                status = f"'{folder}': {loaded} files loaded, {chunks} chunks"
                if newly_processed > 0:
                    status += f", {newly_processed} newly processed"
                if errors > 0:
                    status += f", {errors} errors"
                logger.info(status)
                for err in result["errors"]:
                    logger.error(f"Error in {err['file']}: {err['error']}")
        logger.info(f"Vectorstore ready: {total_loaded} files, {total_chunks} chunks loaded.")
        if total_newly_processed > 0:
            logger.info(f"{total_newly_processed} files were newly processed and marked in registry")
        save_vectorstore()
    elif active_vs is not None:
        logger.info(f"Vector store loaded from disk with {active_vs.index.ntotal} chunks.")
    else:
        logger.warning("No active vector store available — rebuild may be required.")

    # Start the background scheduler for periodic sync tasks
    logger.info(f"Starting background scheduler for {settings.SYNC_INTERVAL_SECONDS} seconds interval...")
    start_scheduler()

    # Register file-source scan on the SAME scheduler and interval
    logger.info("Registering file-source auto-scan job...")
    register_filesource_scan_job()

    # Initialize long-term memory store (LangMem)
    logger.info("Initializing long-term memory store...")
    try:
        await init_memory_store()
        logger.info("Long-term memory store ready.")
    except Exception as exc:
        logger.warning("Long-term memory store init failed (non-fatal): %s", exc)

    yield
    # Shutdown
    logger.info("Shutting down...")

    # Persist and close long-term memory store
    logger.info("Persisting long-term memory store...")
    await shutdown_memory_store()

    # Close the conversation checkpointer connection
    logger.info("Closing conversation checkpointer...")
    await shutdown_checkpointer()

    # Stop the background scheduler (stops all jobs including file-source scan)
    stop_scheduler()

    logger.info("Saving vector stores to disk...")
    vsm.save_both()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="A chatbot api endpoints that understands the intent behind a user's query and retrives relevent documents/chunks  and exact pages/sections from an uploaded document repository, using semantic search and RAG.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    if _STATIC.exists():
        application.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    # Include public API router (chatbot, files, search, vector_db)
    application.include_router(api_router, prefix="/api/v1")

    # Include Sync Management API (protected by admin auth via middleware)
    application.include_router(
        sync_manager_router, prefix="/api/v1/sync", tags=["Sync Management"]
    )

    # Include File Source Configuration API (protected by admin auth via middleware)
    application.include_router(
        filesource_router, prefix="/api/v1/filesources", tags=["File Sources"]
    )

    # Include Admin API (auth, users, logs) under /chat/admin/api
    application.include_router(
        admin_api_router, prefix="/chat/admin/api", tags=["Admin"]
    )

    return application


app = create_application()


# ── Professional error pages ──────────────────────────────────────────

_ERROR_TEMPLATE: str | None = None

_ERROR_META = {
    400: ("Bad Request", "The server could not understand the request. Please check your input and try again."),
    401: ("Unauthorized", "You need to be authenticated to access this resource. Please log in and try again."),
    403: ("Forbidden", "You do not have permission to access this resource."),
    404: ("Page Not Found", "The page you are looking for doesn't exist or has been moved."),
    405: ("Method Not Allowed", "The request method is not supported for this endpoint."),
    422: ("Validation Error", "The request data did not pass validation. Please check your input."),
    500: ("Internal Server Error", "Something went wrong on our end. Please try again later."),
}


def _render_error_page(status_code: int, request: Request, detail: str | None = None) -> HTMLResponse:
    global _ERROR_TEMPLATE
    if _ERROR_TEMPLATE is None:
        tpl_path = _STATIC / "error.html"
        _ERROR_TEMPLATE = tpl_path.read_text() if tpl_path.exists() else "<h1>{{STATUS_CODE}}</h1><p>{{MESSAGE}}</p>"

    title, default_msg = _ERROR_META.get(status_code, ("Error", "An unexpected error occurred."))
    message = detail if detail and not detail.startswith("{") else default_msg
    html = (
        _ERROR_TEMPLATE
        .replace("{{STATUS_CODE}}", str(status_code))
        .replace("{{TITLE}}", title)
        .replace("{{MESSAGE}}", message)
        .replace("{{PATH}}", str(request.url.path))
    )
    return HTMLResponse(content=html, status_code=status_code)


def _is_api_request(request: Request) -> bool:
    """Return True for API calls (expect JSON, not HTML)."""
    path = request.url.path
    accept = request.headers.get("accept", "")
    if path.startswith("/api/") or path.startswith("/chat/admin/api/"):
        return True
    if "application/json" in accept and "text/html" not in accept:
        return True
    return False


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Return a styled HTML error page for browser requests, JSON for API calls."""
    if _is_api_request(request):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    detail = exc.detail if isinstance(exc.detail, str) else None
    return _render_error_page(exc.status_code, request, detail)


@app.exception_handler(RequestValidationError)
async def custom_validation_exception_handler(request: Request, exc: RequestValidationError):
    if _is_api_request(request):
        import json as _json

        from fastapi.responses import JSONResponse

        def _safe(obj):
            if isinstance(obj, bytes):
                return obj.decode(errors="replace")
            if isinstance(obj, Exception):
                return str(obj)
            raise TypeError
        return JSONResponse(
            status_code=422,
            content=_json.loads(_json.dumps({"detail": exc.errors()}, default=_safe)),
        )
    return _render_error_page(422, request)


@app.exception_handler(ChatbotError)
async def chatbot_error_handler(request: Request, exc: ChatbotError):
    """Map chatbot/RAG errors to structured JSON responses."""
    logger.error("ChatbotError on %s: [%s] %s", request.url.path, exc.error_code, exc.message)
    status_code = 502 if isinstance(exc, LLMConnectionError) else 500
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": {
                "message": exc.message,
                "error_code": exc.error_code,
            }
        },
    )


@app.exception_handler(Exception)
async def custom_generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    if _is_api_request(request):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    return _render_error_page(500, request)


# ── Admin auth middleware for existing sync & filesource APIs ──────────


@app.middleware("http")
async def admin_api_auth_middleware(request: Request, call_next):
    """Protect admin pages and admin-only APIs with JWT verification.

    - Admin HTML pages  (/chat/admin/* except /chat/admin/login)
      → redirect to login page when unauthenticated.
    - Admin-scoped APIs (/api/v1/sync/*, /api/v1/filesources/*)
      → return 401 JSON when unauthenticated.
    - Everything else (chat, chatbot API, files, search, health, docs)
      → untouched.
    """
    path = request.url.path

    # Paths that never require auth
    public_paths = ("/chat/admin/login", "/chat/admin/api/auth/login")
    if any(path == p for p in public_paths):
        return await call_next(request)

    # Determine if this path needs protection
    protected_api_prefixes = ("/api/v1/sync", "/api/v1/filesources")
    is_admin_page = (
        (path == "/chat/admin" or path.startswith("/chat/admin/"))
        and not path.startswith("/chat/admin/api/")
        and path != "/chat/admin/login"
    )
    is_protected_api = any(path.startswith(p) for p in protected_api_prefixes)

    if is_admin_page or is_protected_api:
        from app.admin.security import decode_admin_token

        token = None
        cookie_token = request.cookies.get("admin_token")
        if cookie_token:
            token = cookie_token
        else:
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                token = auth[7:]

        if not token or decode_admin_token(token) is None:
            if is_admin_page:
                login_url = "/chat/admin/login?next=" + path
                return RedirectResponse(url=login_url, status_code=302)
            else:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=401,
                    content={"detail": {"message": "Admin authentication required", "error_code": "AUTH_REQUIRED"}},
                )

    response = await call_next(request)
    return response


# ── Public routes ─────────────────────────────────────────────────────


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - redirect to chat page."""
    return {"message": "Welcome to Semantic Document Discovery", "chat_url": "/chat"}


@app.get("/chat", tags=["Chat_Frontend"])
async def chat_page():
    """Serve the chat interface page."""
    chat_file = _STATIC / "chat.html"
    if chat_file.exists():
        return FileResponse(chat_file, media_type="text/html")
    return {"error": "Chat page not found"}


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ── Admin page routes (all under /chat/admin/*) ──────────────────────


@app.get("/chat/admin/login", tags=["Admin Pages"])
async def admin_login_page():
    """Serve the admin login page."""
    page = _STATIC / "admin-login.html"
    if page.exists():
        return FileResponse(page, media_type="text/html")
    return {"error": "Login page not found"}


@app.get("/chat/admin/sync", tags=["Admin Pages"])
async def admin_sync_page():
    """Serve the sync administration panel."""
    page = _STATIC / "admin-sync.html"
    if page.exists():
        return FileResponse(page, media_type="text/html")
    return {"error": "Sync admin page not found"}


@app.get("/chat/admin/filesource", tags=["Admin Pages"])
async def admin_filesource_page():
    """Serve the file source configuration panel."""
    page = _STATIC / "admin-filesource.html"
    if page.exists():
        return FileResponse(page, media_type="text/html")
    return {"error": "File source admin page not found"}


@app.get("/chat/admin/users", tags=["Admin Pages"])
async def admin_users_page():
    """Serve the user management page."""
    page = _STATIC / "admin-users.html"
    if page.exists():
        return FileResponse(page, media_type="text/html")
    return {"error": "Users admin page not found"}


@app.get("/chat/admin/logs", tags=["Admin Pages"])
async def admin_logs_page():
    """Serve the activity log viewer."""
    page = _STATIC / "admin-logs.html"
    if page.exists():
        return FileResponse(page, media_type="text/html")
    return {"error": "Logs admin page not found"}


@app.get("/chat/admin/settings", tags=["Admin Pages"])
async def admin_settings_page():
    """Serve the runtime settings page."""
    page = _STATIC / "admin-settings.html"
    if page.exists():
        return FileResponse(page, media_type="text/html")
    return {"error": "Settings admin page not found"}


# ── Backward-compatible redirects ────────────────────────────────────


@app.get("/chat/admin", tags=["Admin Redirects"])
async def redirect_old_admin(request: Request):
    """Redirect /chat/admin to /chat/admin/sync (auth enforced by middleware)."""
    return RedirectResponse(url="/chat/admin/sync", status_code=302)


@app.get("/admin/filesource-config", tags=["Admin Redirects"])
async def redirect_old_filesource():
    """Redirect legacy /admin/filesource-config to /chat/admin/filesource."""
    return RedirectResponse(url="/chat/admin/filesource", status_code=302)


@app.get("/admin", tags=["Admin Redirects"])
@app.get("/admin/{rest:path}", tags=["Admin Redirects"])
async def redirect_admin_catchall():
    """Redirect any /admin* path to the admin login."""
    return RedirectResponse(url="/chat/admin/login", status_code=302)
