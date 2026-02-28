"""API router for admin authentication, user management, and activity logs.

All endpoints are mounted under ``/chat/admin/api`` by main.py.
"""

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.admin.dependencies import get_current_admin, require_admin, require_superadmin
from app.admin.exceptions import (
    AdminBaseException,
    AuthenticationError,
    UserAlreadyExistsError,
    UserInactiveError,
    UserNotFoundError,
)
from app.admin.schemas import (
    ActivityLogListResponse,
    ActivityLogOut,
    AdminUserOut,
    CreateUserRequest,
    LoginRequest,
    MeResponse,
    TokenResponse,
    UpdateUserRequest,
    UserListResponse,
)
from app.admin.security import create_admin_token
from app.admin.service import (
    authenticate_user,
    create_user,
    delete_user,
    get_activity_logs,
    list_users,
    log_activity,
    update_user,
)
from app.core.config import settings

logger = logging.getLogger("app.admin")
router = APIRouter()

IST = timezone(timedelta(hours=5, minutes=30))


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── Auth ──────────────────────────────────────────────────────


@router.post(
    "/auth/login",
    response_model=TokenResponse,
    summary="Admin login",
    tags=["Admin Auth"],
)
async def login(body: LoginRequest, request: Request):
    """Authenticate and return a JWT."""
    ip = _client_ip(request)
    try:
        user = authenticate_user(body.username, body.password)
    except (AuthenticationError, UserInactiveError) as exc:
        log_activity(body.username, "LOGIN_FAILED", {"reason": exc.message}, ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": exc.message, "error_code": exc.error_code},
        )

    token = create_admin_token(data={"sub": user.username, "role": user.role})
    log_activity(user.username, "LOGIN", {"role": user.role}, ip)

    response = JSONResponse(
        content={
            "access_token": token,
            "token_type": "bearer",
            "username": user.username,
            "role": user.role,
            "expires_in_minutes": settings.ADMIN_TOKEN_EXPIRE_MINUTES,
        }
    )
    response.set_cookie(
        key="admin_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=settings.ADMIN_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    return response


@router.post("/auth/logout", summary="Admin logout", tags=["Admin Auth"])
async def logout(request: Request, admin=Depends(get_current_admin)):
    ip = _client_ip(request)
    log_activity(admin.username, "LOGOUT", None, ip)
    response = JSONResponse(content={"message": "Logged out"})
    response.delete_cookie("admin_token", path="/")
    return response


@router.get(
    "/auth/me",
    response_model=MeResponse,
    summary="Current admin info",
    tags=["Admin Auth"],
)
async def me(admin=Depends(require_admin)):
    return MeResponse(
        username=admin.username,
        role=admin.role,
        is_active=admin.is_active,
        created_at=admin.created_at,
    )


# ── User Management ───────────────────────────────────────────
# Both admin and superadmin can manage users.
# Guardrail: only superadmin may assign/change the "superadmin" role.


def _enforce_role_guardrail(caller_role: str, target_role: str | None) -> None:
    """Raise 403 if a non-superadmin tries to grant the superadmin role."""
    if target_role == "superadmin" and caller_role != "superadmin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": "Only a superadmin can assign the superadmin role", "error_code": "FORBIDDEN"},
        )


@router.get(
    "/users",
    response_model=UserListResponse,
    summary="List admin users",
    tags=["Admin Users"],
)
async def list_admin_users(admin=Depends(require_admin)):
    users = list_users()
    return UserListResponse(
        users=[
            AdminUserOut(
                id=u.id,
                username=u.username,
                role=u.role,
                is_active=u.is_active,
                created_at=u.created_at,
                updated_at=u.updated_at,
                created_by=u.created_by,
            )
            for u in users
        ],
        total=len(users),
    )


@router.post(
    "/users",
    response_model=AdminUserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create admin user",
    tags=["Admin Users"],
)
async def create_admin_user(body: CreateUserRequest, request: Request, admin=Depends(require_admin)):
    _enforce_role_guardrail(admin.role, body.role)
    ip = _client_ip(request)
    try:
        user = create_user(body.username, body.password, body.role, created_by=admin.username)
    except UserAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"message": exc.message, "error_code": exc.error_code})

    log_activity(admin.username, "CREATE_USER", {"target": body.username, "role": body.role}, ip)
    return AdminUserOut(
        id=user.id,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
        created_by=user.created_by,
    )


@router.put(
    "/users/{user_id}",
    response_model=AdminUserOut,
    summary="Update admin user",
    tags=["Admin Users"],
)
async def update_admin_user(user_id: int, body: UpdateUserRequest, request: Request, admin=Depends(require_admin)):
    _enforce_role_guardrail(admin.role, body.role)
    ip = _client_ip(request)
    try:
        user = update_user(
            user_id,
            password=body.password,
            role=body.role,
            is_active=body.is_active,
        )
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"message": exc.message, "error_code": exc.error_code})

    changes = {k: v for k, v in body.model_dump().items() if v is not None and k != "password"}
    if body.password is not None:
        changes["password"] = "***changed***"
    log_activity(admin.username, "UPDATE_USER", {"target_id": user_id, "changes": changes}, ip)

    return AdminUserOut(
        id=user.id,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
        created_by=user.created_by,
    )


@router.delete(
    "/users/{user_id}",
    summary="Deactivate admin user",
    tags=["Admin Users"],
)
async def deactivate_admin_user(user_id: int, request: Request, admin=Depends(require_admin)):
    ip = _client_ip(request)
    try:
        user = delete_user(user_id)
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"message": exc.message, "error_code": exc.error_code})

    log_activity(admin.username, "DEACTIVATE_USER", {"target_id": user_id, "username": user.username}, ip)
    return {"message": f"User '{user.username}' deactivated"}


# ── Activity Logs ─────────────────────────────────────────────


@router.get(
    "/logs",
    response_model=ActivityLogListResponse,
    summary="View admin activity logs",
    tags=["Admin Logs"],
)
async def view_logs(
    page: int = 1,
    page_size: int = 50,
    username: str = None,
    action: str = None,
    admin=Depends(require_admin),
):
    logs, total = get_activity_logs(page=page, page_size=page_size, username=username, action=action)
    return ActivityLogListResponse(
        logs=[
            ActivityLogOut(
                id=log.id,
                username=log.username,
                action=log.action,
                detail=log.detail,
                ip_address=log.ip_address,
                timestamp=log.timestamp,
            )
            for log in logs
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


# ── Runtime Settings ──────────────────────────────────────────


@router.get(
    "/settings",
    summary="Get all runtime settings",
    tags=["Admin Settings"],
)
async def get_settings(admin=Depends(require_admin)):
    from app.admin.runtime_config import rc
    return {"settings": rc.get_all(mask_secrets=True)}


@router.put(
    "/settings",
    summary="Update runtime settings",
    tags=["Admin Settings"],
)
async def update_settings(body: dict, request: Request, admin=Depends(require_admin)):
    from app.admin.runtime_config import rc

    updates = body.get("settings", {})
    if not updates or not isinstance(updates, dict):
        raise HTTPException(status_code=400, detail={"message": "Provide {\"settings\": {\"key\": \"value\", ...}}"})

    ip = _client_ip(request)

    from app.chatbot.agent.llm import force_reinit
    from app.vectorstore.vectorstore import vsm

    embedding_keys = {"embedding_provider", "embedding_model", "embedding_api_key", "embedding_dimensions"}
    toggle_key = "use_remote_models"
    model_keys = {
        toggle_key, "llm_provider", "llm_model_name",
        "llm_base_url", "llm_api_key", "llm_temperature",
    } | embedding_keys

    changed_keys = set(updates.keys())
    count = rc.bulk_set(updates, changed_by=admin.username)

    if model_keys & changed_keys:
        force_reinit()

    if toggle_key in changed_keys:
        new_mode = "remote" if updates[toggle_key] == "true" else "local"
        vsm.switch_mode(new_mode)

    if embedding_keys & changed_keys:
        vsm.rebuild_status["rebuild_required"] = True

    log_activity(admin.username, "UPDATE_SETTINGS", {"keys": list(changed_keys)}, ip)
    return {"message": f"{count} setting(s) updated", "updated": count}


# ── Model Configuration ──────────────────────────────────────


@router.get(
    "/settings/active-model",
    summary="Get active model configuration",
    tags=["Admin Settings"],
)
async def get_active_model(admin=Depends(require_admin)):
    from app.chatbot.agent.llm import get_active_config
    return get_active_config()


@router.post(
    "/settings/test-connection",
    summary="Test LLM and embedding model connectivity",
    tags=["Admin Settings"],
)
async def test_model_connection(request: Request, admin=Depends(require_admin)):
    """Invoke both the LLM and embedding model with a trivial input to verify connectivity."""
    ip = _client_ip(request)
    results: dict = {}

    try:
        from app.chatbot.agent.llm import get_model
        llm = get_model()
        response = await llm.ainvoke(
            [{"role": "user", "content": "Say hello in one word."}]
        )
        results["llm"] = {
            "status": "success",
            "response": str(response.content)[:200],
        }
    except Exception as exc:
        logger.warning("LLM test-connection failed: %s", exc)
        results["llm"] = {"status": "error", "message": str(exc)[:500]}

    try:
        from app.chatbot.agent.llm import get_embeddings
        emb = get_embeddings()
        vector = emb.embed_query("connectivity test")
        results["embeddings"] = {
            "status": "success",
            "dimensions": len(vector),
        }
    except Exception as exc:
        logger.warning("Embedding test-connection failed: %s", exc)
        results["embeddings"] = {"status": "error", "message": str(exc)[:500]}

    log_activity(admin.username, "TEST_MODEL_CONNECTION", results, ip)
    return results


# ── Vector Store Rebuild ──────────────────────────────────────


@router.get(
    "/settings/rebuild-status",
    summary="Get vector store rebuild status and index info",
    tags=["Admin Settings"],
)
async def rebuild_status(admin=Depends(require_admin)):
    from app.vectorstore.vectorstore import vsm
    return vsm.get_status()


@router.post(
    "/settings/rebuild-vectorstore",
    summary="Rebuild the vector store for the target mode",
    tags=["Admin Settings"],
)
async def rebuild_vectorstore(
    request: Request,
    body: dict = None,
    admin=Depends(require_admin),
):
    """Kick off a full vector store rebuild as a background task.

    Body (optional): ``{"target_mode": "remote"}``
    Defaults to the currently active mode.
    """
    from app.vectorstore.vectorstore import vsm

    if vsm.rebuild_status.get("status") == "running":
        raise HTTPException(status_code=409, detail={"message": "A rebuild is already in progress"})

    target_mode = (body or {}).get("target_mode", vsm.mode)
    if target_mode not in ("local", "remote"):
        raise HTTPException(status_code=400, detail={"message": f"Invalid target_mode: {target_mode}"})

    ip = _client_ip(request)
    log_activity(admin.username, "REBUILD_VECTORSTORE", {"target_mode": target_mode}, ip)

    import asyncio
    asyncio.create_task(vsm.rebuild(target_mode))

    return {
        "message": f"Rebuild started for {target_mode} mode. Poll /settings/rebuild-status for progress.",
        "target_mode": target_mode,
    }
