"""FastAPI boundary: authenticated, bounded URL input and real model inference."""
from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import hmac
from ipaddress import ip_address
import json
import logging
import threading
import uuid

from fastapi import FastAPI, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from inference.clinical_review import build_clinical_review
from inference.clinical_workflow import (
    CareEvaluateRequest,
    ClinicalWorkflowStore,
    CompareRequest,
    IntakeRequest,
    evaluate_care_rules,
)
from inference.hospital_system import (
    AgentQueryRequest,
    HospitalImageUploadRequest,
    HospitalSystemStore,
    PatientUpsertRequest,
    PortalChatRequest,
)
from inference.image_io import InputImageError, URLFetchPolicy, fetch_and_decode_image
from inference.local_llm import LocalExplanationUnavailable
from inference.schemas import FundusResponse, ScreenRequest, failure_response
from inference.settings import Settings

LOGGER = logging.getLogger("retina.inference")
KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _local_report_request(request: Request) -> bool:
    if request.client is None or request.url.hostname not in LOOPBACK_HOSTS:
        return False
    try:
        if not ip_address(request.client.host).is_loopback:
            return False
    except ValueError:
        return False
    origin = request.headers.get("origin")
    return origin is None or origin == f"{request.url.scheme}://{request.url.netloc}"


def safe_json(payload, status_code=200):
    # Reject non-finite numbers even inside output dictionaries.
    json.dumps(payload, allow_nan=False)
    return JSONResponse(payload, status_code=status_code)


class RequestBoundary:
    """Read at most max_body_bytes before JSON parsing, including chunked bodies."""
    def __init__(self, app, max_body_bytes):
        self.app, self.max_body_bytes = app, max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def identified_send(message):
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []),
                            (b"x-request-id", request_id.encode()), (b"cache-control", b"no-store")]}
            await send(message)

        if scope["method"] == "POST" and scope["path"] == "/v1/screen_fundus":
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > self.max_body_bytes:
                    response = safe_json(failure_response("input_error", "请求正文超过大小限制。", request_id,
                                                         reason="request_too_large"), 413)
                    return await response(scope, receive, identified_send)
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
            consumed = False

            async def buffered_receive():
                nonlocal consumed
                if not consumed:
                    consumed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            return await self.app(scope, buffered_receive, identified_send)
        return await self.app(scope, receive, identified_send)


def create_app(settings=None, *, engine=None, fetcher=None, visual_origin=None, report_explainer=None):
    settings = settings or Settings.from_env()
    policy = URLFetchPolicy(**settings.image_policy)
    gate = threading.BoundedSemaphore(settings.max_concurrent_requests)
    report_explanation_gate = threading.BoundedSemaphore(1)
    injected_engine = engine
    origin = visual_origin or settings.visual_origin
    delivery = None
    if origin:
        from inference.visual_delivery import VisualDelivery
        delivery = VisualDelivery(origin)
    workflow = ClinicalWorkflowStore(settings.workflow_db)
    hospital = HospitalSystemStore(settings.hospital_root, public_origin=origin)

    async def expire_views():
        while True:
            await asyncio.sleep(60)
            delivery.expire()

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = injected_engine
        if not settings.api_key:
            LOGGER.error("inference_disabled: RETINA_API_KEY missing")
        else:
            try:
                if app.state.engine is None:
                    import torch
                    from inference.engine import FundusEngine
                    from inference.registry import ModelRegistry
                    torch.set_num_threads(settings.cpu_threads)
                    registry = ModelRegistry.from_file(settings.registry, weights_root=settings.weights_root)
                    research = None
                    if settings.research_root is not None:
                        from inference.research_experts import ResearchExperts
                        device = settings.device if settings.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
                        research = ResearchExperts(settings.research_root, device=device)
                    app.state.engine = FundusEngine(registry, device=settings.device,
                                                   precision=settings.precision, routing=settings.routing,
                                                   research=research)
                await run_in_threadpool(app.state.engine.load)
            except Exception as error:
                app.state.engine = None
                LOGGER.error("model_startup_failed: %s", type(error).__name__)
        cleaner = asyncio.create_task(expire_views()) if delivery is not None else None
        try:
            yield
        finally:
            if cleaner is not None:
                cleaner.cancel()
                await asyncio.gather(cleaner, return_exceptions=True)
            workflow.close()
            hospital.close()
            app.state.engine = None

    app = FastAPI(title="Retinal screening inference", version="0.1.0", lifespan=lifespan,
                  docs_url="/docs" if settings.docs_enabled else None, redoc_url=None,
                  openapi_url="/openapi.json" if settings.docs_enabled else None)
    app.state.engine = injected_engine
    app.add_middleware(RequestBoundary, max_body_bytes=settings.max_body_bytes)

    def request_id(request):
        return getattr(request.state, "request_id", uuid.uuid4().hex)

    def authorized(key):
        return bool(settings.api_key and key and hmac.compare_digest(key.encode("utf-8"), settings.api_key.encode("utf-8")))

    def ready():
        return bool(settings.api_key and app.state.engine is not None and app.state.engine.is_ready)

    @app.get("/v1/view/{token}/{name}", include_in_schema=False)
    def view(token: str, name: str):
        from inference.visual_delivery import MEDIA
        assessment_record = None
        if delivery is not None and name == "report.html":
            linked_id = delivery.get_request_id(token)
            if linked_id:
                assessment_record = workflow.get_assessment(linked_id)
        payload = delivery.read(token, name, assessment_record=assessment_record) if delivery else None
        if payload is None:
            return Response(status_code=404)
        headers = {"Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff",
                   "X-Robots-Tag": "noindex, nofollow, noarchive",
                   "Content-Security-Policy": "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"}
        return Response(payload, media_type=MEDIA[name], headers=headers)

    @app.get("/v1/demo", include_in_schema=False)
    @app.get("/v1/portal", include_in_schema=False)
    def demo_workbench_page():
        from inference.portal_pages import render_dual_portal_html

        html_doc = render_dual_portal_html(ready=ready())
        headers = {
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "X-Robots-Tag": "noindex, nofollow, noarchive",
            "Content-Security-Policy": (
                "default-src 'self'; img-src 'self' data: blob:; "
                "style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                "connect-src 'self'; base-uri 'none'; form-action 'self'"
            ),
        }
        return Response(html_doc.encode("utf-8"), media_type="text/html; charset=utf-8", headers=headers)

    @app.get("/v1/imaging_room", include_in_schema=False)
    def imaging_room_page():
        from inference.portal_pages import render_imaging_room_html

        html_doc = render_imaging_room_html(ready=ready())
        headers = {
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "X-Robots-Tag": "noindex, nofollow, noarchive",
            "Content-Security-Policy": (
                "default-src 'self'; img-src 'self' data: blob:; "
                "style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                "connect-src 'self'; base-uri 'none'; form-action 'self'"
            ),
        }
        return Response(html_doc.encode("utf-8"), media_type="text/html; charset=utf-8", headers=headers)

    @app.get("/v1/demo/sample/{sample_id}", include_in_schema=False)
    def demo_sample_image(sample_id: str):
        from inference.demo_workbench import get_sample_image_bytes

        blob = get_sample_image_bytes(sample_id)
        if blob is None:
            return Response(status_code=404)
        return Response(blob, media_type="image/jpeg", headers={"X-Content-Type-Options": "nosniff"})

    @app.post("/v1/demo/run", include_in_schema=False)
    def demo_run_workflow(body: dict):
        from inference.demo_workbench import DemoRunRequest, run_demo_workflow

        try:
            req = DemoRunRequest.model_validate(body)
        except Exception:
            return safe_json({"ok": False, "message": "演示请求参数格式有误，请检查输入字段。"}, 422)
        if req.mode != "no_image":
            if not ready():
                return safe_json({"ok": False, "message": "推理模型尚未就绪，可先切换「无图咨询」体验分诊与科普。"}, 503)
            if not gate.acquire(blocking=False):
                return safe_json({"ok": False, "message": "服务正在处理另一张图像，请稍候片刻重试。"}, 429)
        try:
            data = run_demo_workflow(
                req,
                engine=app.state.engine,
                workflow=workflow,
                delivery=delivery,
                policy=policy,
                fetcher=fetcher,
            )
            return safe_json(data, 200)
        except InputImageError as error:
            return safe_json({"ok": False, "message": str(error)}, 422)
        except Exception as error:
            LOGGER.error("demo_run_failed: %s", type(error).__name__)
            return safe_json({"ok": False, "message": "演示执行失败，请检查输入图片或日志。"}, 500)
        finally:
            if req.mode != "no_image":
                gate.release()

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error):
        # Never echo a signed URL, image payload or API key through Pydantic details.
        if authorized(request.headers.get("x-api-key")):
            LOGGER.warning("invalid_screen_request request_id=%s body_type=%s has_image_url=%s validation=%s",
                           request_id(request), type(error.body).__name__,
                           isinstance(error.body, dict) and "image_url" in error.body,
                           sorted({issue["type"] for issue in error.errors()}))
        if request.url.path != "/v1/screen_fundus":
            return safe_json({"ok": False, "data": None,
                              "error": {"code": "invalid_request", "message": "请求参数校验失败，请检查字段名、类型或长度限制。"},
                              "request_id": request_id(request)}, 422)
        return safe_json(failure_response("input_error", "请求参数无效；仅接受 image_url 字段。",
                                        request_id(request), reason="invalid_request"), 422)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error):
        return safe_json(failure_response("error", "请求的接口或方法不可用。", request_id(request),
                                         reason="http_error"), error.status_code)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error):
        LOGGER.error("request_failed: %s", type(error).__name__)
        return safe_json(failure_response("error", "推理服务发生内部错误。", request_id(request),
                                         reason="internal_error"), 500)

    @app.get("/healthz", tags=["health"])
    def health():
        return {"alive": True}

    @app.get("/readyz", tags=["health"])
    def readiness():
        return safe_json({"ready": ready()}, 200 if ready() else 503)

    @app.get("/v1/models", tags=["models"])
    def models(request: Request, key: str | None = Security(KEY_HEADER)):
        if not authorized(key):
            return safe_json(failure_response("error", "接口认证失败。", request_id(request), reason="unauthorized"), 401)
        if not ready():
            return safe_json(failure_response("tool_unavailable", "模型尚未准备完成。", request_id(request)), 503)
        return {"models": app.state.engine.model_info()}

    @app.post("/v1/screen_fundus", response_model=FundusResponse, operation_id="screen_fundus", tags=["screening"],
              responses={code: {"model": FundusResponse} for code in (401, 413, 415, 422, 429, 500, 503)})
    def screen(body: ScreenRequest, request: Request, key: str | None = Security(KEY_HEADER)):
        rid = request_id(request)
        if not authorized(key):
            return safe_json(failure_response("error", "接口认证失败。", rid, reason="unauthorized"), 401)
        if not ready():
            return safe_json(failure_response("tool_unavailable", "模型尚未准备完成。", rid), 503)
        if not gate.acquire(blocking=False):
            response = failure_response("tool_unavailable", "服务正在处理其他图像，请稍后重试。", rid)
            response["warnings"].append("service_busy")
            return safe_json(response, 429)
        decoded = None
        try:
            decoded = (fetcher or fetch_and_decode_image)(body.image_url, policy=policy)
            response = app.state.engine.screen(decoded.image, request_id=rid, image_sha256=decoded.sha256)
            response["warnings"].extend(decoded.warnings)
            response = FundusResponse.model_validate(response).model_dump(exclude_unset=True)
            summary = response.get("report_summary")
            intake = {"eye": body.eye, "symptoms": body.symptoms, "history": body.history}
            if summary is not None:
                summary["clinical_decision"] = evaluate_care_rules(response, intake)["clinical_decision"]
            if delivery is not None and response["status"] in {"ok", "partial"}:
                try:
                    response["visual_report"] = delivery.create(
                        decoded.image, {**response, "_report_intake": intake}, decoded.sha256, rid
                    )
                except Exception as error:
                    LOGGER.warning("visual_report_unavailable request_id=%s reason=%s", rid, type(error).__name__)
                    response["warnings"].append("visual_report_unavailable")
            if summary is not None:
                summary["clinical_review"] = build_clinical_review(response)
                visual = response.get("visual_report") or {}
                if visual.get("reference_grade") is not None:
                    summary["clinical_decision"] = evaluate_care_rules(response, intake)["clinical_decision"]
            if response["status"] in {"ok", "partial", "quality_rejected"}:
                workflow.record_screening(
                    response,
                    eye=body.eye,
                    symptoms=body.symptoms,
                    history=body.history,
                )
            status_code = {"tool_unavailable": 503, "error": 500}.get(response["status"], 200)
            return safe_json(response, status_code)
        except InputImageError as error:
            LOGGER.warning("screen_image_rejected request_id=%s reason=%s", rid, error.code)
            code = error.code
            if decoded is not None and code in {"image_uniform", "image_fully_transparent"}:
                # AgentArts stops a conversation on non-2xx tool status. A decoded
                # photo with no visible detail is an input assessment, not a tool outage.
                from inference.report_summary import build_report_summary
                response = failure_response("input_error", str(error), rid, reason=code,
                                            image_sha256=decoded.sha256)
                skipped = response["results"]["quality"]
                response.update(contract_version="research-20261122.1",
                                research_results={name: dict(skipped) for name in ("multilabel_11", "glaucoma")})
                response["report_summary"] = build_report_summary(response["results"], response["research_results"])
                response["report_summary"]["clinical_review"] = build_clinical_review(response)
                intake = {"eye": body.eye, "symptoms": body.symptoms, "history": body.history}
                response["report_summary"]["clinical_decision"] = evaluate_care_rules(response, intake)["clinical_decision"]
                return safe_json(FundusResponse.model_validate(response).model_dump(exclude_unset=True))
            status = 413 if code in {"image_too_large", "image_pixel_limit"} else 415 if code in {
                "unsupported_image_format", "image_animated"} else 422
            return safe_json(failure_response("input_error", str(error), rid, reason=code), status)
        except Exception as error:
            LOGGER.error("inference_failed: %s", type(error).__name__)
            return safe_json(failure_response("error", "图像分析失败，请稍后重试或人工复核。", rid,
                                             reason="internal_error", image_sha256=decoded.sha256 if decoded else None), 500)
        finally:
            try:
                if decoded is not None:
                    decoded.image.close()
            finally:
                gate.release()

    @app.get("/v1/context", operation_id="get_context", tags=["workflow"])
    def get_context(request: Request, session_id: str = "default", key: str | None = Security(KEY_HEADER)):
        rid = request_id(request)
        if not authorized(key):
            return safe_json(failure_response("error", "接口认证失败。", rid, reason="unauthorized"), 401)
        data = workflow.get_context(session_id=session_id)
        return safe_json({"ok": True, "data": data, "error": None, "request_id": rid}, 200)

    @app.get("/v1/assessments/{assessment_id}", operation_id="get_assessment", tags=["workflow"])
    def get_assessment(assessment_id: str, request: Request, key: str | None = Security(KEY_HEADER)):
        rid = request_id(request)
        if not authorized(key):
            return safe_json(failure_response("error", "接口认证失败。", rid, reason="unauthorized"), 401)
        record = workflow.get_assessment(assessment_id)
        if record is None:
            return safe_json({"ok": False, "data": None,
                              "error": {"code": "assessment_not_found", "message": "未找到对应的筛查评估记录。"},
                              "request_id": rid}, 404)
        return safe_json({"ok": True, "data": record, "error": None, "request_id": rid}, 200)

    @app.post("/v1/intake", operation_id="record_intake", tags=["workflow"])
    def record_intake(body: IntakeRequest, request: Request, key: str | None = Security(KEY_HEADER)):
        rid = request_id(request)
        if not authorized(key):
            return safe_json(failure_response("error", "接口认证失败。", rid, reason="unauthorized"), 401)
        try:
            data = workflow.record_intake(body)
        except KeyError:
            return safe_json({"ok": False, "data": None,
                              "error": {"code": "assessment_not_found", "message": "未找到对应的筛查评估记录，请先上传眼底图完成筛查。"},
                              "request_id": rid}, 404)
        return safe_json({"ok": True, "data": data, "error": None, "request_id": rid}, 200)

    @app.post("/v1/care/evaluate", operation_id="evaluate_care", tags=["workflow"])
    def evaluate_care(body: CareEvaluateRequest, request: Request, key: str | None = Security(KEY_HEADER)):
        rid = request_id(request)
        if not authorized(key):
            return safe_json(failure_response("error", "接口认证失败。", rid, reason="unauthorized"), 401)
        try:
            plan = workflow.evaluate_care(body)
        except KeyError:
            return safe_json({"ok": False, "data": None,
                              "error": {"code": "assessment_not_found", "message": "未找到对应的筛查评估记录，请先上传眼底图完成筛查。"},
                              "request_id": rid}, 404)
        return safe_json({"ok": True, "data": plan, "error": None, "request_id": rid}, 200)

    @app.post("/v1/comparisons", operation_id="compare_assessments", tags=["workflow"])
    def compare_assessments(body: CompareRequest, request: Request, key: str | None = Security(KEY_HEADER)):
        rid = request_id(request)
        if not authorized(key):
            return safe_json(failure_response("error", "接口认证失败。", rid, reason="unauthorized"), 401)
        try:
            result = workflow.compare_assessments(body)
        except KeyError:
            return safe_json({"ok": False, "data": None,
                              "error": {"code": "assessment_not_found", "message": "至少有一条评估记录不存在，无法进行对比。"},
                              "request_id": rid}, 404)
        return safe_json({"ok": True, "data": result, "error": None, "request_id": rid}, 200)

    # --- Hospital imaging sync, dual-audience persistent reports & single-query AgentArts routes ---

    @app.get("/v1/his/patients", include_in_schema=False)
    def his_list_patients():
        return safe_json({"ok": True, "patients": hospital.list_patients_brief()}, 200)

    @app.get("/v1/his/patients/{phone}", include_in_schema=False)
    def his_get_patient(phone: str):
        overview = hospital.get_patient_overview(phone)
        if overview is None:
            return safe_json({"ok": False, "message": "未找到该手机号对应的患者档案。"}, 404)
        return safe_json({"ok": True, **overview}, 200)

    @app.post("/v1/his/patients", include_in_schema=False)
    def his_upsert_patient(body: PatientUpsertRequest):
        overview = hospital.upsert_patient(body)
        return safe_json({"ok": True, **overview}, 200)

    @app.post("/v1/his/images/upload", include_in_schema=False)
    def his_upload_image(body: HospitalImageUploadRequest):
        try:
            res = hospital.upload_hospital_image(body)
            return safe_json(res, 200)
        except (ValueError, InputImageError) as err:
            return safe_json({"ok": False, "message": str(err)}, 422)

    @app.get("/v1/his/reports/{report_id}/{name}", include_in_schema=False)
    def his_view_report(report_id: str, name: str, request: Request):
        if name == "patient.html":
            doc = hospital.render_patient_report_html(report_id)
            if doc is None:
                return Response(status_code=404)
            return Response(doc.encode("utf-8"), media_type="text/html; charset=utf-8",
                            headers={"X-Content-Type-Options": "nosniff"})
        if name == "doctor.html":
            explanation_label = report_explainer.label if report_explainer and _local_report_request(request) else None
            doc = hospital.render_doctor_report_html(
                report_id, report_explanation_label=explanation_label
            )
            if doc is None:
                return Response(status_code=404)
            return Response(doc.encode("utf-8"), media_type="text/html; charset=utf-8",
                            headers={"X-Content-Type-Options": "nosniff"})
        if name in {"original.jpg", "overlay.jpg"}:
            blob = hospital.read_report_media(report_id, name)
            if blob is None:
                return Response(status_code=404)
            return Response(blob, media_type="image/jpeg", headers={"X-Content-Type-Options": "nosniff"})
        return Response(status_code=404)

    @app.post("/v1/his/reports/{report_id}/report-explanation", include_in_schema=False)
    def his_report_explanation(report_id: str, request: Request):
        if report_explainer is None or not _local_report_request(request):
            return safe_json({"ok": False, "message": "请从本机医生端打开报告后生成解读。"}, 404)
        report = hospital.get_report(report_id)
        if report is None:
            return safe_json({"ok": False, "message": "未找到对应的眼底报告。"}, 404)
        if not report_explanation_gate.acquire(blocking=False):
            return safe_json({"ok": False, "message": "模型正在解读另一份报告，请稍后重试。"}, 429)
        try:
            result = report_explainer.explain(report["screening"], report["care_plan"]["clinical_decision"])
        except LocalExplanationUnavailable as error:
            return safe_json({"ok": False, "message": str(error) or "模型暂未完成解读，请稍后重试。"}, 503)
        finally:
            report_explanation_gate.release()
        return safe_json({"ok": True, **result})

    @app.post("/v1/his/portal/chat", include_in_schema=False)
    def his_portal_chat(body: PortalChatRequest):
        from inference.portal_pages import handle_portal_chat

        if not gate.acquire(blocking=False):
            return safe_json({"ok": False, "reply": "推理服务正忙，请稍候片刻再试。"}, 429)
        try:
            return safe_json(handle_portal_chat(hospital, body, app.state.engine), 200)
        except RuntimeError as err:
            return safe_json({"ok": False, "reply": str(err)}, 503)
        finally:
            gate.release()

    @app.post("/v1/his/agent/patient/session", tags=["hospital_agent"])
    def his_agent_patient_session(body: AgentQueryRequest, request: Request, key: str | None = Security(KEY_HEADER)):
        rid = request_id(request)
        if not authorized(key):
            return safe_json(failure_response("error", "接口认证失败。", rid, reason="unauthorized"), 401)
        from inference.portal_pages import handle_patient_session_query

        return safe_json(handle_patient_session_query(hospital, body.query), 200)

    @app.post("/v1/his/agent/patient/report", tags=["hospital_agent"])
    def his_agent_patient_report(body: AgentQueryRequest, request: Request, key: str | None = Security(KEY_HEADER)):
        rid = request_id(request)
        if not authorized(key):
            return safe_json(failure_response("error", "接口认证失败。", rid, reason="unauthorized"), 401)
        from inference.portal_pages import handle_patient_report_query

        if not gate.acquire(blocking=False):
            return safe_json({"ok": False, "reply": "当前推理队列正忙，请稍候几秒再试。"}, 429)
        try:
            return safe_json(handle_patient_report_query(hospital, body.query, app.state.engine), 200)
        except RuntimeError as err:
            return safe_json({"ok": False, "reply": str(err)}, 503)
        finally:
            gate.release()

    @app.post("/v1/his/agent/patient/symptom", tags=["hospital_agent"])
    def his_agent_patient_symptom(body: AgentQueryRequest, request: Request, key: str | None = Security(KEY_HEADER)):
        rid = request_id(request)
        if not authorized(key):
            return safe_json(failure_response("error", "接口认证失败。", rid, reason="unauthorized"), 401)
        from inference.portal_pages import handle_patient_symptom_query

        return safe_json(handle_patient_symptom_query(hospital, body.query), 200)

    @app.post("/v1/his/agent/doctor/patient", tags=["hospital_agent"])
    def his_agent_doctor_patient(body: AgentQueryRequest, request: Request, key: str | None = Security(KEY_HEADER)):
        rid = request_id(request)
        if not authorized(key):
            return safe_json(failure_response("error", "接口认证失败。", rid, reason="unauthorized"), 401)
        from inference.portal_pages import handle_doctor_patient_query

        return safe_json(handle_doctor_patient_query(hospital, body.query), 200)

    @app.post("/v1/his/agent/doctor/diagnose", tags=["hospital_agent"])
    def his_agent_doctor_diagnose(body: AgentQueryRequest, request: Request, key: str | None = Security(KEY_HEADER)):
        rid = request_id(request)
        if not authorized(key):
            return safe_json(failure_response("error", "接口认证失败。", rid, reason="unauthorized"), 401)
        from inference.portal_pages import handle_doctor_diagnose_query

        if not gate.acquire(blocking=False):
            return safe_json({"ok": False, "reply": "当前推理队列正忙，请稍候再试。"}, 429)
        try:
            return safe_json(handle_doctor_diagnose_query(hospital, body.query, app.state.engine), 200)
        except RuntimeError as err:
            return safe_json({"ok": False, "reply": str(err)}, 503)
        finally:
            gate.release()

    @app.post("/v1/his/agent/doctor/records", tags=["hospital_agent"])
    def his_agent_doctor_records(body: AgentQueryRequest, request: Request, key: str | None = Security(KEY_HEADER)):
        rid = request_id(request)
        if not authorized(key):
            return safe_json(failure_response("error", "接口认证失败。", rid, reason="unauthorized"), 401)
        from inference.portal_pages import handle_doctor_records_query

        return safe_json(handle_doctor_records_query(hospital, body.query), 200)

    return app
