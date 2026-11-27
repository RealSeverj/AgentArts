"""Hospital imaging room page, patient/doctor dual portal UI, and single-query agent workflow handlers."""
from __future__ import annotations

from typing import Any

from inference.clinical_workflow import _coerce_str_list, _detect_urgent_symptoms, _extract_predictions
from inference.hospital_system import (
    EYE_CN,
    EYE_MEDICAL,
    HospitalSystemStore,
    PortalChatRequest,
    extract_eye,
    extract_phone,
)
from inference.report_levels import display_report_level, report_level_css


def handle_patient_session_query(store: HospitalSystemStore, query: str) -> dict[str, Any]:
    phone = extract_phone(query) or store.get_active_phone("patient")
    if not phone:
        return {
            "ok": True,
            "bound": False,
            "message": "请您先告诉我您的手机号（例如 13800138001），我帮您验证并调出您的专属眼健康档案。",
        }
    overview = store.get_patient_overview(phone)
    if overview is None:
        return {
            "ok": True,
            "bound": False,
            "phone": phone,
            "message": f"抱歉，医院系统里暂时还没有查到手机号 {phone} 的建档记录，您可以先去医院拍片室或通过拍片室系统完成建档同步。",
        }
    store.set_active_phone("patient", phone)
    p = overview["patient"]
    pending = [EYE_CN.get(e, e) for e in overview["pending_eyes"]]
    pending_tip = f"医院刚刚新同步了您的{'、'.join(pending)}检查照片，如果您想看解读随时告诉我。" if pending else ""
    return {
        "ok": True,
        "bound": True,
        "phone": phone,
        "patient_name": p["name"],
        "chronic_history": p["chronic_history"],
        "eye_history": p["eye_history"],
        "pending_eyes": overview["pending_eyes"],
        "available_eyes": overview["available_eyes"],
        "message": (
            f"{p['name']}您好！我是您的个人眼健康管家。我已经帮您关联好健康档案"
            f"（既往记录：{p['chronic_history'] or '暂无特殊慢病'}）。"
            f"{pending_tip}请问今天有什么可以帮您的？无论是查检查报告、看眼睛恢复进展，还是咨询日常护眼或突然的不适，我都可以陪您聊聊。"
        ),
    }


def handle_patient_report_query(store: HospitalSystemStore, query: str, engine: Any) -> dict[str, Any]:
    phone = extract_phone(query) or store.get_active_phone("patient")
    if not phone:
        return {
            "ok": True,
            "bound": False,
            "has_image": False,
            "reply": "查报告前，请您先告诉我您建档用的手机号，我这就帮您看一眼医院最新的检查照片。",
        }
    store.set_active_phone("patient", phone)
    eye = extract_eye(query)
    try:
        res = store.analyze_or_fetch_eye(phone=phone, eye=eye, engine=engine)
    except KeyError:
        return {"ok": False, "bound": False, "reply": f"系统里暂时没有找到 {phone} 的档案记录。"}
    reply = store.build_patient_reply(res)
    rep = res.get("report")
    decision = ((rep or {}).get("care_plan") or {}).get("clinical_decision") or {}
    return {
        "ok": True,
        "bound": True,
        "phone": phone,
        "patient_name": res["patient"]["name"],
        "eye": res.get("eye"),
        "has_image": res.get("has_image"),
        "has_new_image": res.get("has_new_image"),
        "patient_report_url": rep["urls"]["patient_report_url"] if rep else None,
        "clinical_level": decision.get("level"),
        "comparison": rep.get("comparison") if rep else None,
        "reply": reply,
    }


def handle_patient_symptom_query(store: HospitalSystemStore, query: str) -> dict[str, Any]:
    phone = extract_phone(query) or store.get_active_phone("patient")
    patient_ctx: dict[str, Any] = {"bound": False}
    eye_summaries: list[str] = []
    if phone:
        overview = store.get_patient_overview(phone)
        if overview is not None:
            store.set_active_phone("patient", phone)
            p = overview["patient"]
            patient_ctx = {
                "bound": True,
                "phone": phone,
                "name": p["name"],
                "age": p["age"],
                "chronic_history": p["chronic_history"],
                "eye_history": p["eye_history"],
            }
            seen_eyes: set[str] = set()
            for r in overview["reports"]:
                if r["eye"] in seen_eyes:
                    continue
                seen_eyes.add(r["eye"])
                rep = store.get_report(r["report_id"])
                if not rep:
                    continue
                _, reached, dr_grade = _extract_predictions(rep["screening"])
                eye_cn = EYE_CN.get(r["eye"], r["eye"])
                eye_summaries.append(
                    f"{eye_cn}最近记录({rep['analyzed_utc'][:10]})：糖网{dr_grade if dr_grade is not None else '--'}级，"
                    f"关注项:{'、'.join(sorted(reached)) if reached else '未见明显异常'}"
                )

    urgent_hits = _detect_urgent_symptoms(_coerce_str_list(query))
    return {
        "ok": True,
        "patient": patient_ctx,
        "recent_eye_summaries": eye_summaries,
        "urgent_detected": bool(urgent_hits),
        "urgent_hits": urgent_hits,
        "guidance": (
            "已触发紧急眼底预警症状，无论既往报告结果如何，请立刻建议患者停止剧烈活动并尽快前往眼科急诊排查视网膜裂孔、脱离或眼底出血。"
            if urgent_hits
            else "暂未命中急性危重红线，请结合患者上述个人慢病史与最近眼底记录给出温和、贴切的解释和复查建议。"
        ),
    }


def handle_doctor_patient_query(store: HospitalSystemStore, query: str) -> dict[str, Any]:
    phone = extract_phone(query) or store.get_active_phone("doctor")
    if not phone:
        return {
            "ok": True,
            "bound": False,
            "message": "请输入本次需要会诊的患者手机号（例如 13800138001），系统将自动调取患者档案与院内眼底影像。",
        }
    overview = store.get_patient_overview(phone)
    if overview is None:
        return {
            "ok": True,
            "bound": False,
            "phone": phone,
            "message": f"未查询到手机号 {phone} 的患者档案，请核对手机号或通知拍片室先完成建档传图。",
        }
    store.set_active_phone("doctor", phone)
    p = overview["patient"]
    left_imgs = [im for im in overview["images"] if im["eye"] == "left"]
    right_imgs = [im for im in overview["images"] if im["eye"] == "right"]

    def _eye_status(lst: list[dict[str, Any]], label: str) -> str:
        if not lst:
            return f"{label}（未拍摄）"
        pending_n = sum(1 for x in lst if x["status"] == "pending")
        if pending_n:
            return f"{label}（共{len(lst)}次，含{pending_n}张待分析新影像）"
        return f"{label}（共{len(lst)}次，均已分析归档）"

    return {
        "ok": True,
        "bound": True,
        "phone": phone,
        "patient": p,
        "available_eyes": overview["available_eyes"],
        "pending_eyes": overview["pending_eyes"],
        "images": overview["images"],
        "reports": overview["reports"],
        "message": (
            f"您正在会诊：**{p['name']}**（{p['gender']}，{p['age']}岁，手机号 {p['phone']}）\n"
            f"- **既往慢病史**：{p['chronic_history'] or '未记录'}\n"
            f"- **既往眼科史**：{p['eye_history'] or '未记录'}\n"
            f"- **院内影像同步情况**：{_eye_status(left_imgs, '左眼')} / {_eye_status(right_imgs, '右眼')}"
        ),
    }


def handle_doctor_diagnose_query(store: HospitalSystemStore, query: str, engine: Any) -> dict[str, Any]:
    phone = extract_phone(query) or store.get_active_phone("doctor")
    if not phone:
        return {
            "ok": True,
            "bound": False,
            "reply": "请先输入会诊患者的手机号，再指定需要辅助诊断的左眼或右眼。",
        }
    store.set_active_phone("doctor", phone)
    eye = extract_eye(query)
    try:
        res = store.analyze_or_fetch_eye(phone=phone, eye=eye, engine=engine)
    except KeyError:
        return {"ok": False, "bound": False, "reply": f"未查询到患者 {phone} 的院内档案。"}
    reply = store.build_doctor_reply(res)
    rep = res.get("report")
    decision = ((rep or {}).get("care_plan") or {}).get("clinical_decision")
    return {
        "ok": True,
        "bound": True,
        "phone": phone,
        "patient_name": res["patient"]["name"],
        "eye": res.get("eye"),
        "has_image": res.get("has_image"),
        "has_new_image": res.get("has_new_image"),
        "doctor_report_url": rep["urls"]["doctor_report_url"] if rep else None,
        "clinical_decision": decision,
        "clinical_level": display_report_level(decision) if rep else None,
        "comparison": rep.get("comparison") if rep else None,
        "reply": reply,
    }


def handle_doctor_records_query(store: HospitalSystemStore, query: str) -> dict[str, Any]:
    phone = extract_phone(query) or store.get_active_phone("doctor")
    if not phone:
        return {"ok": True, "bound": False, "reply": "请先提供需要调阅报告的患者手机号。"}
    overview = store.get_patient_overview(phone)
    if overview is None:
        return {"ok": True, "bound": False, "reply": f"未找到患者 {phone} 的建档记录。"}
    store.set_active_phone("doctor", phone)
    eye = extract_eye(query)
    reports = [r for r in overview["reports"] if eye is None or r["eye"] == eye]
    p = overview["patient"]
    if not reports:
        eye_txt = EYE_MEDICAL.get(eye or "", "眼底")
        return {
            "ok": True,
            "bound": True,
            "phone": phone,
            "reports": [],
            "reply": f"患者 {p['name']}（{phone}）名下暂无已完成的{eye_txt}分析报告；如有待分析影像请指令执行辅助诊断。",
        }
    lines = [f"为您调出患者 **{p['name']}**（{phone}）的归档临床辅助报告："]
    for idx, r in enumerate(reports[:4], 1):
        eye_txt = EYE_MEDICAL.get(r["eye"], r["eye"])
        grade_txt = f"DR {r['dr_grade']}级" if r.get("dr_grade") is not None else "分期未完成"
        level = display_report_level({"level": r.get("clinical_level")})
        decision_txt = " · ".join(filter(None, (level, r.get("clinical_title"))))
        details = "，".join(filter(None, (r["analyzed_utc"][:10], grade_txt, decision_txt)))
        lines.append(
            f"{idx}. {eye_txt}（{details}）："
            f"[打开临床报告]({r['doctor_report_url']})"
        )
    return {
        "ok": True,
        "bound": True,
        "phone": phone,
        "patient_name": p["name"],
        "reports": reports[:4],
        "reply": "\n".join(lines),
    }


def handle_portal_chat(store: HospitalSystemStore, req: PortalChatRequest, engine: Any) -> dict[str, Any]:
    """Unified interactive chat handler backing the /v1/demo patient & doctor dual portal."""
    q = (req.query or "").strip()
    phone = (req.phone or "").strip() or extract_phone(q)
    combined_q = f"{phone or ''} {q}".strip()

    if req.role == "doctor":
        if phone and req.action == "bind":
            return {"ok": True, "role": "doctor", **handle_doctor_patient_query(store, phone)}
        if not phone:
            phone = store.get_active_phone("doctor")
        if not phone:
            return {"ok": True, "role": "doctor", "reply": "请先在左侧选择或输入需要会诊的患者手机号。"}

        eye_arg = "" if req.eye == "auto" else ("左眼" if req.eye == "left" else "右眼")
        if req.action == "report_or_diagnose" or any(k in q for k in ("分析", "诊断", "判读", "新片", "辅助")):
            res = handle_doctor_diagnose_query(store, f"{phone} {eye_arg} {q}", engine)
            return {"ok": True, "role": "doctor", **res, "overview": store.get_patient_overview(phone)}
        if any(k in q for k in ("调出", "调阅", "报告", "记录", "历史", "对比")):
            res = handle_doctor_records_query(store, f"{phone} {eye_arg} {q}")
            return {"ok": True, "role": "doctor", **res, "overview": store.get_patient_overview(phone)}
        return {"ok": True, "role": "doctor", **handle_doctor_patient_query(store, f"{phone} {q}")}

    # Patient role
    if req.action == "bind":
        return {"ok": True, "role": "patient", **handle_patient_session_query(store, combined_q)}
    if phone:
        store.set_active_phone("patient", phone)
    else:
        phone = store.get_active_phone("patient")

    if req.action == "education" or any(k in q for k in ("青光眼", "白内障", "飞蚊症怎么", "注意什么", "科普", "怎么发生", "豹纹")):
        topic_reply = _patient_education_reply(q)
        return {"ok": True, "role": "patient", "phone": phone, "reply": topic_reply}

    if req.action == "symptom" or any(k in q for k in ("今早", "突然", "黑影", "闪光", "模糊", "疼", "痛", "遮挡", "怎么回事", "咋回事")):
        ctx = handle_patient_symptom_query(store, f"{phone or ''} {q}")
        p = ctx["patient"]
        name_prefix = f"{p['name']}您好，" if p.get("bound") else "您好，"
        history_note = (
            f"特别是结合您本身有「{p.get('chronic_history') or '慢性基础关注'}」以及既往检查底子（{'；'.join(ctx['recent_eye_summaries']) or '暂未见新异常'}），"
            if p.get("bound") else ""
        )
        if ctx["urgent_detected"]:
            reply = (
                f"{name_prefix}听到您描述的「{'、'.join(ctx['urgent_hits'])}」情况，这属于需要立刻重视的眼底警示信号！\n\n"
                f"突然出现眼前明显黑影飘动、闪光感或像被黑幕遮挡一样，可能是玻璃体剧烈牵拉视网膜，甚至需要排除视网膜裂孔或少量出血。{history_note}"
                "今天请先不要做剧烈运动、不要用力低头或揉眼睛，尽快让家人陪您去医院眼科看急诊、查一下散瞳眼底最稳妥。"
            )
        else:
            reply = (
                f"{name_prefix}感谢您把眼睛的感受告诉我。{history_note}"
                "目前从您的描述里没有听到特别紧急的视网膜脱离警示信号，但如果有持续视物模糊或看直线发弯，还是建议早点到眼科让医生当面查一下眼底和黄斑光学相干断层扫描（OCT），心里最踏实。"
            )
        return {"ok": True, "role": "patient", "phone": phone, "reply": reply, "symptom_context": ctx}

    eye_arg = "" if req.eye == "auto" else ("左眼" if req.eye == "left" else "右眼")
    if req.action == "report_or_diagnose" or any(k in q for k in ("报告", "情况", "左眼", "右眼", "解读", "结果", "检查", "新照片")):
        res = handle_patient_report_query(store, f"{phone or ''} {eye_arg} {q}", engine)
        return {"ok": True, "role": "patient", **res, "overview": store.get_patient_overview(phone) if phone else None}

    return {"ok": True, "role": "patient", **handle_patient_session_query(store, combined_q)}


def _patient_education_reply(query: str) -> str:
    if "青光眼" in query:
        return (
            "青光眼可以简单理解成眼球里面的“水压（眼压）”排不顺畅、压力升高，慢慢压迫到负责传图像的视神经。\n\n"
            "平常生活中建议您重点注意这几件小事：\n"
            "1. **不要关着灯长时间低头看手机或追剧**，黑暗环境瞳孔散大容易堵塞房水排水通道；\n"
            "2. **喝水少量多次**，不要一口气猛灌一大瓶水；\n"
            "3. **保持心情平稳**，少熬夜和长时间俯卧低头；\n"
            "4. 如果有青光眼家族史或超过 40 岁，每年体检时顺便测一下眼压。\n\n"
            "推荐延伸科普资源（后续可在华为云平台接通市场音视频检索 MCP 自动丰富）：\n"
            "- 国家卫健委健康中国专题：《青光眼早期几乎没感觉？定期查眼压和眼底最关键》\n"
            "- 科普视频检索建议关键词：`中华医学会 青光眼预防与日常护眼`"
        )
    if "豹纹" in query or "玻璃膜疣" in query:
        return (
            "体检单上写着“豹纹状眼底”，听起来像疾病名字，其实大多只是因为眼睛近视或者年龄增长，视网膜下面一层正常的脉络膜血管隐约透出来，看起来像浅棕色的纹理，**并不是发炎或长肿瘤**，定期每年检查一次眼底就好。\n\n"
            "如果写着“玻璃膜疣”，这是眼底深处随代谢出现的小小微黄沉着物，不等于立刻会得严重黄斑变性，去眼科做一次黄斑 OCT 确认稳定状态就安心了。"
        )
    return (
        "很高兴为您解答眼健康小常识！日常保护眼底微血管和视网膜最重要的是三件事：\n"
        "1. 管住血压和血糖波动，避免指标大起大落伤到眼底细小毛细血管；\n"
        "2. 高度近视或糖尿病朋友每年做 1 次眼底照相或散瞳检查；\n"
        "3. 户外强光下佩戴防紫外线墨镜，平时避免剧烈撞击眼部。如果您想了解具体某种眼病（如糖网、青光眼、飞蚊症、黄斑问题），随时告诉我！"
    )


def render_imaging_room_html(*, ready: bool) -> str:
    """Render the standalone Hospital Imaging Room sync page (/v1/imaging_room)."""
    status_text = "PACS 影像中心与六模型引擎就绪" if ready else "PACS 影像同步已就绪（模型未开启）"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>眼科拍片室影像采集与建档同步系统</title>
<style>
  *{{box-sizing:border-box}}
  body{{margin:0;background:#f1f5f9;color:#0f172a;font:14px/1.6 "PingFang SC","Microsoft YaHei",sans-serif}}
  .top{{background:#0f4c81;color:#fff;padding:15px 24px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}}
  .top h1{{margin:0;font-size:18px}}
  .nav a{{color:#fff;text-decoration:none;background:rgba(255,255,255,0.16);padding:7px 14px;border-radius:7px;font-size:13px;font-weight:600}}
  .nav a:hover{{background:rgba(255,255,255,0.28)}}
  .grid{{max-width:1220px;margin:20px auto;padding:0 18px;display:grid;grid-template-columns:460px 1fr;gap:18px;align-items:start}}
  .card{{background:#fff;border:1px solid #dbe3ee;border-radius:10px;padding:18px 20px;box-shadow:0 1px 4px rgba(15,23,42,.04)}}
  h2{{margin:0 0 12px;font-size:15.5px;color:#0a365c;border-bottom:1px solid #e2e8f0;padding-bottom:8px}}
  label{{display:block;font-weight:600;font-size:12.5px;margin:10px 0 4px}}
  input,select{{width:100%;padding:8px 10px;border:1px solid #cbd5e1;border-radius:6px;font:inherit}}
  .row2{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
  .btn{{border:none;background:#0f4c81;color:#fff;padding:10px 16px;border-radius:7px;font-weight:700;cursor:pointer;margin-top:12px;width:100%}}
  .btn:hover{{background:#0a365c}}
  .btn-sub{{background:#0284c7;margin-top:6px;width:auto;padding:7px 12px;font-size:12.5px}}
  .chips{{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}}
  .chip{{border:1px solid #cbd5e1;background:#f8fafc;border-radius:999px;padding:4px 10px;font-size:12px;cursor:pointer}}
  .chip:hover{{background:#e0f2fe;border-color:#7dd3fc}}
  table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}}
  th,td{{border:1px solid #e2e8f0;padding:7px 9px;text-align:left}}
  th{{background:#f8fafc}}
  .tag{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11.5px;font-weight:600}}
  .tag-pending{{background:#fef3c7;color:#92400e}}
  .tag-done{{background:#dcfce7;color:#166534}}
  .msg{{margin-top:10px;padding:10px 12px;border-radius:7px;font-size:13px;display:none}}
  .msg.ok{{display:block;background:#f0fdf4;border:1px solid #86efac;color:#166534}}
  .msg.err{{display:block;background:#fef2f2;border:1px solid #fca5a5;color:#991b1b}}
  @media (max-width:920px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header class="top">
  <div>
    <h1>医院眼科拍片室 · 眼底影像采集与云端同步系统</h1>
    <div style="font-size:12.5px;color:#cbd5e1;">输入患者手机号自动检索档案或建档，拍照上传左/右眼图像后自动进入医生与患者智能体待分析队列（{status_text}）</div>
  </div>
  <div class="nav">
    <a href="/v1/demo">返回医患双端智能体门户</a>
  </div>
</header>

<main class="grid">
  <section class="card">
    <h2>1. 患者身份确认与拍片上传</h2>
    <label>预置示例患者（点击快速代入）</label>
    <div class="chips" id="seedChips"></div>

    <label for="phoneInput">患者手机号（唯一档案标识）</label>
    <div style="display:flex;gap:8px;align-items:center;">
      <input id="phoneInput" value="13800138001" placeholder="输入11位手机号">
      <button type="button" class="btn btn-sub" id="lookupBtn" style="margin-top:0;white-space:nowrap;">检索/核验</button>
    </div>

    <div class="row2">
      <div><label for="nameInput">患者姓名（新手机号必填）</label><input id="nameInput" placeholder="例如：张建国"></div>
      <div><label for="ageInput">年龄</label><input id="ageInput" type="number" value="62"></div>
    </div>
    <div class="row2">
      <div>
        <label for="genderSelect">性别</label>
        <select id="genderSelect"><option value="男">男</option><option value="女">女</option><option value="未知">未知</option></select>
      </div>
      <div>
        <label for="eyeSelect">本次拍摄眼别</label>
        <select id="eyeSelect"><option value="left">左眼（OS）</option><option value="right">右眼（OD）</option></select>
      </div>
    </div>
    <label for="chronicInput">既往慢病史</label>
    <input id="chronicInput" placeholder="例如：2型糖尿病8年，高血压5年">
    <label for="eyeHistoryInput">既往眼科记录</label>
    <input id="eyeHistoryInput" placeholder="例如：半年前左眼曾查出中度糖网">

    <label for="sourceSelect">眼底相机图像来源</label>
    <select id="sourceSelect">
      <option value="sample">使用拍片室校准样片（用于快速演示）</option>
      <option value="upload">从本机选择真实眼底彩色照片上传</option>
    </select>
    <div id="sampleBox">
      <label for="sampleSelect">选择拍片室测试样片</label>
      <select id="sampleSelect">
        <option value="public_proxy">常规眼健康复查照（阴性/改善对照）</option>
        <option value="idrid_03">中重度糖网复查照（阳性病变对照）</option>
      </select>
    </div>
    <div id="uploadBox" style="display:none;">
      <label for="fileInput">选择眼底图像文件（JPEG / PNG / WEBP）</label>
      <input type="file" id="fileInput" accept="image/jpeg,image/png,image/webp">
    </div>
    <label for="noteInput">检查备注</label>
    <input id="noteInput" value="门诊眼科拍片室最新采集眼底照">

    <button type="button" class="btn" id="syncBtn">拍照采集并同步至患者档案库</button>
    <div id="syncMsg" class="msg"></div>
  </section>

  <section class="card">
    <h2>2. 该患者院内已同步影像与分析状态实时看板</h2>
    <div id="patientSummary" style="font-size:13px;color:#475569;margin-bottom:10px;">请输入手机号点击“检索/核验”查看档案详情。</div>
    <table>
      <thead><tr><th>拍摄时间</th><th>眼别</th><th>来源说明</th><th>状态</th><th>对应报告</th></tr></thead>
      <tbody id="imgTableBody"></tbody>
    </table>
  </section>
</main>

<script>
(() => {{
  const el = (id) => document.getElementById(id);
  let uploadB64 = null;

  function esc(s) {{
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
  }}

  async function loadBriefList() {{
    const r = await fetch("/v1/his/patients");
    const d = await r.json();
    const box = el("seedChips");
    box.innerHTML = "";
    for (const p of (d.patients || [])) {{
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.textContent = `${{p.name}} (${{p.phone}})`;
      b.onclick = () => {{ el("phoneInput").value = p.phone; lookupPatient(); }};
      box.appendChild(b);
    }}
  }}

  async function lookupPatient() {{
    const phone = el("phoneInput").value.trim();
    if (!phone) return;
    const r = await fetch(`/v1/his/patients/${{encodeURIComponent(phone)}}`);
    const body = el("imgTableBody");
    if (r.status === 404) {{
      el("nameInput").value = "";
      el("chronicInput").value = "";
      el("eyeHistoryInput").value = "";
      el("patientSummary").textContent = `手机号 ${{phone}} 尚未在医院建档。请在左侧填写姓名后直接上传拍片，系统将自动完成建档。`;
      body.innerHTML = "";
      return;
    }}
    const data = await r.json();
    const p = data.patient;
    el("nameInput").value = p.name || "";
    el("ageInput").value = p.age || 50;
    el("genderSelect").value = p.gender || "未知";
    el("chronicInput").value = p.chronic_history || "";
    el("eyeHistoryInput").value = p.eye_history || "";
    el("patientSummary").innerHTML = `<strong>${{esc(p.name)}}</strong>（${{esc(p.gender)}}，${{esc(p.age)}}岁）｜慢病史：${{esc(p.chronic_history || "无")}}`;

    body.innerHTML = "";
    for (const im of (data.images || [])) {{
      const isPending = im.status === "pending";
      const tag = isPending ? "<span class='tag tag-pending'>待智能体触发分析</span>" : "<span class='tag tag-done'>已出报告</span>";
      const links = im.report_urls
        ? `<a href="${{esc(im.report_urls.doctor_report_url)}}" target="_blank">医生版</a> / <a href="${{esc(im.report_urls.patient_report_url)}}" target="_blank">患者版</a>`
        : "--";
      body.innerHTML += `<tr><td>${{esc(im.captured_utc.slice(0,19).replace("T"," "))}}</td><td>${{im.eye === "left" ? "左眼" : "右眼"}}</td><td>${{esc(im.note)}}</td><td>${{tag}}</td><td>${{links}}</td></tr>`;
    }}
  }}

  el("sourceSelect").addEventListener("change", (e) => {{
    const isUp = e.target.value === "upload";
    el("uploadBox").style.display = isUp ? "block" : "none";
    el("sampleBox").style.display = isUp ? "none" : "block";
  }});

  el("fileInput").addEventListener("change", (ev) => {{
    const f = ev.target.files && ev.target.files[0];
    if (!f) return;
    const rd = new FileReader();
    rd.onload = () => {{ uploadB64 = rd.result; }};
    rd.readAsDataURL(f);
  }});

  el("lookupBtn").addEventListener("click", lookupPatient);

  el("syncBtn").addEventListener("click", async () => {{
    const msg = el("syncMsg");
    msg.className = "msg";
    msg.style.display = "none";
    const payload = {{
      phone: el("phoneInput").value.trim(),
      eye: el("eyeSelect").value,
      source: el("sourceSelect").value,
      sample_id: el("sampleSelect").value,
      image_base64: el("sourceSelect").value === "upload" ? uploadB64 : null,
      note: el("noteInput").value.trim(),
      name: el("nameInput").value.trim() || null,
      age: Number(el("ageInput").value) || null,
      gender: el("genderSelect").value,
      chronic_history: el("chronicInput").value.trim(),
      eye_history: el("eyeHistoryInput").value.trim(),
    }};
    const r = await fetch("/v1/his/images/upload", {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(payload),
    }});
    const d = await r.json();
    if (!r.ok || !d.ok) {{
      msg.textContent = d.message || (d.error && d.error.message) || "同步失败，请检查手机号（11位）和姓名/图片。";
      msg.className = "msg err";
      msg.style.display = "block";
      return;
    }}
    msg.innerHTML = `已成功将新拍摄的<strong>${{payload.eye === "left" ? "左眼" : "右眼"}}</strong>照片同步至患者名下（状态：待分析）。现在可以切回 <a href="/v1/demo">医患双端智能体门户</a> 直接指令分析或解读新报告！`;
    msg.className = "msg ok";
    msg.style.display = "block";
    await loadBriefList();
    await lookupPatient();
  }});

  loadBriefList();
  lookupPatient();
}})();
</script>
</body>
</html>"""


def render_dual_portal_html(*, ready: bool) -> str:
    """Render the unified Patient管家 + Doctor会诊台 portal at /v1/demo."""
    status_pill = "六模型眼底推理引擎在线就绪" if ready else "离线档案浏览模式（推理引擎加载中）"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>视健康·眼底筛查与随访多智能体演示工作台</title>
<style>
  :root{{--bg:#f3f6fa;--card:#fff;--ink:#0f172a;--muted:#475569;--line:#dbe3ee;--pri:#0f4c81;--pri2:#0a365c;--warm:#0d9488}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.65 "PingFang SC","Microsoft YaHei",sans-serif}}
  .topbar{{background:linear-gradient(120deg,#0a365c 0%,#0f4c81 65%,#0d9488 100%);color:#fff;padding:14px 24px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}}
  .topbar h1{{margin:0;font-size:18px}}
  .topbar p{{margin:2px 0 0;font-size:12px;color:#cbd5e1}}
  .top-actions{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
  .role-tabs{{display:inline-flex;background:rgba(15,23,42,.28);padding:3px;border-radius:8px;gap:4px}}
  .role-btn{{border:none;background:transparent;color:#e2e8f0;padding:7px 14px;border-radius:6px;font-weight:700;font-size:13px;cursor:pointer}}
  .role-btn.active{{background:#fff;color:var(--pri2)}}
  .ext-link{{color:#fff;text-decoration:none;border:1px solid rgba(255,255,255,.35);padding:6px 12px;border-radius:7px;font-size:12.5px;font-weight:600}}
  .ext-link:hover{{background:rgba(255,255,255,.15)}}
  .shell{{max-width:1360px;margin:18px auto 30px;padding:0 18px;display:grid;grid-template-columns:380px 1fr;gap:18px;align-items:start}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;box-shadow:0 1px 4px rgba(15,23,42,.04)}}
  .card+.card{{margin-top:14px}}
  h2{{margin:0 0 10px;font-size:15px;color:var(--pri2);border-bottom:1px solid var(--line);padding-bottom:7px}}
  label{{display:block;font-weight:600;font-size:12.5px;margin:9px 0 4px}}
  input,select,textarea{{width:100%;padding:8px 10px;border:1px solid #cbd5e1;border-radius:6px;font:inherit}}
  .btn{{border:none;background:var(--pri);color:#fff;padding:9px 14px;border-radius:7px;font-weight:700;cursor:pointer}}
  .btn:hover{{background:var(--pri2)}}
  .btn-warm{{background:var(--warm)}}
  .chips{{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}}
  .chip{{border:1px solid #cbd5e1;background:#f8fafc;color:#1e293b;padding:5px 10px;border-radius:999px;font-size:12px;cursor:pointer}}
  .chip:hover{{background:#e2e8f0}}
  .chat-box{{height:480px;overflow-y:auto;padding:14px;background:#f8fafc;border:1px solid var(--line);border-radius:9px;display:flex;flex-direction:column;gap:12px}}
  .bubble{{max-width:86%;padding:11px 15px;border-radius:10px;white-space:pre-wrap;line-height:1.7}}
  .bubble.user{{align-self:flex-end;background:#0f4c81;color:#fff}}
  .bubble.agent{{align-self:flex-start;background:#fff;border:1px solid #dbe3ee;color:#0f172a}}
  .bubble.agent a{{color:#0284c7;font-weight:700}}
  .composer{{display:flex;gap:8px;margin-top:10px}}
  .composer input{{flex:1}}
  .img-list{{font-size:12.5px;margin-top:8px;display:grid;gap:8px}}
  .img-row{{border:1px solid var(--line);border-radius:7px;padding:8px 10px;background:#fafcff;display:flex;justify-content:space-between;align-items:center;gap:8px}}
  .badge{{padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}}
  .b-pend{{background:#fef3c7;color:#92400e}}
  .b-done{{background:#dcfce7;color:#166534}}
  .level-chip{{display:inline-block;padding:1px 7px;border:1px solid;border-radius:999px;font-size:11px;font-weight:700;white-space:nowrap}}
  .img-report-status{{flex-shrink:0;text-align:right}}
  .img-report-status .level-chip{{margin-left:4px}}
  .bubble .level-chip{{margin-bottom:5px}}
  {report_level_css()}
  @media (max-width:960px){{.shell{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header class="topbar">
  <div>
    <h1>视健康·院内影像同步与医患双端智能体演示门户</h1>
    <p>手机号绑定院内影像中心 · 医生临床辅助诊断 + 患者温暖解读与问诊 · {status_pill}</p>
  </div>
  <div class="top-actions">
    <div class="role-tabs">
      <button type="button" class="role-btn active" data-role="patient">患者端：个人眼健康管家</button>
      <button type="button" class="role-btn" data-role="doctor">医生端：临床辅助会诊台</button>
    </div>
    <a class="ext-link" href="/v1/imaging_room">打开拍片室同步系统</a>
  </div>
</header>

<main class="shell">
  <section>
    <div class="card">
      <h2 id="leftTitle">1. 患者手机号登录（免传图关联档案）</h2>
      <div class="chips" id="seedPatients"></div>
      <label for="phoneInput" id="phoneLabel">您的手机号</label>
      <div style="display:flex;gap:8px;">
        <input id="phoneInput" value="13800138001">
        <button type="button" class="btn" id="bindBtn" style="white-space:nowrap;">确认关联</button>
      </div>
      <label for="eyeSelect">指定眼别（可选，不选则智能匹配新片）</label>
      <select id="eyeSelect">
        <option value="auto">智能匹配（优先待分析新片或最新记录）</option>
        <option value="left">左眼</option>
        <option value="right">右眼</option>
      </select>
      <div id="patientProfileBox" style="margin-top:12px;font-size:13px;color:#334155;"></div>
    </div>

    <div class="card">
      <h2>2. 院内影像同步状态（来自医院拍片室）</h2>
      <div class="img-list" id="hisImagesBox">暂无记录。</div>
    </div>
  </section>

  <section class="card">
    <h2 id="chatTitle">视健康·患者端个人眼健康管家对话窗</h2>
    <div class="chat-box" id="chatBox"></div>

    <div class="chips" id="quickChips"></div>

    <div class="composer">
      <input id="chatInput" placeholder="输入您的问题，例如：能帮我看下左眼最新的检查情况吗？">
      <button type="button" class="btn" id="sendBtn">发送</button>
    </div>
  </section>
</main>

<script>
(() => {{
  const el = (id) => document.getElementById(id);
  let role = new URLSearchParams(window.location.search).get("role") === "doctor" ? "doctor" : "patient";

  function esc(s) {{
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
  }}

  function localDoctorReportUrl(url) {{
    const match = /^(?:https?:\\/\\/[^/]+)?(\\/v1\\/his\\/reports\\/[A-Za-z0-9_-]+\\/doctor\\.html)$/.exec(url);
    return match ? match[1] : url;
  }}

  function linkifyMarkdown(text) {{
    let safe = esc(text);
    safe = safe.replace(/\\*\\*(.+?)\\*\\*/g, "<strong>$1</strong>");
    safe = safe.replace(/\\[([^\\]]+)\\]\\((https?:\\/\\/[^)]+|\\/v1\\/[^)]+)\\)/g, (_match, label, url) => {{
      const href = role === "doctor" ? localDoctorReportUrl(url) : url;
      return `<a href="${{href}}" target="_blank" rel="noopener">${{label}}</a>`;
    }});
    return safe;
  }}

  function levelChip(level, title) {{
    if (!/^L[0-4]$/.test(level || "")) return "";
    return `<span class="level-chip" data-level="${{level}}" title="${{esc(title || "分诊级别")}}">${{level}}</span>`;
  }}

  function addBubble(who, text, level, title) {{
    const div = document.createElement("div");
    div.className = "bubble " + who;
    const chip = who === "agent" ? levelChip(level, title) : "";
    div.innerHTML = who === "agent" ? (chip ? chip + "<br>" : "") + linkifyMarkdown(text) : esc(text);
    el("chatBox").appendChild(div);
    el("chatBox").scrollTop = el("chatBox").scrollHeight;
  }}

  async function loadSeeds() {{
    const r = await fetch("/v1/his/patients");
    const d = await r.json();
    const box = el("seedPatients");
    box.innerHTML = "";
    for (const p of (d.patients || [])) {{
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.textContent = `${{p.name}} · ${{p.phone}}`;
      b.onclick = () => {{ el("phoneInput").value = p.phone; bindCurrent(); }};
      box.appendChild(b);
    }}
  }}

  function renderOverview(data) {{
    if (!data || !data.patient) return;
    const p = data.patient;
    el("patientProfileBox").innerHTML =
      `<div><strong>${{esc(p.name)}}</strong>（${{esc(p.gender)}}，${{esc(p.age)}}岁）</div>` +
      `<div>慢病史：${{esc(p.chronic_history || "无")}}</div>` +
      `<div>眼科史：${{esc(p.eye_history || "无")}}</div>`;
    const box = el("hisImagesBox");
    box.innerHTML = "";
    const reports = new Map((data.reports || []).map((report) => [report.report_id, report]));
    for (const im of (data.images || [])) {{
      const isPend = im.status === "pending";
      const badge = isPend ? '<span class="badge b-pend">今日新片·待分析</span>' : '<span class="badge b-done">已分析</span>';
      const report = reports.get(im.latest_report_id);
      const reportUrl = role === "doctor"
        ? localDoctorReportUrl(im.report_urls?.doctor_report_url)
        : im.report_urls?.patient_report_url;
      const referralChip = role === "doctor" && report && reportUrl
        ? levelChip(report.clinical_level ?? "L0", report.clinical_title || "待复核")
        : "";
      const repLink = reportUrl
        ? `<a href="${{esc(reportUrl)}}" target="_blank" rel="noopener">${{role === "doctor" ? "查看临床报告" : "查看报告"}}</a>`
        : "";
      box.innerHTML += `<div class="img-row"><div><strong>${{im.eye === "left" ? "左眼" : "右眼"}}</strong> · ${{esc(im.note)}}<br><span style="color:#64748b">${{esc(im.captured_utc.slice(0,19).replace("T"," "))}}</span></div><div class="img-report-status">${{badge}}${{referralChip}}<br>${{repLink}}</div></div>`;
    }}
  }}

  function refreshRoleUI() {{
    document.querySelectorAll(".role-btn").forEach((b) => b.classList.toggle("active", b.dataset.role === role));
    const chips = el("quickChips");
    chips.innerHTML = "";
    el("chatBox").innerHTML = "";
    if (role === "patient") {{
      el("leftTitle").textContent = "1. 患者手机号登录（免传图关联档案）";
      el("phoneLabel").textContent = "您的手机号";
      el("chatTitle").textContent = "视健康·患者端个人眼健康管家对话窗";
      el("chatInput").placeholder = "例如：能帮我看下左眼最新的检查情况吗？";
      addBubble("agent", "您好！请问您今天需要什么帮助呢？无论是眼科问题的答疑解惑、还是病情进展的追踪与解读，我都可以帮您回答！");
      const presets = [
        ["帮我看下左眼最新报告（自动测新片+比进展）", "report_or_diagnose", "left", "你能帮我看一下我左眼的最新报告吗？"],
        ["帮我看下右眼检查结果", "report_or_diagnose", "right", "你能帮我解读一下我的右眼当前是什么情况吗？"],
        ["今早起床突然眼前有黑影飘和闪光感", "symptom", "auto", "我今早起床突然觉得眼前有黑影飘，还有点闪光感，这是咋回事呀？"],
        ["青光眼一般是怎么发生的？平常注意什么", "education", "auto", "青光眼一般是怎么发生的呀，平常要注意些什么呢？"],
      ];
      for (const [label, action, eye, q] of presets) {{
        const b = document.createElement("button");
        b.type = "button";
        b.className = "chip";
        b.textContent = label;
        b.onclick = () => sendChat(q, action, eye);
        chips.appendChild(b);
      }}
    }} else {{
      el("leftTitle").textContent = "1. 会诊患者检索（医生端工作站）";
      el("phoneLabel").textContent = "请输入会诊患者手机号";
      el("chatTitle").textContent = "视健康·医生端临床辅助会诊对话窗";
      el("chatInput").placeholder = "例如：请你辅助诊断一下病人左眼的病情";
      addBubble("agent", "医生您好，请输入需要会诊的患者手机号，系统将自动调取院内拍片室同步影像。");
      const presets = [
        ["开始会诊当前患者", "bind", "auto", "查询会诊患者档案"],
        ["辅助诊断患者左眼待分析影像", "report_or_diagnose", "left", "请你辅助诊断一下病人左眼的病情"],
        ["调出患者全部已归档临床报告", "auto", "auto", "请你帮我调出病人已经完成的临床报告"],
      ];
      for (const [label, action, eye, q] of presets) {{
        const b = document.createElement("button");
        b.type = "button";
        b.className = "chip";
        b.textContent = label;
        b.onclick = () => sendChat(q, action, eye);
        chips.appendChild(b);
      }}
    }}
  }}

  async function bindCurrent() {{
    const phone = el("phoneInput").value.trim();
    if (!phone) return;
    const r = await fetch(`/v1/his/patients/${{encodeURIComponent(phone)}}`);
    if (r.ok) {{
      renderOverview(await r.json());
    }}
    await sendChat(role === "doctor" ? `会诊患者 ${{phone}}` : `你好，我的手机号是 ${{phone}}`, "bind", el("eyeSelect").value);
  }}

  async function sendChat(text, action = "auto", eyeOverride = null) {{
    const q = (text || "").trim();
    if (!q) return;
    addBubble("user", q);
    el("sendBtn").disabled = true;
    try {{
      const payload = {{
        role,
        phone: el("phoneInput").value.trim() || null,
        action,
        eye: eyeOverride || el("eyeSelect").value,
        query: q,
      }};
      const r = await fetch("/v1/his/portal/chat", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify(payload),
      }});
      const d = await r.json();
      const isDoctorReport = role === "doctor" && d.role === "doctor" && d.doctor_report_url;
      const reportLevel = isDoctorReport ? d.clinical_level : null;
      addBubble("agent", d.reply || d.message || "已完成处理。", reportLevel, d.clinical_decision?.title);
      if (d.overview) {{
        renderOverview(d.overview);
      }} else if (d.patient || d.images) {{
        renderOverview({{patient: d.patient, images: d.images, reports: d.reports}});
      }}
    }} finally {{
      el("sendBtn").disabled = false;
    }}
  }}

  document.querySelectorAll(".role-btn").forEach((b) => {{
    b.addEventListener("click", () => {{
      role = b.dataset.role;
      refreshRoleUI();
      bindCurrent();
    }});
  }});
  el("bindBtn").addEventListener("click", bindCurrent);
  el("sendBtn").addEventListener("click", () => {{
    const v = el("chatInput").value;
    el("chatInput").value = "";
    sendChat(v);
  }});
  el("chatInput").addEventListener("keydown", (e) => {{
    if (e.key === "Enter") {{
      const v = el("chatInput").value;
      el("chatInput").value = "";
      sendChat(v);
    }}
  }});

  loadSeeds();
  refreshRoleUI();
  bindCurrent();
}})();
</script>
</body>
</html>"""
