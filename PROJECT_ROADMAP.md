# 眼底筛查与随访多智能体系统：总体进展、技术架构与演进规划（唯一权威基线文档）

> 本文档整合了原《眼底筛查多智能体比赛设计方案.pdf》的核心技术路线与系统当前代码实现，后续开发、联调、演示与答辩均以本文档为唯一权威基线，不再依赖外部 PDF。

---

## 一、项目定位与三层受众扩展体系

### 1.1 核心产品定位
本系统是面向华为云 AgentArts（智果）平台的**眼底图像辅助筛查与全流程随访多智能体系统**。系统打通“图像质控 → 六模型专家与四类病灶推理 → 结构化证据沉淀 → 知识溯源解读 → 症状优先随访决策 → 动态 A4 报告与 PDF 导出 → 同眼跨期对照”的闭环链路。

**核心安全与医学边界**：
- 本系统承担辅助筛查、结构化证据整合、就医分诊与报告解读职责；**临床诊断与治疗方案决定由执业眼科医生完成**。
- 模型分类得分（0~1 原始分或 0~100 标度显示分）**未经人群后验概率校准**，严禁将模型得分读作“患病百分比概率”，亦不将“未达到研究筛查线”等同于“绝对正常、排除一切眼病”。

### 1.2 三层受众扩展设计（从专业糖网延伸至大众体检与慢病照护）
为了扩大系统受众并提升基层适用性，系统在统一冻结证据层（`EvidencePacket`）之上，支持三类受众的差异化沟通语态与无图咨询扩展：

| 受众类型 | 核心痛点与诉求 | 系统适配模式 |
| :--- | :--- | :--- |
| **1. 大众体检与无症状初筛人群** | 拿到体检眼底照或纸质报告上的 `TSLN`、`DN`、`ODC`、`LS` 等缩写感到恐慌；或仅有飞蚊/视疲劳不适想先咨询 | **通俗定心解读模式 (`patient`)**：用比喻说清良性体征与疾病界限、消除对“豹纹状眼底/陈旧激光斑”的误解、支持无图症状分诊与纸质报告名词答疑 |
| **2. 慢病患者与陪诊家属** | 糖尿病、高血压患者或视力不佳老年受检者的子女，需要清晰可执行的复查窗口、就医前清单与居家高危红线预警 | **家属陪诊照护模式 (`caregiver`)**：突出急症优先拦截、红/黄/绿就医紧迫度、诊前携带资料清单、居家监测要点与可向医生提问的问题清单 |
| **3. 基层全科与体检科医生** | 需要在短时间内完成海量眼底初筛复核、核对多模型一致性、定位可疑病灶并决定是否向上转诊 | **基层专业阅片模式 (`clinician`)**：完整呈现 `EvidencePacket` 全量 13 项指标、与冻结研究阈值对比明细、四类病灶定位热区叠加图及专病转诊指征 |

---

## 二、视觉模型底座与标签字典体系

### 2.1 六模型专家与四类病灶分割清单（全部权重经 SHA-256 锁定）

| 模块名称 | 任务定位 | 模型标识 (`model_id`) | 权重文件（相对仓库根目录） |
| :--- | :--- | :--- | :--- |
| **模型 1：图像质控** | 评估眼底照片可判读性（Good / Usable / Reject），不达标则拦截下游 | `quality_eyeq_vitb14_392_best11_20261114` | `02_模型训练与评估/01_在用模型/02_眼底图像质控/quality_best.pt` |
| **模型 2：糖网五级分级** | 糖尿病视网膜病变 0~4 级专病分期 | `dr_ddr_vitl14_crop336_best24_20261114` | `02_模型训练与评估/01_在用模型/01_糖网五级分级/best.pt` |
| **模型 3：五标签多异常** | DR、AMD、MYOPIC_FUNDUS、OPTIC_DISC_CUPPING、ABNORMAL 快速筛查 | `multidisease_rfmid_vitb14_392_best21_20261114` | `02_模型训练与评估/01_在用模型/03_眼底多异常五标签/multidisease_best.pt` |
| **模型 4：视盘视杯分割** | 二维视盘（Disc）与视杯（Cup）解剖轮廓勾画，辅助模型 5 视盘 ROI 定位 | `disc_cup_refuge_unet32_512_best33_20261114` | `02_模型训练与评估/01_在用模型/04_视盘视杯分割/disc_refuge_best.pt` |
| **模型 5：十一标签集成** | 全局 ViT-L + 视盘 ROI 双分支 ViT-B 集成（含视盘 ROI 代码与源码 SHA-256 校验） | `rfmid11_vitl_vitb_disc_mean_20261122` | `02_模型训练与评估/07_模型5云端优化结果_20261122/{large_full,base_disc}/best.pt` |
| **模型 6：青光眼专病筛选** | 青光眼单标签辅助研究筛选 | `glaucoma_historical_vitb392_selected_20261122` | `02_模型训练与评估/05_本轮训练结果_20261122/glaucoma_partial/best.pt` |
| **病灶分割器（四类）** | 糖网 4 类典型病灶定位叠加图（MA 微动脉瘤、HE 出血、EX 硬性渗出、SE 棉絮斑） | `inference_runtime/lesion_localization` | `inference_runtime/lesion_localization/model.safetensors` |

> **关键运行保护锁说明**：
> - `scripts/predict_model5_ensemble_original.py` 硬编码校验 `scripts/prepare_screening_optic_disc_rois.py` 的 SHA-256 摘要，并校验 `training/screening_optimization.py` 与 `training/screening_training.py` 与云端 `run.json` 的源码 SHA-256 一致性；
> - `portable/models.json`、`02_模型训练与评估/`、`cloud_upload/pretrained/dinov2_vitb14_pretrain.pth` 与上述脚本属于**不可移动的运行时核心保护资产**。

### 2.2 标准化标签字典与概念边界
系统在 `inference/report_summary.py` 与 `inference/clinical_workflow.py` 中严格区分四类标签概念，绝不混淆疾病诊断、影像体征与治疗痕迹：

| 标签代码 | 中文名称 | 概念类别 | 所属模型 | 冻结研究筛查线（0~1 原始分） | 核心解读边界 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `DR` | 糖尿病视网膜病变 | 疾病 | 模型 3（另有模型 2 0~4 级专病分级） | `0.043763` | 模型 2 负责 0~4 级分期，模型 3 提供补充筛查线 |
| `AMD` | 年龄相关性黄斑变性 | 疾病 | 模型 3 / 模型 5 | `0.005201` (M3) / `0.006724` (M5) | 需黄斑 OCT 确认，不与 `DN` 混同 |
| `RVO` | 视网膜静脉阻塞 | 疾病 | 模型 5 | `0.011820` | 属血管高危提示，触发 1~2 周内专科就诊建议 |
| `CSR` | 中心性浆液性脉络膜视网膜病变 | 疾病 | 模型 5 | `0.072070` | 需结合黄斑 OCT 核对视网膜下积液 |
| `glaucoma` | 青光眼 | 疾病 | 模型 6 | `0.149283` | 仅作研究辅助筛查，确诊需眼压、视野与 RNFL OCT |
| `DN` | 玻璃膜疣 | 影像表现 | 模型 5 | `0.067511` | 早期黄斑沉着物表现，**不直接等同于湿性 AMD** |
| `OPTIC_DISC_CUPPING` | 视盘凹陷表型 | 影像表现 | 模型 3 / 模型 5 | `0.000013` (M3) / `0.000966` (M5) | 生理性大视杯亦可出现，**不直接等同于青光眼** |
| `ODE` | 视盘水肿 | 影像表现 | 模型 5 | `0.030557` | 高优先提示，需尽早眼科评估视神经及颅压因素 |
| `ODP` | 视盘苍白 | 影像表现 | 模型 5 | `0.000934` | 可能受曝光影响或提示视神经萎缩，需原图复核 |
| `MYOPIC_FUNDUS` | 近视性眼底改变 | 影像表现 | 模型 3 / 模型 5 | `0.063370` (M3) / `0.073954` (M5) | 仅反映眼底结构表型，**不可反推近视具体屈光度数** |
| `TSLN` | 豹纹状眼底 | 影像表现 | 模型 5 | `0.055573` | 高度近视与中老年常见良性底色纹理，避免过度焦虑 |
| `LS` | 激光瘢痕 | 治疗痕迹 | 模型 5 | `0.000993` | **代表既往光凝治疗斑痕，绝不能误判为新发眼病** |
| `ABNORMAL` | 总体异常 | 综合标签 | 模型 3 / 模型 5 | `0.841816` (M3) / `0.777078` (M5) | 独立总体异常判断头，不由单项得分相加得出 |

---

## 三、统一证据对象、SQLite 持久化与 REST API 契约

### 3.1 四种核心职责分离架构
1. **图像识别 (`inference/engine.py`, `inference/research_experts.py`, `inference/lesion_segmenter.py`)**：只产生客观图像分数、分期与热区叠加图，严禁编造缺失阶段结果。
2. **筛查标记 (`inference/report_summary.py`)**：确定性对比冻结研究筛查线，生成 `达到研究筛查线 / 未达到研究筛查线`。
3. **症状优先与规则随访 (`inference/clinical_workflow.py`)**：维护确定性规则引擎与 SQLite 状态机，生成 `EvidencePacket` (`schema_version: "evidence.v1"`)、优先级别、缺失项清单与 `[RULE_*]` / `[SRC_*]` 追溯。
4. **多智能体解释与交互 (`agent/prompts/` + 华为云知识库 `knowledge_base/`)**：严格只读工具证据，依据知识库给出有引用的自然语言解析，绝不篡改数值或优先级。

### 3.2 SQLite 核心数据表 (`inference_runtime/clinical_workflow.sqlite3`)
- `assessments`：保存每次筛查的 `assessment_id`、`session_id`、`created_utc`、`image_sha256`、`eye` (`left` / `right` / `unknown`)、`intake_revision`、`intake_json`、`screening_json` 与完整 `evidence_packet_json`。
- `care_plans`：版本化保存每次规则评估产出的 `plan_id`、`assessment_id`、`version`、`care_priority` 与结构化 `plan_json`。
- `session_links`：维护当前会话绑定的 `active_assessment_id` 与 `active_plan_id`，支持多轮续聊精准找回检查上下文。

### 3.3 已上线 6 个核心 API 插件工具与 Web 路由

| 工具 / 路由 | 方法与路径 | 功能说明与防错设计 |
| :--- | :--- | :--- |
| `screen_fundus` | `POST /v1/screen_fundus` | 上传 `image_url` 执行六模型与四类病灶全套推理。**支持可选原子化绑定参数 `eye`、`symptoms`、`history`**，彻底杜绝双图连续上传时的并行 `record_intake` 竞态错位问题。 |
| `get_context` | `GET /v1/context` | 读取当前会话活跃检查摘要、最新随访计划以及最近 5 次检查 `recent_assessments` 列表，用于追问、双眼对照与历史切换。 |
| `get_assessment` | `GET /v1/assessments/{assessment_id}` | 按 ID 读取冻结的完整 `EvidencePacket` 与 `latest_care_plan`。 |
| `record_intake` | `POST /v1/intake` | 补充或更正已有检查的左右眼别、症状、起病时间、慢病/激光史及既有医嘱；递增 `intake_revision` 并实时重算 `EvidencePacket`。支持字符串或列表入参以兼容平台网关。 |
| `evaluate_care` | `POST /v1/care/evaluate` | 运行确定性随访规则引擎并持久化新版本 `care_plan`，返回就医优先级、复查时间、监测指标与诊前提问清单及实时 `report_url`。 |
| `compare_assessments` | `POST /v1/comparisons` | **同眼严格校验对比**：强制校验两侧均非 `unknown` 且眼别相同、质控均通过、模型版本差提示，并附加“得分差异不等同病情进展”约束说明。 |
| 动态 A4 报告与 PDF 导出 | `GET /v1/view/{token}/report.html` | 实时查询 SQLite 最新 `intake`（眼别、症状、病史）与 `latest_care_plan`，生成含 DR 0~4 进度条、双图切换与 `@page A4 portrait` 一键打印 PDF 的独立页面。 |
| 集成可视化演示工作台 | `GET /v1/demo` & `POST /v1/demo/run` | 面向答辩与用户直观体验的一站式 Web 工作台，支持三层受众切换、内置双样本/自定义图片测试、多 Agent 协同链路 Trace 与一键跳转 A4 报告。 |

### 3.4 确定性随访规则优先级顺序
`evaluate_care_rules` 返回 `clinical_decision`，包含级别、各触发项的证据与医学依据、就诊时限和去向、建议检查。病例总览采用最高级别，同级采用更早的时限。

| 优先级 | 触发条件与筛查流程时限 |
|---|---|
| L4 · 立即眼科急诊 | 急症症状优先于图像质控和分级，立即就医（`urgent_ophthalmology`）。 |
| 需核实图像质量 | 质控未完成或拒识时，重拍或由医生复核（`retake_image`）。 |
| L4 · 优先转诊 | 视盘水肿线索建议 24 小时内评估，临床怀疑颅内压增高时当日急诊。RVO、糖网 3–4 级或重度糖网升阶复核预警，尽快转诊眼底病专科，筛查流程参考 1–2 周内（`prompt_ophthalmology`）。 |
| L3 · 限期专科复核 | 糖网 2 级、青光眼风险、视盘凹陷或苍白、AMD、CSC、近视性眼底改变，2 周至 1 个月内复核（`scheduled_ophthalmology`）。 |
| L2 · 常规观察 | 激光瘢痕 3–6 个月，糖网 1 级或玻璃膜疣 6–12 个月，豹纹状眼底 12 个月（`routine_followup`）。 |
| L1 · 低风险 | 质控合格、糖网 0 级，且全部多病种筛查项目均低于研究筛查线，12 个月复查（`annual_screening`）。 |
| 待专科复核 | 缺少有效分期或筛查结果，或仅见糖网筛查与专病分期不一致、综合异常等未分级线索时，复阅后确定级别（`scheduled_ophthalmology`）。 |

升阶复核可使用结构化 `dr_severe_review_alert` 或本项目公开教学病例的同图专家标注与模型分期差异。该项只提升转诊优先级，原始糖网模型分期照常显示。

---

## 四、华为云 AgentArts 多智能体（Multi-Agent）协作架构

### 4.1 两种运行形态并存（单智能体全能版 + 多智能体协同版）
- **形态 A（单智能体基线 `A0`）**：执行 `python agent/prepare_agent.py --origin https://retina.example.com --output agent/agent_ready_local` 生成 `agent_fundus.jsonl`。将示例域名换成实际部署地址后，可导入华为云联调 API 工具与医学知识库。
- **形态 B（多智能体正式比赛架构 `A1 + A2 + A3 + A4`）**：在 AgentArts“多智能体应用”中由 `A1 主控导诊` 统一接收对话并按固定路由规则分发至三个专业子 Agent：

```text
用户消息 / 眼底图片 / 体检症状咨询
              │
              ▼
  [A1 主控导诊智能体 (唯一入口)]
   ├─ 入口急症关键词抢先检查 & 受众身份识别 (大众 / 患者家属 / 基层医生)
   ├─ 无图纯咨询 / 体检单名词解释 ──────────▶ [A2 医学解读与问答 (W20)] + [A4 慢病科普]
   ├─ 新传眼底图 ──▶ screen_fundus (原子绑定 eye/symptoms/history)
   │                    └─▶ [A2 医学解读与问答 (W20)] (零虚构解读 + [SRC_*] 溯源 + 原图/病灶图)
   ├─ 追问意义 / 查看数值 / 要 PDF ─────────▶ [A2 医学解读与问答 (W20)]
   ├─ 补充症状眼别 / 问何时复查 / 就医准备 ─▶ [A3 随访与急症分诊 (W30)] (record_intake + evaluate_care + [RULE_*])
   └─ 同眼历史比较 / 双眼汇总 / 慢病护眼科普 ▶ [A4 历史对比与科普教育 (W40/W60)] (compare_assessments)
```

### 4.2 跨 Agent 状态传递原则
1. **传递 ID 而非复制长聊天记录**：各 Agent 之间仅传递 `assessment_id`、`session_id` 与结构化字段引用，杜绝在自然语言摘要传递中丢失小数分数、眼别或版本号。
2. **受众语态贯穿全链**：A1 识别到受众偏好（如子女代父母咨询、基层医生阅片、大众体检初筛）后，A2/A3/A4 均在事实零篡改的前提下，调整术语展开深度与行动清单颗粒度。
3. **严格零 Emoji 与规范排版**：所有 Agent 输出与 HTML 报告均禁用 Emoji，采用 Markdown 三级标题、加粗要素列表、标准化表格与底部在线 PDF 报告链接。

---

## 五、医学知识库结构 (`knowledge_base/`)

| 文档 | 核心覆盖主题 | 关键 `[SRC_*]` 引用编号 |
| :--- | :--- | :--- |
| `01_标签字典与体征边界.md` | 13 项标签四类概念区分、激光瘢痕 `LS` 防误判、玻璃膜疣 `DN` vs `AMD`、视盘凹陷 `ODC` vs 青光眼、`TSLN` 不推屈光度 | `SRC_DICT_OVERVIEW`, `SRC_DICT_LS`, `SRC_DICT_DN_AMD`, `SRC_DICT_ODC_GLAUCOMA`, `SRC_DICT_MACULA`, `SRC_DICT_DISC`, `SRC_DICT_MYOPIA` |
| `02_糖网分期与专病随访依据.md` | 糖网 0~4 级临床含义、RVO/ODE 高危指征、青光眼专科检查路径（眼压/视野/RNFL OCT）、非确诊声明 | `SRC_DR_GUIDELINE_2024`, `SRC_RVO_ODE_CONSENSUS`, `SRC_GLAUCOMA_SCREEN_2024` |
| `03_眼科急症警示症状与筛查边界.md` | 突发飞蚊/闪光/幕帘遮挡/骤降眼痛急症优先依据、Reject 质控重拍要求、模型覆盖边界（不能替代周边视网膜裂孔散瞳检查） | `SRC_NEI_RD_URGENT`, `SRC_QC_RETAKE`, `SRC_SCREEN_SCOPE` |
| `04_大众体检与慢病护眼科普.md` | 三层受众沟通模板、良性体检常见词去焦虑解释、无眼底图纯文本症状与纸质报告答疑、糖尿病/高血压全身管理与家属陪诊清单 | `SRC_EDU_CHECKUP_BENIGN`, `SRC_EDU_CHRONIC_CARE`, `SRC_EDU_CAREGIVER_GUIDE`, `SRC_EDU_NO_IMAGE_TRIAGE` |

---

## 六、目录结构规范（精简后）

- `inference/`：FastAPI 服务、六模型引擎、四类病灶分割、SQLite 临床工作流、动态 A4 PDF 报告与集成 Web 演示工作台（`/v1/demo`）。
- `portable/`：一键启动入口 (`portable/run.py`) 与便携模型注册表 (`portable/models.json`)。
- `agent/`：华为云 AgentArts OpenAPI 契约、`prepare_agent.py` 构建器、`export_dify_workflows.py`、`dify_workflows/`（6 套标准化 Dify Workflow DSL）与 `A0~A4` 多智能体提示词。
- `knowledge_base/`：导入华为云知识库的 4 份结构化医学与科普文档。
- `configs/`、`02_模型训练与评估/`、`cloud_upload/`、`training/`、`scripts/`（仅保留 6 个核心运行时与基础设施文件）：模型推理与校验锁依赖。
- `tests/`：核心推理、临床工作流、API、病灶分割与可视化报告自动化测试集。
- `examples/`：演示用典型样例眼底图片。
- `_archive/`：统一归档的历史云端训练脚本、旧部署脚本、数据集下载脚本、离线评估报告及原设计方案 PDF。

---

## 七、标准演进与比赛演示验收清单

### 7.1 医患双端 + 拍片室完整演示闭环剧本
1. **拍片室建档与拍片同步（`/v1/imaging_room`）**：输入手机号（如 `13800138001` 自动回显张建国档案，或输入新手机号现场建档），上传或选择左/右眼眼底照同步至医院 PACS，状态为 `pending` 待分析。
2. **医生端临床辅助会诊（`视健康-医生端辅助诊断` / `/v1/demo` 医生视角）**：输入会诊患者手机号，自动调取患者病史与左/右眼待分析影像；指令辅助诊断左眼后实时跑完六模型与四类病灶定位，输出临床建议、指南依据与临床版 A4 PDF 报告（`/v1/his/reports/{id}/doctor.html`）。
3. **患者端免传图报告解读与跨期对比（`视健康-患者端管家` / `/v1/demo` 患者视角）**：患者无需上传照片，直接询问“能帮我看下左眼最新报告吗”，系统自动检出院内新片完成推理与既往基线对比，用温和亲切口语给出结论、复查建议与患者通俗版 PDF 解读单（`/v1/his/reports/{id}/patient.html`）。
4. **患者端个体化症状问诊与日常科普**：咨询“今早突然眼前有黑影飘和闪光感”，自动关联本人糖尿病史与既往眼底底子优先触发急诊提醒；咨询“青光眼是怎么发生的、平时注意什么”输出生活化比喻科普与权威资源推荐。

## 八、当前整体进展与启动方式（截至 2026-11-24）

### 8.1 已完成能力盘点
1. **视觉模型推理基座**：六个专病与多标签模型（图像质控、糖网 0~4 级分期、五标签异常、视盘视杯分割、十一标签集成、青光眼筛选）+ 四类病灶分割（MA/HE/EX/SE）已在 `conda py310` GPU 环境稳定封装与校验锁定。
2. **院内 PACS 闭环数据中心 (`inference/hospital_system.py`)**：以患者手机号为唯一档案键，打通“拍片室上传 -> 待分析队列 -> 推理归档 -> 同眼进展对比 -> 双版本持久化 A4 PDF 报告（患者通俗版 `patient.html` + 医生临床版 `doctor.html`）”。预置 3 位典型病例（`13800138001` 张建国、`13800138002` 林晓雯、`13800138003` 李秀英）。
3. **双端可视化演示平台 (`inference/portal_pages.py`)**：
   - 拍片室影像同步模拟端：`/v1/imaging_room`；
   - 医患双端一体化演示门户：`/v1/demo`（及 `/v1/portal`）。
4. **华为云 AgentArts 双应用工作流资产 (`agent/export_dify_workflows.py`)**：按“方式 1”拆分为 **`视健康-患者端管家`**（`P00/P10/P20/P30`）与 **`视健康-医生端辅助诊断`**（`D00/D10/D20`）+ 通用 `W90`；全部采用 **Dify `0.6.0` 单一必选入参 `query`**，彻底消除多槽位参数提取器失败问题。
5. **医学与科普知识库 (`knowledge_base/`)**：共 5 份结构化知识文档（含患者亲切通俗科普文档 `05_常见眼病日常科普问答_患者亲切版.md`）。

### 8.2 一键启动与后续待办
- **根目录一键启动**：`python run.py` 启动本机推理及双端门户。公网联调时运行 `python run.py up --origin https://retina.example.com --ssh-host retina-server`，并替换为实际 HTTPS 地址与 SSH 主机别名。按 `Ctrl+C` 停止服务。
- **后续控制台联调待办**：在华为云 AgentArts 平台完成 5 份知识库上传、导入 `agent/agent_ready_local/dify_workflows/` 下的 `.yml` 工作流并装配两个多智能体应用；如有需要可在 `P30` 中接入华为市场眼科视频搜索 MCP。
