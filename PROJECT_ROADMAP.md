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
`evaluate_care_rules` 按以下严格顺序命中（高位优先，**症状永远优先于图像未检出结果**）：
1. `RULE_URGENT_SYMPTOM_01` → `urgent_ophthalmology`（立即眼科急诊评估：突发大量飞蚊/黑影、闪光感、幕帘样遮挡、突发视力骤降、剧烈眼痛伴头痛恶心；带否定前缀如“无飞蚊”自动豁免） `[SRC_NEI_RD_URGENT]`
2. `RULE_QC_RETAKE_01` → `retake_image`（图像质量 Reject 或未完成质控，拦截一切正常/阴性结论并引导重拍） `[SRC_QC_RETAKE]`
3. `RULE_DR_SEVERE_OR_VASC_01` → `prompt_ophthalmology`（糖网 3~4 级，或 `RVO`/`ODE` 达到筛查线，建议 1~2 周内专科面诊） `[SRC_DR_GUIDELINE_2024, SRC_RVO_ODE_CONSENSUS]`
4. `RULE_MODERATE_OR_SPECIALIST_02` → `scheduled_ophthalmology`（糖网 2 级，或 `AMD`/`CSR`/`glaucoma`/`ODP`/`OPTIC_DISC_CUPPING` 达线，建议 1 个月内专科复核） `[SRC_DR_GUIDELINE_2024, SRC_DICT_DN_AMD, SRC_DICT_MACULA, SRC_DICT_ODC_GLAUCOMA, SRC_GLAUCOMA_SCREEN_2024]`
5. `RULE_MILD_OR_PHENOTYPE_03` → `routine_followup`（糖网 1 级，或 `DN`/`MYOPIC_FUNDUS`/`TSLN`/`LS`/`ABNORMAL` 达线，建议 3~6 个月常规随访及病史核对） `[SRC_DR_GUIDELINE_2024, SRC_DICT_LS, SRC_DICT_MYOPIA]`
6. `RULE_ROUTINE_ANNUAL_04` → `annual_screening`（覆盖范围内未达筛查线，建议常规 6~12 个月或每年眼健康随访，并提示模型覆盖边界） `[SRC_SCREEN_SCOPE]`

---

## 四、华为云 AgentArts 多智能体（Multi-Agent）协作架构

### 4.1 两种运行形态并存（单智能体全能版 + 多智能体协同版）
- **形态 A（单智能体基线 `A0`）**：通过 `agent/prepare_agent.py` 一键生成 `agent/agent_ready_severj/agent_fundus.jsonl`，直接导入华为云即可使用全部 6 个 API 工具与医学知识库，用于单通道快速联调与兜底。
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

### 7.1 四大核心演示剧本
1. **剧本一（双图左右眼原子防错位 + 四类病灶定位 + PDF 导出）**：上传含糖网 2/3 级伴微动脉瘤/出血渗出的左眼样例图并说明“左眼，糖尿病 8 年”，紧接着上传正常对照右眼图并说明“右眼”。展示两眼独立建档互不串台、双眼对照表格与一键导出标准 A4 PDF 报告。
2. **剧本二（Symptom-first 症状优先急症拦截）**：在同一病例中补充“右眼今天突然出现大量飞蚊和闪光感”，即使右眼照片分类未达阳性筛查线，规则引擎仍立即升级为 `urgent_ophthalmology`（`[RULE_URGENT_SYMPTOM_01]`）并置顶急诊提示。
3. **剧本三（大众体检与无图科普咨询）**：切换为大众体检/家属陪诊视角，提问“体检报告写了豹纹状眼底（TSLN）和激光瘢痕（LS），是不是得重病了？需要立刻手术吗？”展示 A2+A4 结合知识库去焦虑科普与无图场景下的规范边界。
4. **剧本四（Web 演示工作台一站式 Trace 展示 `/v1/demo`）**：通过 `https://severj.top/v1/demo` 展示三类受众切换、多 Agent 协作流转轨迹（A1 → W10 → A2 → A3 → A4）、确定性规则依据 `[RULE_*]` 与一键导出 A4 临床报告。

### 7.2 后续扩展方向（P2）
- 医生复核确认闭环台与人工修订记录导出（用于离线错误分析）；
- 面向微信/短信/小程序的主动随访到期提醒调度器；
- 引入经独立数据集验证的严格 OOD 域外检测与温度缩放概率校准模块。
