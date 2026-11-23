# 眼底筛查与随访多智能体系统（AgentArts）

面向华为云 AgentArts（智果）平台的眼底图像辅助筛查、证据溯源解读、症状优先随访决策与大众护眼科普系统。

- **唯一权威架构与规划文档**：请先阅读 [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)（已完整沉淀原方案设计、技术架构、标签字典、接口契约、多智能体协同链路与验收剧本）。
- **知识库文档**：见 [knowledge_base/README.md](knowledge_base/README.md)（共 4 份结构化医学与大众科普文档）。
- **多智能体与插件构建**：见 [agent/多智能体编排与受众扩展指南.md](agent/多智能体编排与受众扩展指南.md)。

## 快速启动与演示入口

使用 `conda py310` 环境运行本地推理后端与集成演示工作台：

```powershell
# 启动推理服务与公网演示端
C:\Users\Severj\anaconda3\envs\py310\python.exe portable/run.py serve --public-origin https://severj.top

# 运行精简后的全量单元测试
C:\Users\Severj\anaconda3\envs\py310\python.exe -m unittest discover -s tests
```

- **集成可视化演示工作台（Web Demo）**：`http://127.0.0.1:8000/v1/demo`（公网映射：`https://severj.top/v1/demo`）
- **健康与就绪检查**：`GET /healthz`、`GET /readyz`
- **华为云 AgentArts 导入产物**：运行 `agent/prepare_agent.py` 生成的 `agent/agent_ready_severj/agent_fundus.jsonl` 与 `retina_tools.openapi.json`

## 目录说明

- `inference/`：FastAPI 推理边界、六模型路由、四类病灶分割、SQLite 临床工作流规则引擎、动态 A4 PDF 报告及 Web 演示工作台
- `portable/`：一键离线演示与服务端启动入口 (`portable/run.py`)
- `agent/`：AgentArts API 插件 Schema、JSONL 导出脚本与 A0~A4 提示词
- `knowledge_base/`：导入华为云平台知识库的医学指南、标签字典与大众科普文档
- `tests/`：核心推理、工作流规则、报告渲染与 API 自动化测试集
- `02_模型训练与评估/`、`cloud_upload/`、`configs/`、`training/`、`scripts/`：经 SHA-256 校验锁定的模型权重与预处理/集成推理模块
- `_archive/`：历史云端训练脚本、旧部署脚本、数据准备脚本及原方案文档归档
