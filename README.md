# AgentArts 眼底筛查与随访系统

面向眼底图像辅助筛查与随访的 FastAPI 项目。包含视觉模型路由、医患门户、拍片室、临床辅助报告、报告解读及 AgentArts／Dify 工作流模板。

这份源码包保留应用代码、模型索引、锁定配置和测试。图像推理和预置演示需要分别恢复模型权重与演示素材。数据库和报告在运行时生成，凭证通过启动环境配置。

## 本机启动

在 Windows 上使用 64 位 Python 3.12，从项目根目录运行以下命令。安装脚本会在项目内创建 `.venv`。

```powershell
py -3.12 portable/install.py --backend cpu
.\.venv\Scripts\python.exe portable/run.py serve --device cpu --host 127.0.0.1 --port 8766
```

使用 NVIDIA GPU 时，以 `--backend cu118` 安装 CUDA 版依赖，启动时传入 `--device cuda`。

- [本机医生端](http://127.0.0.1:8766/v1/demo?role=doctor)
- [本机患者端](http://127.0.0.1:8766/v1/demo)
- [本机拍片室](http://127.0.0.1:8766/v1/imaging_room)
- 服务与模型状态 [服务存活](http://127.0.0.1:8766/healthz) · [视觉推理就绪](http://127.0.0.1:8766/readyz)

缺少权重时可浏览门户。补齐依赖和权重后，`/readyz` 返回 `{"ready":true}` 即可进行图像筛查。

## 恢复模型与演示素材

- 按 [模型权重索引.json](模型权重索引.json) 的目标相对路径放置既有权重，并核对 SHA-256。核心视觉权重由 [portable/models.json](portable/models.json) 指定。模型 5 和模型 6 使用 `02_模型训练与评估/` 内保留的锁定 JSON，权重和预训练文件需按索引恢复。
- 四类病灶分割权重位于 `inference_runtime/lesion_localization/model.safetensors`。
- 如需使用项目预置演示，从原始部署资料恢复 `examples/IDRiD_03.jpg` 与 `inference/reference_cases/idrid_03/masks.png`。另一个演示图位于 `inference_runtime/local_gpu/retina_public_sample_proxy.jpg`。公开样例的来源和许可见 [examples/IDRiD-LICENSE.txt](examples/IDRiD-LICENSE.txt)。

## 启用报告解读

**华为 MaaS DeepSeek**。在启动服务的 PowerShell 中设置 `MAAS_API_KEY`。本机解读入口使用回环地址，`RETINA_PUBLIC_ORIGIN` 留空。

```powershell
$key = Read-Host '华为 MaaS API Key' -AsSecureString
$env:MAAS_API_KEY = [System.Net.NetworkCredential]::new('', $key).Password
$key.Dispose()
$env:RETINA_PUBLIC_ORIGIN = $null

.\.venv\Scripts\python.exe portable/run.py serve --device cpu --host 127.0.0.1 --port 8766 --maas-llm
```

在医生端生成或打开报告，点击 **生成解读**。视觉推理在本机完成，MaaS 根据报告的结构化要点生成文字分析，这一步需要联网。

**本地 Qwen**。将本地运行时和模型放入 `inference_runtime/local_llm/`，包括 `llama.cpp/llama-server.exe` 及其配套文件、`qwen2.5-1.5b-ollama.gguf`。使用 `--local-llm` 启动，默认模型端口为 `8091`。

## AgentArts／Dify 工作流

`agent/dify_workflows/` 提供医患端导入模板。使用自己的 HTTPS 服务地址生成一份导入包：

```powershell
.\.venv\Scripts\python.exe agent/export_dify_workflows.py --origin https://retina.example.com --output output/dify_workflows
```

导出的工作流默认使用 API 密钥占位符。导入后在部署环境配置对应的 API Key。更多接口与编排说明见 [多智能体编排与受众扩展指南](agent/多智能体编排与受众扩展指南.md)。

## 目录与测试

| 目录 | 内容 |
| --- | --- |
| `inference/` | API、视觉推理、分诊规则、医患门户及报告 |
| `portable/` | 依赖安装、模型索引与本机启动入口 |
| `agent/`、`knowledge_base/` | 工作流、接口契约、提示词及知识库 |
| `configs/`、`training/`、`scripts/` | 模型配置、训练与评估代码 |
| `02_模型训练与评估/` | 推理所需的锁定配置和选择记录 |
| `tests/` | 推理、报告、工作流及 API 测试 |

```powershell
.\.venv\Scripts\python.exe -m pip install "httpx>=0.27,<1" "PyYAML>=6,<7"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

测试使用临时生成的合成图像与模拟推理。架构和接口定义见 [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)。
