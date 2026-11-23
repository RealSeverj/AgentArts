# 在新电脑上生成医生阅片报告

将整个迁移包解压到有写入权限的文件夹，保留目录结构。从该文件夹打开 PowerShell。以下 Windows 命令使用 64 位 Python 3.12，与迁移前工程使用的 Python 主次版本相同。首次安装依赖需要联网。模型权重已随包提供，本地照片推理直接读取这些文件。

## 安装

有兼容 CUDA 11.8 的 NVIDIA 显卡和驱动时，运行

```powershell
py -3.12 portable/install.py --backend cu118
```

仅使用 CPU 时，运行

```powershell
py -3.12 portable/install.py --backend cpu
```

两种方式均在解压目录下创建 `.venv`。CPU 推理对内存需求较高，首次启动包含所有权重的指纹校验。Linux 使用 `python3.12 portable/install.py --backend cpu` 或 `cu118`，之后用 `.venv/bin/python` 替换示例中的 Windows 解释器路径。Linux 未在本次迁移验收中实机验证。

## 用一张公开样图验收

```powershell
.\.venv\Scripts\python.exe portable/run.py demo --image examples/IDRiD_03.jpg --output output/case-03
```

打开 `output/case-03/report.html`。这张公开图在原服务实测中质控为可用，糖网专病模型预测 2 级，同图专家标注为 3 级，分割模型返回四类病灶候选区。检查报告的模型分期、专家分期、图例及并排图像，并检查 `result.json` 中各阶段状态和 `visual_report.localization_status`。实测参考值对应这一张原始文件；不同硬件上的浮点得分可有细微差异。

读取自己的本地照片时，将 `--image` 换成实际路径。每次指定一个新的输出目录；报告 HTML、图像和原始结果保存在一起。输出到自选文件夹，完整报告和候选区可离线查看。质量不足时，报告会给出质控提示。

## 提供智果筛查接口

```powershell
.\.venv\Scripts\python.exe portable/run.py serve --public-origin https://你的实际推理域名
```

把命令中的示例域名改为你管理的实际 HTTPS 地址。该地址应将 `/v1/screen_fundus`、`/v1/view/` 和健康检查路径转发至本机服务，供智果和医生浏览器访问。命令默认监听 `127.0.0.1:8000`，适用于同一台机器上的 HTTPS 反向代理或隧道。已有单独的网关时，可按部署需要设置 `--host` 和 `--port`。前台服务可用 Ctrl+C 结束。

服务首次启动默认生成本机接口密钥，保存于当前用户目录的 `.retina-screening/api-key.txt`。已有自建密钥时可通过环境变量 `RETINA_API_KEY` 设置，UTF-8 长度至少 32 字节。导入智果应用后，在插件中设置实际域名、`POST /v1/screen_fundus` 和 Header 鉴权 `X-API-Key`；模型供应商另用目标账号的 MaaS 凭据。仅在配置鉴权时取用本机密钥。

在另一 PowerShell 窗口检查就绪状态。设置了环境变量作为密钥时，使用该变量的值进行接口鉴权。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/readyz
$apiKey = if ($env:RETINA_API_KEY) { $env:RETINA_API_KEY } else { (Get-Content -Raw (Join-Path $env:USERPROFILE '.retina-screening/api-key.txt')).Trim() }
Invoke-RestMethod http://127.0.0.1:8000/v1/models -Headers @{ 'X-API-Key' = $apiKey }
```

最后从目标账号的智果开发预览上传公开样图。验收标准为工具 `screen_fundus` 成功、返回医生报告、两张图片和双图阅片页均可打开。仅调试本机 API 可省略 `--public-origin`，这时 API 返回结构化筛查结果；需要本地双图时使用上面的 `demo` 命令。
