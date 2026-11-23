Windows 本地四模型推理启动工具

本目录只管理本项目的本地推理进程，不安装依赖，不修改智能体配置，也不关闭云服务器。
不设置开机自启或任务计划；需要时手动启动，不需要时手动停止。

双击使用
- start.cmd：后台启动本地四模型，启动请求返回后可关闭包装器窗口；首次加载需要时间。
- status.cmd：查看是否 ready=true。
- stop.cmd：仅停止经身份核验的本项目本地推理进程。
包装器只为当次 PowerShell 进程使用 ExecutionPolicy Bypass，不修改系统或用户执行策略。
服务运行时四模型常驻显存；停止本服务后其显存和内存由系统回收，其他程序的占用不受影响。
临时 HTTPS 隧道是另一部分，需要分别启动和停止；这些包装器只管理推理 API。

前提
- 项目根目录：本目录上两级，可包含中文路径。
- CUDA 推理环境：项目根目录下 .inference-gpu-venv\Scripts\python.exe。
- API 密钥：inference_runtime\retina_api_key.dpapi，Windows 当前用户 DPAPI 原始二进制。
- 审计权重清单：inference_runtime\local_models.json，四项绝对路径。
- 本地配置：inference_runtime\local_gpu\service.json。weights_root=null，避免给绝对路径重复加根目录。
- 四个模型使用原有备份，不复制、不覆盖、不重新下载权重。

在 PowerShell 中执行（从项目根目录）

  & '.\deployment\local_windows\start.ps1' -WaitSeconds 0
  & '.\deployment\local_windows\status.ps1'
  & '.\deployment\local_windows\stop.ps1'

start 默认最多等待 180 秒；自动操作时建议使用 -WaitSeconds 0 启动，再逐次查询 status。
重复 start 如果确认已有本项目进程，会报告 already_running。需要重启时，先 stop 再 start。
如果 Windows 执行策略阻止脚本，请由操作者在当前终端采用适当的单次运行方式，不必更改系统策略。

运行方式
- 只监听 http://127.0.0.1:8000，不直接监听外部网卡。
- device=cuda、precision=fp16（自动混合精度），不会静默切换为 CPU。
- 四模型常驻同一 GPU，单张图依次推理；并发请求槽为 1。
- Uvicorn 访问日志关闭。启动日志位于 inference_runtime\local_gpu\server.stdout.log 与 server.stderr.log。
- 本地入口在 POST /v1/screen_fundus 响应完成后，原子更新 last_screen_request.json，仅记录响应请求ID、HTTP状态、UTC、实际PID和实例UUID，便于与智果调用记录核对；不记录请求正文、图像地址、密钥、IP或推理结果。此辅助记录失败不会影响HTTP响应。
- GET /validation/public-retina.jpg 仅提供固定的公开 scikit-image retina 验收样例。每次读取后校验完整内容的 SHA256 必须为 38a07f36f27f095e818aea7b96d34202c05176d30253c66733f2e00379e9e0e6 且不超过 2MiB，否则返回404；没有动态文件路径或文件列表，不提供个人图像，不是任意文件托管。该路由不列入OpenAPI，POST筛查鉴权和图像下载公网校验保持不变。
- 通过当前 Windows 用户解密 API Key，仅传入子进程环境；密钥不写入命令行、配置或日志。
- 启动脚本会恢复调用进程原有环境，并清理子进程的 RETINA_REGISTRY / RETINA_WEIGHTS_ROOT 继承覆盖。
- run_server.py 用文件锁限制单实例，并记录自己的实际 PID 与随机实例 UUID。
- stop 会核对本项目脚本路径、Python 进程、随机 UUID 与启动时间；不匹配时拒绝停止，不按端口杀进程。
- status 的 ready=true 同时要求本项目进程身份通过校验与 /readyz 返回 true。

本轮已核实的部署状态（2026-11-14，单张公开样例）
- 本机 CUDA/FP16 真实四模型已通过：四阶段 completed、响应契约与模型来源校验通过，单次进程内 HTTP 请求 0.641 秒。PyTorch 峰值 reserved 为 2524971008 字节（约 2.35GiB），不等于整机全部显存占用，也不代表所有图片或并发请求的峰值。
- 停止本项目推理进程后，整张 GPU 的观测占用从 4306MiB 降至 2262MiB，云端未改变。该记录包含桌面等其他进程，不能把剩余占用算作本项目未释放，也不能把差额当成所有场景下的模型显存。
- 当前临时 HTTPS 公网入口已通过全套检查：TLS 证书验证、healthz、readyz、无密钥 HTTP401、私网图片源 HTTP422、真实四阶段 HTTP200、原图哈希及响应 request_id 与请求头匹配。此次公网请求 10.719 秒，含网络与图片获取，不是纯模型耗时。
- 原智果应用调用本地已验收：同一个插件3b542825-32b0-4b02-aae9-485325dfdad3发布v20261114223121，原应用89f83965-a1db-45f4-aa08-9c4583e1ed20升级后移除旧绑定并重新添加，旧版本绑定图标计数为0，最后自动保存22:38:12。原应用仍为“开发中”，未提交版本、未公开发布。公开URL及paperclip上传同一公开样例均实际调用本地四模型，原始工具trace为errCode=0、data.status=ok、四阶段completed；请求ID、原图SHA与本地进程记录匹配。此次没有新增训练、临床准确率或并发性能验收。

回执位置（项目根目录下）
  inference_runtime\local_gpu\real_gpu_smoke.receipt.json
  inference_runtime\local_gpu\verification_public_final\verification.json
  inference_runtime\local_gpu\stop_release_verification.json

保持隧道，只重启 API
临时隧道 cloudflared 转发到固定地址 http://127.0.0.1:8000，不按推理进程 PID 转发。需要重启 API 时，可先 stop.cmd 再 start.cmd，保持隧道进程运行，不执行 tunnel-stop.ps1；随后等 status.cmd 显示 ready=true，再验证同一 HTTPS 地址。API 停止期间公网图片分析暂时不可用，但仅重启 API 通常不需要更换隧道域名或插件 URL。
本轮 tunnel-state.json 的 origin_pid=260016 是隧道启动时记录的来源进程快照，当前 server_process.json 的 API PID=275908；这两者不同不妨碍固定端口继续转发，也不表示模型服务仍在使用旧 PID。实际进程管理以脚本对各自进程的所有权核验和当前监听为准，不手改历史 origin_pid 来伪装一致。这些 PID 只是本轮快照，后续启动会变化。
如果停止并重新启动临时隧道、电脑重启或隧道退出，域名可能更换，需从 tunnel-status.ps1 读取实际地址，并重新做公网验证及智果插件 URL 更新。隧道存活只说明转发进程仍在，不能代替 API ready 与鉴权检查。

本轮智果本地端到端验收记录
当前插件origin：https://retina.example.invalid
筛查接口：https://retina.example.invalid/v1/screen_fundus
- 公开URL输入：总18.62秒，工具15.66秒，request_id=c4e918c99c47493a98b56f9c07bbab8b。
- 原应用paperclip上传公开样例：总11.12秒，工具3.63秒，request_id=2e33ef407d21434aabbdc1bc7c295089。
- 两次源图SHA均为38a07f36f27f095e818aea7b96d34202c05176d30253c66733f2e00379e9e0e6。两次均由本机API PID275908、实例a243b650-8efb-441f-8058-f7ee14f5a3e7处理；last_screen_request.json当前记录第二次附件请求。上述耗时是不同调用的单次观测，不是延迟承诺。
- API最终核实ready=true、owner=true；隧道running=true、PID279352。API及隧道均保留运行，没有设置自动启动。整张GPU观测4593/8188MiB包含其他桌面程序，不能视为模型独占显存。

证据位于项目根目录inference_runtime\local_gpu\：
  agentarts_local_e2e_receipt.json
  agentarts_public_url_tool_evidence.txt
  agentarts_attachment_tool_evidence.txt
  agentarts_local_report_success.jpg
其中截图对应公开样例附件报告成功11.12秒；原始工具证据与请求记录用于确认本机处理，不能只凭报告文本判断已切换。
本地插件OpenAPI位于delivery\agentarts_integration_20261114\screen_fundus_plugin_openapi_local_3.0.json，仅servers使用当前临时隧道。原云端screen_fundus_plugin_openapi_3.0.json保留。更换隧道地址后应同步更新插件并发布、绑定和重新验收；不需要更换四模型权重。

本轮已确认原智果应用通过当前HTTPS入口调用本地。旧云配置保留为回滚历史，本轮未停止或删除旧云实例；旧地址超时不等于实例已关机，其当前实例状态未查询。此说明不代表已完成所有图片/附件类型、临床性能或并发负载验收。
关闭本地电脑、睡眠、断网或停止本地服务后，依赖此服务的图片分析不可用；报告文字和智能体配置不会因此丢失。
