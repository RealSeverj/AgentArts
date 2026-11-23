本机推理的临时HTTPS入口（Cloudflare Quick Tunnel）

用途：供智果API插件调用本机 http://127.0.0.1:8000；仅用于公开测试图联调与开发演示。
本机四模型服务必须已启动、/readyz返回ready=true、无API Key访问/v1/models返回401。
脚本不会打开Windows入站端口，不改防火墙、系统PATH、登录状态或云服务器。

准备的官方程序：
inference_runtime/local_gpu/tools/cloudflared-windows-amd64-2026.11.0.exe
官方SHA256：d5c1d82031da6253dcdcff64c944cd0fe063016edeb2e4fd088075fc9bd52d0a
来源与校验记录：同目录cloudflared_download_receipt.json。

在PowerShell中从项目根目录执行：
  & '.\deployment\local_windows\tunnel-start.ps1'
  & '.\deployment\local_windows\tunnel-status.ps1'
  & '.\deployment\local_windows\tunnel-stop.ps1'

启动脚本会校验程序SHA256，核验本地监听与API鉴权，再以隐藏窗口启动cloudflared。
cloudflared使用显式空配置、INFO日志、IPv4/TCP HTTP2连接；不使用账户凭据或命名隧道。
脚本不读取API密钥文件，启动子进程前会移除RETINA_API_KEY/MAAS_API_KEY/TUNNEL_*环境变量。
接口仍由原有X-API-Key鉴权。不要给API插件启用邮件验证码认证，该方式不支持非交互客户端。
日志不启用DEBUG、HTTP访问日志或请求体记录；DEBUG会输出请求头，不能用于带密钥联调。
状态文件及INFO启动日志位于inference_runtime/local_gpu/tunnel-state.json和tunnel_logs/。
停止脚本仅停止已核验路径、进程创建时间及参数的本项目cloudflared进程，不停止推理服务。

拿到public_url不等于联调通过；需要先验证公网/readyz为200、无密钥访问/v1/models为401、
带密钥公开图片推理通过、原智果应用插件实际调用成功，然后再考虑关闭GPU云服务器。
配置智果时，只更新插件服务基准URL为本次public_url；/v1/screen_fundus和X-API-Key保持原语义。

限制：
- 域名是随机的*.trycloudflare.com，每次新建隧道都会变化；旧地址停止进程后失效。
- 无可用性保证，最多200个同时进行的请求，不支持SSE；本项目插件使用普通POST JSON。
- 电脑、推理服务、隧道和网络都需要持续运行。没有本机模型服务时，智果仍可文字对话，但图片推理不可用。
- Quick Tunnel不出现在Cloudflare账户的隧道列表。长期部署应使用已验证的固定域名方案。
- HTTPS在Cloudflare边缘终止，API请求和响应经过Cloudflare；这里先用公开测试图验证，不宣称患者数据生产方案已验收。

官方依据（核对日期：2026-11-14）：
https://developers.cloudflare.com/tunnel/downloads/
https://github.com/cloudflare/cloudflared/releases/tag/2026.11.0
https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/
https://developers.cloudflare.com/tunnel/reference/run-parameters/
