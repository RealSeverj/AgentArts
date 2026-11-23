"""python -m inference serve|openapi (no training or checkpoint downloads)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    spec = commands.add_parser("openapi")
    spec.add_argument("--output", required=True, type=Path)
    spec.add_argument("--server-url", help="Only the real future HTTPS deployment URL")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run("inference.app:create_app", factory=True, host=args.host, port=args.port,
                    workers=1, access_log=False)
    else:
        from inference.app import create_app
        document = create_app().openapi()
        if args.server_url:
            from urllib.parse import urlsplit
            value = urlsplit(args.server_url)
            if value.scheme != "https" or not value.hostname or value.username or value.password or value.query or value.fragment:
                parser.error("--server-url must be a real HTTPS origin or base path without credentials/query")
            document["servers"] = [{"url": args.server_url.rstrip("/")}]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
