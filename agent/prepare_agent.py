"""Materialize the AgentArts import files for a new inference origin (no credentials)."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def origin_url(value):
    url = urlsplit(value)
    try:
        host = url.hostname
        _ = url.port
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Invalid host or port") from exc
    if (url.scheme != "https" or not host or url.username is not None or url.password is not None
            or url.path not in ("", "/") or url.fragment or url.query
            or any(c.isspace() for c in value) or host == "localhost" or host.endswith(".invalid")
            or host in {"127.0.0.1", "0.0.0.0", "::1"}):
        raise argparse.ArgumentTypeError("Supply the public HTTPS origin without a path, credentials, or query")
    return "https://" + url.netloc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True, type=origin_url)
    parser.add_argument("--output", type=Path, default=Path("agent_ready"),
                        help="Create a new directory for the configured files")
    parser.add_argument("--api-key", default=None,
                        help="Optional X-API-Key to embed in agent_fundus.jsonl for local debugging")
    parser.add_argument("--embed-local-key", action="store_true",
                        help="Read ~/.retina-screening/api-key.txt and embed X-API-Key for local debugging")
    args = parser.parse_args()
    template_root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    api_key = args.api_key
    if args.embed_local_key and not api_key:
        key_path = Path.home() / ".retina-screening" / "api-key.txt"
        if not key_path.is_file():
            parser.error(f"Local API key file not found: {key_path}")
        api_key = key_path.read_text(encoding="ascii").strip()
    records = [json.loads(line) for line in (template_root / "agent_template.jsonl")
               .read_text(encoding="utf-8").splitlines() if line.strip()]
    tool, = (record for record in records if record["resource_type"] == "tool")
    plugin = json.loads(tool["dsl"])
    for definition in (plugin, tool["metadata"]):
        request = json.loads(definition["request_info"])
        assert request["basic_info"]["host"] == "fundus.example.invalid"
        request["basic_info"]["host"] = urlsplit(args.origin).netloc
        definition["request_info"] = compact(request)
        if api_key:
            for item in definition.get("auth_info", {}).get("auth_keys", []):
                if item.get("target_name") == "X-API-Key":
                    item["auth_key"] = api_key
    tool["dsl"] = compact(plugin)
    if api_key:
        agent_record = next(record for record in records if record["resource_type"] == "agent")
        for tool_ref in agent_record.get("dsl", {}).get("tools", []):
            for item in tool_ref.get("auth_info", {}).get("auth_keys", []):
                if item.get("target_name") == "X-API-Key":
                    item["auth_key"] = api_key
    schema = json.loads((template_root / "screen_fundus.openapi.json").read_text(encoding="utf-8"))
    schema["servers"][0]["url"] = args.origin
    multi_path = template_root / "retina_tools.openapi.json"
    multi_schema = None
    if multi_path.is_file():
        multi_schema = json.loads(multi_path.read_text(encoding="utf-8"))
        multi_schema["servers"][0]["url"] = args.origin
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        new_files = Path(temporary) / "prepared"
        new_files.mkdir()
        (new_files / "agent_fundus.jsonl").write_text(
            "".join(compact(item) + "\n" for item in records), encoding="utf-8")
        (new_files / "screen_fundus.openapi.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if multi_schema is not None:
            (new_files / "retina_tools.openapi.json").write_text(
                json.dumps(multi_schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        new_files.rename(output)
    print(f"Created {output}")


if __name__ == "__main__":
    main()

