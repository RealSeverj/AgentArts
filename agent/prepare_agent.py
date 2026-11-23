"""Materialize the AgentArts import files for a new inference origin."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

TOOL_IDS = {
    "screen_fundus": "dfe381565e634a81",
    "get_context": "dfe381565e634a82",
    "get_assessment": "dfe381565e634a83",
    "record_intake": "dfe381565e634a84",
    "evaluate_care": "dfe381565e634a85",
    "compare_assessments": "dfe381565e634a86",
}


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


def _build_tool_entries(multi_schema, original_output_schema):
    tool_info = []
    input_schemas = []
    output_schemas = []
    is_input_list = []
    is_output_list = []
    intf_type = []
    test_status = []
    agent_tool_specs = []

    for path, methods in multi_schema.get("paths", {}).items():
        for method, op in methods.items():
            op_id = op["operationId"]
            tid = TOOL_IDS[op_id]
            tool_info.append({
                "tool_id": tid,
                "tool_display_name": op_id,
                "tool_chinese_name": op_id,
                "tool_desc": op.get("description") or op.get("summary") or op_id,
                "method": method.upper(),
                "path": path,
            })

            props = {}
            required = []
            agent_params = []

            if method.upper() == "POST":
                ct_desc = "固定 HTTP 请求头。每次调用均填写 application/json。"
                props["Content-Type"] = {
                    "default": "application/json",
                    "location": "Headers",
                    "validate_rule": "",
                    "validate_type": "CHAR",
                    "validated": False,
                    "type": "string",
                    "description": ct_desc,
                }
                required.append("Content-Type")
                agent_params.append({
                    "default": "application/json",
                    "description": ct_desc,
                    "name": "Content-Type",
                    "required": True,
                    "type": "string",
                    "value": {"content": "application/json", "type": "literal"},
                    "visibility": True,
                })

            for param in op.get("parameters", []):
                name = param["name"]
                loc = {"header": "Headers", "query": "Query", "path": "Path"}.get(param["in"], "Query")
                pschema = param.get("schema", {})
                prop = {
                    "location": loc,
                    "validated": False,
                    "type": "string",
                    "description": param.get("description", ""),
                }
                default_val = pschema.get("default")
                if default_val is not None:
                    prop["default"] = default_val
                    prop["validate_rule"] = ""
                    prop["validate_type"] = "CHAR"
                props[name] = prop
                is_req = bool(param.get("required")) or name == "Content-Type"
                if is_req:
                    required.append(name)
                agent_params.append({
                    "default": default_val if default_val is not None else "",
                    "description": prop["description"],
                    "name": name,
                    "required": is_req,
                    "type": "string",
                    "value": {"content": default_val if default_val is not None else "", "type": "literal"},
                    "visibility": True,
                })

            body_schema = ((op.get("requestBody") or {}).get("content") or {}).get("application/json", {}).get("schema")
            if body_schema:
                ref = body_schema.get("$ref")
                if ref:
                    ref_name = ref.rsplit("/", 1)[-1]
                    body_schema = multi_schema["components"]["schemas"][ref_name]
                body_required = set(body_schema.get("required", []))
                for name, bprop in body_schema.get("properties", {}).items():
                    prop = {
                        "location": "Body",
                        "validated": False,
                        "type": "string",
                        "description": bprop.get("description", ""),
                    }
                    default_val = bprop.get("default")
                    if default_val is not None:
                        prop["default"] = default_val
                        prop["validate_rule"] = ""
                        prop["validate_type"] = "CHAR"
                    props[name] = prop
                    is_req = name in body_required
                    if is_req:
                        required.append(name)
                    agent_params.append({
                        "default": default_val if default_val is not None else "",
                        "description": prop["description"],
                        "name": name,
                        "required": is_req,
                        "type": "string",
                        "value": {"content": default_val if default_val is not None else "", "type": "literal"},
                        "visibility": True,
                    })

            input_schemas.append({
                "tool_id": tid,
                "input_schema": compact({"type": "object", "properties": props, "required": required}),
            })
            if tid == TOOL_IDS["screen_fundus"] and original_output_schema is not None:
                output_schemas.append(original_output_schema)
            else:
                output_schemas.append({
                    "tool_id": tid,
                    "output_schema": compact({"type": "object", "properties": {}, "required": []}),
                })
            is_input_list.append({"tool_id": tid, "is_input_list": False})
            is_output_list.append({"tool_id": tid, "is_output_list": False})
            intf_type.append({"tool_id": tid, "intf_type": "blocking"})
            test_status.append({"tool_id": tid, "test_status": 1})
            agent_tool_specs.append({
                "tool_id": tid,
                "op_id": op_id,
                "desc": op.get("description") or op.get("summary") or op_id,
                "tool_parameter": compact(agent_params),
            })

    return {
        "tool_ids": [item["tool_id"] for item in tool_info],
        "tool_info": tool_info,
        "input_schema": compact(input_schemas),
        "output_schema": compact(output_schemas),
        "is_input_list": compact(is_input_list),
        "is_output_list": compact(is_output_list),
        "intf_type": compact(intf_type),
        "test_status": compact(test_status),
        "agent_tool_specs": agent_tool_specs,
    }


def _apply_multi_tool(records, template_root, multi_schema):
    agent_record = next(record for record in records if record["resource_type"] == "agent")
    tool_record = next(record for record in records if record["resource_type"] == "tool")
    plugin = json.loads(tool_record["dsl"])
    orig_outputs = json.loads(plugin["output_schema"])
    orig_screen_output = next((x for x in orig_outputs if x.get("tool_id") == TOOL_IDS["screen_fundus"]), None)

    entries = _build_tool_entries(multi_schema, orig_screen_output)
    plugin_id = tool_record["resource_id"]

    for definition in (plugin, tool_record["metadata"]):
        request = json.loads(definition["request_info"])
        request["tool_info"] = entries["tool_info"]
        definition["request_info"] = compact(request)
        definition["input_schema"] = entries["input_schema"]
        definition["output_schema"] = entries["output_schema"]
        definition["is_input_list"] = entries["is_input_list"]
        definition["is_output_list"] = entries["is_output_list"]
        definition["intf_type"] = entries["intf_type"]
        definition["test_status"] = entries["test_status"]
        if "tool_ids" in definition:
            definition["tool_ids"] = entries["tool_ids"]
    tool_record["dsl"] = compact(plugin)

    base_tool_ref = dict(agent_record["dsl"]["tools"][0])
    new_tools = []
    for spec in entries["agent_tool_specs"]:
        ref = dict(base_tool_ref)
        ref["tool_id"] = f"{plugin_id}#{spec['tool_id']}"
        ref["tool_display_name"] = spec["op_id"]
        ref["tool_chinese_name"] = spec["op_id"]
        ref["desc"] = spec["desc"]
        ref["tool_parameter"] = spec["tool_parameter"]
        new_tools.append(ref)
    agent_record["dsl"]["tools"] = new_tools

    model_id = agent_record["metadata"]["model_deployment_id"]
    agent_record["level2_resources"] = ["100", *[ref["tool_id"] for ref in new_tools], model_id]

    prompt_path = template_root / "prompts" / "A0_全功能单智能体提示词.txt"
    if prompt_path.is_file():
        prompt_text = prompt_path.read_text(encoding="utf-8")
        agent_record["metadata"]["instructions"] = prompt_text
        agent_record["dsl"]["instructions"] = prompt_text
        agent_record["metadata"]["description"] = (
            "眼底筛查与随访智能助手：支持眼底图像质控、六模型筛查与四类病灶定位、"
            "症状优先评估、结构化随访计划生成与同眼历史对比。"
        )
        agent_record["dsl"]["description"] = agent_record["metadata"]["description"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True, type=origin_url)
    parser.add_argument("--output", type=Path, default=Path("agent_ready"),
                        help="Create a new directory for the configured files")
    parser.add_argument("--api-key", default=None,
                        help="Optional X-API-Key to embed in agent_fundus.jsonl for local debugging")
    parser.add_argument("--embed-local-key", action="store_true",
                        help="Read ~/.retina-screening/api-key.txt and embed X-API-Key for local debugging")
    parser.add_argument("--single-tool", action="store_true",
                        help="Keep legacy single screen_fundus tool instead of 6-tool multi-workflow bundle")
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
        if not args.single_tool:
            _apply_multi_tool(records, template_root, multi_schema)

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
