"""Validate exported dual-app Dify 0.6.0 Workflow DSL (.yml) packages for AgentArts."""
from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path
import re
import tempfile
import unittest

import yaml

from agent.export_dify_workflows import (
    DEFAULT_ORIGIN,
    PLACEHOLDER_API_KEY,
    WORKFLOW_BUILDERS,
    export_dify_workflows,
)

ROOT = Path(__file__).resolve().parents[1]
REPO_DIFY_DIR = ROOT / "agent" / "dify_workflows"
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


class DifyWorkflowDslTests(unittest.TestCase):
    def _assert_valid_dsl(self, path: Path, *, expected_api_key: str):
        raw = path.read_text(encoding="utf-8")
        self.assertFalse(EMOJI_RE.findall(raw), f"Found emoji in {path.name}")
        self.assertNotIn("&id", raw, f"Unexpected YAML anchor in {path.name}")
        self.assertNotIn("*id", raw, f"Unexpected YAML alias in {path.name}")

        doc = yaml.safe_load(raw)
        self.assertEqual(doc["kind"], "app")
        self.assertEqual(doc["version"], "0.6.0")
        self.assertEqual(doc["dependencies"], [])
        self.assertEqual(doc["app"]["mode"], "workflow")

        graph = doc["workflow"]["graph"]
        self.assertEqual(graph["viewport"], {"x": 0, "y": 0, "zoom": 1})
        nodes = graph["nodes"]
        edges = graph["edges"]
        self.assertLessEqual(len(nodes), 150)

        node_map = {n["id"]: n for n in nodes}
        self.assertEqual(len(node_map), len(nodes), f"Duplicate node ids in {path.name}")
        start_nodes = [n for n in nodes if n["data"]["type"] == "start"]
        end_nodes = [n for n in nodes if n["data"]["type"] == "end"]
        self.assertEqual(len(start_nodes), 1)
        self.assertGreaterEqual(len(end_nodes), 1)

        # Single mandatory 'query' parameter avoids multi-slot extraction failures in AgentArts controller.
        start_vars = start_nodes[0]["data"]["variables"]
        self.assertEqual([v["variable"] for v in start_vars], ["query"])
        self.assertTrue(start_vars[0]["required"])

        indegree = {nid: 0 for nid in node_map}
        adj: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            src, dst = edge["source"], edge["target"]
            self.assertIn(src, node_map)
            self.assertIn(dst, node_map)
            self.assertEqual(edge["type"], "custom")
            self.assertEqual(edge["data"]["sourceType"], node_map[src]["data"]["type"])
            self.assertEqual(edge["data"]["targetType"], node_map[dst]["data"]["type"])
            adj[src].append(dst)
            indegree[dst] += 1

        queue = deque(nid for nid, deg in indegree.items() if deg == 0)
        visited = 0
        while queue:
            cur = queue.popleft()
            visited += 1
            for nxt in adj[cur]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        self.assertEqual(visited, len(node_map), f"Cycle detected in {path.name}")

        end_selectors: list[tuple[str, ...]] = []
        for node in nodes:
            if node["data"]["type"] == "end":
                for out in node["data"].get("outputs", []):
                    end_selectors.append(tuple(out["value_selector"]))
            if node["data"]["type"] == "http-request":
                data = node["data"]
                self.assertTrue(data["url"].startswith(DEFAULT_ORIGIN))
                self.assertIn(f"X-API-Key: {expected_api_key}", data["headers"])
                self.assertIn("Content-Type: application/json", data["headers"])
                self.assertEqual(data["body"]["type"], "json")
                body_text = data["body"]["data"][0]["value"]
                self.assertEqual(json.loads(body_text), {"query": "{{#start.query#}}"})
        self.assertEqual(len(end_selectors), len(set(end_selectors)), f"Duplicate end value_selector in {path.name}")

    def test_repo_sanitized_dual_app_dify_workflows_valid(self):
        self.assertTrue(REPO_DIFY_DIR.is_dir())
        files = sorted(p.relative_to(REPO_DIFY_DIR).as_posix() for p in REPO_DIFY_DIR.rglob("*.yml"))
        self.assertEqual(files, sorted(WORKFLOW_BUILDERS))
        for rel in files:
            self._assert_valid_dsl(REPO_DIFY_DIR / rel, expected_api_key=PLACEHOLDER_API_KEY)

    def test_export_with_custom_api_key_and_origin(self):
        custom_key = "custom-test-key-123456"
        custom_origin = "https://retina.example.org"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bundle"
            paths = export_dify_workflows(out_dir, origin=custom_origin, api_key=custom_key)
            self.assertEqual(len(paths), len(WORKFLOW_BUILDERS))
            p10 = (out_dir / "patient" / "P10_患者报告温暖解读.yml").read_text(encoding="utf-8")
            self.assertIn(f"{custom_origin}/v1/his/agent/patient/report", p10)
            self.assertIn(f"X-API-Key: {custom_key}", p10)
            self.assertNotIn(PLACEHOLDER_API_KEY, p10)


if __name__ == "__main__":
    unittest.main()
