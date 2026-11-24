"""Validate exported Dify Workflow DSL (.yml) files for AgentArts import compatibility."""
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
        types = {n["data"]["type"] for n in nodes}
        self.assertIn("start", types)
        self.assertIn("end", types)

        # Verify DAG (no cycles) and edge handle integrity
        indegree = {nid: 0 for nid in node_map}
        adj: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            src, dst = edge["source"], edge["target"]
            self.assertIn(src, node_map)
            self.assertIn(dst, node_map)
            self.assertEqual(edge["type"], "custom")
            src_node = node_map[src]
            self.assertEqual(edge["data"]["sourceType"], src_node["data"]["type"])
            self.assertEqual(edge["data"]["targetType"], node_map[dst]["data"]["type"])
            if src_node["data"]["type"] == "if-else":
                valid_handles = {c["case_id"] for c in src_node["data"]["cases"]} | {"false"}
                self.assertIn(edge["sourceHandle"], valid_handles)
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

        # Verify HTTP node settings & JSON body template validity
        end_selectors: list[tuple[str, ...]] = []
        for node in nodes:
            if node["data"]["type"] == "end":
                for out in node["data"].get("outputs", []):
                    end_selectors.append(tuple(out["value_selector"]))
            if node["data"]["type"] == "http-request":
                data = node["data"]
                self.assertTrue(data["url"].startswith(DEFAULT_ORIGIN))
                self.assertIn(f"X-API-Key: {expected_api_key}", data["headers"])
                if data["method"] == "post":
                    self.assertIn("Content-Type: application/json", data["headers"])
                    self.assertEqual(data["body"]["type"], "json")
                    body_text = data["body"]["data"][0]["value"]
                    json.loads(body_text)
        # AgentArts merges multiple end nodes and names inputs from value_selector; duplicates cause import errors.
        self.assertEqual(len(end_selectors), len(set(end_selectors)), f"Duplicate end value_selector in {path.name}")

    def test_repo_sanitized_dify_workflows_present_and_valid(self):
        self.assertTrue(REPO_DIFY_DIR.is_dir())
        files = sorted(p.name for p in REPO_DIFY_DIR.glob("*.yml"))
        self.assertEqual(files, sorted(WORKFLOW_BUILDERS))
        for filename in files:
            self._assert_valid_dsl(REPO_DIFY_DIR / filename, expected_api_key=PLACEHOLDER_API_KEY)

    def test_export_with_custom_api_key_and_origin(self):
        custom_key = "custom-test-key-123456"
        custom_origin = "https://retina.example.org"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bundle"
            paths = export_dify_workflows(out_dir, origin=custom_origin, api_key=custom_key)
            self.assertEqual(len(paths), len(WORKFLOW_BUILDERS))
            w10 = (out_dir / "W10_眼底筛查与首屏解读.yml").read_text(encoding="utf-8")
            self.assertIn(f"{custom_origin}/v1/screen_fundus", w10)
            self.assertIn(f"X-API-Key: {custom_key}", w10)
            self.assertNotIn(PLACEHOLDER_API_KEY, w10)


if __name__ == "__main__":
    unittest.main()
