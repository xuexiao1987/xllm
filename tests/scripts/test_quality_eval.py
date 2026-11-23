# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Comparison safety and optional real harness/HTTP integration tests."""

import argparse
import copy
import hashlib
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tools import quality_eval


def _fixture(full: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "xllm_eval": {
            "protocol": {"limit": None if full else 2},
            "metadata": {key: "a" * 64 for key in quality_eval.IDENTITY_FIELDS},
            "length_limited": 0,
            "response_count": 1860 if full else 4,
        },
        "results": {},
        "samples": {},
    }
    result["xllm_eval"]["metadata"]["purpose"] = "real-checkpoint"
    for name, spec in quality_eval.TASKS.items():
        metric = spec["primary"].split(",")[0]
        result["results"][name] = {f"{metric},{f}": 1.0 for f in spec["filters"]}
        result["samples"][name] = [
            {
                "doc_id": i,
                "filter": f,
                "doc_hash": str(i),
                "prompt_hash": str(i),
                "target_hash": "target",
                "resps": [["answer"]],
                metric: 1,
            }
            for i in range(spec["count"] if full else 2)
            for f in spec["filters"]
        ]
    return result


class QualityComparisonTest(unittest.TestCase):
    def test_rejects_wrong_cached_content_even_with_correct_row_count(self) -> None:
        task = SimpleNamespace(eval_docs=[{}] * 1319, config=SimpleNamespace(num_fewshot=8))
        with self.assertRaisesRegex(ValueError, "Dataset content differs"):
            quality_eval._dataset_manifest({"gsm8k_cot": task})

    def test_full_results_require_explicit_acceptance_budget(self) -> None:
        result = _fixture(full=True)
        self.assertEqual(quality_eval._compare(result, result, None)["status"], "review_required")
        self.assertEqual(quality_eval._compare(result, result, 0)["status"], "within_configured_budget")

    def test_small_subset_never_passes_acceptance(self) -> None:
        result = _fixture()
        self.assertEqual(quality_eval._compare(result, result, 100)["status"], "ineligible_for_acceptance")

    def test_reports_paired_regression_without_counting_gsm_filters_twice(self) -> None:
        reference = _fixture(full=True)
        candidate = copy.deepcopy(reference)
        candidate["samples"]["gsm8k_cot"][1]["exact_match"] = 0
        candidate["samples"]["gsm8k_cot"][1]["resps"] = [["wrong"]]
        candidate["results"]["gsm8k_cot"]["exact_match,flexible-extract"] = 1 - 1 / 1319
        report = quality_eval._compare(reference, candidate, 0)
        self.assertEqual(report["status"], "regression")
        self.assertEqual(report["tasks"]["gsm8k_cot"]["samples"], 1319)
        self.assertEqual(report["tasks"]["gsm8k_cot"]["regressed_ids"], [0])
        self.assertEqual(report["tasks"]["gsm8k_cot"]["text_changed_ids"], [0])

    def test_rejects_mismatched_inputs_and_missing_filters(self) -> None:
        reference = _fixture()
        for field in ("prompt_hash", "target_hash", "doc_hash"):
            candidate = copy.deepcopy(reference)
            candidate["samples"]["gsm8k_cot"][0][field] = "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                quality_eval._compare(reference, candidate, None)
        candidate = copy.deepcopy(reference)
        candidate["samples"]["gsm8k_cot"].pop()
        with self.assertRaises(ValueError):
            quality_eval._compare(reference, candidate, None)

    def test_rejects_duplicates_weights_protocol_and_nonfinite_scores(self) -> None:
        reference = _fixture()
        candidates = [copy.deepcopy(reference) for _ in range(4)]
        candidates[0]["samples"]["ifeval"][1] = candidates[0]["samples"]["ifeval"][0]
        candidates[1]["xllm_eval"]["metadata"]["weights_sha256"] = "b" * 64
        candidates[2]["xllm_eval"]["protocol"]["limit"] = 1
        candidates[3]["results"]["ifeval"]["prompt_level_strict_acc,none"] = float("nan")
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                quality_eval._compare(reference, candidate, None)

    def test_truncation_synthetic_and_missing_responses_block_acceptance(self) -> None:
        reference = _fixture(full=True)
        for key, value in (("length_limited", 1), ("response_count", 1859)):
            candidate = copy.deepcopy(reference)
            candidate["xllm_eval"][key] = value
            self.assertEqual(quality_eval._compare(reference, candidate, 0)["status"], "ineligible_for_acceptance")
        candidate = copy.deepcopy(reference)
        candidate["xllm_eval"]["metadata"]["purpose"] = "pipeline-test"
        self.assertEqual(quality_eval._compare(reference, candidate, 0)["status"], "ineligible_for_acceptance")

    def test_malformed_api_outputs_fail_instead_of_becoming_answers(self) -> None:
        for content in (None, 0, [], {}):
            with self.subTest(content=content), self.assertRaises(ValueError):
                quality_eval._validate_response(
                    {"choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}]}
                )
        with self.assertRaises(ValueError):
            quality_eval._validate_response({"error": "unavailable"})
        with self.assertRaises(ValueError):
            quality_eval._validate_response({"choices": [{"index": 0, "message": {}, "finish_reason": "stop"}]})
        self.assertEqual(
            quality_eval._validate_response(
                {"choices": [{"index": 0, "message": {"content": "partial"}, "finish_reason": "length"}]}
            ),
            "length",
        )


@unittest.skipUnless(
    os.environ.get("XLLM_QUALITY_EVAL_INTEGRATION") == "1", "requires pinned harness and downloaded public datasets"
)
class HarnessIntegrationTest(unittest.TestCase):
    def test_interrupted_http_run_resumes_only_exact_completed_requests(self) -> None:
        from aiohttp import ClientResponseError
        from requests.exceptions import HTTPError

        state = {"requests": 0, "fail": True}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state["requests"] += 1
                if state["fail"] and state["requests"] == 3:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                answer = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                body = json.dumps(
                    {"choices": [{"index": 0, "message": {"content": answer}, "finish_reason": "stop"}]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        for concurrency in (1, 2):
            with self.subTest(concurrency=concurrency), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                metadata = {key: "a" * 64 for key in (*quality_eval.IDENTITY_FIELDS, *quality_eval.RUN_FIELDS)}
                metadata["purpose"] = "pipeline-test"
                (root / "metadata.json").write_text(json.dumps(metadata))
                args = argparse.Namespace(
                    metadata=root / "metadata.json",
                    endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                    model="pipeline-test",
                    output=root / "interrupted",
                    resume=None,
                    limit=2,
                    concurrency=concurrency,
                    timeout=10,
                    max_tokens=None,
                    temperature=1.0,
                    top_p=0.95,
                    seed=1234,
                )
                state.update(requests=0, fail=True)
                with self.assertRaises((ClientResponseError, HTTPError)):
                    quality_eval._run(args)
                records = [
                    json.loads(line) for line in (args.output / "requests-responses.jsonl").read_text().splitlines()
                ]
                self.assertGreaterEqual(len(records), 2)
                self.assertLess(len(records), 4)
                for record in records:
                    expected = hashlib.sha256(json.dumps(record["request"], sort_keys=True).encode()).hexdigest()
                    self.assertEqual(record["response"]["choices"][0]["message"]["content"], expected)
                state["fail"] = False
                requests_before = state["requests"]
                args.resume, args.output = args.output, root / "resumed"
                quality_eval._run(args)
                result = json.loads((args.output / "results.json").read_text())
                self.assertEqual(result["xllm_eval"]["response_count"], 4)
                self.assertEqual(result["xllm_eval"]["reused_responses"], len(records))
                self.assertEqual(state["requests"] - requests_before, 4 - len(records))
                args.resume, args.output = None, root / "fresh"
                quality_eval._run(args)
                fresh = json.loads((args.output / "results.json").read_text())
                for name in quality_eval.TASKS:
                    self.assertEqual(quality_eval._sample_index(result, name), quality_eval._sample_index(fresh, name))
                args.resume, args.output = root / "resumed", root / "wrong-seed"
                args.seed += 1
                before = state["requests"]
                with self.assertRaisesRegex(ValueError, "Resume identity differs: protocol"):
                    quality_eval._run(args)
                self.assertEqual(state["requests"], before)
                args.seed -= 1
                args.output = root / "wrong-engine"
                metadata["engine_tree_sha256"] = "b" * 64
                args.metadata.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "Resume identity differs: metadata"):
                    quality_eval._run(args)
                self.assertEqual(state["requests"], before)

    def test_official_tasks_score_two_http_runs_and_detect_changed_answer(self) -> None:
        self._check_http_runs(temperature=0.0, top_p=1.0, seed=1234)

    def test_sampled_runs_send_parameters_and_record_distinct_protocol(self) -> None:
        self._check_http_runs(temperature=1.0, top_p=0.95, seed=42)

    def test_empty_answers_are_scored_without_losing_samples_or_using_reasoning(self) -> None:
        self._check_http_runs(temperature=0.0, top_p=1.0, seed=1234, candidate_text="")

    def _check_http_runs(
        self, temperature: float, top_p: float, seed: int, candidate_text: str = "The answer is 999."
    ) -> None:
        requests: list[dict[str, Any]] = []
        response = {"text": "The answer is 18."}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(payload)
                body = json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "message": {"content": response["text"], "reasoning_content": "The answer is 18."},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {key: "a" * 64 for key in (*quality_eval.IDENTITY_FIELDS, *quality_eval.RUN_FIELDS)}
            metadata["purpose"] = "pipeline-test"
            (root / "metadata.json").write_text(json.dumps(metadata))
            for name, text in (("reference", "The answer is 18."), ("candidate", candidate_text)):
                response["text"] = text
                quality_eval._run(
                    argparse.Namespace(
                        metadata=root / "metadata.json",
                        endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                        model="pipeline-test",
                        output=root / name,
                        resume=None,
                        limit=2,
                        concurrency=2,
                        timeout=10,
                        max_tokens=None,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed,
                    )
                )
            reference = json.loads((root / "reference/results.json").read_text())
            candidate = json.loads((root / "candidate/results.json").read_text())
            self.assertEqual(candidate["xllm_eval"]["response_count"], 4)
            self.assertEqual(candidate["xllm_eval"]["empty_answers"], 4 if not candidate_text.strip() else 0)
            self.assertEqual(reference["xllm_eval"]["empty_answers"], 0)
            self.assertTrue(all(row["resps"] == [[candidate_text]] for row in candidate["samples"]["gsm8k_cot"]))
            report = quality_eval._compare(reference, candidate, 0)
            self.assertEqual(report["status"], "ineligible_for_acceptance")
            self.assertEqual(report["tasks"]["gsm8k_cot"]["regressed_ids"], [0])
            self.assertEqual(report["tasks"]["gsm8k_cot"]["metrics"]["exact_match,flexible-extract"]["delta_pp"], -50)
            self.assertEqual(len(requests), 8)
            self.assertTrue(
                all(r["temperature"] == temperature and r["top_p"] == top_p and r["seed"] == seed for r in requests)
            )
            protocol = reference["xllm_eval"]["protocol"]
            self.assertEqual(protocol["sampling_seed"], seed)
            self.assertEqual(protocol["seed"], 1234)
            for config in protocol["generation_kwargs"].values():
                self.assertEqual(config["temperature"], temperature)
                self.assertEqual(config["top_p"], top_p)
                self.assertEqual(config["do_sample"], temperature > 0)
            changed_protocol = copy.deepcopy(candidate)
            changed_protocol["xllm_eval"]["protocol"]["sampling_seed"] = seed + 1
            with self.assertRaisesRegex(ValueError, "Evaluation protocols differ"):
                quality_eval._compare(reference, changed_protocol, None)
            self.assertEqual({r["max_tokens"] for r in requests}, {1024, 1280})
            self.assertTrue(all(isinstance(r["messages"], list) for r in requests))


if __name__ == "__main__":
    unittest.main()
