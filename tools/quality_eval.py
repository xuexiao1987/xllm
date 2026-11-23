# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pinned official benchmark scoring and paired xLLM engine comparisons."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from collections import defaultdict, deque
from contextvars import ContextVar
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.logger import logger

HARNESS_VERSION = "0.4.13"
SCORING_PACKAGES = ("lm-eval", "datasets", "numpy", "nltk", "langdetect", "regex", "word2number")
PUNKT_ENGLISH_SHA256 = "3efaaeecafe626e3462d4d5e800be9a087bd377eb6b29ac570095fb9df68f24e"
TASKS = {
    "gsm8k_cot": {
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "count": 1319,
        "fewshot": 8,
        "max_tokens": 1024,
        "primary": "exact_match,flexible-extract",
        "filters": ["strict-match", "flexible-extract"],
        "yaml": "gsm8k/gsm8k-cot.yaml",
        "rows_sha256": "e5cba797d83336f35da00c37e973eb7e56b60c14b841e22d2af3acad2f3a47c9",
    },
    "ifeval": {
        "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "count": 541,
        "fewshot": 0,
        "max_tokens": 1280,
        "primary": "prompt_level_strict_acc,none",
        "filters": ["none"],
        "yaml": "ifeval/ifeval.yaml",
        "rows_sha256": "8f08f4c6f2abee71c2a642516debcd291c7f756da6a25790e8ed203e80bab078",
    },
}
IDENTITY_FIELDS = ("weights_sha256", "tokenizer_sha256", "chat_template_sha256", "precision", "reasoning_policy")
RUN_FIELDS = ("engine_revision", "engine_tree_sha256", "topology", "hardware", "launch_command")


def _write_json(path: Path, value: Any) -> None:
    from lm_eval.utils import handle_non_serializable

    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=handle_non_serializable) + "\n")


def _load_tasks() -> dict[str, Any]:
    import langdetect
    import lm_eval
    import nltk
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml

    resource = Path(nltk.data.find("tokenizers/punkt_tab/english"))
    digest = hashlib.sha256()
    for path in sorted(resource.iterdir()):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != PUNKT_ENGLISH_SHA256:
        raise ValueError("NLTK sentence tokenizer differs from pinned evaluation resources")
    langdetect.DetectorFactory.seed = 1234
    version = importlib.metadata.version("lm_eval")
    if version != HARNESS_VERSION:
        raise ValueError(f"Expected lm-eval {HARNESS_VERSION}, installed {version}")
    configs = []
    for spec in TASKS.values():
        # The loader and scoring functions belong to the pinned harness wheel.
        config = load_yaml(Path(lm_eval.__file__).parent / "tasks" / spec["yaml"])
        config["dataset_kwargs"] = {"revision": spec["revision"]}
        configs.append(config)
    return TaskManager(include_defaults=False).load(configs)["tasks"]


def _dataset_manifest(tasks: dict[str, Any]) -> dict[str, Any]:
    manifest = {}
    for name, task in tasks.items():
        rows = list(task.eval_docs)
        spec = TASKS[name]
        if len(rows) != spec["count"] or task.config.num_fewshot != spec["fewshot"]:
            raise ValueError(f"Official task changed: {name}")
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        # datasets can choose another cached revision in offline mode.
        if digest != spec["rows_sha256"]:
            raise ValueError(f"Dataset content differs from pinned revision: {name}")
        manifest[name] = {
            **spec,
            "dataset": task.config.dataset_path,
            "split": task.config.test_split,
            "rows_sha256": digest,
            "task_version": task.config.metadata["version"],
        }
    return manifest


def _sample_index(result: dict[str, Any], name: str) -> dict[tuple[int, str], Any]:
    rows = result["samples"][name]
    indexed = {(row["doc_id"], row["filter"]): row for row in rows}
    limit = result["xllm_eval"]["protocol"]["limit"]
    spec = TASKS[name]
    count = min(limit, spec["count"]) if limit else spec["count"]
    expected = {(i, f) for i in range(count) for f in spec["filters"]}
    if len(indexed) != len(rows) or indexed.keys() != expected:
        raise ValueError(f"Incomplete/duplicate sample or filter coverage: {name}")
    return indexed


def _metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    for key in (*IDENTITY_FIELDS, *RUN_FIELDS):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"Missing nonempty metadata field: {key}")
        if key.endswith("sha256") and (len(data[key]) != 64 or any(c not in "0123456789abcdef" for c in data[key])):
            raise ValueError(f"Expected lowercase SHA256: {key}")
    if data.get("purpose") not in ("real-checkpoint", "pipeline-test"):
        raise ValueError("purpose must be real-checkpoint or pipeline-test")
    return data


def _validate_response(output: dict[str, Any]) -> str:
    choices = output.get("choices", [])
    if len(choices) != 1 or choices[0].get("index") != 0:
        raise ValueError("Expected exactly one indexed completion")
    choice = choices[0]
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Missing or non-string final content")
    reason = choice.get("finish_reason")
    if reason not in ("stop", "length"):
        raise ValueError(f"Unexpected finish_reason: {reason}")
    return reason


def _request_key(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _load_response_cache(directory: Path | None, manifest: dict[str, Any]) -> dict[str, deque[dict[str, Any]]]:
    cached: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    if directory is None:
        return cached
    previous = json.loads((directory / "manifest.json").read_text())
    for key in ("protocol", "metadata", "model_alias"):
        if previous[key] != manifest[key]:
            raise ValueError(f"Resume identity differs: {key}")
    # Completion order is not request order. Only replay responses recorded
    # with their exact HTTP payload; old response-only logs cannot be resumed.
    with (directory / "requests-responses.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            _validate_response(record["response"])
            cached[_request_key(record["request"])].append(record["response"])
    return cached


def _run(args: argparse.Namespace) -> None:
    from lm_eval import simple_evaluate
    from lm_eval.models.openai_completions import LocalChatCompletion
    from lm_eval.tasks import TaskManager

    metadata = _metadata(args.metadata)
    if not args.endpoint.endswith("/v1/chat/completions"):
        raise ValueError("endpoint must end with /v1/chat/completions")
    tasks = _load_tasks()
    datasets = _dataset_manifest(tasks)
    args.output.mkdir(parents=True, exist_ok=False)
    response_path = args.output / "responses.jsonl"
    journal_path = args.output / "requests-responses.jsonl"
    request_payload: ContextVar[dict[str, Any]] = ContextVar("quality_request_payload")
    cached_responses: dict[str, deque[dict[str, Any]]] = {}
    finish_reasons: list[str] = []
    empty_answers = 0
    reused_responses = 0

    class AuditedChatCompletion(LocalChatCompletion):
        """Keep API failures visible instead of scoring placeholder answers."""

        def _create_payload(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
            payload = super()._create_payload(messages, **kwargs)
            request_payload.set(payload)
            return payload

        def _cached_output(self, messages: Any, kwargs: dict[str, Any]) -> dict[str, Any] | None:
            nonlocal reused_responses
            payload = self._create_payload(self.create_message(messages), seed=self._seed, **copy.deepcopy(kwargs))
            cached = cached_responses.get(_request_key(payload))
            if cached:
                reused_responses += 1
                return cached.popleft()
            return None

        def model_call(self, messages: Any, **kwargs: Any) -> Any:
            cached = self._cached_output(messages, {"eos": self.eos_string, **kwargs})
            if cached is not None:
                return cached
            return super().model_call(messages=messages, **kwargs)

        async def amodel_call(self, session: Any, sem: Any, messages: Any, **kwargs: Any) -> Any:
            cached = self._cached_output(messages, kwargs)
            if cached is not None:
                return self.parse_generations(cached)
            return await super().amodel_call(session=session, sem=sem, messages=messages, **kwargs)

        def parse_generations(self, outputs: Any, **kwargs: Any) -> list[str]:
            nonlocal empty_answers
            batch = outputs if isinstance(outputs, list) else [outputs]
            for output in batch:
                reason = _validate_response(output)
                with journal_path.open("a") as stream:
                    stream.write(
                        json.dumps({"request": request_payload.get(), "response": output}, ensure_ascii=False) + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                with response_path.open("a") as stream:
                    stream.write(json.dumps(output, ensure_ascii=False) + "\n")
                finish_reasons.append(reason)
                # A valid empty answer is scored by the task, not retried or
                # dropped. Reasoning text must never replace the final answer.
                empty_answers += not output["choices"][0]["message"]["content"].strip()
            return super().parse_generations(outputs, **kwargs)

    model = AuditedChatCompletion(
        model=args.model,
        base_url=args.endpoint,
        tokenizer_backend=None,
        tokenized_requests=False,
        num_concurrent=args.concurrency,
        batch_size=1,
        max_retries=1,
        timeout=args.timeout,
        seed=args.seed,
    )
    for name, task in tasks.items():
        # Preserve each official task's stop strings and answer extraction rules.
        task.set_config(
            "generation_kwargs",
            {
                **task.config.generation_kwargs,
                "max_gen_toks": args.max_tokens or TASKS[name]["max_tokens"],
                "temperature": args.temperature,
                "top_p": args.top_p,
                "do_sample": args.temperature > 0,
            },
        )
    protocol = {
        "harness_version": HARNESS_VERSION,
        "scoring_versions": {name: importlib.metadata.version(name) for name in SCORING_PACKAGES},
        "punkt_english_sha256": PUNKT_ENGLISH_SHA256,
        "datasets": datasets,
        "seed": 1234,
        "sampling_seed": args.seed,
        "apply_chat_template": True,
        "fewshot_as_multiturn": False,
        "limit": args.limit,
        "concurrency": args.concurrency,
        "generation_kwargs": {name: task.config.generation_kwargs for name, task in tasks.items()},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    environment = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (args.output / "environment.txt").write_text(environment)
    manifest = {
        "protocol": protocol,
        "metadata": metadata,
        "endpoint": args.endpoint,
        "model_alias": args.model,
        "status": "running",
    }
    cached_responses = _load_response_cache(args.resume, manifest)
    _write_json(args.output / "manifest.json", manifest)
    result = simple_evaluate(
        model=model,
        tasks=list(tasks.values()),
        limit=args.limit,
        apply_chat_template=True,
        fewshot_as_multiturn=False,
        log_samples=True,
        random_seed=1234,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
        bootstrap_iters=1000,
        task_manager=TaskManager(include_defaults=False),
    )
    if result is None:
        raise RuntimeError("No benchmark results returned")
    result["xllm_eval"] = {
        "protocol": protocol,
        "metadata": metadata,
        "response_count": len(finish_reasons),
        "length_limited": finish_reasons.count("length"),
        "empty_answers": empty_answers,
        "reused_responses": reused_responses,
    }
    for name in TASKS:
        _sample_index(result, name)
    _write_json(args.output / "results.json", result)
    manifest = json.loads((args.output / "manifest.json").read_text())
    manifest["status"] = "complete"
    _write_json(args.output / "manifest.json", manifest)
    logger.info("Saved official scores and per-sample results to %s", args.output)


def _compare(reference: dict[str, Any], candidate: dict[str, Any], max_drop_pp: float | None) -> dict[str, Any]:
    ref_meta, new_meta = reference["xllm_eval"], candidate["xllm_eval"]
    if ref_meta["protocol"] != new_meta["protocol"]:
        raise ValueError("Evaluation protocols differ; rerun with identical settings")
    for key in IDENTITY_FIELDS:
        if ref_meta["metadata"][key] != new_meta["metadata"][key]:
            raise ValueError(f"Model identity differs: {key}")
    if set(reference["samples"]) != set(TASKS) or set(candidate["samples"]) != set(TASKS):
        raise ValueError("Expected both official tasks")
    report: dict[str, Any] = {"status": "review_required", "tasks": {}}
    eligible = True
    for meta in (ref_meta, new_meta):
        expected = sum(spec["count"] for spec in TASKS.values())
        eligible &= (
            meta["protocol"]["limit"] is None
            and meta["length_limited"] == 0
            and meta["metadata"]["purpose"] == "real-checkpoint"
            and meta["response_count"] == expected
        )
    for name, spec in TASKS.items():
        left = _sample_index(reference, name)
        right = _sample_index(candidate, name)
        changed, lost, gained = [], [], []
        metric, filter_name = spec["primary"].split(",")
        # GSM8K logs each document twice, once per official answer filter.
        for doc_id, row_filter in left:
            old, new = left[(doc_id, row_filter)], right[(doc_id, row_filter)]
            for key in ("doc_hash", "prompt_hash", "target_hash"):
                if old[key] != new[key]:
                    raise ValueError(f"Input/target mismatch: {name}/{doc_id}/{key}")
            if row_filter != filter_name:
                continue
            if old["resps"] != new["resps"]:
                changed.append(doc_id)
            old_value, new_value = old[metric], new[metric]
            if old_value > new_value:
                lost.append(doc_id)
            elif old_value < new_value:
                gained.append(doc_id)
        metrics = {}
        for key, old_value in reference["results"][name].items():
            if "," not in key or "_stderr," in key:
                continue
            new_value = candidate["results"][name][key]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (old_value, new_value)):
                raise ValueError(f"Non-finite metric: {name}/{key}")
            metrics[key] = {"reference": old_value, "candidate": new_value, "delta_pp": (new_value - old_value) * 100}
        report["tasks"][name] = {
            "samples": len(left) // len(spec["filters"]),
            "metrics": metrics,
            "text_changed_ids": changed,
            "primary_metric": spec["primary"],
            "regressed_ids": lost,
            "improved_ids": gained,
        }
    if not eligible:
        report["status"] = "ineligible_for_acceptance"
    elif max_drop_pp is not None:
        report["status"] = "within_configured_budget"
        # Apply an explicitly chosen budget to every reported quality metric.
        if any(m["delta_pp"] < -max_drop_pp for task in report["tasks"].values() for m in task["metrics"].values()):
            report["status"] = "regression"
    report["max_drop_pp"] = max_drop_pp
    report["reference"] = ref_meta["metadata"]
    report["candidate"] = new_meta["metadata"]
    return report


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--endpoint", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--metadata", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--resume", type=Path, help="Previous output directory with an identical protocol and engine identity"
    )
    run.add_argument("--limit", type=_positive_int)
    run.add_argument("--concurrency", type=_positive_int, default=1)
    run.add_argument("--timeout", type=_positive_int, default=600)
    run.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 keeps greedy diagnostics; set explicitly for model-quality sampling",
    )
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument(
        "--seed", type=int, default=1234, help="API sampling seed; dataset/few-shot selection remains fixed"
    )
    run.add_argument(
        "--max-tokens",
        type=_positive_int,
        help="Override both output budgets; rerun both endpoints with the same value",
    )
    compare = commands.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--max-drop-pp", type=float)
    args = parser.parse_args()
    if args.command == "prepare":
        _write_json(args.output, _dataset_manifest(_load_tasks()))
        logger.info("Prepared pinned datasets: %s", args.output)
    elif args.command == "run":
        if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
            parser.error("--temperature must be finite and in [0, 2]")
        if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
            parser.error("--top-p must be finite and in (0, 1]")
        _run(args)
    else:
        if args.max_drop_pp is not None and (not math.isfinite(args.max_drop_pp) or args.max_drop_pp < 0):
            parser.error("--max-drop-pp must be finite and nonnegative")
        report = _compare(
            json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()), args.max_drop_pp
        )
        _write_json(args.output, report)
        logger.info("Comparison status: %s; report: %s", report["status"], args.output)
        if report["status"] in ("regression", "ineligible_for_acceptance"):
            sys.exit(2)


if __name__ == "__main__":
    _main()
