#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import copy
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from datasets import Dataset, concatenate_datasets, load_dataset
from vllm import SamplingParams

from evaluation.livecodebench_utils import (
    compute_metrics_from_results,
    lcb_run,
    map_to_example,
    post_process_code,
    translate_private_test_cases,
)

LCB_PROMPT_WITHOUT_STARTER_CODE = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

Question: {problem_description}

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
```python
  # YOUR CODE HERE
```"""

LCB_PROMPT_WITH_STARTER_CODE = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

Question: {problem_description}

You will use the following starter code to write the solution to the problem and enclose your code within delimiters as follows.
```python
{entry_point}
```"""


def has_code(response):
    pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    return matches


def filter_by_contest_date(example):
    target_months = ["2025-02", "2025-03", "2025-04", "2025-05"]
    return example["contest_date"][:7] in target_months


class LiveCodeBenchV6:
    """
    LiveCodeBench V6 - Benchmark for evaluating code generation capabilities of LLMs
    on competitive programming problems from recent contests (Feb-May 2025).
    """

    def __init__(
        self,
        llm,
        tokenizer,
        max_tokens: int = 32768,
        n_repeat: int = 20,
        sampling_params: Optional[Dict[str, Any]] = None,
        seed: Optional[List[int]] = None,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.n_repeat = n_repeat
        self.sampling_params = sampling_params if sampling_params is not None else {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
        }
        self.seed = seed if seed is not None else [0, 1234, 1234, 1234]
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        """Run the full benchmark: load data, generate solutions, evaluate."""
        ds = self.load_questions()
        examples = list(ds)
        self.generate(examples)
        results = self.evaluate(examples)
        return results

    def generate(self, examples):
        """Generate solution completions using vLLM."""
        all_outputs = []
        stop_token_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []

        for i in range(self.n_repeat):
            seed = self.seed[0] + i

            prompts = []
            for example in examples:
                if example["is_stdin"]:
                    prompt_text = LCB_PROMPT_WITHOUT_STARTER_CODE.format(
                        problem_description=example["prompt"]
                    )
                else:
                    prompt_text = LCB_PROMPT_WITH_STARTER_CODE.format(
                        problem_description=example["prompt"],
                        entry_point=example["entry_point"],
                    )

                messages = [{"role": "user", "content": prompt_text}]
                templated = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(templated)

            sampling_params = SamplingParams(
                max_tokens=self.max_tokens,
                seed=seed,
                stop_token_ids=stop_token_ids,
                **self.sampling_params,
            )

            self.logger.info(f"Generating responses (repeat {i + 1}/{self.n_repeat})...")
            outputs = self.llm.generate(prompts, sampling_params)
            texts = [o.outputs[0].text for o in outputs]
            all_outputs.append(texts)

        for example, per_example_outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(per_example_outputs)
            example["model_answers"] = [has_code(o) for o in per_example_outputs]

    @staticmethod
    def check_correctness(problem: Dict, completion: str, timeout: float, is_extracted: bool = False) -> Dict:
        """Evaluate functional correctness by running the test suite."""
        result_list = lcb_run(problem, completion, timeout, is_extracted)
        details = [r[0] for r in result_list]
        all_passed = all(details)
        return {
            "all_passed": all_passed,
            "result_list": result_list,
            "test_cases": problem["test"],
        }

    def evaluate_single_example(self, example):
        """Evaluate a single example by running its code against test cases."""
        try:
            response_entry = {
                "task_id": example.get("task_id"),
                "prompt": example.get("prompt", ""),
                "entry_point": example.get("entry_point", ""),
                "is_stdin": example.get("is_stdin", False),
                "content": example["model_answer"],
                "difficulty": example["difficulty"],
                "correctness": None,
                "reason": None,
                "test_input": None,
                "test_output": None,
                "test_expected": None,
                "num_tests_passed": 0,
                "num_tests_failed": 0,
                "test_results": [],
            }

            code_filter_result = example["model_answer"]

            if not code_filter_result or len(code_filter_result) == 0:
                response_entry["correctness"] = False
                response_entry["reason"] = "Does not contain code component."
                return response_entry

            try:
                last_code = code_filter_result[-1]
                problem_to_check = copy.deepcopy(example)

                self.logger.debug(f"Evaluating {example['difficulty']} problem...")

                curr_res = self.check_correctness(
                    problem=problem_to_check,
                    completion=post_process_code(last_code),
                    timeout=6,
                    is_extracted=not problem_to_check["is_stdin"],
                )

                self.logger.debug(f"Result for {example['difficulty']}: {curr_res['all_passed']}")

                result_list = curr_res["result_list"]
                test_cases = curr_res["test_cases"]

                num_passed = sum(1 for r in result_list if r[0])
                num_failed = len(result_list) - num_passed

                response_entry["test_results"] = [1 if r[0] else 0 for r in result_list]
                response_entry["num_tests_passed"] = num_passed
                response_entry["num_tests_failed"] = num_failed
                response_entry["correctness"] = curr_res["all_passed"]

                if not curr_res["all_passed"]:
                    response_entry["reason"] = "Code is incorrect."

                    for idx, (passed, output_error, output_value, time_elapsed) in enumerate(result_list):
                        if not passed and idx < len(test_cases):
                            test_case = test_cases[idx]
                            response_entry["test_input"] = str(test_case.get("input", ""))
                            response_entry["test_expected"] = str(test_case.get("output", ""))
                            response_entry["test_output"] = str(output_value)
                            break
                else:
                    response_entry["reason"] = ""

            except Exception as e:
                self.logger.error(f"Error evaluating {example['difficulty']} example: {str(e)}")
                response_entry["correctness"] = False
                response_entry["reason"] = f"Evaluation error: {str(e)}"

            return response_entry

        except Exception as outer_e:
            self.logger.error(f"Outer error in evaluate_single_example: {str(outer_e)}")
            return {
                "task_id": example.get("task_id"),
                "prompt": example.get("prompt", ""),
                "entry_point": example.get("entry_point", ""),
                "is_stdin": example.get("is_stdin", False),
                "content": example.get("model_answer"),
                "difficulty": example.get("difficulty"),
                "correctness": False,
                "reason": f"Critical error: {str(outer_e)}",
                "test_input": None,
                "test_output": None,
                "test_expected": None,
                "num_tests_passed": 0,
                "num_tests_failed": 0,
                "test_results": [],
            }

    def evaluate(self, examples):
        """Evaluate generated solutions using parallel thread execution."""
        self.logger.info(f"Evaluating {len(examples)} examples...")
        self.logger.warning("Expect some output leaks from code/test execution into stdout")

        # Organize completions by repeat index
        examples_by_repeat = defaultdict(list)
        for example in examples:
            for i, (output, answers) in enumerate(zip(example["model_outputs"], example["model_answers"])):
                example_copy = example.copy()
                example_copy["model_answer"] = answers
                example_copy["model_output"] = output
                example_copy.pop("model_outputs", None)
                example_copy.pop("model_answers", None)
                examples_by_repeat[i].append(example_copy)

        all_repeat_results = []
        num_questions = len(examples)

        for repeat_idx, repeat_examples in examples_by_repeat.items():
            results = []
            with ThreadPoolExecutor(max_workers=32) as executor:
                future_to_example = {}
                for i, example in enumerate(repeat_examples):
                    future = executor.submit(self.evaluate_single_example, example)
                    future_to_example[future] = (i, example)

                results = [None] * len(repeat_examples)
                for future in as_completed(future_to_example):
                    idx, example = future_to_example[future]
                    try:
                        result = future.result()
                        results[idx] = (result, example)
                    except Exception as e:
                        self.logger.error(f"Future error for example {idx}: {str(e)}")
                        results[idx] = (
                            {
                                "task_id": example.get("task_id"),
                                "prompt": example.get("prompt", ""),
                                "entry_point": example.get("entry_point", ""),
                                "is_stdin": example.get("is_stdin", False),
                                "content": example["model_answer"],
                                "difficulty": example["difficulty"],
                                "correctness": False,
                                "reason": f"Future error: {str(e)}",
                                "test_input": None,
                                "test_output": None,
                                "test_expected": None,
                                "num_tests_passed": 0,
                                "num_tests_failed": 0,
                                "test_results": [],
                            },
                            example,
                        )

            all_repeat_results.append(results)

        final_metrics = {}

        # Compute pass@k metrics
        self.logger.info("Computing pass@k metrics...")

        results_by_task_id = defaultdict(list)
        results_by_task_id_and_difficulty = defaultdict(lambda: defaultdict(list))

        for repeat_results in all_repeat_results:
            for result, example in repeat_results:
                task_id = result["task_id"]
                difficulty = result["difficulty"]
                test_results = result.get("test_results", [])

                if test_results:
                    results_by_task_id[task_id].append(test_results)
                    results_by_task_id_and_difficulty[difficulty][task_id].append(test_results)
                else:
                    num_test_cases = len(example.get("test", [])) if "test" in example else 1
                    num_test_cases = max(num_test_cases, 1)
                    self.logger.debug(
                        f"Task {task_id} ({difficulty}): empty test_results, "
                        f"treating as all {num_test_cases} tests failed"
                    )
                    failed_results = [0] * num_test_cases
                    results_by_task_id[task_id].append(failed_results)
                    results_by_task_id_and_difficulty[difficulty][task_id].append(failed_results)

        k_list = [1]
        if self.n_repeat >= 10:
            k_list.append(5)
        if self.n_repeat >= 20:
            k_list.append(10)
        if self.n_repeat >= 32:
            k_list.append(16)
        if self.n_repeat >= 40:
            k_list.append(20)
        if self.n_repeat >= 64:
            k_list.append(32)

        self.logger.info(f"Computing pass@k metrics for k={k_list} (n_repeat={self.n_repeat})")

        if results_by_task_id:
            pass_at_k_overall = compute_metrics_from_results(dict(results_by_task_id), k_list=k_list)

            for k in k_list:
                key = f"pass@{k}"
                if key in pass_at_k_overall:
                    final_metrics[key] = pass_at_k_overall[key]
                    self.logger.info(f"Overall {key}: {pass_at_k_overall[key]:.2%}")

        for difficulty in results_by_task_id_and_difficulty:
            if difficulty in results_by_task_id_and_difficulty:
                diff_results = dict(results_by_task_id_and_difficulty[difficulty])
                if diff_results:
                    pass_at_k_diff = compute_metrics_from_results(diff_results, k_list=k_list)

                    for k in k_list:
                        key = f"pass@{k}"
                        if key in pass_at_k_diff:
                            final_metrics[f"pass@{k}_{difficulty}"] = pass_at_k_diff[key]
                            self.logger.info(f"{key} {difficulty}: {pass_at_k_diff[key]:.2%}")

        final_metrics["examples"] = [result for result, _ in results]
        final_metrics["num_total"] = num_questions
        final_metrics["num_repeat"] = self.n_repeat

        return final_metrics

    def load_questions(self) -> Dataset:
        """Load LiveCodeBenchV6 questions from HuggingFace."""
        self.logger.info("Loading LiveCodeBenchV6 questions from livecodebench/code_generation_lite...")
        cpu_count = os.cpu_count()
        lcb_codegen = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True)
        self.logger.info(f"Loaded {len(lcb_codegen)} problems from livecodebench/code_generation_lite")
        ds = lcb_codegen.filter(filter_by_contest_date)
        self.logger.info(f"{len(ds)} problems after date filter (Feb-May 2025)")
        # Avoids "pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays" when mapping
        processed_shards = []
        num_shards = 4
        for i in range(num_shards):
            shard = ds.shard(num_shards=num_shards, index=i)
            shard = shard.map(
                lambda example: {"private_test_cases": translate_private_test_cases(example["private_test_cases"])},
                num_proc=cpu_count,
            )
            shard = shard.map(map_to_example, remove_columns=ds.column_names, load_from_cache_file=False)
            processed_shards.append(shard)
        ds = concatenate_datasets(processed_shards)
        return ds
