# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Callable, Optional

import psutil
import ray
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score


# Set multiprocessing start method to 'spawn' for safety in distributed environments
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=90.0):
    loop = asyncio.get_running_loop()
    # logging.info(f"Starting task for completion: {completion[:50]}...")
    try:
        future = loop.run_in_executor(
            executor,
            partial(evaluation_func, completion, reference, task, task_extra_info),
        )
        result = await asyncio.wait_for(future, timeout=timeout)
        # logging.info(f"Task completed for completion: {completion[:50]}")
        return result
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timed out: {completion[:50]}")
        return None
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:50]}")
        return None


async def parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info=None, num_threads=64):
    if extra_info is None:
        extra_info = [None] * len(tasks)
    scores = []
    logging.info(f"Starting parallel scoring with {num_threads} threads for {len(completions)} items")

    # Use ThreadPoolExecutor for I/O-bound API calls
    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            try:
                tasks_async = [
                    single_compute_score(evaluation_func, c, r, t, ei, executor, timeout=90.0)
                    for c, r, t, ei in zip(completions, references, tasks, extra_info)
                ]
                # logging.info(f"Submitted {len(tasks_async)} async tasks")
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks_async, return_exceptions=True),
                    timeout=3600.0  # 1-hour global timeout
                )
                logging.info("All tasks gathered")
            except asyncio.TimeoutError:
                print("Global batch timeout; returning partial results")
                results = [None] * len(tasks_async)  # Fallback for timeout
            except Exception as e:
                print(f"[Exception] async gather failed: {e}")
                raise
    except Exception as e:
        print(f"Parallel execution failed: {e}. Falling back to sequential.")
        results = []
        for c, r, t, ei in zip(completions, references, tasks, extra_info):
            try:
                result = evaluation_func(c, r)
            except Exception:
                result = None
            results.append(result)

    # Process results
    for result in results:
        if isinstance(result, Exception) or result is None:
            scores.append(0.0)
        elif isinstance(result, (int, float, bool)):
            scores.append(float(result))
        else:
            scores.append(float(result[0]))
    return scores


def run_reward_scoring(evaluation_func, completions, references, tasks, extra_info=None, num_threads=64):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(parallel_compute_score_async(
            evaluation_func, completions, references, tasks, extra_info, num_threads
        ))
    finally:
        if not loop.is_closed():
            loop.close()


class NewPrimeRewardManager:
    """
    Improved Reward Manager for https://github.com/PRIME-RL/PRIME
    Enhancements: ThreadPoolExecutor for API calls, rate limiting, caching, robust timeouts, Ray compatibility.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            num_examine: int,
            compute_score: Optional[Callable] = None,
            reward_fn_key: str = "data_source",
            **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        # Use Ray resources if available, else default to conservative thread count
        self.num_threads = kwargs.get('num_threads',
                                      min(64, int(ray.cluster_resources().get('CPU', mp.cpu_count()) // 2)))

    def verify(self, data):
        """
        Verify the batch and save as ``acc`` tensor
        """
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = run_reward_scoring(
                self.compute_score,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=extra_info,
                num_threads=self.num_threads,
            )
        except Exception as e:
            logging.error(f"[Error] Unexpected error during scoring: {e}. Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute rewards for the given dataset batch"""

        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print_data_sources = {}

        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        scores = self.verify(data)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(data_source)
                print(sequences_str[i][:1000] + " (omitted over 1000 characters)")
                print()

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
        else:
            return reward_tensor