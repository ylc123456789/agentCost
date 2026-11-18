"""Standalone Inspect AI task: one synthetic prompt, original solver, local CPU model."""
import json
import os
from pathlib import Path

import inspect_ai
import torch
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelAPI, ModelOutput, ModelUsage, modelapi
from inspect_ai.solver import generate
from transformers import GPT2Config, GPT2LMHeadModel


def recorder():
    if "CAPABILITY_EVAL_RECORDER" in os.environ:
        from capability_eval import Recorder

        return Recorder.from_environment()
    return None


REC = recorder()


@modelapi("local-tiny-cpu")
def tiny_backend():
    class LocalTiny(ModelAPI):
        def __init__(self, model_name, base_url=None, api_key=None, config=None, **kwargs):
            super().__init__(model_name, base_url=base_url, api_key=api_key, config=config)
            torch.set_num_threads(1)
            torch.manual_seed(23)
            self.actor = GPT2LMHeadModel(GPT2Config(vocab_size=259, n_positions=256, n_embd=16, n_layer=1,
                                                   n_head=1, pad_token_id=0, bos_token_id=1, eos_token_id=2)).eval()

        async def generate(self, input, tools, tool_choice, config):
            if tools or len(input) != 1:
                raise ValueError("standalone example expects exactly one message and no tools")
            tokens = [1, *[value + 3 for value in input[0].text.encode("utf-8")]]
            ids = torch.tensor([tokens])
            if REC:
                REC.reserve("inspect_generate")
            with torch.inference_mode():
                output = self.actor.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=3,
                                             do_sample=False, pad_token_id=0, use_cache=True)
            new_ids = output[0, len(tokens):].tolist()
            # Use a lossless token list in the synthetic result, not a quality claim.
            result = ModelOutput.from_content(self.model_name, json.dumps(new_ids))
            result.usage = ModelUsage(input_tokens=len(tokens), output_tokens=len(new_ids),
                                      total_tokens=len(tokens) + len(new_ids))
            if REC:
                REC.measure("inspect_input_tokens", len(tokens), "tokens", data={"scope": "sdk_backend"})
                REC.measure("inspect_output_tokens", len(new_ids), "tokens", data={"scope": "sdk_backend"})
            return result

    return LocalTiny


def main():
    prompt = "What is 2 + 2?"
    if "CAPABILITY_EVAL_REQUEST" in os.environ:
        prompt = json.loads(Path(os.environ["CAPABILITY_EVAL_REQUEST"]).read_text())["task"]["prompt"]
    log, = inspect_ai.eval(inspect_ai.Task(dataset=[Sample(input=prompt, id="inspect-local-1")],
                                          solver=generate(cache=False)), model="local-tiny-cpu/test", display="none",
                           max_samples=1, retry_on_error=0, epochs=1, score=False, log_dir="inspect-log", log_format="json")
    if log.status != "success" or not log.samples or not log.samples[0].output:
        raise RuntimeError(f"Inspect run ended with {log.status}: {log.error}")
    print(json.dumps({"runner": "inspect_ai", "status": log.status, "sample_id": log.samples[0].id,
                      "completion": log.samples[0].output.completion}, sort_keys=True))


if __name__ == "__main__":
    main()
