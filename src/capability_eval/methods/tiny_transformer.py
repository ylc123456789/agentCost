"""CPU integration fixture. Synthetic weights, no training or remote artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..contracts import SolveResult
from ..io import ContractError, digest, hash_file

WIDTH = 32
SLOTS = 3
TARGET = "c_attn"


def save_base(output, runtime):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.set_num_threads(runtime["threads"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(runtime["seed"])
        config = GPT2Config(vocab_size=259, n_positions=runtime["max_input"] + runtime["max_new_tokens"],
                            n_embd=WIDTH, n_layer=2, n_head=2, pad_token_id=0, bos_token_id=1,
                            eos_token_id=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        model = GPT2LMHeadModel(config).eval()
        model.save_pretrained(output, safe_serialization=True)


def save_prefix(output, runtime):
    import torch
    from safetensors.torch import save_file

    generator = torch.Generator(device="cpu").manual_seed(runtime["seed"] + 1)
    tensor = torch.randn((1, SLOTS, WIDTH), generator=generator, dtype=torch.float32) * 0.12
    save_file({"prefix": tensor}, str(output / "prefix.safetensors"))


def _base(base_path):
    from transformers import GPT2LMHeadModel

    return GPT2LMHeadModel.from_pretrained(base_path, local_files_only=True,
                                           use_safetensors=True, attn_implementation="eager").eval()


def save_adapter(output, base_path, runtime):
    import torch
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(_base(base_path), LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4,
                                                        target_modules=[TARGET], fan_in_fan_out=True))
    generator = torch.Generator(device="cpu").manual_seed(runtime["seed"] + 2)
    with torch.no_grad():
        for key, param in model.named_parameters():
            if "lora_A" in key or "lora_B" in key:
                param.copy_(torch.rand(param.shape, generator=generator) * 0.4 - 0.2)
    model.save_pretrained(output, safe_serialization=True)
    # PEFT emits an optional model card; it is not needed to load the actor.
    (output / "README.md").unlink(missing_ok=True)


def _sha_bytes(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _max_gap(left, right):
    return float((left - right).abs().max().item())


class TinyTransformer:
    def load(self, state, runtime_spec, recorder):
        import torch

        torch.set_num_threads(runtime_spec["threads"])
        self.torch = torch
        self.runtime = runtime_spec
        self.files = state.files
        self.carriers = set(state.carrier)
        self.base_path = Path(self.files["base_model"]["path"]).parent
        self.model = _base(self.base_path)
        self.prefix = None
        self.skill = None
        if "text" in self.carriers:
            self.skill = Path(self.files["text"]["path"]).read_text(encoding="utf-8")
        if "input_embeds" in self.carriers:
            from safetensors.torch import load_file

            self.prefix = load_file(self.files["prefix"]["path"], device="cpu")["prefix"]
            if tuple(self.prefix.shape) != (1, SLOTS, self.model.config.n_embd) or not self.prefix.isfinite().all():
                raise ContractError("prefix incompatible with the actor embedding space")
            if not self.prefix.count_nonzero():
                raise ContractError("expected a nonzero prefix")
        if "actor_lora" in self.carriers:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, Path(self.files["lora"]["path"]).parent,
                                                  is_trainable=False).eval()

    def solve(self, public_task, context, recorder):
        torch = self.torch
        with recorder.span("render_and_embed", role="composer"):
            prompt = (f"Skill:\n{self.skill}\n" if self.skill is not None else "") + f"Task:\n{public_task.prompt}\nAnswer:"
            tokens = [1] + [code + 3 for code in prompt.encode("utf-8")]
            n_prefix = 0 if self.prefix is None else self.prefix.shape[1]
            if len(tokens) + n_prefix > self.runtime["max_input"]:
                raise ContractError("input exceeds frozen context limit; no silent truncation")
            text_ids = torch.tensor([tokens], dtype=torch.long)
            model_ids = text_ids
            opts = {}
            with torch.inference_mode():
                if n_prefix:
                    model_ids = torch.cat((torch.zeros((1, n_prefix), dtype=torch.long), text_ids), dim=1)
                    embedding = self.model.get_input_embeddings()(text_ids)
                    opts["inputs_embeds"] = torch.cat((self.prefix.to(embedding.dtype), embedding), dim=1)
            mask = torch.ones_like(model_ids)
            self.tensors = (text_ids, model_ids, mask, opts)
        recorder.measure("input_text_tokens", len(tokens), "tokens", data={"scope": "actor_byte_tokens_with_bos"})
        recorder.measure("prefix_slots", n_prefix, "slots", data={"scope": "actor_continuous_input"})
        with recorder.span("greedy_generate", role="actor"), torch.inference_mode():
            recorder.reserve("generate")
            output = self.model.generate(input_ids=model_ids, attention_mask=mask, **opts,
                                         max_new_tokens=self.runtime["max_new_tokens"], do_sample=False,
                                         pad_token_id=0, eos_token_id=2, use_cache=True)
        if not torch.equal(output[:, :model_ids.shape[1]], model_ids):
            raise ContractError("generate returned a different output boundary")
        new_ids = output[0, model_ids.shape[1]:].tolist()
        raw = bytes(i - 3 for i in new_ids if 3 <= i < 259).decode("utf-8", errors="replace")
        recorder.measure("output_tokens", len(new_ids), "tokens", data={"scope": "new_actor_tokens"})
        lora = self.files.get("lora")
        prefix = self.files.get("prefix")
        return SolveResult(raw, raw, {
            "fixture": "synthetic_tiny_gpt2_byte_tokenizer", "training_calls": 0,
            "base_model_sha256": self.files["base_model"]["sha256"],
            "rendered_prompt": prompt, "rendered_prompt_sha256": _sha_bytes(prompt),
            "text_token_ids_sha256": digest(tokens), "input_text_tokens": len(tokens),
            "input_model_positions": model_ids.shape[1], "new_token_start": model_ids.shape[1],
            "output_token_ids": new_ids, "termination": "eos" if new_ids[-1:] == [2] else "max_new_tokens",
            "mask_shape": list(mask.shape), "mask_values_sha256": digest(mask.tolist()), "positions": "0..n-1",
            "kv_cache_policy": "request-only", "generation": "greedy",
            "text_sha256": self.files.get("text", {}).get("sha256"),
            "prefix": ({"sha256": prefix["sha256"], "positions": [0, n_prefix], "shape": list(self.prefix.shape),
                        "dtype": str(self.prefix.dtype), "injected_shape": list(opts["inputs_embeds"].shape),
                        "nonzero_values": int(self.prefix.count_nonzero())} if prefix else None),
            "actor_lora": ({"sha256": lora["sha256"], "target_module": TARGET, "mode": "dynamic_peft",
                            "enabled": self.model.get_model_status().enabled is True} if lora else None),
        })

    def probe(self, recorder):
        torch = self.torch
        text_ids, ids, mask, opts = self.tensors
        proof = {}
        with torch.inference_mode():
            if self.prefix is not None:
                with recorder.span("prefix_logits_controls", role="actor"):
                    embeddings = opts["inputs_embeds"]
                    zero = embeddings.clone()
                    zero[:, :SLOTS] = 0
                    recorder.reserve("probe")
                    injected = self.model(inputs_embeds=embeddings, attention_mask=mask, use_cache=False).logits[:, -1]
                    recorder.reserve("probe")
                    zero_logits = self.model(inputs_embeds=zero, attention_mask=mask, use_cache=False).logits[:, -1]
                    recorder.reserve("probe")
                    removed = self.model(input_ids=text_ids, attention_mask=torch.ones_like(text_ids),
                                         use_cache=False).logits[:, -1]
                    proof["prefix_logits"] = {"perturbation_max_abs": _max_gap(injected, zero_logits),
                                               "removed_max_abs": _max_gap(injected, removed), "threshold": 1e-6}
            if "actor_lora" in self.carriers:
                with recorder.span("lora_logits_controls", role="actor"):
                    args = dict(input_ids=ids, attention_mask=mask, use_cache=False)
                    recorder.reserve("probe")
                    enabled = self.model(**args).logits[:, -1]
                    with self.model.disable_adapter():
                        recorder.reserve("probe")
                        disabled = self.model(**args).logits[:, -1]
                    recorder.reserve("probe")
                    independent = _base(self.base_path)(**args).logits[:, -1]
                    proof["lora_logits"] = {"enabled_vs_disabled_max_abs": _max_gap(enabled, disabled),
                                             "disabled_vs_base_max_abs": _max_gap(disabled, independent),
                                             "reactivated": self.model.get_model_status().enabled is True,
                                             "threshold": 1e-6}
        return proof

    def close(self):
        self.model = None
