"""Instruction normalizer (method B1): map ANY phrasing to the canonical training instruction.

The LLM only CHOOSES an item and a bin colour from a closed list (JSON); the canonical sentence
is then built from a fixed template, so the VLA always receives one of the 9 training phrasings.
Few-shot examples use training-style wording only (never the evaluation wording).

Run in the LeRobot env (PYTHONPATH unset).
"""
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import phrasing

# keyword matching, so slightly different answers ("PET bottle", "ペットボトル") are accepted.
# order matters: check bottle / cup before can.
ITEM_WORDS = [("bottle", ["bottle", "ペットボトル"]),
              ("paper", ["cup", "paper", "紙コップ", "コップ"]),
              ("can", ["can", "缶"])]
COLOR_WORDS = [("bin_blue", ["blue", "青"]),
               ("bin_green", ["green", "緑"]),
               ("bin_orange", ["orange", "オレンジ", "橙"])]


def _match(value, table):
    v = str(value).strip().lower()
    if not v or "unknown" in v:
        return None
    for key, words in table:
        if any(w in v for w in words):
            return key
    return None

SYSTEM = ("You control a trash-sorting robot. On the table there are exactly three items: "
          "can, plastic bottle, paper cup. There are exactly three bins: blue, green, orange.\n"
          "Read the user's request and decide which item should go into which bin.\n"
          'Answer with JSON only: {"item": "<can|plastic bottle|paper cup|unknown>", '
          '"color": "<blue|green|orange|unknown>"}\n'
          'Use "unknown" if the request does not clearly name one of them.\n'
          'Requests may be in Japanese. In this robot, the Japanese words mean: '
          '青 / 青い = blue (NOT green), 緑 = green, オレンジ / 橙 = orange, '
          '缶 = can, ペットボトル = plastic bottle, 紙コップ = paper cup.')

# training-style phrasings only, so the evaluation wording is never shown to the LLM
EXAMPLES = [
    ("put the soda can into the green box", {"item": "can", "color": "green"}),
    ("drop the aluminum can in the blue tray", {"item": "can", "color": "blue"}),
    ("throw the PET bottle into the orange container", {"item": "plastic bottle", "color": "orange"}),
    ("the used paper cup goes in the blue tray", {"item": "paper cup", "color": "blue"}),
    ("move the arm a little to the left", {"item": "unknown", "color": "unknown"}),
]


class InstructionNormalizer:
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct", device="cuda"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    def _messages(self, text):
        msgs = [{"role": "system", "content": SYSTEM}]
        for req, ans in EXAMPLES:
            msgs.append({"role": "user", "content": f"Request: {req}"})
            msgs.append({"role": "assistant", "content": json.dumps(ans)})
        msgs.append({"role": "user", "content": f"Request: {text}"})
        return msgs

    @torch.inference_mode()
    def parse(self, text):
        """-> (item key or None, bin id or None, raw LLM output)"""
        prompt = self.tok.apply_chat_template(self._messages(text), tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**ids, max_new_tokens=40, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        m = re.search(r"\{.*?\}", raw, re.S)
        try:
            ans = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            ans = {}
        item = _match(ans.get("item", ""), ITEM_WORDS)
        bin_id = _match(ans.get("color", ""), COLOR_WORDS)
        return item, bin_id, raw

    def normalize(self, text):
        """-> (canonical instruction or None, item key, bin id, raw LLM output)"""
        item, bin_id, raw = self.parse(text)
        if item is None or bin_id is None:
            return None, item, bin_id, raw
        return phrasing.canonical(item, bin_id), item, bin_id, raw
