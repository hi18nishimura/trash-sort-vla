"""SmolVLA policy server. Run in the LeRobot env (PYTHONPATH unset):

  source ~/lerobot-env/bin/activate && unset PYTHONPATH
  python tools/policy_server.py --ckpt outputs/train/smolvla_trash_v1/checkpoints/last/pretrained_model

Protocol (multiprocessing.connection, localhost, plain python types only):
  {"cmd": "reset"}                                          -> {"ok": True}
  {"cmd": "act", "image": bytes(HxWx3 uint8 RGB), "h", "w",
   "state": [8 floats], "task": str}                        -> {"action": [8 floats], "ms": float}
"""
import argparse
import time
from multiprocessing.connection import Listener

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".../checkpoints/<step|last>/pretrained_model")
    ap.add_argument("--port", type=int, default=6000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--normalizer", default="",
                    help="LLM id for instruction normalisation, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-action-steps", type=int, default=10,
                    help="execute this many actions of each predicted chunk before re-predicting")
    args = ap.parse_args()

    policy = SmolVLAPolicy.from_pretrained(args.ckpt).to(args.device).eval()
    policy.config.n_action_steps = args.n_action_steps
    policy.reset()
    pre, post = make_pre_post_processors(
        policy.config, args.ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device}})
    print(f"policy loaded from {args.ckpt} (n_action_steps={args.n_action_steps})", flush=True)
    norm = None
    if args.normalizer:
        from normalizer import InstructionNormalizer
        norm = InstructionNormalizer(args.normalizer, args.device)
        print(f"normalizer loaded: {args.normalizer}", flush=True)

    listener = Listener(("127.0.0.1", args.port), authkey=b"trash-vla")
    print(f"listening on 127.0.0.1:{args.port}", flush=True)
    while True:
        conn = listener.accept()
        print("client connected", flush=True)
        try:
            while True:
                msg = conn.recv()
                if msg["cmd"] == "normalize":
                    if norm is None:
                        conn.send({"canonical": msg["text"], "raw": "(no normalizer)"})
                    else:
                        canon, _, _, raw = norm.normalize(msg["text"])
                        conn.send({"canonical": canon, "raw": raw})
                    continue
                if msg["cmd"] == "reset":
                    policy.reset()
                    conn.send({"ok": True})
                    continue
                t0 = time.time()
                img = np.frombuffer(msg["image"], dtype=np.uint8).reshape(msg["h"], msg["w"], 3)
                img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
                state = torch.tensor(msg["state"], dtype=torch.float32).unsqueeze(0)
                batch = {"observation.images.camera1": img_t,   # same rename as in training
                         "observation.state": state,
                         "task": [msg["task"]]}
                with torch.inference_mode():
                    action = post(policy.select_action(pre(batch)))
                conn.send({"action": action.squeeze(0).float().cpu().tolist(),
                           "ms": (time.time() - t0) * 1000})
        except (EOFError, ConnectionResetError):
            print("client disconnected", flush=True)
            conn.close()


if __name__ == "__main__":
    main()
