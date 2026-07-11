#!/usr/bin/env python3
"""
Minimal command-line chat with mansa-nano. Weights are pulled from the Hugging Face Hub.

Interactive:
  python -m scripts.chat_cli

Single prompt:
  python -m scripts.chat_cli -p "What is the capital of France?"

Disable the persona system prompt:
  python -m scripts.chat_cli --system-prompt "" -p "Write a haiku about the ocean."
"""
import argparse

from nanochat.common import compute_init, autodetect_device_type, DEFAULT_SYSTEM_PROMPT
from nanochat.engine import Engine
from nanochat.model_loader import load_from_hf, DEFAULT_HF_REPO

parser = argparse.ArgumentParser(description='Chat with mansa-nano')
parser.add_argument('--hf-repo', type=str, default=DEFAULT_HF_REPO, help='Hugging Face repo id hosting the weights')
parser.add_argument('--local-dir', type=str, default=None, help='Local dir with checkpoint files (skips HF download)')
parser.add_argument('-s', '--step', type=int, default=None, help='Checkpoint step to load (default: latest on disk)')
parser.add_argument('--revision', type=str, default=None, help='HF revision (branch/tag/commit)')
parser.add_argument('-p', '--prompt', type=str, default='', help='Single-shot prompt, get one response and exit')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Max tokens to generate per turn')
parser.add_argument('--system-prompt', type=str, default=DEFAULT_SYSTEM_PROMPT, help="Persona system prompt prepended to the first user message (empty string to disable)")
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type: cuda|cpu|mps. empty => autodetect')
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_from_hf(
    device, phase="eval", repo_id=args.hf_repo, local_dir=args.local_dir, step=args.step, revision=args.revision
)

bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

engine = Engine(model, tokenizer)

print("\nmansa-nano Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
print("-" * 50)

conversation_tokens = [bos]

while True:
    if args.prompt:
        user_input = args.prompt
    else:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break
    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue
    if not user_input:
        continue

    # Prepend the persona system prompt to the first user message only.
    is_first_user_message = conversation_tokens == [bos]
    message_text = user_input
    if args.system_prompt and is_first_user_message:
        message_text = args.system_prompt + "\n\n" + user_input

    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(message_text))
    conversation_tokens.append(user_end)

    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
        token = token_column[0]
        if token == assistant_end or token == bos:
            break
        response_tokens.append(token)
        print(tokenizer.decode([token]), end="", flush=True)
    print()

    response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    if args.prompt:
        break
