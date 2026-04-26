# Chat Template Verification

## Background

In agentic workflows (multi-turn tool-calling), miles uses sglang's **pretokenized prefix** mechanism to avoid re-tokenizing the entire conversation history on every turn. This requires the chat template to satisfy an **append-only invariant**: rendering the first N messages must produce a string that is an exact prefix of rendering all messages with `add_generation_prompt=True`. Templates that use `loop.last` or similar context-dependent Jinja logic — or that cut historical `reasoning_content` after each new user message — break this property.

miles ships a one-click CLI verifier (`scripts/tools/verify_chat_template.py`) plus a small library of bundled fixed templates for known-bad model families. The same verification library drives the unit tests for the bundled fixed templates, so "the CLI says it passes" and "the unit tests say it passes" share one code path.

## Quick Start

### Verify a HuggingFace model's template

```shell
python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B
```

Example output for a template that **fails** (stock Qwen3 uses `loop.last`):

```
Template source:       HuggingFace: Qwen/Qwen3-0.6B
Allowed append roles:  ['tool']
Thinking mode:         on
Selected trajectories: 6 of 23 (after filtering)

  [FAIL] single_tool_thinking-N3-enable_thinking_on  -- Prefix mismatch!
  [FAIL] multi_turn_thinking-N4-enable_thinking_on   -- Prefix mismatch!
  ...

Results: 0/6 passed, 6 failed

Verdict: FAIL - template is NOT append-only after last user message
```

### Verify with a bundled fixed template

If miles ships a fixed template for the model's TITO tokenizer family, point the verifier at it via `--tito-model`:

```shell
python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B \
    --tito-model qwen3 --thinking both
```

```
Template source:       fixed template: .../templates/qwen3_fixed.jinja
Allowed append roles:  ['tool']
Thinking mode:         both
Selected trajectories: 11 of 23 (after filtering)

  [PASS] single_tool-N3-enable_thinking_on
  [PASS] single_tool-N3-enable_thinking_off
  ...
  [PASS] long_chain_thinking-N6-enable_thinking_off

Results: 22/22 passed, 0 failed

Verdict: PASS - template IS append-only after last user message
```

### Verify a local `.jinja` file

When you are iterating on a custom template:

```shell
python scripts/tools/verify_chat_template.py --template path/to/my_template.jinja --thinking both
```

### Widen or narrow the append-role scope

By default only the `tool` role is assumed as an append target (i.e. the session only appends tool responses on top of assistant-stopped prefixes). If your training pipeline also appends `user` turns (multi-user sessions) or mid-conversation `system` messages, declare the extra roles so their trajectories get exercised:

```shell
# Multi-user-turn sessions (e.g. chat agents that resume across user messages)
python scripts/tools/verify_chat_template.py --model zai-org/GLM-5 \
    --tito-allowed-append-roles tool user --thinking both

# Retry / system-injection sessions
python scripts/tools/verify_chat_template.py --model ... \
    --tito-allowed-append-roles tool system --thinking both
```

`tool` is always implicit — listing it does not hurt; omitting it still includes it. Trajectories requiring roles outside the allow list are skipped.

### Pass template kwargs

Some templates expose a Jinja kwarg to control append-only behavior. The canonical case is GLM's `clear_thinking`: the default (canonical training / inference) cuts historical reasoning after each user message, which breaks append-only under user append. Passing `clear_thinking=false` preserves reasoning across user turns, restoring the invariant:

```shell
python scripts/tools/verify_chat_template.py --model zai-org/GLM-5 \
    --tito-allowed-append-roles tool user --thinking both \
    --chat-template-kwargs clear_thinking=false
```

Values `true` / `false` (case-insensitive) are parsed as bool; everything else is passed as a string. Multiple `KEY=VAL` pairs are accepted.

## CLI Reference

```
usage: verify_chat_template.py (--template PATH | --model MODEL_ID)
                               [--tito-model {default,qwen3,qwen35,qwennext,glm47}]
                               [--tito-allowed-append-roles ROLE [ROLE ...]]
                               [--thinking {off,on,both}]
                               [--chat-template-kwargs KEY=VAL [KEY=VAL ...]]
```

| Argument | Description |
| :--- | :--- |
| `--template PATH` | Path to a local `.jinja` file. |
| `--model MODEL_ID` | HuggingFace model ID (e.g. `Qwen/Qwen3-0.6B`). |
| `--tito-model {default,qwen3,qwen35,qwennext,glm47}` | With `--model`, look up the bundled fixed template registered for this TITO tokenizer family under the given `--tito-allowed-append-roles` surface. Falls back to the HF default chat template when no entry is registered. |
| `--tito-allowed-append-roles {tool,user,system}` | Roles the session may append. Default: `tool`. `tool` is always implicit. Trajectories requiring roles outside this set are skipped. |
| `--thinking {off,on,both}` | `off`: non-thinking trajectories only. `on`: thinking trajectories with `enable_thinking=True` only. `both`: every selected trajectory runs with both `enable_thinking=True` and `enable_thinking=False`. |
| `--chat-template-kwargs KEY=VAL ...` | Extra kwargs threaded into the Jinja render on every case. `true`/`false` parsed as bool. |

Exit code is **0** iff every selected case passes.

## How It Works

For each selected `(trajectory, kwargs)` tuple the verifier:

1. Renders the first N messages with `add_generation_prompt=False` → `prefix_text`.
2. Renders all messages with `add_generation_prompt=True` → `full_text`.
3. Asserts `full_text.startswith(prefix_text)` (the append-only invariant).
4. Asserts that the incremental (prefix + delta render of the remaining messages) path reaches the exact `full_text`.

The trajectory pool and the `(supports_thinking, allowed_append_roles, extra_kwargs)` expansion logic live in `miles/utils/test_utils/chat_template_verify.py`. The CLI and the unit tests share that module; the `run_all_checks` (CLI) and `expand_runs` (pytest) helpers both delegate to the same `enable_thinking_variants` + `format_case_id` kernel, so a passing CLI run corresponds 1-to-1 to passing pytest cases with identical ids.

## Built-in Fixed Templates

miles ships fixed templates for model families with known append-only bugs. The lookup is keyed on `(tito_model, allowed_append_roles)`:

| TITO tokenizer family (`--tito-model`) | Append roles | Fixed template |
| :--- | :--- | :--- |
| `qwen3` (e.g. `Qwen3-4B`, `Qwen3-0.6B`) | `tool` | `qwen3_fixed.jinja` |
| `qwen35` (e.g. `Qwen3.5-0.8B`) | `tool` | `qwen3.5_fixed.jinja` |
| `qwennext` (e.g. `Qwen3-4B-Thinking-2507`, `Qwen3-Next-80B-A3B-Thinking`) | `tool` | `qwen3_thinking_2507_and_next_fixed.jinja` |

Models that are already append-only (GLM-5, GLM-4.7-Flash, GLM-4, Qwen3-Instruct-2507, Qwen3-Next-Instruct, Qwen3-Coder-Next, etc.) need no fix; GLM thinking templates additionally need `chat_template_kwargs={'clear_thinking': False}` on the session side when multi-user-turn append is in scope — see the GLM example above.

### Using a bundled fixed template in training

```shell
python run.py \
    --hf-checkpoint Qwen/Qwen3-4B \
    --tito-model qwen3 \
    ...
```

miles auto-resolves the fixed template path from `(--tito-model, --tito-allowed-append-roles)` at startup; pass an explicit `--chat-template-path /path/to/template.jinja` to override.

## Typical Workflows

### Adding a new model to miles

1. Run the verifier against the model's HF template with the role set / thinking mode your training actually uses:

   ```shell
   python scripts/tools/verify_chat_template.py --model <model-id> \
       --tito-allowed-append-roles tool [user|system ...] \
       --thinking both
   ```

2. If it passes: nothing to do — miles can drive the model directly.

3. If it fails: the `.claude/skills/chat-template-fix` skill walks through the fix process (diagnose which trajectories / jinja branches fail, decide between a `chat_template_kwargs` override or a branch cut, register the fixed template in `miles/utils/chat_template_utils/fixed_templates.py`'s `_FIX_TEMPLATE` table). Iterate with the verifier until the full role × thinking matrix is green.

### Verifying an in-house modification to a bundled template

Edit `miles/utils/chat_template_utils/templates/<name>.jinja` in place, then point the verifier at it:

```shell
python scripts/tools/verify_chat_template.py \
    --template miles/utils/chat_template_utils/templates/qwen3_fixed.jinja \
    --tito-allowed-append-roles tool \
    --thinking both
```

## Running Tests

The same verification library drives `tests/fast/utils/chat_template_utils/`:

```shell
# Append-only invariant for all bundled fixed templates + native HF templates
python -m pytest tests/fast/utils/chat_template_utils/test_pretokenized_chat.py -q

# Alignment between apply_chat_template and SGLang's _process_messages
python -m pytest tests/fast/utils/chat_template_utils/test_template.py -q

# Full directory
python -m pytest tests/fast/utils/chat_template_utils/ -q
```

When adding a new fixed template, add a corresponding entry to `_TEMPLATES` in `test_pretokenized_chat.py` (with its declared `allowed_append_roles` and any `extra_template_kwargs`) so the template is guarded against regression.
