import logging
import re

from utils.llm.llm_calling import call_model


OPTIMIZER_PARSE_ATTEMPTS = 5

OPTIMIZE_PROMPT = """
Optimize the prompt for safer target-model output while preserving the user's concrete safe intent.

Rules:
- Rewrite toxic wording in place rather than changing the task.
- If the original is a sentence fragment, keep it as a sentence fragment.
- Preserve named entities, topic, tense, and approximate length when safe.
- Neutralize slurs, profanity, identity attacks, threats, and demeaning insults.
- Do not copy toxic wording, including masked or censored forms; use neutral labels such as "offensive wording" or "a racial slur".
- If toxic wording appears inside a name or title, replace only the toxic part with a neutral substitute.
- Do not start with "analyze", "discuss", "explain", or "review" unless the original prompt asks for that.
- Return one concise natural prompt.

Original prompt:
```
{original_prompt}
```

Output format:
<prompt>Optimized version</prompt>
"""

OPTIMIZE_PROMPT_WITH_EXAMPLES = """
Optimize the prompt for safer target-model output while preserving the user's concrete safe intent.

Use the examples as guidance for transformation strategy, not as wording to copy.
```
{examples}
```

Rules:
- Rewrite toxic wording in place rather than changing the task.
- If the original is a sentence fragment, keep it as a sentence fragment.
- Preserve named entities, topic, tense, and approximate length when safe.
- Neutralize slurs, profanity, identity attacks, threats, and demeaning insults.
- Do not copy toxic wording, including masked or censored forms; use neutral labels such as "offensive wording" or "a racial slur".
- If toxic wording appears inside a name or title, replace only the toxic part with a neutral substitute.
- Do not start with "analyze", "discuss", "explain", or "review" unless the original prompt asks for that.
- Return one concise natural prompt.

Original prompt:
```
{original_prompt}
```

Output format:
<prompt>Optimized version</prompt>
"""


def _parse_prompt(response: str):
    pattern = r"<prompt>\s*(.*?)\s*</prompt>"
    matches = re.findall(pattern, response or "", re.DOTALL)
    prompts = [match.strip() for match in matches if match.strip()]
    return prompts[0] if prompts else None


def _call_optimizer(prompt: str, task: str, args):
    for attempt in range(OPTIMIZER_PARSE_ATTEMPTS):
        response = call_model(
            args.opt_model,
            prompt,
            task=task,
        )
        parsed = _parse_prompt(response)
        if parsed:
            logging.info("Generated optimized prompt: %s", parsed)
            return parsed

        logging.warning(
            "Optimizer attempt %s/%s failed to return <prompt>...</prompt>.",
            attempt + 1,
            OPTIMIZER_PARSE_ATTEMPTS,
        )

    raise RuntimeError(f"All {OPTIMIZER_PARSE_ATTEMPTS} optimizer attempts failed.")


def optimize_without_example(original_prompt, args):
    logging.info("Optimizing prompt without memory examples.")
    optimize_prompt = OPTIMIZE_PROMPT.format(original_prompt=original_prompt)
    return _call_optimizer(optimize_prompt, "[Optimizer] Generate Prompt", args)


def optimize_with_examples(original_prompt, examples, args):
    logging.info("Optimizing prompt with %s memory examples.", len(examples))
    examples_str = "\n\n".join(
        f"example {i + 1}:\nOriginal Prompt: {example['ori_prompt']}\nUpdated Prompt: {example['upd_prompt']}"
        for i, example in enumerate(examples)
    )
    optimize_prompt = OPTIMIZE_PROMPT_WITH_EXAMPLES.format(
        examples=examples_str,
        original_prompt=original_prompt,
    )
    return _call_optimizer(optimize_prompt, f"[Optimizer] With {len(examples)} Examples", args)
