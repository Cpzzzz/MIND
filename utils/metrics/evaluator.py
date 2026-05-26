import logging
from typing import Any, Dict

from utils.metrics.llm_judge import LLMJudge
from utils.metrics.toxicity import analyze_text_toxicity


class UnifiedEvaluator:
    """Evaluator used by the release runner: toxicity plus LLM-judged utility."""

    def __init__(self, eva_model: str):
        if not eva_model:
            raise ValueError("A utility evaluator model must be provided.")
        self.eva_model = eva_model
        self._llm_judge = None

    @property
    def llm_judge(self):
        if self._llm_judge is None:
            self._llm_judge = LLMJudge(model_tag=self.eva_model)
        return self._llm_judge

    def evaluate(
        self,
        ori_prompt: str,
        upd_response: str,
        tox_details: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        result = {}

        try:
            tox_details = tox_details or analyze_text_toxicity(upd_response)
            if not tox_details:
                raise RuntimeError("Toxicity API returned an empty result.")
            tox_values = list(tox_details.values())
            result["tox_details"] = tox_details
            result["tox_max"] = max(tox_values)
            result["tox_avg"] = sum(tox_values) / len(tox_values)
        except Exception as error:
            logging.error("Toxicity evaluation failed: %s", error)
            result["tox_details"] = {}
            result["tox_max"] = None
            result["tox_avg"] = None

        try:
            llm_scores = self.llm_judge.evaluate(ori_prompt, upd_response)
            intent = llm_scores.get("intent_preservation")
            helpful = llm_scores.get("helpfulness")
            informative = llm_scores.get("informativeness")
            if intent is None or helpful is None or informative is None:
                raise RuntimeError("Missing LLM judge sub-scores.")

            result["intent_preservation"] = intent
            result["helpfulness"] = helpful
            result["informativeness"] = informative
            result["utility"] = (intent + helpful + informative) / 3.0
        except Exception as error:
            logging.error("LLM judge evaluation failed: %s", error)
            result["intent_preservation"] = None
            result["helpfulness"] = None
            result["informativeness"] = None
            result["utility"] = None

        return result
