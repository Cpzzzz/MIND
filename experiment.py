import logging
from types import SimpleNamespace
from typing import Any, Dict

import requests

from methods.mind.mind import MIND
from utils.llm.llm_calling import call_model, get_call_stats, reset_call_stats
from utils.memory.db import MemorySystem, RemoteMemorySystem
from utils.metrics.toxicity import analyze_text_toxicity


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _toxicity_metrics(text: str) -> Dict[str, Any]:
    tox_details = analyze_text_toxicity(text)
    if not tox_details:
        return {
            "tox_details": {},
            "tox_max": None,
            "tox_avg": None,
        }
    values = list(tox_details.values())
    return {
        "tox_details": tox_details,
        "tox_max": max(values),
        "tox_avg": sum(values) / len(values),
    }


def _build_mind_args(args, memory_system):
    return SimpleNamespace(
        llm_model=args.target_model,
        opt_model=args.optimizer_model,
        eva_model=args.judge_model,
        embedding_model=args.embedding_model,
        memory_system=memory_system,
        T=args.T,
        tau=args.tau,
        mr_alpha=args.mr_alpha,
        mr_beta=args.mr_beta,
        mr_gamma=args.mr_gamma,
        mm_alpha=args.mm_alpha,
        mm_beta=args.mm_beta,
        lambda_val=args.lambda_val,
        n_examples=args.k,
        dataset="user_prompt",
        split="interactive",
        exp_id="interactive",
        config_id="interactive",
    )


def _build_memory_system(args):
    if args.memory_mode == "server":
        try:
            response = requests.get(f"{args.memory_server_url.rstrip('/')}/active/all", timeout=5)
            response.raise_for_status()
        except Exception as error:
            raise RuntimeError(
                "Memory server is not reachable. Start it with "
                "`python -m utils.memory.memory_server` or use `--memory-mode local`."
            ) from error
        logging.info("Using shared memory server. MM parameters are configured when the server starts.")
        return RemoteMemorySystem(args.memory_server_url)

    return MemorySystem(
        persist_directory=args.memory_dir,
        active_cap=args.M,
        update_batch_size=args.N,
        embedding_model=args.embedding_model,
        mm_alpha=args.mm_alpha,
        mm_beta=args.mm_beta,
        lambda_val=args.lambda_val,
    )


def run_mind_for_prompt(args) -> Dict[str, Any]:
    setup_logging(args.verbose)

    memory_system = _build_memory_system(args)

    reset_call_stats()
    ori_response = call_model(
        args.target_model,
        args.prompt,
        task="[Initial] Generate",
    )
    if not ori_response:
        raise RuntimeError("Initial target model returned no response.")

    init_tox = _toxicity_metrics(ori_response)
    init_usage = get_call_stats()
    if init_tox["tox_max"] is None:
        raise RuntimeError("Initial toxicity evaluation failed.")

    result = {
        "status": "initial_clean",
        "ori_prompt": args.prompt,
        "ori_response": ori_response,
        "ori_tox": init_tox["tox_details"],
        "ori_tox_max": init_tox["tox_max"],
        "ori_tox_avg": init_tox["tox_avg"],
        "initial_usage": init_usage,
    }

    if init_tox["tox_max"] <= args.tau:
        return result

    mind_args = _build_mind_args(args, memory_system)
    rets = SimpleNamespace(
        ori_prompt=args.prompt,
        ori_response=ori_response,
        ori_tox=init_tox["tox_details"],
        tox_max=init_tox["tox_max"],
        init_metrics={
            "tox_details": init_tox["tox_details"],
            "tox_max": init_tox["tox_max"],
            "tox_avg": init_tox["tox_avg"],
            "utility": None,
        },
    )

    reset_call_stats()
    MIND.mind(mind_args, rets)
    mind_usage = get_call_stats()

    best_record = getattr(rets, "best_record", {}) or {}
    result.update(
        {
            "status": "mind_detoxified",
            "optimized_prompt": getattr(rets, "upd_prompt", None),
            "final_response": getattr(rets, "upd_response", None),
            "final_tox_max": getattr(rets, "upd_tox_max", None),
            "final_tox": best_record.get("upd_tox", {}),
            "utility": best_record.get("utility"),
            "intent_preservation": best_record.get("intent_preservation"),
            "helpfulness": best_record.get("helpfulness"),
            "informativeness": best_record.get("informativeness"),
            "Q_d": best_record.get("Q_d"),
            "delta_tox": best_record.get("delta_tox"),
            "memory_saved": best_record.get("memory_saved"),
            "mind_usage": mind_usage,
        }
    )
    return result
