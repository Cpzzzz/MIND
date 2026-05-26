import argparse
import json

from dotenv import load_dotenv

from experiment import run_mind_for_prompt


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def probability(value):
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Run MIND on a single user prompt.")
    parser.add_argument("--prompt", required=True, help="User prompt to detoxify through MIND when needed.")

    parser.add_argument("--target-model", default="openrouter/meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--optimizer-model", default="siliconflow/Qwen/Qwen3.6-27B")
    parser.add_argument("--judge-model", default="openrouter/deepseek/deepseek-v4-flash")
    parser.add_argument("--embedding-model", default="siliconflow/BAAI/bge-m3")

    parser.add_argument("--T", type=positive_int, default=3, help="Maximum MIND optimization iterations.")
    parser.add_argument("--tau", type=probability, default=0.1, help="Toxicity threshold for accepting a response.")
    parser.add_argument("--mr-alpha", type=nonnegative_float, default=1.0, help="Similarity weight in memory retrieval (MR).")
    parser.add_argument("--mr-beta", type=nonnegative_float, default=1.0, help="Quality weight in memory retrieval (MR).")
    parser.add_argument("--mr-gamma", type=nonnegative_float, default=1.0, help="Diversity weight in memory retrieval (MR).")
    parser.add_argument("--mm-alpha", type=nonnegative_float, default=1.0, help="Quality weight in MM. In server mode, set this on memory_server.")
    parser.add_argument("--mm-beta", type=nonnegative_float, default=1.0, help="Diversity weight in MM. In server mode, set this on memory_server.")
    parser.add_argument("--lambda-val", dest="lambda_val", type=probability, default=0.5, help="Toxicity-utility tradeoff in Q_d.")
    parser.add_argument("--M", type=positive_int, default=300, help="Active memory size used by MR/MM. In server mode, set this on memory_server.")
    parser.add_argument("--N", type=positive_int, default=100, help="Number of newly collected examples that trigger one MM update. In server mode, set this on memory_server.")
    parser.add_argument("--k", type=positive_int, default=3, help="Number of memory examples retrieved by MR.")

    parser.add_argument("--memory-mode", choices=["server", "local"], default="server")
    parser.add_argument("--memory-server-url", default="http://127.0.0.1:6001")
    parser.add_argument("--memory-dir", default="artifacts/memory/shared")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def print_result(result):
    print("\n=== MIND Result ===")
    print(f"Status: {result['status']}")
    print(f"Initial toxicity: {result['ori_tox_max']:.4f}")
    print("\nOriginal response:")
    print(result["ori_response"])

    if result["status"] == "initial_clean":
        print("\nInitial response is already below tau; MIND optimization was not needed.")
        return

    final_tox = result.get("final_tox_max")
    utility = result.get("utility")
    qd = result.get("Q_d")
    print("\nOptimized prompt:")
    print(result.get("optimized_prompt"))
    print("\nFinal response:")
    print(result.get("final_response"))
    print("\nMetrics:")
    print(f"Final toxicity: {final_tox:.4f}" if final_tox is not None else "Final toxicity: N/A")
    print(f"Utility: {utility:.4f}" if utility is not None else "Utility: N/A")
    print(f"Q_d: {qd:.4f}" if qd is not None else "Q_d: N/A")
    print(f"Memory saved: {result.get('memory_saved')}")


def main():
    load_dotenv()
    args = parse_args()
    result = run_mind_for_prompt(args)
    print_result(result)
    if args.verbose:
        print("\nRaw result:")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
