"""
Test Scenarios: Chatbot vs ReAct Agent
Runs real test cases with real LLM and generates logs for Lab 3 reports.

Usage:
  python test_scenarios.py --provider google
  python test_scenarios.py --provider local
"""
import argparse
import json
import os
import sys
import time
import warnings

# Fix Unicode for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning, module="google")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from src.agent.agent import ReActAgent
from src.agent.chatbot import BaselineChatbot
from src.core.gemini_provider import GeminiProvider
from src.core.local_provider import LocalProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker
from src.tools.movie_booking_tools import get_tools


# ──────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "TC01",
        "name": "Multi-step booking (happy path)",
        "input": "Tìm phim hành động gần Royal City, 2 vé tối nay dưới 250k",
        "expect": "multi-step: recommend -> hold -> promo -> final answer",
    },
    {
        "id": "TC02",
        "name": "Simple genre query",
        "input": "Có phim tình cảm nào đang chiếu ở Hà Nội không?",
        "expect": "recommend showtimes with horror genre",
    },
    {
        "id": "TC03",
        "name": "Specific movie + budget",
        "input": "Tôi muốn xem Dune Part Two, rạp gần Cầu Giấy, ngân sách 200k cho 2 người",
        "expect": "search for Dune, hold seats, apply promo",
    },
    {
        "id": "TC04",
        "name": "Vague request (stress test)",
        "input": "Đặt vé xem phim tối nay đi",
        "expect": "agent should ask for clarification or use defaults",
    },
    {
        "id": "TC05",
        "name": "Student discount scenario",
        "input": "Tôi là sinh viên, muốn xem phim hài gần Thanh Xuân, 1 vé thôi, càng rẻ càng tốt",
        "expect": "should use student discount",
    },
]


def build_llm(provider: str):
    if provider == "google":
        api_key = os.getenv("GEMINI_API_KEY")
        return GeminiProvider(model_name="gemini-2.0-flash", api_key=api_key)
    elif provider == "local":
        model_path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")
        return LocalProvider(model_path=model_path, n_ctx=2048)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def run_chatbot_test(llm, test_case: dict) -> dict:
    """Run a single test case through the baseline chatbot."""
    chatbot = BaselineChatbot(llm)
    tracker.reset()

    start = time.perf_counter()
    answer = chatbot.chat(test_case["input"])
    duration_ms = int((time.perf_counter() - start) * 1000)

    summary = tracker.summary()
    return {
        "test_id": test_case["id"],
        "test_name": test_case["name"],
        "mode": "chatbot",
        "input": test_case["input"],
        "answer": answer,
        "duration_ms": duration_ms,
        "metrics": summary,
    }


def run_agent_test(llm, test_case: dict) -> dict:
    """Run a single test case through the ReAct agent."""
    agent = ReActAgent(llm=llm, tools=get_tools(), max_steps=6)
    tracker.reset()

    start = time.perf_counter()
    answer = agent.run(test_case["input"])
    duration_ms = int((time.perf_counter() - start) * 1000)

    summary = tracker.summary()
    return {
        "test_id": test_case["id"],
        "test_name": test_case["name"],
        "mode": "agent",
        "input": test_case["input"],
        "answer": answer,
        "duration_ms": duration_ms,
        "steps": len(agent.history),
        "history": agent.history,
        "metrics": summary,
    }


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run test scenarios for Lab 3")
    parser.add_argument("--provider", choices=["google", "local"], default="google")
    parser.add_argument("--cases", type=str, default="all",
                        help="Comma-separated test case IDs (e.g. TC01,TC02) or 'all'")
    args = parser.parse_args()

    llm = build_llm(args.provider)
    print(f"Provider: {args.provider} | Model: {llm.model_name}")
    print("=" * 70)

    # Filter test cases
    if args.cases == "all":
        cases = TEST_CASES
    else:
        ids = [c.strip().upper() for c in args.cases.split(",")]
        cases = [tc for tc in TEST_CASES if tc["id"] in ids]

    all_results = []

    for tc in cases:
        print(f"\n{'='*70}")
        print(f"[{tc['id']}] {tc['name']}")
        print(f"Input: {tc['input']}")
        print(f"{'='*70}")

        # Run chatbot
        print("\n--- Chatbot Baseline ---")
        try:
            cb_result = run_chatbot_test(llm, tc)
            print(f"Answer: {cb_result['answer'][:200]}...")
            print(f"Duration: {cb_result['duration_ms']}ms | Tokens: {cb_result['metrics']['total_tokens']}")
            all_results.append(cb_result)
        except Exception as e:
            print(f"[ERROR] Chatbot failed: {e}")
            all_results.append({"test_id": tc["id"], "mode": "chatbot", "error": str(e)})

        # Small delay to avoid rate limiting
        time.sleep(1)

        # Run agent
        print("\n--- ReAct Agent ---")
        try:
            ag_result = run_agent_test(llm, tc)
            print(f"Answer: {ag_result['answer'][:200]}...")
            print(f"Duration: {ag_result['duration_ms']}ms | Steps: {ag_result['steps']} | Tokens: {ag_result['metrics']['total_tokens']}")
            all_results.append(ag_result)
        except Exception as e:
            print(f"[ERROR] Agent failed: {e}")
            all_results.append({"test_id": tc["id"], "mode": "agent", "error": str(e)})

        # Delay between test cases
        time.sleep(2)

    # Save results
    os.makedirs("test_results", exist_ok=True)
    result_file = f"test_results/results_{args.provider}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {result_file}")
    print(f"{'='*70}")

    # Print summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'ID':<6} {'Mode':<10} {'Duration':<12} {'Tokens':<10} {'Steps':<7} {'Status'}")
    print("-" * 65)
    for r in all_results:
        if "error" in r:
            print(f"{r['test_id']:<6} {r['mode']:<10} {'N/A':<12} {'N/A':<10} {'N/A':<7} ERROR: {r['error'][:30]}")
        else:
            steps = r.get("steps", "-")
            print(f"{r['test_id']:<6} {r['mode']:<10} {r['duration_ms']:<12} {r['metrics']['total_tokens']:<10} {str(steps):<7} OK")


if __name__ == "__main__":
    main()
