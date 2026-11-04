"""Agent Evaluation Package Entry Point."""

import sys
from pathlib import Path

def main() -> None:
    print("=" * 70)
    print(" LangChain Agent Evaluation System (DeepEval + Ollama Cloud)")
    print("=" * 70)
    print("\nAvailable commands:")
    print("  1. Run full 7-metric evaluation suite:")
    print("     python evaluations/eval_all.py\n")
    print("  2. Run individual metric evaluation:")
    print("     python evaluations/task_completion.py")
    print("     python evaluations/tool_correctness.py")
    print("     python evaluations/answer_correctness.py\n")
    print("  3. Run interactive refund agent CLI:")
    print("     python agent/agent.py\n")
    print("=" * 70)

if __name__ == "__main__":
    main()
