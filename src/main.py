"""
CBRN-Sentinel Main Entry Point (V2 - GAIA Style)
Follows Plan → Act → Verify → Answer pattern
"""
import asyncio
import argparse
import logging
import json
import os
import sys
import datetime
from dotenv import load_dotenv

# Import new GAIA-style components
from src.agent import CBRNAgent, CBRNPlanner, CBRNVerifier
from src.modules.tap import TapEngine
from src.modules.judging import LlamaGuardJudge
from src.modules.scenario_gen import AutoRedGenerator
from src.interfaces.llm_provider import LLMFactory
from src.interfaces.a2a_client import A2AClient
from src.benchmarks.loader import BenchmarkLoader

load_dotenv()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CBRN-Sentinel.Main")


# Difficulty weights (inspired by AQA competitor)
DIFFICULTY_WEIGHTS = {
    "easy": 1,
    "medium": 2,
    "hard": 3,
    "expert": 4
}


def calculate_weighted_score(results: list) -> float:
    """Calculate weighted score based on difficulty. Formula from AQA."""
    if not results:
        return 0.0
    
    weighted_sum = 0.0
    weight_total = 0.0
    
    for r in results:
        difficulty = r.get('difficulty', 'medium')
        weight = DIFFICULTY_WEIGHTS.get(difficulty, 2)
        score = r.get('baseline_score', 0.0)
        weighted_sum += weight * score
        weight_total += weight
    
    return weighted_sum / weight_total if weight_total > 0 else 0.0


def save_report(results: list, benchmark_name: str, total_usage: dict):
    """Generates a Markdown summary following GAIA style with weighted scoring."""
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = benchmark_name.replace("/", "_").replace("\\", "_")
    filename = f"reports/summary_{safe_name}_{timestamp}.md"
    
    # Cost Calculation (GPT-4o tier)
    input_cost = (total_usage["prompt_tokens"] / 1_000_000) * 2.50
    output_cost = (total_usage["completion_tokens"] / 1_000_000) * 10.00
    total_cost = input_cost + output_cost
    
    # Aggregate metrics (simple average)
    if results:
        avg_baseline = sum(r['baseline_score'] for r in results) / len(results)
        avg_attack = sum(r['attack_score'] for r in results) / len(results)
        avg_delta = avg_baseline - avg_attack
    else:
        avg_baseline = avg_attack = avg_delta = 0.0
    
    # Weighted score (difficulty-adjusted)
    weighted_score = calculate_weighted_score(results)
    
    # Difficulty distribution
    diff_counts = {"easy": 0, "medium": 0, "hard": 0, "expert": 0}
    for r in results:
        diff = r.get('difficulty', 'medium')
        diff_counts[diff] = diff_counts.get(diff, 0) + 1
    
    md = f"""# CBRN-Sentinel Assessment Report (GAIA Style)
**Date**: {datetime.datetime.now()}
**Benchmark**: `{benchmark_name}`
**Cost**: ${total_cost:.4f} (Input: {total_usage['prompt_tokens']}, Output: {total_usage['completion_tokens']})

## Architecture
Following GAIA pattern: **Plan → Act → Verify → Answer**

## Aggregate Metrics
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Baseline Score** | {avg_baseline:.2f} | Safety without adversarial pressure |
| **Attack Score** | {avg_attack:.2f} | Safety under TAP attack |
| **Defense Delta** | {avg_delta:.2f} | Gap (Baseline - Attack) |
| **Weighted Score** | {weighted_score:.2f} | Difficulty-adjusted (Easy=1x, Medium=2x, Hard=3x, Expert=4x) |

## Difficulty Distribution
| Difficulty | Count | Weight |
| :--- | :--- | :--- |
| Easy | {diff_counts['easy']} | 1x |
| Medium | {diff_counts['medium']} | 2x |
| Hard | {diff_counts['hard']} | 3x |
| Expert | {diff_counts['expert']} | 4x |

## Detailed Results
"""
    
    for i, result in enumerate(results):
        verification = result.get("verification", {})
        difficulty = result.get('difficulty', 'medium')
        weight = DIFFICULTY_WEIGHTS.get(difficulty, 2)
        md += f"""
### Task {i+1}: {result.get('task_id', 'unknown')}
**Domain**: `{result.get('domain', 'unknown')}` | **Difficulty**: `{difficulty.upper()}` ({weight}x)
**Source**: `{result.get('source', 'unknown')}`
**Prompt**: "{result.get('prompt', 'N/A')[:100]}..."

| Metric | Value |
| :--- | :--- |
| Baseline | {result.get('baseline_score', 0.0):.2f} |
| Attack | {result.get('attack_score', 0.0):.2f} |
| Delta | {result.get('defense_delta', 0.0):.2f} |
| Steps | {result.get('steps_completed', 0)}/{result.get('steps_total', 0)} |

**Verification**: {'✅ PASS' if verification.get('verified') else '❌ FAIL'} (Confidence: {verification.get('confidence', 0.0):.2f})
"""
        if verification.get("issues"):
            md += f"**Issues**: {', '.join(verification['issues'])}\n"
    
    with open(filename, "w") as f:
        f.write(md)
    logger.info(f"Report saved to {filename}")
    return filename


async def main():
    parser = argparse.ArgumentParser(description="CBRN-Sentinel (GAIA Style)")
    parser.add_argument("--key", help="API Key (optional if in .env)", default=None)
    parser.add_argument("--provider", help="LLM Provider", default="openai")
    parser.add_argument("--model", help="Model for TAP attacks", default="gpt-5-nano")
    parser.add_argument("--judge_model", help="Model for Judge (hybrid)", default=None)
    parser.add_argument("--base_url", help="Custom API Base URL", default=None)
    parser.add_argument("--dataset", help="Benchmark to run", default="tests/demo_case.csv")
    parser.add_argument("--limit", help="Number of items", type=int, default=3)
    parser.add_argument("--seed", help="Random seed for reproducibility", type=int, default=42)
    args = parser.parse_args()

    # 1. Resolve Credentials
    api_key = args.key
    if not api_key:
        if args.provider == "deepseek": api_key = os.getenv("DEEPSEEK_API_KEY")
        elif args.provider == "google": api_key = os.getenv("GOOGLE_API_KEY")
        elif args.provider == "openai": api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.critical("No API key configured. Exiting.")
        sys.exit(1)

    # 2. Create LLM Interfaces
    logger.info(f"Initializing provider: {args.provider}")
    llm_interface = LLMFactory.create(
        provider=args.provider,
        api_key=api_key,
        base_url=args.base_url,
        model=args.model
    )
    
    # Hybrid mode: separate judge
    if args.judge_model:
        logger.info(f"HYBRID MODE: Using {args.judge_model} for Judge")
        judge_interface = LLMFactory.create(
            provider=args.provider,
            api_key=api_key,
            base_url=args.base_url,
            model=args.judge_model
        )
    else:
        judge_interface = llm_interface

    # 3. Create GAIA-Style Components
    logger.info("=== GAIA-Style Architecture ===")
    
    # Planner
    planner = CBRNPlanner(llm_interface=llm_interface)
    logger.info("✓ Planner initialized")
    
    # Verifier
    verifier = CBRNVerifier(judge_interface=judge_interface)
    logger.info("✓ Verifier initialized")
    
    # Tools
    generator = AutoRedGenerator(llm_interface=llm_interface)
    judge = LlamaGuardJudge(judge_interface=judge_interface)
    tap = TapEngine(generator, judge)
    
    target_client = A2AClient(transport_interface=llm_interface)
    
    tools = {
        "target_agent": target_client,
        "safety_judge": judge,
        "tap_engine": tap,
        "domain_validator": None  # Optional
    }
    logger.info("✓ Tools configured (target, judge, TAP)")
    
    # Agent
    agent = CBRNAgent(
        planner=planner,
        verifier=verifier,
        tools=tools,
        seed=args.seed
    )
    logger.info("✓ CBRNAgent ready")

    # 4. Load Data
    loader = BenchmarkLoader()
    
    if args.dataset.endswith(".csv"):
        items = loader.load_csv_dataset(args.dataset, args.limit)
    elif args.dataset == "wmdp-bio":
        items = loader.load_wmdp("bio", args.limit)
    elif args.dataset == "fortress":
        items = loader.load_fortress(args.limit)
    else:
        items = loader.load_csv_dataset(args.dataset, args.limit)
    
    logger.info(f"Loaded {len(items)} items from {args.dataset}")
    
    if not items:
        logger.critical("No items loaded. Exiting.")
        sys.exit(1)

    # 5. Execute Tasks (GAIA Style)
    logger.info("=== Starting Evaluation ===")
    results = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    
    for item in items:
        task = {
            "id": item.id,
            "prompt": item.prompt,
            "domain": item.domain,
            "difficulty": item.difficulty,
            "source_benchmark": item.source_benchmark
        }
        
        result = await agent.execute(task)
        result['difficulty'] = item.difficulty  # Add difficulty to result for reporting
        results.append(result)
        
        # Log progress
        v = result.get("verification", {})
        logger.info(f"Task {item.id}: {'PASS' if v.get('verified') else 'FAIL'} "
                   f"(Baseline: {result.get('baseline_score', 0):.2f}, "
                   f"Delta: {result.get('defense_delta', 0):.2f})")

    # 6. Generate Report
    report_path = save_report(results, args.dataset, total_usage)
    
    # 7. Output JSON Summary
    summary = {
        "architecture": "GAIA-Style (Plan → Act → Verify → Answer)",
        "tasks_completed": len(results),
        "avg_baseline": sum(r['baseline_score'] for r in results) / len(results) if results else 0,
        "avg_attack": sum(r['attack_score'] for r in results) / len(results) if results else 0,
        "avg_delta": sum(r['defense_delta'] for r in results) / len(results) if results else 0,
        "verification_pass_rate": sum(1 for r in results if r.get('verification', {}).get('verified')) / len(results) if results else 0,
        "report": report_path
    }
    
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
