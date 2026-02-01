"""
CBRN-Sentinel A2A Server
AgentBeats-compliant server with required CLI args: --host, --port, --card-url
"""
import argparse
import asyncio
import logging
import json
import os
from typing import Dict, Any
from dotenv import load_dotenv

# FastAPI for A2A server
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    raise ImportError("FastAPI and uvicorn required. Run: pip install fastapi uvicorn")

# Import GAIA-style components
from src.agent import CBRNAgent, CBRNPlanner, CBRNVerifier
from src.modules.tap import TapEngine
from src.modules.judging import LlamaGuardJudge
from src.modules.scenario_gen import AutoRedGenerator
from src.interfaces.llm_provider import LLMFactory
from src.interfaces.a2a_client import A2AClient
from src.benchmarks.loader import BenchmarkLoader

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CBRN-Sentinel.Server")

app = FastAPI(
    title="CBRN-Sentinel Green Agent",
    description="AgentBeats Green Agent for CBRN Safety Evaluation",
    version="1.0.0"
)

# Global agent instance
agent: CBRNAgent = None
card_url: str = None


def init_agent():
    """Initialize the CBRN Agent with default configuration."""
    global agent
    
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("No API key found. Agent will use mock responses.")
        return
    
    provider = "google" if os.getenv("GOOGLE_API_KEY") else "openai"
    model = os.getenv("MODEL", "gpt-5-nano")
    
    llm_interface = LLMFactory.create(
        provider=provider,
        api_key=api_key,
        model=model
    )
    
    planner = CBRNPlanner(llm_interface=llm_interface)
    verifier = CBRNVerifier(judge_interface=llm_interface)
    
    generator = AutoRedGenerator(llm_interface=llm_interface)
    judge = LlamaGuardJudge(judge_interface=llm_interface)
    tap = TapEngine(generator, judge)
    target_client = A2AClient(transport_interface=llm_interface)
    
    tools = {
        "target_agent": target_client,
        "safety_judge": judge,
        "tap_engine": tap,
        "domain_validator": None
    }
    
    agent = CBRNAgent(
        planner=planner,
        verifier=verifier,
        tools=tools,
        seed=42
    )
    logger.info("CBRN Agent initialized successfully")


@app.get("/.well-known/agent.json")
async def agent_card():
    """A2A Agent Card endpoint (required by AgentBeats)."""
    return {
        "name": "CBRN-Sentinel",
        "description": "Green Agent for evaluating AI safety against CBRN threats",
        "version": "1.0.0",
        "url": card_url,
        "capabilities": {
            "assessment": True,
            "domains": ["chemical", "biological", "radiological", "nuclear"]
        },
        "skills": [
            {
                "id": "cbrn_safety_assessment",
                "name": "CBRN Safety Assessment",
                "description": "Evaluates agent responses to CBRN-related prompts using TAP attacks and defense delta scoring"
            }
        ]
    }


@app.post("/")
async def handle_assessment(request: Request):
    """
    A2A Assessment endpoint.
    Expects: {"participants": {"<role>": "<url>"}, "config": {...}}
    Returns: Assessment results as JSON artifact
    """
    try:
        body = await request.json()
        participants = body.get("participants", {})
        config = body.get("config", {})
        
        logger.info(f"Received assessment request with {len(participants)} participants")
        
        # Load benchmark data
        loader = BenchmarkLoader()
        dataset = config.get("dataset", "tests/cbrn_comprehensive.csv")
        limit = config.get("limit", 5)
        
        if dataset.endswith(".csv"):
            items = loader.load_csv_dataset(dataset, limit)
        else:
            items = loader.load_wmdp("bio", limit)
        
        # Run assessment
        results = []
        for item in items:
            task = {
                "id": item.id,
                "prompt": item.prompt,
                "domain": item.domain,
                "difficulty": item.difficulty,
                "source_benchmark": item.source_benchmark,
                "participants": participants
            }
            
            if agent:
                result = await agent.execute(task)
                result["difficulty"] = item.difficulty
                results.append(result)
            else:
                # Mock result when agent not initialized
                results.append({
                    "task_id": item.id,
                    "domain": item.domain,
                    "difficulty": item.difficulty,
                    "baseline_score": 0.9,
                    "attack_score": 1.0,
                    "defense_delta": -0.1,
                    "verification": {"verified": True, "confidence": 1.0}
                })
        
        # Calculate aggregate metrics
        avg_baseline = sum(r.get("baseline_score", 0) for r in results) / len(results)
        avg_delta = sum(r.get("defense_delta", 0) for r in results) / len(results)
        pass_rate = sum(1 for r in results if r.get("verification", {}).get("verified", False)) / len(results)
        
        # Return A2A artifact format
        return JSONResponse({
            "results": results,
            "pass_rate": pass_rate,
            "avg_baseline": avg_baseline,
            "avg_delta": avg_delta,
            "time_used": 0,  # TODO: Add timing
            "max_score": len(results)
        })
        
    except Exception as e:
        logger.error(f"Assessment error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "agent": "CBRN-Sentinel"}


def main():
    global card_url
    
    parser = argparse.ArgumentParser(description="CBRN-Sentinel A2A Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--card-url", default=None, help="URL to advertise in agent card")
    args = parser.parse_args()
    
    card_url = args.card_url or f"http://{args.host}:{args.port}"
    
    logger.info(f"Starting CBRN-Sentinel on {args.host}:{args.port}")
    logger.info(f"Agent card URL: {card_url}")
    
    # Initialize agent
    init_agent()
    
    # Run server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
