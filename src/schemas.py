from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class EvaluationItem(BaseModel):
    """
    Represents a single test case from a benchmark (e.g., WMDP, SciCode).
    """
    id: str
    domain: str # Allow flexible domains (MMLU, HarmBench, etc.)
    source_benchmark: str
    prompt: str
    system_prompt: Optional[str] = None
    reference_answer: Optional[str] = None
    is_adversarial: bool = False
    is_refusal_test: bool = False # If True, Refusal = Bad (Over-refusal)
    difficulty: Literal["easy", "medium", "hard", "expert"] = "medium"  # For weighted scoring
    metadata: Dict[str, Any] = {}

class EvaluationVerdict(BaseModel):
    """
    The judgment produced by Aegis.
    """
    score_utility: float = Field(..., ge=0.0, le=1.0)
    score_safety: float = Field(..., ge=0.0, le=1.0)
    refusal_detected: bool
    jailbreak_detected: bool
    memory_poisoning_detected: bool = False
    feedback: str
    cot_analysis: Optional[str] = None

class EvaluationResult(BaseModel):
    """
    The final artifact for a single test case execution.
    """
    item_id: str
    agent_response: str
    agent_cot: Optional[str] = None
    verdict: EvaluationVerdict
    execution_time_ms: float
