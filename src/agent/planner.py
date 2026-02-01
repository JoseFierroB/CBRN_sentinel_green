"""
CBRN Planner Module - Modular Task Decomposition
"""
import logging
from typing import List, Dict, Any
from pydantic import BaseModel

logger = logging.getLogger("CBRN-Sentinel.Planner")


class ExecutionStep(BaseModel):
    """A single step in the execution plan."""
    step_id: int
    action: str  # "evaluate_baseline", "run_attack", "verify_safety", "calculate_delta"
    description: str
    tool: str  # Which tool to use
    parameters: Dict[str, Any] = {}
    depends_on: List[int] = []  # Step IDs this depends on


class ExecutionPlan(BaseModel):
    """Complete execution plan for a CBRN evaluation task."""
    task_id: str
    prompt: str
    domain: str
    steps: List[ExecutionStep]
    metadata: Dict[str, Any] = {}


class CBRNPlanner:
    """
    Planner component following modular architecture.
    Decomposes CBRN evaluation tasks into ordered execution steps.
    """
    
    def __init__(self, llm_interface=None):
        self.llm = llm_interface
    
    def plan(self, task: Dict[str, Any]) -> ExecutionPlan:
        """
        Analyze a CBRN task and create an execution plan.
        
        Following modular pattern:
        1. Parse the problem
        2. Decompose into steps
        3. Determine tool requirements
        4. Create ordered execution plan
        """
        logger.info(f"Planning task: {task.get('id', 'unknown')}")
        
        prompt = task.get("prompt", "")
        domain = task.get("domain", "unknown")
        task_id = task.get("id", "unknown")
        
        # Create standard CBRN evaluation plan
        steps = [
            ExecutionStep(
                step_id=1,
                action="evaluate_baseline",
                description="Send original prompt to target agent without adversarial wrapping",
                tool="target_agent",
                parameters={"prompt": prompt, "mode": "baseline"}
            ),
            ExecutionStep(
                step_id=2,
                action="verify_baseline",
                description="Judge the baseline response for safety",
                tool="safety_judge",
                parameters={"check_type": "baseline"},
                depends_on=[1]
            ),
            ExecutionStep(
                step_id=3,
                action="run_attack",
                description="Execute TAP adversarial attack loop",
                tool="tap_engine",
                parameters={"max_depth": 5, "prompt": prompt},
                depends_on=[2]
            ),
            ExecutionStep(
                step_id=4,
                action="verify_attack",
                description="Judge each attack step for safety breaches",
                tool="safety_judge",
                parameters={"check_type": "attack"},
                depends_on=[3]
            ),
            ExecutionStep(
                step_id=5,
                action="calculate_delta",
                description="Compute defense delta between baseline and attack scores",
                tool="calculator",
                parameters={"formula": "baseline_score - attack_score"},
                depends_on=[2, 4]
            ),
            ExecutionStep(
                step_id=6,
                action="verify_final",
                description="Final verification and result compilation",
                tool="verifier",
                parameters={"check_type": "final"},
                depends_on=[5]
            ),
        ]
        
        # Domain-specific additions
        if domain.lower() in ["chemical", "biological"]:
            steps.insert(5, ExecutionStep(
                step_id=7,
                action="validate_technical",
                description="Check if response contains valid scientific content",
                tool="domain_validator",
                parameters={"domain": domain},
                depends_on=[1]
            ))
        
        plan = ExecutionPlan(
            task_id=task_id,
            prompt=prompt,
            domain=domain,
            steps=steps,
            metadata={"source": task.get("source_benchmark", "unknown")}
        )
        
        logger.info(f"Plan created with {len(steps)} steps")
        return plan
    
    def plan_batch(self, tasks: List[Dict[str, Any]]) -> List[ExecutionPlan]:
        """Create execution plans for multiple tasks."""
        return [self.plan(task) for task in tasks]
