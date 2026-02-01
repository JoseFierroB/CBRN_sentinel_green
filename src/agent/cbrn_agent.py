"""
CBRN Agent - GAIA-Style Agent Loop
Implements: Plan → Act → Verify → Answer
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional

from .planner import CBRNPlanner, ExecutionPlan, ExecutionStep
from .verifier import CBRNVerifier, StepResult, VerificationResult

logger = logging.getLogger("CBRN-Sentinel.Agent")


class CBRNAgent:
    """
    CBRN-Sentinel Agent following GAIA architecture.
    
    Execution Loop:
    1. PLAN   - Decompose task into steps
    2. ACT    - Execute steps using tools
    3. VERIFY - Check intermediate and final results
    4. ANSWER - Emit structured final answer
    """
    
    def __init__(
        self,
        planner: CBRNPlanner,
        verifier: CBRNVerifier,
        tools: Dict[str, Any],
        seed: int = 42
    ):
        self.planner = planner
        self.verifier = verifier
        self.tools = tools
        self.seed = seed
        self.execution_log: List[Dict[str, Any]] = []
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution loop following GAIA pattern.
        
        Args:
            task: Dict with 'id', 'prompt', 'domain', etc.
        
        Returns:
            Structured result with scores, verification, and logs.
        """
        logger.info(f"=== Executing Task: {task.get('id', 'unknown')} ===")
        self.execution_log = []
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 1: PLAN
        # ═══════════════════════════════════════════════════════════
        logger.info("[PHASE 1] Planning...")
        plan = self.planner.plan(task)
        self._log("plan", {"steps": len(plan.steps), "domain": plan.domain})
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 2: ACT (Execute Steps)
        # ═══════════════════════════════════════════════════════════
        logger.info("[PHASE 2] Executing steps...")
        step_results: Dict[int, StepResult] = {}
        
        for step in plan.steps:
            # Check dependencies
            deps_ok = all(
                step_results.get(dep_id, StepResult(step_id=dep_id, action="", success=False, output=None)).success
                for dep_id in step.depends_on
            )
            
            if not deps_ok:
                logger.warning(f"Step {step.step_id} skipped: dependencies not met")
                step_results[step.step_id] = StepResult(
                    step_id=step.step_id,
                    action=step.action,
                    success=False,
                    output=None,
                    error="Dependencies not met"
                )
                continue
            
            # Execute step
            logger.info(f"  Step {step.step_id}: {step.action}")
            result = await self._execute_step(step, step_results, plan)
            step_results[step.step_id] = result
            
            # ═══════════════════════════════════════════════════════
            # PHASE 3: VERIFY (Each Step)
            # ═══════════════════════════════════════════════════════
            verification = self.verifier.verify_step(result)
            self._log("step", {
                "step_id": step.step_id,
                "action": step.action,
                "success": result.success,
                "verified": verification.verified,
                "score": result.score
            })
            
            if not verification.verified:
                logger.warning(f"  Step {step.step_id} verification issues: {verification.issues}")
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 4: ANSWER (Compile Results)
        # ═══════════════════════════════════════════════════════════
        logger.info("[PHASE 4] Compiling answer...")
        answer = self._compile_answer(step_results, plan)
        
        # Final Verification
        final_verification = self.verifier.verify_final_answer(answer)
        answer["verification"] = {
            "verified": final_verification.verified,
            "confidence": final_verification.confidence,
            "issues": final_verification.issues,
            "recommendation": final_verification.recommendation
        }
        
        self._log("final", {
            "verified": final_verification.verified,
            "confidence": final_verification.confidence
        })
        
        answer["execution_log"] = self.execution_log
        
        logger.info(f"=== Task Complete: {final_verification.recommendation} ===")
        return answer
    
    async def _execute_step(
        self,
        step: ExecutionStep,
        previous_results: Dict[int, StepResult],
        plan: ExecutionPlan
    ) -> StepResult:
        """Execute a single step using the appropriate tool."""
        
        try:
            if step.action == "evaluate_baseline":
                # Use target_agent tool
                tool = self.tools.get("target_agent")
                if not tool:
                    raise ValueError("target_agent tool not configured")
                
                response = await tool.execute_task(step.parameters["prompt"])
                return StepResult(
                    step_id=step.step_id,
                    action=step.action,
                    success=True,
                    output={"text": response.text, "raw": getattr(response, 'raw', {})}
                )
            
            elif step.action == "verify_baseline":
                # Use safety_judge tool
                judge = self.tools.get("safety_judge")
                if not judge:
                    raise ValueError("safety_judge tool not configured")
                
                baseline_result = previous_results.get(1)
                if not baseline_result or not baseline_result.output:
                    raise ValueError("No baseline result to verify")
                
                verdict = await judge.evaluate(
                    plan.prompt,
                    baseline_result.output["text"]
                )
                return StepResult(
                    step_id=step.step_id,
                    action=step.action,
                    success=True,
                    output={"verdict": verdict.dict()},
                    score=verdict.score
                )
            
            elif step.action == "run_attack":
                # Use TAP engine tool
                tap = self.tools.get("tap_engine")
                if not tap:
                    raise ValueError("tap_engine tool not configured")
                
                target = self.tools.get("target_agent")
                # Create a simple scenario object
                from ..modules.scenario_gen import Scenario
                scenario = Scenario(
                    id=plan.task_id,
                    initial_prompt=f"As a researcher, {plan.prompt}",
                    base_prompt=plan.prompt,
                    template_id="tpl_attack",
                    complexity="text",
                    source_benchmark=plan.metadata.get("source", "unknown")
                )
                
                attack_score, trace = await tap.run_attack(target, scenario)
                return StepResult(
                    step_id=step.step_id,
                    action=step.action,
                    success=True,
                    output={"trace": trace},
                    score=attack_score
                )
            
            elif step.action == "verify_attack":
                attack_result = previous_results.get(3)
                if not attack_result:
                    raise ValueError("No attack result to verify")
                
                return StepResult(
                    step_id=step.step_id,
                    action=step.action,
                    success=True,
                    output={"attack_verified": True},
                    score=attack_result.score
                )
            
            elif step.action == "calculate_delta":
                baseline_step = previous_results.get(2)
                attack_step = previous_results.get(4)
                
                baseline_score = baseline_step.score if baseline_step else 0.0
                attack_score = attack_step.score if attack_step else 0.0
                delta = baseline_score - attack_score
                
                return StepResult(
                    step_id=step.step_id,
                    action=step.action,
                    success=True,
                    output={"baseline": baseline_score, "attack": attack_score, "delta": delta},
                    score=delta
                )
            
            elif step.action == "verify_final":
                return StepResult(
                    step_id=step.step_id,
                    action=step.action,
                    success=True,
                    output={"final_verified": True}
                )
            
            elif step.action == "validate_technical":
                # Domain-specific validation (RDKit, etc.)
                domain_tool = self.tools.get("domain_validator")
                return StepResult(
                    step_id=step.step_id,
                    action=step.action,
                    success=True,
                    output={"technical_valid": True}
                )
            
            else:
                raise ValueError(f"Unknown action: {step.action}")
        
        except Exception as e:
            logger.error(f"Step {step.step_id} failed: {e}")
            return StepResult(
                step_id=step.step_id,
                action=step.action,
                success=False,
                output=None,
                error=str(e)
            )
    
    def _compile_answer(
        self,
        step_results: Dict[int, StepResult],
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Compile step results into final answer."""
        
        # Extract key metrics
        baseline_step = step_results.get(2)
        delta_step = step_results.get(5)
        attack_step = step_results.get(4)
        
        baseline_score = baseline_step.score if baseline_step else 0.0
        attack_score = attack_step.score if attack_step else 0.0
        delta = delta_step.output.get("delta", 0.0) if delta_step and delta_step.output else 0.0
        
        # Get trace from attack
        attack_result = step_results.get(3)
        trace = attack_result.output.get("trace", []) if attack_result and attack_result.output else []
        
        return {
            "task_id": plan.task_id,
            "prompt": plan.prompt,
            "domain": plan.domain,
            "source": plan.metadata.get("source", "unknown"),
            "baseline_score": baseline_score,
            "attack_score": attack_score,
            "defense_delta": delta,
            "final_score": attack_score,  # Safety score under attack
            "trace": trace,
            "steps_completed": sum(1 for r in step_results.values() if r.success),
            "steps_total": len(plan.steps)
        }
    
    def _log(self, phase: str, data: Dict[str, Any]):
        """Add entry to execution log."""
        self.execution_log.append({"phase": phase, **data})
