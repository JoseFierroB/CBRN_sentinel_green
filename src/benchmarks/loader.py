import logging
import os
from typing import List
from ..schemas import EvaluationItem

logger = logging.getLogger("CBRN-Sentinel.Loader")

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

class BenchmarkLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def _load_hf_dataset(self, repo_id: str, subset: str = None, split: str = "test", limit: int = 10, streaming: bool = True) -> Any:
        """Helper to safely load HF datasets with logging and strict error handling."""
        if not load_dataset: return None
        name = f"{repo_id} ({subset})" if subset else repo_id
        logger.info(f"Loading {name}...")
        try:
            return load_dataset(repo_id, subset, split=split, streaming=streaming)
        except Exception as e:
            logger.critical(f"Error loading {name}: {e}")
            raise e

    def load_wmdp(self, subset: str = "bio", limit: int = 10) -> List[EvaluationItem]:
        ds = self._load_hf_dataset("cais/wmdp", f"wmdp-{subset}", limit=limit)
        if not ds: return []
        
        items = []
        count = 0
        for row in ds:
            if count >= limit: break
            choices_str = "\n".join([f"{i}. {c}" for i, c in enumerate(row['choices'])])
            prompt = f"{row['question']}\n\nChoices:\n{choices_str}\n\nAnswer with the index."
            
            items.append(EvaluationItem(
                id=f"wmdp_{subset}_{count}",
                domain=subset.capitalize(),
                source_benchmark="wmdp",
                prompt=prompt,
                reference_answer=str(row['answer']),
                is_adversarial=True
            ))
            count += 1
        return items

    def load_fortress(self, limit: int = 10) -> List[EvaluationItem]:
        ds = self._load_hf_dataset("ScaleAI/fortress_public", split="train", limit=limit)
        if not ds: return []
        
        items = []
        count = 0
        for row in ds:
            if count >= limit: break
            items.append(EvaluationItem(
                id=f"fortress_{count}",
                domain="Adversarial",
                source_benchmark="fortress",
                prompt=row.get('prompt', row.get('question', row.get('text', ''))),
                system_prompt="You are a helpful assistant.",
                is_adversarial=True
            ))
            count += 1
        return items

    def load_gpqa(self, limit: int = 10) -> List[EvaluationItem]:
        ds = self._load_hf_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", limit=limit)
        if not ds: return []
        
        items = []
        count = 0
        for row in ds:
            if count >= limit: break
            prompt = f"{row['Question']}\n\nAnswer this PhD-level question."
            items.append(EvaluationItem(
                id=f"gpqa_{count}",
                domain="Reasoning",
                source_benchmark="gpqa",
                prompt=prompt,
                reference_answer=row['Correct Answer'],
                is_adversarial=False
            ))
            count += 1
        return items

    def load_scicode(self, limit: int = 10) -> List[EvaluationItem]:
        ds = self._load_hf_dataset("scicode-bench/SciCode", split="test", limit=limit)
        if not ds: return []
        
        items = []
        count = 0
        for row in ds:
            if count >= limit: break
            items.append(EvaluationItem(
                id=f"scicode_{count}",
                domain="Scientific Coding",
                source_benchmark="scicode",
                prompt=row['problem_text'],
                reference_answer=row.get('solution_code', ''),
                is_adversarial=False
            ))
            count += 1
        return items

    def load_cyberseceval(self, limit: int = 10) -> List[EvaluationItem]:
        return self.load_wmdp("cyber", limit)

    def load_harmbench(self, limit: int = 10) -> List[EvaluationItem]:
        ds = self._load_hf_dataset("walledai/HarmBench", split="train", limit=limit)
        if not ds: return []
        
        items = []
        count = 0
        for row in ds:
            if count >= limit: break
            prompt = row.get('behavior', row.get('prompt', ''))
            items.append(EvaluationItem(
                id=f"harmbench_{count}",
                domain="General Harm",
                source_benchmark="harmbench",
                prompt=prompt,
                is_adversarial=True
            ))
            count += 1
        return items

    def load_csv_dataset(self, file_path: str, limit: int = 10) -> List[EvaluationItem]:
        """
        Loads custom CSV.
        Expected headers: prompt, domain (optional), reference (optional)
        """
        if not os.path.exists(file_path):
            logger.error(f"CSV not found: {file_path}")
            return []
            
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            items = []
            count = 0
            for _, row in df.iterrows():
                if count >= limit: break
                
                # Parse difficulty, default to medium
                difficulty = str(row.get('difficulty', 'medium')).lower()
                if difficulty not in ['easy', 'medium', 'hard', 'expert']:
                    difficulty = 'medium'
                
                items.append(EvaluationItem(
                    id=f"custom_{count}",
                    domain=row.get('domain', 'Custom'),
                    source_benchmark="custom_csv",
                    prompt=row['prompt'],
                    reference_answer=row.get('reference', ''),
                    difficulty=difficulty,
                    is_adversarial=True
                ))
                count += 1
            return items
        except Exception as e:
            logger.critical(f"Error loading CSV: {e}")
            raise e

    def load_mmlu(self, limit: int = 10) -> List[EvaluationItem]:
        # Using TIGER-Lab/MMLU-Pro
        ds = self._load_hf_dataset("TIGER-Lab/MMLU-Pro", split="test", limit=limit)
        if not ds: return []
        
        items = []
        count = 0
        for row in ds:
            if count >= limit: break
            
            # Filter for STEM
            category = row.get('category', '').lower()
            if not any(x in category for x in ["chem", "bio", "phys"]):
                continue
            
            prompt = f"{row['question']}\n\nChoices:\n{row['options']}\n\nAnswer with the letter/index."
            items.append(EvaluationItem(
                id=f"mmlu_pro_{count}",
                domain=f"MMLU-{category}",
                source_benchmark="mmlu-pro",
                prompt=prompt,
                reference_answer=str(row.get('answer', '')),
                is_adversarial=False
            ))
            count += 1
        return items

    def load_xstest(self, limit: int = 10) -> List[EvaluationItem]:
        ds = self._load_hf_dataset("natolambert/xstest-v2-copy", split="prompts", limit=limit)
        if not ds: return []
        
        items = []
        count = 0
        for row in ds:
            if count >= limit: break
            
            if row.get('type') in ['contrast_safe', 'safe']:
                items.append(EvaluationItem(
                    id=f"xstest_{count}",
                    domain="Over-Refusal",
                    source_benchmark="xstest",
                    prompt=row['prompt'],
                    is_adversarial=False,
                    is_refusal_test=True
                ))
                count += 1
        return items

    def load_lab_bench(self, subset: str = "CloningScenarios", limit: int = 10) -> List[EvaluationItem]:
        ds = self._load_hf_dataset("futurehouse/lab-bench", subset, split="train", limit=limit)
        if not ds: return []
        
        items = []
        count = 0
        for row in ds:
            if count >= limit: break
            
            question = row.get('question', row.get('prompt', ''))
            answer = row.get('answer', row.get('target', ''))
            
            if subset == "FigQA" and "image" in row:
                question += " [Image Context Unavailable]"

            items.append(EvaluationItem(
                id=f"lab_bench_{subset}_{count}",
                domain=f"Lab-{subset}",
                source_benchmark="lab-bench",
                prompt=question,
                reference_answer=str(answer),
                is_adversarial=False
            ))
            count += 1
        return items
