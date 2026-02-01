#!/usr/bin/env python3
"""
Publishes the latest CBRN-Sentinel results to the AgentBeats Leaderboard Repository.
"""
import os
import json
import shutil
import tempfile
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CBRN-Sentinel.Publisher")

# Configuration
RESULTS_REPO = "git@github.com:JoseFierroB/cbrn_sentinel_results.git"
LATEST_RESULTS_FILE = "reports/latest_results.json"
REPO_DIR = "leaderboard_repo"

def run_git_command(command, cwd=None):
    try:
        subprocess.run(command, shell=True, check=True, cwd=cwd, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {e.stderr.decode()}")
        raise

def publish():
    if not os.path.exists(LATEST_RESULTS_FILE):
        logger.error(f"No results found at {LATEST_RESULTS_FILE}. Run an assessment first.")
        return

    with open(LATEST_RESULTS_FILE, 'r') as f:
        results = json.load(f)

    timestamp = results.get("timestamp", "unknown")
    benchmark = results.get("benchmark", "unknown").replace("/", "_")
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(f"Cloning leaderboard repo into {tmp_dir}")
        run_git_command(f"git clone {RESULTS_REPO} .", cwd=tmp_dir)
        
        # Ensure results directory exists in the repo
        os.makedirs(os.path.join(tmp_dir, "results"), exist_ok=True)
        
        # Copy latest results to the repo
        target_file = os.path.join(tmp_dir, "results", f"result_{benchmark}_{timestamp}.json")
        shutil.copy2(LATEST_RESULTS_FILE, target_file)
        
        # Create a cleaner summary for the leaderboard UI if needed
        # (The template usually handles the UI, we just need the JSONs)
        
        # Commit and Push
        logger.info("Committing and pushing results...")
        run_git_command("git add .", cwd=tmp_dir)
        try:
            run_git_command(f"git commit -m 'Add assessment results for {benchmark} at {timestamp}'", cwd=tmp_dir)
            run_git_command("git push origin main", cwd=tmp_dir)
            logger.info("✓ Results published successfully to the leaderboard!")
        except subprocess.CalledProcessError as e:
            if "nothing to commit" in str(e).lower():
                logger.info("No new changes to commit.")
            else:
                raise

if __name__ == "__main__":
    publish()
