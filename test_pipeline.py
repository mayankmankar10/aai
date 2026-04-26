import sys
from agent.agent_pipeline import agent_pipeline

if __name__ == "__main__":
    try:
        print("Testing agent_pipeline...")
        res = agent_pipeline(r"c:\AAI\frontend\public\favicon.svg")
        print(f"Success: {res.success}, Error: {res.error_message}")
    except Exception as e:
        print(f"Exception: {e}")
