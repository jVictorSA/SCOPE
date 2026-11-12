#!/usr/bin/env python
"""
Quick Start example for SCOPE.

This is the simplest possible example showing SCOPE in action.
Run with: python examples/quick_start.py

Results are saved to examples/results/quick_start/<timestamp>/ by default.
Use --output-dir to choose another base directory, or --temp to discard them.
"""
import argparse
import asyncio

from dotenv import load_dotenv

from scope import SCOPEOptimizer
from scope.models import create_openai_model

from _output import add_output_args, run_context

# Load environment variables from .env file
load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SCOPE quick start example")
    add_output_args(parser)
    return parser.parse_args()


async def main(args):
    with run_context("quick_start", args) as run:
        # Create model - API key is automatically loaded from OPENAI_API_KEY env var
        model = run.wrap_model(create_openai_model("gpt-4o-mini"))
        optimizer = SCOPEOptimizer(
            synthesizer_model=model,
            exp_path=run.exp_path,  # Strategic rules persist here
            store_history=True,
        )

        # IMPORTANT: Load previously learned strategic rules at agent initialization
        # This applies cross-task knowledge from previous runs
        base_prompt = "You are a helpful assistant."
        strategic_rules = optimizer.get_strategic_rules_for_agent("my_agent")
        current_prompt = base_prompt + strategic_rules
        print(f"Loaded {len(strategic_rules)} chars of strategic rules")

        # Simulate a step with an error
        result = await optimizer.on_step_complete(
            agent_name="my_agent",
            agent_role="AI Assistant",
            task="Process user request",
            error=Exception("Failed to parse JSON response"),
            model_output="The answer is: yes",
            current_system_prompt=current_prompt,
            task_id="task_001",
        )

        if result:
            guideline, guideline_type = result
            print(f"Generated {guideline_type} guideline:")
            print(f"  {guideline}")
            current_prompt += f"\n\n## Learned Guideline:\n{guideline}"
        else:
            print("No guideline generated")

        run.save_text("final_prompt.md", current_prompt)
        run.save_summary({"statistics": optimizer.get_statistics()})


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
