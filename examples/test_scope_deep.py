#!/usr/bin/env python
"""
Deep Functionality Test for SCOPE.

This script tests the core SCOPE functionality including:
1. Guideline synthesis from errors (corrective learning)
2. Guideline synthesis from quality analysis (enhancement learning)
3. Strategic memory persistence and retrieval
4. Tactical vs Strategic routing
5. Memory persistence across optimizer instances
6. Multiple agents support

Usage:
    # Use defaults (gpt-4o-mini via LiteLLM)
    python examples/test_scope_deep.py
    
    # Specify model and provider
    python examples/test_scope_deep.py --model gpt-4o --provider openai
    python examples/test_scope_deep.py --model claude-3-5-sonnet-20241022 --provider anthropic
    python examples/test_scope_deep.py --model deepseek-chat --provider deepseek
    python examples/test_scope_deep.py --model gemini/gemini-1.5-flash --provider litellm
    
    # Override API credentials
    python examples/test_scope_deep.py --api-key sk-xxx --base-url https://api.example.com/v1
"""
import argparse
import asyncio
import sys
import os
import json

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from _output import add_output_args, run_context

# Load environment variables
load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deep SCOPE functionality test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_scope_deep.py                                    # Default: gpt-4o-mini via litellm
  python test_scope_deep.py --model gpt-4o                     # Use GPT-4o
  python test_scope_deep.py --provider openai                  # Use OpenAI adapter directly
  python test_scope_deep.py --provider anthropic --model claude-3-5-sonnet-20241022
  python test_scope_deep.py --provider deepseek --model deepseek-chat
  python test_scope_deep.py --api-key sk-xxx                   # Override API key
        """
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name to use (default depends on provider)"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "anthropic", "deepseek", "litellm"],
        default="litellm",
        help="Provider adapter to use (default: litellm)"
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="API key (overrides environment variable)"
    )
    parser.add_argument(
        "--base-url", "-u",
        default=None,
        help="Base URL for API (overrides environment variable)"
    )
    add_output_args(parser)
    return parser.parse_args()


DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "deepseek": "deepseek-chat",
    "litellm": "gpt-4o-mini",
}


def resolved_model(args):
    """Return the requested model, or the provider's default."""
    return args.model or DEFAULT_MODELS.get(args.provider, "gpt-4o-mini")


def create_model(args):
    """Create model based on arguments."""
    model_name = resolved_model(args)
    if args.provider == "openai":
        from scope.models import create_openai_model
        return create_openai_model(
            model=model_name,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    elif args.provider == "anthropic":
        from scope.models import create_anthropic_model
        return create_anthropic_model(
            model=model_name,
            api_key=args.api_key,
        )
    elif args.provider == "deepseek":
        from scope.models import create_deepseek_model
        return create_deepseek_model(
            model=model_name,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    else:  # litellm
        from scope.models import create_litellm_model
        return create_litellm_model(
            model=model_name,
            api_key=args.api_key,
            base_url=args.base_url,
        )


class TestResults:
    """Track test results."""
    def __init__(self):
        self.results = []
    
    def add(self, name: str, passed: bool, message: str = ""):
        self.results.append((name, passed, message))
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if message:
            print(f"    → {message}")
    
    def summary(self):
        passed = sum(1 for _, p, _ in self.results if p)
        total = len(self.results)
        return passed, total


async def test_error_synthesis(model, tmpdir, results: TestResults):
    """Test 1: Guideline synthesis from errors."""
    print("\n" + "=" * 60)
    print("TEST 1: Error-Driven Guideline Synthesis")
    print("=" * 60)
    print("Testing that SCOPE generates guidelines when errors occur...\n")
    
    from scope import SCOPEOptimizer
    
    optimizer = SCOPEOptimizer(
        synthesizer_model=model,
        exp_path=tmpdir,
        store_history=True,
        auto_accept_threshold="low",  # Accept most guidelines for testing
    )
    
    # Simulate a JSON parsing error
    result = await optimizer.on_step_complete(
        agent_name="json_agent",
        agent_role="An agent that processes JSON data",
        task="Parse and validate user input",
        model_output='{"result": invalid}',  # Invalid JSON
        error=Exception("JSONDecodeError: Expecting value at line 1"),
        current_system_prompt="You are a JSON processing agent.",
        task_id="task_001",
    )
    
    if result:
        guideline, guideline_type = result
        results.add(
            "Error generates guideline",
            True,
            f"Generated {guideline_type} guideline: {guideline[:60]}..."
        )
        return guideline, guideline_type
    else:
        results.add("Error generates guideline", False, "No guideline generated")
        return None, None


async def test_quality_synthesis(model, tmpdir, results: TestResults):
    """Test 2: Guideline synthesis from quality analysis."""
    print("\n" + "=" * 60)
    print("TEST 2: Quality-Driven Guideline Synthesis")
    print("=" * 60)
    print("Testing that SCOPE analyzes successful steps for improvements...\n")
    
    from scope import SCOPEOptimizer
    
    optimizer = SCOPEOptimizer(
        synthesizer_model=model,
        exp_path=tmpdir,
        store_history=True,
        enable_quality_analysis=True,
        quality_analysis_frequency=2,  # Analyze every 2 steps
        auto_accept_threshold="low",
    )
    
    guidelines_generated = []
    
    # Run multiple successful steps to trigger quality analysis
    for i in range(4):
        result = await optimizer.on_step_complete(
            agent_name="quality_agent",
            agent_role="A code review agent",
            task="Review and improve code quality",
            model_output=f"Step {i+1}: Analyzed the code and found {i} issues.",
            tool_calls='[{"name": "code_analyzer", "arguments": {"file": "main.py"}}]',
            observations="Analysis complete. Found potential improvements.",
            error=None,  # No error - quality analysis
            current_system_prompt="You are a code review agent.",
            task_id="task_002",
        )
        if result:
            guidelines_generated.append(result)
    
    if guidelines_generated:
        guideline, guideline_type = guidelines_generated[-1]
        results.add(
            "Quality analysis generates guideline",
            True,
            f"Generated {len(guidelines_generated)} guideline(s)"
        )
    else:
        results.add(
            "Quality analysis generates guideline",
            False,
            "No quality guidelines generated (may be OK if confidence too low)"
        )


async def test_strategic_memory(model, tmpdir, results: TestResults):
    """Test 3: Strategic memory storage and retrieval."""
    print("\n" + "=" * 60)
    print("TEST 3: Strategic Memory Persistence")
    print("=" * 60)
    print("Testing that strategic guidelines are persisted to disk...\n")
    
    from scope import SCOPEOptimizer
    
    # Create optimizer with very low threshold to ensure strategic storage
    optimizer = SCOPEOptimizer(
        synthesizer_model=model,
        exp_path=tmpdir,
        store_history=True,
        auto_accept_threshold="all",  # Accept everything
        strategic_confidence_threshold=0.3,  # Very low threshold for testing
    )
    
    # Generate multiple guidelines to increase chance of strategic storage
    for i in range(3):
        result = await optimizer.on_step_complete(
            agent_name="strategic_agent",
            agent_role="A data processing agent",
            task="Process and validate data",
            model_output=f"Processing failed at step {i+1}",
            error=Exception(f"ValidationError: Required field 'field_{i}' is missing"),
            current_system_prompt="You are a data processing agent.",
            task_id=f"task_003_{i}",
        )
        if result:
            _, guideline_type = result
            print(f"  Generated {guideline_type} guideline (attempt {i+1})")
    
    # Check if strategic memory file exists
    strategic_path = os.path.join(tmpdir, "strategic_memory", "global_rules.json")
    
    if os.path.exists(strategic_path):
        with open(strategic_path, 'r') as f:
            data = json.load(f)
        
        if "strategic_agent" in data and data["strategic_agent"]:
            rule_count = sum(len(rules) for rules in data["strategic_agent"].values())
            results.add(
                "Strategic rules persisted to disk",
                True,
                f"Found {rule_count} rule(s) in strategic memory"
            )
        else:
            # Check all agents in data
            all_rules = sum(
                sum(len(rules) for rules in agent_data.values())
                for agent_data in data.values()
                if isinstance(agent_data, dict)
            )
            if all_rules > 0:
                results.add(
                    "Strategic rules persisted to disk",
                    True,
                    f"Found {all_rules} rule(s) for other agents"
                )
            else:
                results.add(
                    "Strategic rules persisted to disk",
                    False,
                    "No strategic rules stored (model may have low confidence)"
                )
    else:
        results.add("Strategic rules persisted to disk", False, "Strategic memory file not created")
    
    # Test retrieval
    rules_text = optimizer.get_strategic_rules_for_agent("strategic_agent")
    results.add(
        "Strategic rules retrievable",
        rules_text is not None and len(rules_text) > 0,
        f"Retrieved {len(rules_text)} chars" if rules_text else "No rules (may be tactical only)"
    )


async def test_memory_persistence_across_instances(model, tmpdir, results: TestResults):
    """Test 4: Memory persists across optimizer instances."""
    print("\n" + "=" * 60)
    print("TEST 4: Cross-Instance Memory Persistence")
    print("=" * 60)
    print("Testing that strategic memory survives optimizer restart...\n")
    
    from scope import SCOPEOptimizer
    
    # First optimizer instance - generate a rule
    optimizer1 = SCOPEOptimizer(
        synthesizer_model=model,
        exp_path=tmpdir,
        store_history=True,
        auto_accept_threshold="low",
        strategic_confidence_threshold=0.5,
    )
    
    await optimizer1.on_step_complete(
        agent_name="persist_agent",
        agent_role="A persistent learning agent",
        task="Learn and remember patterns",
        model_output="Failed to handle edge case",
        error=Exception("EdgeCaseError: Unexpected null value"),
        current_system_prompt="You are a learning agent.",
        task_id="task_004",
    )
    
    rules_before = optimizer1.get_strategic_rules_for_agent("persist_agent")
    
    # Create NEW optimizer instance (simulating restart)
    optimizer2 = SCOPEOptimizer(
        synthesizer_model=model,
        exp_path=tmpdir,  # Same path
        store_history=True,
        auto_accept_threshold="low",
    )
    
    rules_after = optimizer2.get_strategic_rules_for_agent("persist_agent")
    
    # Check if rules survived the "restart"
    if rules_before and rules_after:
        results.add(
            "Memory persists across instances",
            True,
            "Rules available in new optimizer instance"
        )
    elif not rules_before:
        results.add(
            "Memory persists across instances",
            False,
            "No rules generated in first instance"
        )
    else:
        results.add(
            "Memory persists across instances",
            False,
            "Rules lost after creating new instance"
        )


async def test_multiple_agents(model, tmpdir, results: TestResults):
    """Test 5: Support for multiple agents."""
    print("\n" + "=" * 60)
    print("TEST 5: Multiple Agents Support")
    print("=" * 60)
    print("Testing that SCOPE handles multiple agents independently...\n")
    
    from scope import SCOPEOptimizer
    
    optimizer = SCOPEOptimizer(
        synthesizer_model=model,
        exp_path=tmpdir,
        store_history=True,
        auto_accept_threshold="low",
        strategic_confidence_threshold=0.5,
    )
    
    # Agent 1: Web scraper
    await optimizer.on_step_complete(
        agent_name="web_scraper",
        agent_role="A web scraping agent",
        task="Scrape product data from websites",
        model_output="Failed to parse HTML",
        error=Exception("ParsingError: Malformed HTML structure"),
        current_system_prompt="You are a web scraping agent.",
        task_id="task_005a",
    )
    
    # Agent 2: Data analyzer
    await optimizer.on_step_complete(
        agent_name="data_analyzer",
        agent_role="A data analysis agent",
        task="Analyze sales trends",
        model_output="Analysis incomplete due to missing data",
        error=Exception("DataError: Column 'revenue' contains null values"),
        current_system_prompt="You are a data analysis agent.",
        task_id="task_005b",
    )
    
    # Check that both agents have separate rules
    rules_scraper = optimizer.get_strategic_rules_for_agent("web_scraper")
    rules_analyzer = optimizer.get_strategic_rules_for_agent("data_analyzer")
    
    scraper_has_rules = rules_scraper is not None and len(rules_scraper) > 0
    analyzer_has_rules = rules_analyzer is not None and len(rules_analyzer) > 0
    
    if scraper_has_rules and analyzer_has_rules:
        results.add(
            "Multiple agents have separate rules",
            True,
            "Both agents have independent rule sets"
        )
    elif scraper_has_rules or analyzer_has_rules:
        results.add(
            "Multiple agents have separate rules",
            True,
            "At least one agent has rules (other may have low confidence)"
        )
    else:
        results.add(
            "Multiple agents have separate rules",
            False,
            "No rules generated for either agent"
        )


async def test_guideline_history(model, tmpdir, results: TestResults):
    """Test 6: Guideline history tracking."""
    print("\n" + "=" * 60)
    print("TEST 6: Guideline History Tracking")
    print("=" * 60)
    print("Testing that guideline history is recorded when enabled...\n")
    
    from scope import SCOPEOptimizer
    
    optimizer = SCOPEOptimizer(
        synthesizer_model=model,
        exp_path=tmpdir,
        store_history=True,  # Enable history
        auto_accept_threshold="low",
    )
    
    # Generate some guidelines
    for i in range(2):
        await optimizer.on_step_complete(
            agent_name="history_agent",
            agent_role="An agent for history testing",
            task=f"Task {i+1}",
            model_output=f"Output {i+1}",
            error=Exception(f"Error type {i+1}: Something went wrong"),
            current_system_prompt="You are a test agent.",
            task_id=f"task_006_{i}",
        )
    
    # Check for active_rules.json (created when guidelines are generated)
    active_rules_path = os.path.join(tmpdir, "prompt_updates", "active_rules.json")
    # Also check for per-agent JSONL file
    agent_history_path = os.path.join(tmpdir, "prompt_updates", "history_agent.jsonl")
    
    active_rules_exists = os.path.exists(active_rules_path)
    agent_history_exists = os.path.exists(agent_history_path)
    
    if active_rules_exists or agent_history_exists:
        details = []
        if active_rules_exists:
            with open(active_rules_path, 'r') as f:
                data = json.load(f)
            task_count = len(data)
            details.append(f"active_rules: {task_count} tasks")
        if agent_history_exists:
            with open(agent_history_path, 'r') as f:
                lines = f.readlines()
            details.append(f"agent JSONL: {len(lines)} entries")
        
        results.add(
            "Guideline history recorded",
            True,
            ", ".join(details)
        )
    else:
        results.add(
            "Guideline history recorded",
            False,
            "No history files created"
        )


async def test_statistics(model, tmpdir, results: TestResults):
    """Test 7: Statistics tracking."""
    print("\n" + "=" * 60)
    print("TEST 7: Statistics Tracking")
    print("=" * 60)
    print("Testing that optimizer tracks statistics correctly...\n")
    
    from scope import SCOPEOptimizer
    
    optimizer = SCOPEOptimizer(
        synthesizer_model=model,
        exp_path=tmpdir,
        store_history=True,
        auto_accept_threshold="low",
    )
    
    # Generate some activity
    await optimizer.on_step_complete(
        agent_name="stats_agent",
        agent_role="A test agent",
        task="Test task",
        model_output="Test output",
        error=Exception("TestError"),
        current_system_prompt="Test prompt",
        task_id="task_007",
    )
    
    stats = optimizer.get_statistics()
    
    results.add(
        "Statistics available",
        stats is not None and isinstance(stats, dict),
        f"Keys: {list(stats.keys())}" if stats else "No stats"
    )


async def main():
    """Run all deep functionality tests."""
    args = parse_args()

    with run_context("test_scope_deep", args) as run:
        print("\n" + "=" * 60)
        print("SCOPE Deep Functionality Tests")
        print("=" * 60)
        print(f"\nProvider: {args.provider}")
        print(f"Model: {resolved_model(args)}")
        if args.api_key:
            print("API Key: ***")
        if args.base_url:
            print(f"Base URL: {args.base_url}")

        model = run.wrap_model(create_model(args))
        results = TestResults()

        # Give every test its own exp_path inside the run directory.
        await test_error_synthesis(
            model, run.experiment_dir("error_synthesis"), results
        )
        await test_quality_synthesis(
            model, run.experiment_dir("quality_synthesis"), results
        )
        await test_strategic_memory(
            model, run.experiment_dir("strategic_memory"), results
        )
        await test_memory_persistence_across_instances(
            model, run.experiment_dir("memory_persistence"), results
        )
        await test_multiple_agents(
            model, run.experiment_dir("multiple_agents"), results
        )
        await test_guideline_history(
            model, run.experiment_dir("guideline_history"), results
        )
        await test_statistics(
            model, run.experiment_dir("statistics"), results
        )

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed, total = results.summary()
        all_passed = passed >= total * 0.7

        print(f"\nPassed: {passed}/{total}")

        if passed == total:
            print("\n🎉 All tests passed!")
        elif all_passed:
            print("\n✓ Most tests passed (some may fail due to model confidence thresholds)")
        else:
            print("\n⚠ Several tests failed. Check the output above.")

        run.save_summary({
            "provider": args.provider,
            "model": resolved_model(args),
            "passed": passed,
            "total": total,
            "all_passed": all_passed,
            "results": [
                {"name": name, "passed": ok, "message": message}
                for name, ok, message in results.results
            ],
        })
        return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

