#!/usr/bin/env python
"""
Simple API Integration Test for SCOPE.

This script tests that the model adapters work correctly with real API calls.

Usage:
    # Use defaults (gpt-4o-mini via LiteLLM, reads from .env)
    python examples/test_simple.py
    
    # Specify model
    python examples/test_simple.py --model gpt-4o
    
    # Use different provider
    python examples/test_simple.py --provider openai --model gpt-4o-mini
    python examples/test_simple.py --provider anthropic --model claude-3-5-sonnet-20241022
    python examples/test_simple.py --provider deepseek --model deepseek-chat
    python examples/test_simple.py --provider litellm --model gemini/gemini-1.5-flash
    
    # Override API credentials
    python examples/test_simple.py --api-key sk-xxx --base-url https://api.example.com/v1
"""
import argparse
import asyncio
import sys
import os

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from _output import add_output_args, run_context

# Load environment variables
load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test SCOPE API integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_simple.py                                    # Default: gpt-4o-mini via litellm
  python test_simple.py --model gpt-4o                     # Use GPT-4o
  python test_simple.py --provider openai                  # Use OpenAI adapter directly
  python test_simple.py --provider anthropic --model claude-3-5-sonnet-20241022
  python test_simple.py --provider deepseek --model deepseek-chat
  python test_simple.py --provider litellm --model gemini/gemini-1.5-flash
  python test_simple.py --api-key sk-xxx                   # Override API key
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
    parser.add_argument(
        "--skip-optimizer",
        action="store_true",
        help="Skip the SCOPEOptimizer test (only test model adapter)"
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


async def test_model_adapter(args, model):
    """Test model adapter with a simple completion."""
    print("\n" + "=" * 50)
    print(f"Testing {args.provider.upper()} Adapter")
    print(f"Model: {resolved_model(args)}")
    print("=" * 50)

    from scope.models import Message

    try:
        messages = [
            Message(role="user", content="Say 'Hello SCOPE!' and nothing else.")
        ]
        
        print("Sending request...")
        response = await model.generate(messages)
        
        print(f"✓ Response received: {response.content}")
        print("✓ Model adapter test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Model adapter test FAILED: {e}")
        return False


async def test_scope_optimizer(args, model, run):
    """Test full SCOPEOptimizer with a real model."""
    print("\n" + "=" * 50)
    print("Testing SCOPEOptimizer Integration")
    print("=" * 50)

    from scope import SCOPEOptimizer

    try:
        optimizer = SCOPEOptimizer(
            synthesizer_model=model,
            exp_path=run.experiment_dir("optimizer"),
            store_history=True,
        )

        print("Simulating a step with error...")

        result = await optimizer.on_step_complete(
            agent_name="test_agent",
            agent_role="A helpful AI assistant",
            task="Answer user questions",
            model_output="The answer is: yes",
            error=Exception("JSON parsing failed: expected object, got string"),
            current_system_prompt="You are a helpful assistant.",
            task_id="test_001",
        )

        if result:
            guideline, guideline_type = result
            print(f"✓ Generated {guideline_type} guideline:")
            print(f"  {guideline[:100]}..." if len(guideline) > 100 else f"  {guideline}")
            print("✓ SCOPEOptimizer integration test PASSED")
            return True
        else:
            print("  No guideline generated (confidence below threshold)")
            print("✓ SCOPEOptimizer integration test PASSED (no errors)")
            return True

    except Exception as e:
        print(f"✗ SCOPEOptimizer integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run API tests."""
    args = parse_args()

    with run_context("test_simple", args) as run:
        print("\n" + "=" * 50)
        print("SCOPE Simple API Test")
        print("=" * 50)
        print(f"\nProvider: {args.provider}")
        print(f"Model: {resolved_model(args)}")
        if args.api_key:
            print("API Key: ***")
        if args.base_url:
            print(f"Base URL: {args.base_url}")

        model = run.wrap_model(create_model(args))
        results = []

        # Test 1: Model Adapter
        results.append(("Model Adapter", await test_model_adapter(args, model)))

        # Test 2: SCOPEOptimizer (optional)
        if not args.skip_optimizer:
            results.append((
                "SCOPEOptimizer",
                await test_scope_optimizer(args, model, run),
            ))

        # Summary
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)

        all_passed = True
        for name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  {name}: {status}")
            if not passed:
                all_passed = False

        print()
        if all_passed:
            print("All tests passed! 🎉")
        else:
            print("Some tests failed. Check the output above for details.")

        run.save_summary({
            "provider": args.provider,
            "model": resolved_model(args),
            "results": [
                {"name": name, "passed": passed} for name, passed in results
            ],
            "all_passed": all_passed,
        })
        return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
