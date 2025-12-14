"""
MAKER: Massively decomposed Agentic processes with K-voting Error Reduction.

This example demonstrates the MAKER workflow pattern for achieving high
reliability through statistical consensus voting.

Based on the paper:
    "Solving a Million-Step LLM Task with Zero Errors"
    Meyerson et al., 2024
    https://arxiv.org/abs/2511.09030

Key Concepts:
-------------
1. **First-to-ahead-by-k Voting**: Multiple samples are drawn from a worker
   agent. The first response to achieve a k-vote margin over all alternatives
   wins. This provides provable error bounds.

2. **Red-Flagging**: Responses that show signs of confusion (too long,
   malformed) are discarded before voting, improving effective success rate.

3. **Cost-Effective Reliability**: By trading compute (multiple samples) for
   accuracy (statistical consensus), cheap models can achieve high reliability.

When to Use MAKER:
------------------
MAKER is designed for **long chains of simple steps** where errors compound:

Good use cases:
- **ETL pipelines**: 1000s of row transformations - one bad parse = corrupted data
- **Code migration**: 1000s of file changes - one syntax error = build fails
- **Document processing**: 1000s of pages - one missed field = compliance failure
- **Data validation**: Millions of records - one wrong validation = bad data in prod
- **Automated testing**: 1000s of assertions - one false positive = wasted debugging
- **Cost optimization**: Cheap model + voting can replace expensive model

When NOT to use MAKER:
- Single classifications (just use a good model - 95% accuracy is fine)
- Creative/open-ended tasks (no "correct" answer to vote on)
- Complex reasoning (need smarter model, not more samples)
- Tasks where occasional errors are acceptable

The Math:
---------
- 95% per-step accuracy over 100 steps = 0.6% overall success (0.95^100)
- 99.9% per-step accuracy (with MAKER) over 100 steps = 90% overall success
- For million-step tasks, even 99% per-step fails; MAKER enables 99.99%+

Demo Use Case:
--------------
This example shows customer message intent classification. While modern LLMs
are quite consistent on this task (you'll see 3:0 votes), the mechanism
demonstrates how voting works. In production with harder tasks or longer
chains, MAKER's value becomes critical.

Usage:
------
    uv run examples/workflows/maker.py

Try modifying k (voting margin) and observe how it affects reliability vs cost.
"""

import asyncio

from fast_agent import FastAgent

fast = FastAgent("MAKER Example")


# Define a classifier using a cheap model (Haiku) - may give inconsistent results
# on ambiguous messages, which is why we wrap it with MAKER for reliability
@fast.agent(
    name="classifier",
    model="claude-3-haiku-20240307", 
    instruction="""You are a customer support intent classifier.
Classify the customer message into exactly one of: COMPLAINT, QUESTION, REQUEST, FEEDBACK.
Respond with ONLY the single word classification, nothing else.

Examples:
- "This product is broken!" → COMPLAINT
- "How do I reset my password?" → QUESTION
- "Please cancel my subscription" → REQUEST
- "Just wanted to say I love the new feature" → FEEDBACK""",
)
# Wrap with MAKER for reliable, consistent classification
@fast.maker(
    name="reliable_classifier",
    worker="classifier",
    k=3,  # Require 3-vote margin for consensus
    max_samples=10,  # Max attempts before falling back to plurality
    match_strategy="normalized",  # Ignore case/whitespace differences
    red_flag_max_length=20,  # Discard verbose responses (should be one word)
)
async def main():
    """Demonstrate MAKER voting for reliable intent classification."""
    async with fast.run() as agent:
        print("=" * 70)
        print("MAKER: Massively decomposed Agentic processes")
        print("       with K-voting Error Reduction")
        print("=" * 70)
        print()
        print("This example classifies customer messages where intent is ambiguous.")
        print("MAKER voting ensures consistent routing even for edge cases.")
        print()

        # Ambiguous customer messages where intent is unclear
        test_cases = [
            "I've been waiting for 3 days now.",              # Complaint? Status question?
            "Can someone explain how this works?",            # Question? Request for help?
            "This isn't what I expected.",                    # Complaint? Feedback?
            "I'd like to speak to a manager.",                # Complaint? Request?
            "Why does this keep happening?",                  # Complaint? Question?
            "Just wanted to let you know about this.",        # Feedback? Complaint?
            "Is there any way to get a refund?",              # Question? Request?
            "The new update changed everything.",             # Complaint? Feedback?
        ]

        # Collect all results first
        results = []
        for text in test_cases:
            result = await agent.reliable_classifier.send(text)
            stats = agent.reliable_classifier.last_result
            results.append((text, result, stats))

        # Display all results together
        print("-" * 70)
        print(f"{'Text':<50} {'Result':<10} {'Samples':<8} {'Votes'}")
        print("-" * 70)

        for text, result, stats in results:
            votes_str = ""
            samples = ""
            if stats:
                votes_str = ", ".join(f"{k}:{v}" for k, v in stats.votes.items())
                samples = str(stats.total_samples)

            print(f"{text:<50} {result:<10} {samples:<8} {votes_str}")

        print("-" * 70)
        print()
        print("Notice how MAKER provides consistent routing decisions even for")
        print("ambiguous messages by voting across multiple samples.")
        print()

        # Summary statistics
        total_samples = sum(r[2].total_samples for r in results if r[2])
        all_converged = all(r[2].converged for r in results if r[2])
        print("Summary:")
        print(f"  - Total API calls: {total_samples}")
        print(f"  - All converged: {all_converged}")
        print(f"  - Texts classified: {len(results)}")
        print()
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
