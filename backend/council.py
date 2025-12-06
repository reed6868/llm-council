"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Tuple, Optional

from .llm_client import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL, TITLE_MODEL
from .council_config import load_council_config


async def stage1_collect_responses(
    user_query: str,
    models: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question

    Returns:
        List of dicts with 'model' and 'response' keys
    """
    messages = [{"role": "user", "content": user_query}]

    # Allow callers to override the model list (e.g. after validation).
    models_to_use = models if models is not None else COUNCIL_MODELS

    # Query all models in parallel
    responses = await query_models_parallel(models_to_use, messages)

    # Format results
    stage1_results = []
    for model, response in responses.items():
        if response is not None:  # Only include successful responses
            stage1_results.append({
                "model": model,
                "response": response.get('content', '')
            })

    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Only ask models that produced a successful Stage 1 response to rank
    ranking_models = [result["model"] for result in stage1_results]

    # Get rankings from all successful council models in parallel
    responses = await query_models_parallel(ranking_models, messages)

    # Format results
    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed
            })

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    chairman_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2

    Returns:
        Dict with 'model' and 'response' keys
    """
    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

The original question text may include one or more project context sections that look like:
- [PROJECT ROOT] <project:some-id>
- [PROJECT PATH] /some/filesystem/path
- [PROJECT TREE]
  (directory listing)
- [PROJECT CODEBASE]
  (file previews)

Treat these PROJECT ROOT/TREE/CODEBASE sections as the ONLY reliable evidence about any codebase or filesystem path. You MUST:
- Base any statements about specific projects, repositories, or code only on these sections.
- Refer to projects by their logical handle (for example "project:some-id") rather than claiming direct access to raw filesystem paths.
- NOT claim that you have "traversed", "loaded", or "inspected" a repository or path unless it appears in a [PROJECT ROOT]/[PROJECT PATH] block with corresponding TREE/CODEBASE content.
- Treat any lines like "[Error collecting project tree: ...]" or "[Error collecting project codebase: ...]" as meaning that code for that project is NOT available.

When many Stage 1 models explicitly say that they cannot access a path or repository (for example messages containing phrases like "cannot access", "无法访问", or similar), you must treat this as strong evidence that the council as a whole lacks filesystem access. In such cases:
- Do NOT invent details about unseen code or repositories.
- Clearly state that the council does not have direct access to the requested codebase and that any high-level comments are necessarily generic or speculative.

Original Question (including any project context): {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

If the council lacks reliable project context for a requested codebase, your final answer MUST clearly explain this limitation and avoid pretending to have inspected that code.

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Determine which model acts as chairman for this call.
    effective_chairman = chairman_model or CHAIRMAN_MODEL

    # Query the chairman model
    response = await query_model(effective_chairman, messages)

    if response is None:
        # Fallback if chairman fails
        return {
            "model": effective_chairman,
            "response": "Error: Unable to generate final synthesis."
        }

    return {
        "model": effective_chairman,
        "response": response.get('content', '')
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    cfg = load_council_config()
    effective_title_model = cfg.title_model or TITLE_MODEL

    if not effective_title_model:
        return "New Conversation"

    response = await query_model(effective_title_model, messages, timeout=30.0)

    if response is None:
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's question

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
    """
    cfg = load_council_config()

    # If there are no valid council models at all, short-circuit with a clear error.
    if not cfg.council_models:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please check LLM_COUNCIL_MODELS and API keys."
        }, {
            "failures": cfg.failures,
            "runtime_failures": [],
        }

    runtime_failures = []

    # Stage 1: Collect individual responses from the effective council models.
    stage1_results = await stage1_collect_responses(user_query, models=cfg.council_models)

    # Track models that failed to produce a Stage 1 response at runtime.
    stage1_models = [result["model"] for result in stage1_results]
    for spec in cfg.council_models:
        if spec not in stage1_models:
            runtime_failures.append(
                {
                    "model_spec": spec,
                    "stage": "stage1",
                    "error_type": "runtime_failure",
                }
            )

    # If no models responded successfully, return error
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {
            "failures": cfg.failures,
            "runtime_failures": runtime_failures,
        }

    # Stage 2: Collect rankings
    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results)

    # Track models that succeeded in Stage 1 but failed in Stage 2.
    stage2_models = [result["model"] for result in stage2_results]
    for spec in stage1_models:
        if spec not in stage2_models:
            runtime_failures.append(
                {
                    "model_spec": spec,
                    "stage": "stage2",
                    "error_type": "runtime_failure",
                }
            )

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Synthesize final answer
    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results,
        chairman_model=cfg.chairman_model,
    )

    # Prepare metadata
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "failures": cfg.failures,
        "runtime_failures": runtime_failures,
    }

    return stage1_results, stage2_results, stage3_result, metadata
