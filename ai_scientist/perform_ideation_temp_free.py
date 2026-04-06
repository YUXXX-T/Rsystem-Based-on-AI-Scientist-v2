import argparse
import json
import os.path as osp
import re
import traceback
from typing import Any, Dict, List

import sys

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from ai_scientist.llm import (
    AVAILABLE_LLMS,
    create_client,
    get_response_from_llm,
)

from ai_scientist.tools.semantic_scholar import SemanticScholarSearchTool
from ai_scientist.tools.openalex import OpenAlexSearchTool
from ai_scientist.tools.base_tool import BaseTool

# Create tool instances - using OpenAlex (free, no API key needed, generous rate limits)
semantic_scholar_tool = OpenAlexSearchTool()
# Create tool instances
# semantic_scholar_tool = SemanticScholarSearchTool()

# Define tools at the top of the file
tools = [
    semantic_scholar_tool,
    {
        "name": "FinalizeIdea",
        "description": """Finalize your idea by providing the idea details.

The IDEA JSON should include the following fields:
- "Name": A short descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A catchy and informative title for the proposal.
- "Short Hypothesis": A concise statement of the main hypothesis or research question. Clarify the need for this specific direction, ensure this is the best setting to investigate this idea, and there are not obvious other simpler ways to answer the question.
- "Related Work": A brief discussion of the most relevant related work and how the proposal clearly distinguishes from it, and is not a trivial extension.
- "Abstract": An abstract that summarizes the proposal in conference format (approximately 250 words).
- "Experiments": A list of experiments that would be conducted to validate the proposal. Ensure these are simple and feasible. Be specific in exactly how you would test the hypothesis, and detail precise algorithmic changes. Include the evaluation metrics you would use.
- "Risk Factors and Limitations": A list of potential risks and limitations of the proposal.""",
    },
]

# Create a tools dictionary for easy lookup
tools_dict = {tool.name: tool for tool in tools if isinstance(tool, BaseTool)}

# Create a string with the tool descriptions
tool_descriptions = "\n\n".join(
    (
        f"- **{tool.name}**: {tool.description}"
        if isinstance(tool, BaseTool)
        else f"- **{tool['name']}**: {tool['description']}"
    )
    for tool in tools
)

# Extract tool names for the prompt
tool_names = [
    f'"{tool.name}"' if isinstance(tool, BaseTool) else f'"{tool["name"]}"'
    for tool in tools
]
tool_names_str = ", ".join(tool_names)

system_prompt = f"""You are an experienced AI researcher who aims to propose high-impact research ideas resembling exciting grant proposals. Feel free to propose any novel ideas or experiments; make sure they are novel. Be very creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis about the topic. For example, they could involve very interesting and simple interventions or investigations that explore new possibilities or challenge existing assumptions. Clearly clarify how the proposal distinguishes from the existing literature.

Ensure that the proposal does not require resources beyond what an academic lab could afford. These proposals should lead to papers that are publishable at top ML conferences.

You have access to the following tools:

{tool_descriptions}

Respond in the following format:

ACTION:
<The action to take, exactly one of {tool_names_str}>

ARGUMENTS:
<If ACTION is "SearchSemanticScholar", provide the search query as {{"query": "your search query"}}. If ACTION is "FinalizeIdea", provide the idea details as {{"idea": {{ ... }}}} with the IDEA JSON specified below.>

If you choose to finalize your idea, provide the IDEA JSON in the arguments:

IDEA JSON:
```json
{{
  "idea": {{
    "Name": "...",
    "Title": "...",
    "Short Hypothesis": "...",
    "Related Work": "...",
    "Abstract": "...",
    "Experiments": "...",
    "Risk Factors and Limitations": "..."
  }}
}}
```

Ensure the JSON is properly formatted for automatic parsing.

Note: You should perform at least one literature search before finalizing your idea to ensure it is well-informed by existing research."""

# Define the initial idea generation prompt
idea_generation_prompt = """{workshop_description}

Here are the proposals that you have already generated:

'''
{prev_ideas_string}
'''

Begin by generating an interestingly new high-level research proposal that differs from what you have previously proposed.
"""

# Define the reflection prompt
idea_reflection_prompt = """Round {current_round}/{num_reflections}.

In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created.
Include any other factors that you think are important in evaluating the proposal.
Ensure the proposal is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try to refine and improve your proposal.
Stick to the spirit of the original idea unless there are glaring issues.

If you have new information from tools, such as literature search results, incorporate them into your reflection and refine your proposal accordingly.

Results from your last action (if any):

{last_tool_results}
"""


def generate_temp_free_idea(
    idea_fname: str,
    client: Any,
    model: str,
    workshop_description: str,
    max_num_generations: int = 20,
    num_reflections: int = 5,
    reload_ideas: bool = True,
) -> List[Dict]:
    idea_str_archive = []
    # load ideas from file
    if reload_ideas and osp.exists(idea_fname):
        with open(idea_fname, "r") as f:
            idea_str_content = json.load(f)
            for idea in idea_str_content:
                idea_str_archive.append(json.dumps(idea))
            print(f"Loaded {len(idea_str_archive)} ideas from {idea_fname}")
    else:
        print(f"No ideas found in {idea_fname}. Starting from scratch.")

    for gen_idx in range(max_num_generations):
        print()
        print(f"Generating proposal {gen_idx + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            last_tool_results = ""
            idea_finalized = False
            msg_history = []

            for reflection_round in range(num_reflections):
                if reflection_round == 0:
                    # Use the initial idea generation prompt
                    prompt_text = idea_generation_prompt.format(
                        workshop_description=workshop_description,
                        prev_ideas_string=prev_ideas_string,
                    )
                else:
                    # Use the reflection prompt, including tool results if any
                    prompt_text = idea_reflection_prompt.format(
                        current_round=reflection_round + 1,
                        num_reflections=num_reflections,
                        last_tool_results=last_tool_results or "No new results.",
                    )

                response_text, msg_history = get_response_from_llm(
                    prompt=prompt_text,
                    client=client,
                    model=model,
                    system_message=system_prompt,
                    msg_history=msg_history,
                )

                # Parse the LLM's response
                try:
                    # Use regular expressions to extract the components
                    action_pattern = r"ACTION:\s*(.*?)\s*ARGUMENTS:"
                    arguments_pattern = r"ARGUMENTS:\s*(.*?)(?:$|\nTHOUGHT:|\n$)"

                    action_match = re.search(
                        action_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )
                    arguments_match = re.search(
                        arguments_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )

                    if not all([action_match, arguments_match]):
                        raise ValueError("Failed to parse the LLM response.")

                    action = action_match.group(1).strip()
                    arguments_text = arguments_match.group(1).strip()
                    print(f"Action: {action}")
                    print(f"Arguments: {arguments_text}")

                    # If arguments are wrapped in ```json blocks, extract the content
                    if arguments_text.startswith("```json"):
                        arguments_text = re.search(
                            r"```json\s*(.*?)\s*```", arguments_text, re.DOTALL
                        ).group(1)

                    # Process the action and arguments
                    if action in tools_dict:
                        # It's a tool we have defined
                        tool = tools_dict[action]
                        # Parse arguments
                        try:
                            arguments_json = json.loads(arguments_text)
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid arguments JSON for {action}.")

                        # Use the tool
                        try:
                            # Assuming the arguments match the parameters of the tool
                            result = tool.use_tool(**arguments_json)
                            last_tool_results = result
                        except Exception as e:
                            last_tool_results = f"Error using tool {action}: {str(e)}"
                    elif action == "FinalizeIdea":
                        # Parse arguments
                        try:
                            arguments_json = json.loads(arguments_text)
                            idea = arguments_json.get("idea")
                            if not idea:
                                raise ValueError("Missing 'idea' in arguments.")

                            # Append the idea to the archive
                            idea_str_archive.append(json.dumps(idea))
                            print(f"Proposal finalized: {idea}")
                            idea_finalized = True

                            # Save incrementally after each idea to prevent data loss on interruption
                            ideas_so_far = [json.loads(s) for s in idea_str_archive]
                            with open(idea_fname, "w") as f:
                                json.dump(ideas_so_far, f, indent=4)
                            print(f"Incrementally saved {len(ideas_so_far)} ideas to {idea_fname}")

                            break
                        except json.JSONDecodeError:
                            raise ValueError("Invalid arguments JSON for FinalizeIdea.")
                    else:
                        print(
                            "Invalid action. Please specify one of the available tools."
                        )
                        print(f"Available actions are: {tool_names_str}")
                except Exception as e:
                    print(
                        f"Failed to parse LLM response. Response text:\n{response_text}"
                    )
                    traceback.print_exc()
                    break  # Exit the loop if parsing fails

            if idea_finalized:
                continue  # Move to the next idea

        except Exception as e:
            print("Failed to generate proposal:")
            traceback.print_exc()
            continue

    # Save ideas
    ideas = [json.loads(idea_str) for idea_str in idea_str_archive]

    with open(idea_fname, "w") as f:
        json.dump(ideas, f, indent=4)
    print(f"Stored {len(ideas)} ideas in {idea_fname}")
    return ideas


## ============================================================
## Post-processing: Evaluate and Translate Ideas
## ============================================================

evaluate_system_prompt = """You are a senior researcher with expertise in both Machine Learning and Robotics, serving as a reviewer for a top-tier conference (e.g., NeurIPS, ICML, ICLR, ICRA, IROS, AAMAS, RSS). 
You are particularly knowledgeable in multi-agent systems, path planning (MAPF), warehouse automation, and task scheduling.
You will evaluate a research proposal and provide structured scores and feedback.

For each proposal, provide your evaluation in the following JSON format:
```json
{
    "novelty": <1-10>,
    "feasibility": <1-10>,
    "impact": <1-10>,
    "overall_score": "<A+/A/A-/B+/B/B-/C>",
    "strengths": ["strength 1", "strength 2", ...],
    "weaknesses": ["weakness 1", "weakness 2", ...],
    "suggestions": "Brief suggestions for improvement",
    "one_line_summary": "A one-line summary of the core contribution"
}
```

Scoring criteria:
- **Novelty (1-10)**: How original is the idea? Does it offer a genuinely new perspective?
- **Feasibility (1-10)**: Can the proposed experiments be realistically conducted by an academic lab?
- **Impact (1-10)**: If successful, how significant would the contribution be to the field?

Be rigorous but fair. Justify your scores through the strengths and weaknesses."""

evaluate_prompt = """Please evaluate the following research proposal:

**Title**: {title}

**Hypothesis**: {hypothesis}

**Related Work**: {related_work}

**Abstract**: {abstract}

**Experiments**: {experiments}

**Risk Factors**: {risks}

Provide your evaluation as a JSON object."""

translate_system_prompt = """You are a professional academic translator specializing in AI/ML research.
Translate the following research proposal from English to Chinese (简体中文).
Maintain academic rigor and use standard Chinese ML terminology.
Return the result as a JSON object with the exact same keys as the input, but with all values translated to Chinese.
The "Name" field should remain in English (it's a code identifier).
Ensure the JSON is valid and properly formatted."""

translate_prompt = """Translate this research proposal to Chinese (简体中文). Return as valid JSON with the same keys:

```json
{idea_json}
```"""


def evaluate_ideas(
    ideas: List[Dict],
    client: Any,
    model: str,
    output_fname: str,
) -> List[Dict]:
    """Evaluate each idea using LLM and save scored results."""
    evaluated_ideas = []

    for i, idea in enumerate(ideas):
        print(f"\nEvaluating idea {i + 1}/{len(ideas)}: {idea.get('Title', 'Unknown')}")
        try:
            prompt = evaluate_prompt.format(
                title=idea.get("Title", ""),
                hypothesis=idea.get("Short Hypothesis", ""),
                related_work=idea.get("Related Work", ""),
                abstract=idea.get("Abstract", ""),
                experiments=idea.get("Experiments", ""),
                risks=idea.get("Risk Factors and Limitations", ""),
            )

            response_text, _ = get_response_from_llm(
                prompt=prompt,
                client=client,
                model=model,
                system_message=evaluate_system_prompt,
                msg_history=[],
            )

            # Parse the evaluation JSON from response
            eval_json = None
            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                try:
                    eval_json = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            if eval_json is None:
                # Try to find raw JSON
                try:
                    # Find the first { and last }
                    start = response_text.index("{")
                    end = response_text.rindex("}") + 1
                    eval_json = json.loads(response_text[start:end])
                except (ValueError, json.JSONDecodeError):
                    print(f"  Failed to parse evaluation for idea {i + 1}")
                    eval_json = {"error": "Failed to parse evaluation", "raw_response": response_text[:500]}

            evaluated_idea = {**idea, "evaluation": eval_json}
            evaluated_ideas.append(evaluated_idea)
            print(f"  Score: {eval_json.get('overall_score', 'N/A')} | "
                  f"Novelty: {eval_json.get('novelty', 'N/A')}/10 | "
                  f"Feasibility: {eval_json.get('feasibility', 'N/A')}/10 | "
                  f"Impact: {eval_json.get('impact', 'N/A')}/10")

        except Exception as e:
            print(f"  Error evaluating idea {i + 1}: {e}")
            traceback.print_exc()
            evaluated_ideas.append({**idea, "evaluation": {"error": str(e)}})

    # Save evaluated ideas
    with open(output_fname, "w", encoding="utf-8") as f:
        json.dump(evaluated_ideas, f, indent=4, ensure_ascii=False)
    print(f"\nSaved {len(evaluated_ideas)} evaluated ideas to {output_fname}")

    # Print summary ranking
    print("\n" + "=" * 60)
    print("IDEA RANKING SUMMARY")
    print("=" * 60)
    ranked = sorted(
        [(i, e) for i, e in enumerate(evaluated_ideas) if "error" not in e.get("evaluation", {})],
        key=lambda x: (
            x[1]["evaluation"].get("novelty", 0)
            + x[1]["evaluation"].get("feasibility", 0)
            + x[1]["evaluation"].get("impact", 0)
        ),
        reverse=True,
    )
    for rank, (idx, idea) in enumerate(ranked, 1):
        ev = idea["evaluation"]
        total = ev.get("novelty", 0) + ev.get("feasibility", 0) + ev.get("impact", 0)
        print(f"  #{rank} [{ev.get('overall_score', '?')}] (N:{ev.get('novelty')}/F:{ev.get('feasibility')}/I:{ev.get('impact')}={total}) {idea.get('Title', '')[:70]}")
    print("=" * 60)

    return evaluated_ideas


def translate_ideas(
    ideas: List[Dict],
    client: Any,
    model: str,
    output_fname: str,
) -> List[Dict]:
    """Translate each idea to Chinese using LLM and save results."""
    translated_ideas = []

    for i, idea in enumerate(ideas):
        print(f"\nTranslating idea {i + 1}/{len(ideas)}: {idea.get('Title', 'Unknown')[:60]}")
        try:
            # Remove evaluation field if present (translate only the core idea)
            idea_core = {k: v for k, v in idea.items() if k != "evaluation"}
            prompt = translate_prompt.format(idea_json=json.dumps(idea_core, indent=2, ensure_ascii=False))

            response_text, _ = get_response_from_llm(
                prompt=prompt,
                client=client,
                model=model,
                system_message=translate_system_prompt,
                msg_history=[],
            )

            # Parse the translated JSON
            translated = None
            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                try:
                    translated = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            if translated is None:
                try:
                    start = response_text.index("{")
                    end = response_text.rindex("}") + 1
                    translated = json.loads(response_text[start:end])
                except (ValueError, json.JSONDecodeError):
                    print(f"  Failed to parse translation for idea {i + 1}")
                    translated = idea_core  # fallback to original

            # Copy over evaluation if it exists in original
            if "evaluation" in idea:
                translated["evaluation"] = idea["evaluation"]

            translated_ideas.append(translated)
            print(f"  Translated: {translated.get('Title', '')[:60]}")

        except Exception as e:
            print(f"  Error translating idea {i + 1}: {e}")
            traceback.print_exc()
            translated_ideas.append(idea)  # fallback to original

    # Save translated ideas
    with open(output_fname, "w", encoding="utf-8") as f:
        json.dump(translated_ideas, f, indent=4, ensure_ascii=False)
    print(f"\nSaved {len(translated_ideas)} translated ideas to {output_fname}")
    return translated_ideas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI scientist proposals - template free"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--max-num-generations",
        type=int,
        default=1,
        help="Maximum number of proposal generations.",
    )
    parser.add_argument(
        "--workshop-file",
        type=str,
        default="ideas/i_cant_believe_its_not_better.md",
        help="Path to the workshop description file.",
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=5,
        help="Number of reflection rounds per proposal.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip idea generation, only run evaluation and translation on existing ideas.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=True,
        help="Skip idea evaluation (default: True).",
    )
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        default=True,
        help="Skip Chinese translation (default: True).",
    )
    args = parser.parse_args()

    # Create the LLM client
    client, client_model = create_client(args.model)

    with open(args.workshop_file, "r") as f:
        workshop_description = f.read()
    print(f"Using workshop description from {args.workshop_file} for idea generation.")
    print(f"Workshop description:\n{workshop_description}")

    # Create output filename by replacing .md extension with .json
    idea_fname = args.workshop_file.replace(".md", ".json")

    if not args.skip_generation:
        print("Starting idea generation for", idea_fname)
        ideas = generate_temp_free_idea(
            idea_fname=idea_fname,
            client=client,
            model=client_model,
            workshop_description=workshop_description,
            max_num_generations=args.max_num_generations,
            num_reflections=args.num_reflections,
        )
        print(f"{args.workshop_file} generated {len(ideas)} ideas.")
    else:
        print(f"Skipping generation, loading existing ideas from {idea_fname}")
        with open(idea_fname, "r") as f:
            ideas = json.load(f)
        print(f"Loaded {len(ideas)} existing ideas.")

    # Phase 2: Evaluate ideas
    if not args.skip_eval and ideas:
        eval_fname = idea_fname.replace(".json", "_evaluated.json")
        print(f"\n{'='*60}")
        print("PHASE 2: EVALUATING IDEAS")
        print(f"{'='*60}")
        evaluated_ideas = evaluate_ideas(
            ideas=ideas,
            client=client,
            model=client_model,
            output_fname=eval_fname,
        )
    else:
        evaluated_ideas = ideas

    # Phase 3: Translate ideas to Chinese
    if not args.skip_translation and evaluated_ideas:
        cn_fname = idea_fname.replace(".json", "_cn.json")
        print(f"\n{'='*60}")
        print("PHASE 3: TRANSLATING IDEAS TO CHINESE")
        print(f"{'='*60}")
        translate_ideas(
            ideas=evaluated_ideas,
            client=client,
            model=client_model,
            output_fname=cn_fname,
        )

    print("\nAll phases completed.")
