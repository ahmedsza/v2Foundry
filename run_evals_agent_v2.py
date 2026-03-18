"""
Evaluate an EXISTING AI agent with GROUND TRUTH and SIMILARITY EVALUATORS

This script extends the basic evaluation with:

QUALITY EVALUATORS (AI-Assisted):
- Coherence: How well-structured and logical is the response
- Relevance: How relevant is the response to the user's query
- Fluency: How natural and grammatically correct is the response

SIMILARITY EVALUATORS (Ground Truth Comparison):
- F1 Score: Token-level overlap between response and ground truth
- BLEU Score: N-gram precision metric (common in machine translation)
- ROUGE Score: Recall-oriented metric (common in summarization)

SAFETY EVALUATORS (Content Safety):
- Violence: Detects violent content
- Hate/Unfairness: Detects hateful or unfair content
- Sexual: Detects sexual content
- Self-Harm: Detects self-harm content

Data Format:
This script supports JSONL files with ground_truth field:
{
  "query": "How do I get a passport?",
  "ground_truth": "Visit travel.state.gov, complete DS-11 form, provide ID and photo, pay fee"
}

The script uses the new v2.0 Cloud Evaluation API which requires minimal configuration:
- No need to create agents (uses existing agent by name and version)
- No subscription_id, resource_group_name, or project_name needed
- Results automatically appear in Azure AI Foundry portal

Prerequisites:
    pip install azure-ai-projects azure-identity

Environment Variables Required:
    - AZURE_AI_PROJECT_ENDPOINT: The project endpoint from your Azure AI Foundry project
    - MODEL_DEPLOYMENT_NAME: Model deployment name for AI-assisted evaluators
    - EXISTING_AGENT_NAME: Name of the existing agent to evaluate
    - EXISTING_AGENT_VERSION: Version of the existing agent (optional, uses latest if not specified)

Important: Authenticate to Azure using `az login` before running this script.
"""

import os
import json
import time
from typing import Union
from pprint import pprint
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from openai.types.eval_create_params import DataSourceConfigCustom
from openai.types.evals.run_create_response import RunCreateResponse
from openai.types.evals.run_retrieve_response import RunRetrieveResponse

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Continuing with system environment variables...")


def initialize_project_client():
    """Initialize Azure AI Project Client"""
    endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
    
    project_client = AIProjectClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
    )
    return project_client


def load_test_data_from_jsonl(file_path):
    """Load test queries and ground truth from JSONL file"""
    test_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                test_data.append(item)
    
    print(f"Loaded {len(test_data)} test items from {file_path}")
    return test_data


def setup_cloud_evaluators_with_similarity(model_deployment_name, include_similarity=True):
    """Setup cloud evaluators including similarity metrics if ground_truth is available"""
    # Notes: {{sample.output_text}} is the agent's string response
    #        {{sample.output_items}} is structured JSON output including tool call info
    testing_criteria = [
        # ===== QUALITY EVALUATORS (AI-Assisted) =====
        {
            "type": "azure_ai_evaluator",
            "name": "coherence",
            "evaluator_name": "builtin.coherence",
            "initialization_parameters": {"deployment_name": model_deployment_name},
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{sample.output_text}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "relevance",
            "evaluator_name": "builtin.relevance",
            "initialization_parameters": {"deployment_name": model_deployment_name},
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{sample.output_text}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "fluency",
            "evaluator_name": "builtin.fluency",
            "initialization_parameters": {"deployment_name": model_deployment_name},
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{sample.output_text}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "task_adherence",
            "evaluator_name": "builtin.task_adherence",
            "initialization_parameters": {"deployment_name": model_deployment_name},
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{sample.output_items}}",
            },
        },
    ]

    if include_similarity:
        similarity_evaluators = [
            # ===== SIMILARITY / CORRECTNESS EVALUATORS (Ground Truth Comparison) =====
            {
                # AI-assisted semantic similarity: measures whether the response says the same
                # thing as the ground truth, regardless of phrasing. 1-5 score.
                # This is the closest built-in proxy for "correctness".
                "type": "azure_ai_evaluator",
                "name": "similarity",
                "evaluator_name": "builtin.similarity",
                "initialization_parameters": {"deployment_name": model_deployment_name},
                "data_mapping": {
                    "query": "{{item.query}}",
                    "response": "{{sample.output_text}}",
                    "ground_truth": "{{item.ground_truth}}",
                },
            },
            {
                "type": "azure_ai_evaluator",
                "name": "f1_score",
                "evaluator_name": "builtin.f1_score",
                "data_mapping": {
                    "response": "{{sample.output_text}}",
                    "ground_truth": "{{item.ground_truth}}",
                },
            },
            {
                "type": "azure_ai_evaluator",
                "name": "bleu_score",
                "evaluator_name": "builtin.bleu_score",
                "data_mapping": {
                    "response": "{{sample.output_text}}",
                    "ground_truth": "{{item.ground_truth}}",
                },
            },
            {
                "type": "azure_ai_evaluator",
                "name": "rouge_score",
                "evaluator_name": "builtin.rouge_score",
                "initialization_parameters": {"rouge_type": "rougeL"},
                "data_mapping": {
                    "response": "{{sample.output_text}}",
                    "ground_truth": "{{item.ground_truth}}",
                },
            },
        ]
        testing_criteria.extend(similarity_evaluators)

    safety_evaluators = [
        # ===== SAFETY EVALUATORS (Content Safety) =====
        {
            "type": "azure_ai_evaluator",
            "name": "violence",
            "evaluator_name": "builtin.violence",
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{sample.output_text}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "hate_unfairness",
            "evaluator_name": "builtin.hate_unfairness",
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{sample.output_text}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "sexual",
            "evaluator_name": "builtin.sexual",
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{sample.output_text}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "self_harm",
            "evaluator_name": "builtin.self_harm",
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{sample.output_text}}",
            },
        },
    ]
    testing_criteria.extend(safety_evaluators)

    print(f"Configured {len(testing_criteria)} evaluators:")
    print("  - Quality: coherence, relevance, fluency, task_adherence")
    if include_similarity:
        print("  - Similarity/Correctness: similarity (AI), f1_score, bleu_score, rouge_score")
    print("  - Safety: violence, hate_unfairness, sexual, self_harm")

    return testing_criteria


def run_cloud_evaluation(project_client, test_data, testing_criteria, model_deployment_name, agent_name, agent_version=None, has_ground_truth=False):
    """Run cloud evaluation using azure_ai_target_completions - the agent is called by the eval system"""
    print("\n" + "=" * 50)
    print("RUNNING CLOUD EVALUATION WITH GROUND TRUTH" if has_ground_truth else "RUNNING CLOUD EVALUATION")
    print("=" * 50)

    client = project_client.get_openai_client()

    # Build item schema - includes query and optionally ground_truth
    schema_properties = {"query": {"type": "string"}}
    required_fields = ["query"]
    if has_ground_truth:
        schema_properties["ground_truth"] = {"type": "string"}
        required_fields.append("ground_truth")

    data_source_config = DataSourceConfigCustom(
        type="custom",
        item_schema={
            "type": "object",
            "properties": schema_properties,
            "required": required_fields,
        },
        include_sample_schema=True,  # exposes sample.output_text / sample.output_items
    )

    # Create evaluation definition
    print("Creating evaluation definition...")
    eval_name = "agent-evaluation-with-groundtruth" if has_ground_truth else "agent-quality-evaluation"
    eval_object = client.evals.create(
        name=eval_name,
        data_source_config=data_source_config,
        testing_criteria=testing_criteria,  # type: ignore
    )
    print(f"Created evaluation: {eval_object.id}")

    # Build content items - only pass query (and ground_truth if present) as item data
    content = [
        {"item": {k: v for k, v in item.items() if k in ("query", "ground_truth")}}
        for item in test_data
    ]

    # Target references the existing agent by name/version
    target: dict = {"type": "azure_ai_agent", "name": agent_name}
    if agent_version:
        target["version"] = agent_version

    # azure_ai_target_completions: the eval system calls the agent directly per item
    data_source = {
        "type": "azure_ai_target_completions",
        "source": {
            "type": "file_content",
            "content": content,
        },
        "input_messages": {
            "type": "template",
            "template": [
                {"type": "message", "role": "user", "content": {"type": "input_text", "text": "{{item.query}}"}}
            ],
        },
        "target": target,
    }

    # Create evaluation run
    print("Creating evaluation run...")
    run_name = "agent-groundtruth-run" if has_ground_truth else "agent-quality-run"
    eval_run: Union[RunCreateResponse, RunRetrieveResponse] = client.evals.runs.create(
        eval_id=eval_object.id,
        name=f"{run_name} for {agent_name}",
        data_source=data_source,  # type: ignore
    )
    print(f"Created evaluation run: {eval_run.id}")

    # Poll for completion
    print("Waiting for evaluation to complete...")
    while eval_run.status not in ("completed", "failed"):
        eval_run = client.evals.runs.retrieve(run_id=eval_run.id, eval_id=eval_object.id)
        print(f"  Status: {eval_run.status}")
        time.sleep(5)

    if eval_run.status == "completed":
        print(f"\n\u2713 Evaluation run completed successfully!")
        print(f"Result Counts: {eval_run.result_counts}")
        output_items = list(
            client.evals.runs.output_items.list(run_id=eval_run.id, eval_id=eval_object.id)
        )
        return {
            "status": eval_run.status,
            "output_items": output_items,
            "report_url": eval_run.report_url if hasattr(eval_run, "report_url") else None,
        }
    else:
        print("\n\u2717 Evaluation run failed.")
        return {"status": eval_run.status, "error": "Evaluation failed"}


def display_results(results):
    """Display evaluation results"""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    if results["status"] != "completed":
        print(f"\nEvaluation status: {results['status']}")
        if "error" in results:
            print(f"Error: {results['error']}")
        return
    
    # Display report URL
    if results.get("report_url"):
        print(f"\nReport URL: {results['report_url']}")
        print("View detailed results in Azure AI Foundry portal")
    
    # Display output items
    print("\nEvaluation Scores:")
    for item in results["output_items"]:
        pprint(item)
        print("-" * 40)


def main():
    """Main execution function"""
    USE_JSONL_FILE = True
    JSONL_INPUT_FILE = "test_data_with_groundtruth.jsonl"

    # Inline test data fallback
    INLINE_TEST_DATA = [
        {
            "query": "How do I get a new passport?",
            "ground_truth": "To get a new passport: 1) Complete application form DS-11, 2) Provide proof of citizenship and photo ID, 3) Submit passport photo, 4) Pay fee ($130 for book, $30 for card), 5) Submit in person at acceptance facility or passport agency. Processing takes 10-12 weeks normally, 5-7 weeks expedited."
        },
        {
            "query": "What documents do I need for passport renewal?",
            "ground_truth": "For passport renewal you need: 1) Your most recent passport (must be undamaged), 2) Completed form DS-82, 3) One passport photo, 4) Fee payment ($130 for book). You can renew by mail if your passport is less than 15 years old and was issued when you were 16 or older."
        },
        {
            "query": "How long does it take to get a passport?",
            "ground_truth": "Standard passport processing takes 10-12 weeks. Expedited service costs extra $60 and takes 5-7 weeks. For urgent travel, you can visit a passport agency or center for same-day or next-day service with proof of international travel within 14 days."
        },
    ]

    required_env_vars = [
        "AZURE_AI_PROJECT_ENDPOINT",
        "MODEL_DEPLOYMENT_NAME",
        "EXISTING_AGENT_NAME",
    ]
    missing_vars = [var for var in required_env_vars if var not in os.environ]
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        return

    try:
        print("Initializing project client...")
        project_client = initialize_project_client()

        agent_name = os.environ["EXISTING_AGENT_NAME"]
        agent_version = os.environ.get("EXISTING_AGENT_VERSION")
        print(f"\nUsing existing agent: {agent_name}" + (f" (version {agent_version})" if agent_version else " (latest version)"))

        # Load test data
        if USE_JSONL_FILE and os.path.exists(JSONL_INPUT_FILE):
            print(f"\nLoading test data from {JSONL_INPUT_FILE}...")
            test_data = load_test_data_from_jsonl(JSONL_INPUT_FILE)
        else:
            if USE_JSONL_FILE:
                print(f"\nWARNING: {JSONL_INPUT_FILE} not found. Using inline test data instead.")
            else:
                print("\nUsing inline test data...")
            test_data = INLINE_TEST_DATA

        has_ground_truth = any("ground_truth" in item and item["ground_truth"] for item in test_data)
        if has_ground_truth:
            print(f"\n✓ Ground truth detected - will include similarity evaluators")
        else:
            print(f"\n⚠ No ground truth found - similarity evaluators will be skipped")

        print("\nSetting up cloud evaluators...")
        testing_criteria = setup_cloud_evaluators_with_similarity(
            os.environ["MODEL_DEPLOYMENT_NAME"],
            include_similarity=has_ground_truth,
        )

        # Run evaluation - the eval system calls the agent directly via azure_ai_target_completions
        eval_response = run_cloud_evaluation(
            project_client,
            test_data,
            testing_criteria,
            os.environ["MODEL_DEPLOYMENT_NAME"],
            agent_name=agent_name,
            agent_version=agent_version,
            has_ground_truth=has_ground_truth,
        )

        display_results(eval_response)

        print("\n" + "=" * 50)
        print("EVALUATION COMPLETE")
        print("=" * 50)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise


if __name__ == "__main__":
    main()
