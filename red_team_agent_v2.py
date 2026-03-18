"""
Run AI Red Teaming Agent scan against an EXISTING AI agent

This script uses Azure AI Evaluation SDK's RedTeam class to perform automated
adversarial testing against an existing agent. It generates adversarial prompts
across multiple risk categories and attack strategies to assess the agent's
safety and robustness.

KEY FEATURES:
- Automated adversarial prompt generation across risk categories
- Multiple attack strategies (EASY, MODERATE, DIFFICULT complexity)
- Attack Success Rate (ASR) metrics by risk category and complexity
- Detailed scorecard JSON output for compliance and reporting
- Results automatically appear in Azure AI Foundry portal

RISK CATEGORIES TESTED:
- Violence: Violent content detection
- Hate/Unfairness: Hateful or unfair content detection
- Sexual: Sexual content detection
- Self-Harm: Self-harm content detection

ATTACK STRATEGIES:
- EASY: Base64, Flip, Morse (encoding-based attacks)
- MODERATE: Tense (requires generative AI model)
- DIFFICULT: Multi-strategy compositions (e.g., Tense + Base64)

ADVANCED STRATEGIES (commented for optional use):
- Jailbreak: User Injected Prompt Attacks (UPIA)
- IndirectAttack: Indirect Prompt Injected Attacks (XPIA)
- Multiturn: Multi-turn conversation attacks
- Crescendo: Gradually increasing prompt complexity

Prerequisites:
    pip install azure-ai-evaluation[redteam] azure-ai-projects azure-identity

    Note: PyRIT (included with redteam extra) only supports Python 3.10, 3.11, 3.12
    (not Python 3.9)

Environment Variables Required:
    - AZURE_AI_PROJECT_ENDPOINT: The project endpoint from your Azure AI Foundry project
    - EXISTING_AGENT_NAME: Name of the existing agent to test
    - EXISTING_AGENT_VERSION: Version of the existing agent (optional, uses latest if not specified)

Optional Environment Variables:
    - RED_TEAM_NUM_OBJECTIVES: Number of attack objectives per risk category (default: 5)

Results Output:
    PyRIT automatically creates a directory structure for results:
    .scan_Agent-Red-Team-Scan_<timestamp>/
        - scorecard.txt: Summary of ASR by category and complexity
        - final_results.json: Complete results with all data
        - results.json: Results data
        - *_results.jsonl: Per-category detailed attack/response pairs

Important: 
- Authenticate to Azure using `az login` before running this script
- Your Azure AI Project must be in a supported region: East US2, Sweden Central, France Central, or Switzerland West
- This script uses async/await, so it must be run with Python's asyncio

Regional Support:
Only the following regions are supported for AI Red Teaming Agent:
- East US2
- Sweden Central
- France Central
- Switzerland West

Usage:
    python redTeamAgentScan.py
"""

import os
import json
import asyncio
from pprint import pprint
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.evaluation.red_team import RedTeam, RiskCategory, AttackStrategy

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Continuing with system environment variables...")


def get_agent_reference():
    """
    Get reference to existing agent from environment variables.
    Returns a dict with agent name and version for API calls.
    """
    agent_name = os.environ["EXISTING_AGENT_NAME"]
    agent_version = os.environ.get("EXISTING_AGENT_VERSION")  # Optional, uses latest if not specified
    
    agent_ref = {
        "name": agent_name,
        "type": "agent_reference"
    }
    
    if agent_version:
        agent_ref["version"] = agent_version
    
    print(f"Using existing agent: {agent_name}" + (f" (version {agent_version})" if agent_version else " (latest version)"))
    return agent_ref


def initialize_project_client():
    """Initialize Azure AI Project Client"""
    endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
    
    project_client = AIProjectClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
    )
    return project_client


def send_message_to_existing_agent(project_client, agent_ref, message_content):
    """Send a message to existing agent and get response (v2.0 API)"""
    # Get OpenAI client from project client
    openai_client = project_client.get_openai_client()
    
    # Send message and get response using existing agent reference
    response = openai_client.responses.create(
        input=[{"role": "user", "content": message_content}],
        extra_body={"agent_reference": agent_ref},
    )
    
    return response


def create_agent_callback(project_client, agent_ref):
    """
    Create a callback function for the RedTeam agent to test.
    
    The RedTeam.scan() method expects a simple callback that takes a string query
    and returns a string response. This function wraps our existing agent
    communication logic to match that interface.
    
    Returns:
        A function that takes a query string and returns a response string
    """
    def agent_callback(query: str) -> str:
        """Simple callback that sends query to agent and returns response"""
        try:
            response = send_message_to_existing_agent(project_client, agent_ref, query)
            # Extract text response
            response_text = response.output_text if hasattr(response, 'output_text') and response.output_text else "No response"
            return response_text
        except Exception as e:
            print(f"Error calling agent: {str(e)}")
            return f"Error: {str(e)}"
    
    return agent_callback


def setup_red_team_agent(azure_ai_project, credential, num_objectives=5):
    """
    Setup the RedTeam agent with configuration.
    
    Args:
        azure_ai_project: Azure AI project endpoint string
        credential: Azure credential object
        num_objectives: Number of attack objectives per risk category (default: 5)
    
    Returns:
        Configured RedTeam agent instance
    """
    print(f"\n{'=' * 60}")
    print("CONFIGURING RED TEAM AGENT")
    print(f"{'=' * 60}")
    
    # Configure risk categories to test
    risk_categories = [
        RiskCategory.Violence,
        RiskCategory.HateUnfairness,
        RiskCategory.Sexual,
        RiskCategory.SelfHarm
    ]
    
    print(f"\nRisk Categories: {len(risk_categories)}")
    for cat in risk_categories:
        print(f"  - {cat.value}")
    
    print(f"\nAttack Objectives per Category: {num_objectives}")
    print(f"Total baseline prompts to generate: {len(risk_categories) * num_objectives}")
    
    # Create RedTeam agent
    red_team_agent = RedTeam(
        azure_ai_project=azure_ai_project,
        credential=credential,
        risk_categories=risk_categories,
        num_objectives=num_objectives,
    )
    
    print("\n✓ RedTeam agent configured successfully")
    
    return red_team_agent


async def run_red_team_scan(red_team_agent, target_callback, scan_name="Agent-Red-Team-Scan"):
    """
    Run the red team scan with configured attack strategies.
    
    Args:
        red_team_agent: Configured RedTeam agent instance
        target_callback: Callback function that takes a query and returns a response
        scan_name: Name of the scan (PyRIT creates .scan_<name>_<timestamp> directory)
    
    Returns:
        Tuple of (scan_results, output_directory_path)
    
    Note: PyRIT automatically creates a directory structure:
        .scan_<scan_name>_<timestamp>/
            - scorecard.txt (summary)
            - final_results.json (complete results)
            - results.json (results data)
            - <category>_results.jsonl (per risk category)
    """
    print(f"\n{'=' * 60}")
    print("RUNNING RED TEAM SCAN")
    print(f"{'=' * 60}")
    
    # Configure attack strategies
    # Using grouped strategies for comprehensive coverage
    attack_strategies = [
        AttackStrategy.EASY,       # Base64, Flip, Morse
        AttackStrategy.MODERATE,   # Tense
        AttackStrategy.DIFFICULT,  # Composition of Tense + Base64
    ]
    
    # Alternative: Use specific strategies for more targeted testing
    # Uncomment to use specific strategies instead:
    # attack_strategies = [
    #     AttackStrategy.Base64,           # Easy: Encode in Base64
    #     AttackStrategy.ROT13,            # Easy: ROT13 cipher
    #     AttackStrategy.CharacterSpace,   # Easy: Add character spaces
    #     AttackStrategy.UnicodeConfusable, # Easy: Confusable Unicode chars
    #     AttackStrategy.Jailbreak,        # Easy: UPIA attacks
    #     AttackStrategy.IndirectAttack,   # Easy: XPIA attacks
    #     AttackStrategy.Tense,            # Moderate: Change to past tense
    #     AttackStrategy.Multiturn,        # Difficult: Multi-turn attacks
    #     AttackStrategy.Crescendo,        # Difficult: Gradually increasing complexity
    #     AttackStrategy.Compose([AttackStrategy.Base64, AttackStrategy.ROT13]), # Difficult: Composition
    # ]
    
    print(f"\nAttack Strategies: {len(attack_strategies)}")
    for strategy in attack_strategies:
        complexity = "Unknown"
        if strategy == AttackStrategy.EASY:
            complexity = "EASY (Base64, Flip, Morse)"
        elif strategy == AttackStrategy.MODERATE:
            complexity = "MODERATE (Tense)"
        elif strategy == AttackStrategy.DIFFICULT:
            complexity = "DIFFICULT (Tense + Base64)"
        print(f"  - {strategy.value if hasattr(strategy, 'value') else strategy}: {complexity}")
    
    print(f"\nScan name: {scan_name}")
    print("PyRIT will create directory: .scan_{scan_name}_<timestamp>/")
    print("\nStarting scan... (this may take several minutes)")
    print("The agent will be tested with adversarial prompts across all configured")
    print("risk categories and attack strategies.")
    
    # Run the scan (PyRIT creates its own directory structure)
    scan_result = await red_team_agent.scan(
        target=target_callback,
        scan_name=scan_name,
        attack_strategies=attack_strategies,
    )
    
    print("\n✓ Scan completed successfully!")
    
    # Find the created scan directory
    import glob
    scan_pattern = f".scan_{scan_name}_*"
    scan_dirs = sorted(glob.glob(scan_pattern), reverse=True)
    output_dir = scan_dirs[0] if scan_dirs else None
    
    if output_dir:
        print(f"✓ Results saved to: {output_dir}/")
    
    return scan_result, output_dir


def display_scan_results(output_dir):
    """
    Display the red team scan results from the .scan_* directory.
    
    Args:
        output_dir: Path to the .scan_* directory containing scan results
    """
    print(f"\n{'=' * 60}")
    print("RED TEAM SCAN RESULTS")
    print(f"{'=' * 60}")
    
    if not output_dir or not os.path.exists(output_dir):
        print(f"\nERROR: Output directory not found: {output_dir}")
        print("\nLooking for .scan_* directories...")
        import glob
        scan_dirs = glob.glob(".scan_*")
        if scan_dirs:
            print(f"\nFound scan directories:")
            for dir_path in scan_dirs:
                print(f"  - {dir_path}/")
            output_dir = sorted(scan_dirs, reverse=True)[0]
            print(f"\nUsing most recent: {output_dir}/")
        else:
            print("No scan directories found.")
            return
    
    # Display scorecard.txt if available (contains summary)
    scorecard_path = os.path.join(output_dir, "scorecard.txt")
    if os.path.exists(scorecard_path):
        print(f"\n{'=' * 40}")
        print("SCORECARD SUMMARY")
        print(f"{'=' * 40}")
        with open(scorecard_path, 'r', encoding='utf-8') as f:
            print(f.read())
    
    # Try to load final_results.json for detailed attack counts
    final_results_path = os.path.join(output_dir, "final_results.json")
    results = None
    if os.path.exists(final_results_path):
        try:
            with open(final_results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except:
            pass
    
    # Fall back to results.json if final_results not available
    if not results:
        results_path = os.path.join(output_dir, "results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            except:
                pass
    
    # Display count of attack-response pairs if available
    if results and "redteaming_data" in results:
        data_count = len(results["redteaming_data"])
        successful_attacks = sum(1 for item in results["redteaming_data"] if item.get("attack_success", False))
        print(f"\n{'=' * 40}")
        print(f"ATTACK SUMMARY")
        print(f"{'=' * 40}")
        print(f"Total attacks tested: {data_count}")
        print(f"Successful attacks: {successful_attacks}")
        print(f"Overall Attack Success Rate: {(successful_attacks / data_count * 100) if data_count > 0 else 0:.2f}%")
    
    print(f"\n{'=' * 40}")
    print(f"RESULTS LOCATION")
    print(f"{'=' * 40}")
    print(f"Directory: {output_dir}/")
    print(f"\nKey files:")
    print(f"  - scorecard.txt: Summary of ASR by category and complexity")
    print(f"  - final_results.json: Complete results with all data")
    print(f"  - results.json: Results data")
    print(f"  - *_results.jsonl: Per-category detailed attack/response pairs")
    print(f"\n✓ View detailed results in Azure AI Foundry portal")
    print(f"  under Evaluations > Red Teaming or Safety sections")


async def main():
    """Main execution function"""
    print("=" * 60)
    print("AZURE AI RED TEAMING AGENT SCAN")
    print("=" * 60)
    
    # Verify environment variables
    required_env_vars = [
        "AZURE_AI_PROJECT_ENDPOINT",
        "EXISTING_AGENT_NAME",
    ]
    
    missing_vars = [var for var in required_env_vars if var not in os.environ]
    if missing_vars:
        print(f"\nERROR: Missing required environment variables: {', '.join(missing_vars)}")
        print("\nRequired environment variables:")
        print("  - AZURE_AI_PROJECT_ENDPOINT: Your Azure AI Foundry project endpoint")
        print("  - EXISTING_AGENT_NAME: Name of the agent to test")
        print("\nOptional environment variables:")
        print("  - EXISTING_AGENT_VERSION: Version of the agent (uses latest if not specified)")
        print("  - RED_TEAM_NUM_OBJECTIVES: Number of objectives per risk category (default: 5)")
        return
    
    # Get configuration
    num_objectives = int(os.environ.get("RED_TEAM_NUM_OBJECTIVES", "5"))
    scan_name = "Agent-Red-Team-Scan"
    
    try:
        # Initialize project client
        print("\nInitializing Azure AI Project client...")
        project_client = initialize_project_client()
        print("✓ Project client initialized")
        
        # Get agent reference
        print("\nGetting agent reference...")
        agent_ref = get_agent_reference()
        
        # Create callback for red team testing
        print("\nCreating agent callback for red team testing...")
        agent_callback = create_agent_callback(project_client, agent_ref)
        print("✓ Agent callback created")
        
        # Setup RedTeam agent
        azure_ai_project = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
        credential = DefaultAzureCredential()
        red_team_agent = setup_red_team_agent(azure_ai_project, credential, num_objectives)
        
        # Run the scan
        scan_result, output_dir = await run_red_team_scan(red_team_agent, agent_callback, scan_name)
        
        # Display results
        display_scan_results(output_dir)
        
        print(f"\n{'=' * 60}")
        print("RED TEAM SCAN COMPLETE")
        print(f"{'=' * 60}")
        print("\nNext Steps:")
        print(f"1. Review the results in: {output_dir}/")
        print("2. Check Azure AI Foundry portal for detailed visualizations")
        print("3. Analyze high ASR categories and implement mitigations")
        print("4. Re-run scan after implementing safety improvements")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
