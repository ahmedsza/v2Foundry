# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

import asyncio
from dotenv import load_dotenv
from agent_framework import Agent, Annotation
from agent_framework.azure import AzureAIProjectAgentProvider
from azure.identity.aio import AzureCliCredential
from agent_framework.observability import get_tracer, get_meter
from opentelemetry.trace import SpanKind
from opentelemetry.trace.span import format_trace_id

# Load environment variables from .env file
load_dotenv()

# Configure Azure Monitor telemetry with error handling
try:
    from azure.monitor.opentelemetry import configure_azure_monitor
    from agent_framework.observability import create_resource, enable_instrumentation
    
    # read connection string from env var 
    connection_string = os.getenv("AZURE_MONITOR_CONNECTION_STRING")
    if connection_string:
        configure_azure_monitor(
            connection_string=connection_string,
            resource=create_resource(),  # Uses OTEL_SERVICE_NAME, etc.
            enable_live_metrics=True,
        )
        
        # Then activate Agent Framework's telemetry code paths
        # Note: enable_sensitive_data=False to avoid serialization issues
        enable_instrumentation(enable_sensitive_data=False)
        
        # Patch JSON encoder to handle non-serializable objects
        import json
        _original_default = json.JSONEncoder.default
        
        def _custom_default(self, obj):
            try:
                return _original_default(self, obj)
            except TypeError:
                # Return string representation for non-serializable objects
                return f"<{obj.__class__.__name__}>"
        
        json.JSONEncoder.default = _custom_default
        print("✓ Azure Monitor telemetry enabled with custom serialization\n")
    else:
        print("⚠ AZURE_MONITOR_CONNECTION_STRING not set, telemetry disabled\n")
except Exception as e:
    print(f"⚠ Failed to configure telemetry: {e}\n")
    print("Continuing without telemetry...\n")

"""
This sample demonstrates how to create an Azure AI agent that uses Bing Grounding
search to find real-time information from the web with comprehensive citation support.
It shows how to extract and display citations (title, URL, and snippet) from Bing
Grounding responses, enabling users to verify sources and explore referenced content.

Prerequisites:
1. A connected Grounding with Bing Search resource in your Azure AI project
2. Set BING_CONNECTION_ID environment variable
   Example: BING_CONNECTION_ID="your-bing-connection-id"

To set up Bing Grounding:
1. Go to Azure AI Foundry portal (https://ai.azure.com)
2. Navigate to your project's "Connected resources" section
3. Add a new connection for "Grounding with Bing Search"
4. Copy the connection ID and set the BING_CONNECTION_ID environment variable
"""

async def run_agent_with_citations(agent: Agent, user_input: str) -> list[Annotation]:
    """Run agent with user input and return citations.
    
    Args:
        agent: The agent instance to run
        user_input: The user's question or input
        
    Returns:
        List of annotations/citations from the response
    """
    citations: list[Annotation] = []
    
    # Try to use telemetry span if available
    try:
        with get_tracer().start_as_current_span("Scenario: Agent Chat Foundry v2", kind=SpanKind.CLIENT) as current_span:
            print(f"Trace ID: {format_trace_id(current_span.get_span_context().trace_id)}")
            
            try:
                # Stream the response and collect citations
                async for chunk in agent.run(user_input, stream=True):
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                    
                    # Collect citations from Bing Grounding responses
                    for content in getattr(chunk, "contents", []):
                        annotations = getattr(content, "annotations", [])
                        if annotations:
                            citations.extend(annotations)
            
            except Exception as e:
                print(f"\nError during agent execution: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as telemetry_error:
        # If telemetry fails, run without it
        print(f"⚠ Telemetry error: {telemetry_error}")
        print("Running without telemetry tracing...\n")
        
        try:
            # Stream the agent response
            async for chunk in agent.run(user_input, stream=True):
                # Extract and print text content
                if hasattr(chunk, 'text') and chunk.text:
                    print(chunk.text, end="", flush=True)
                
                # Collect citations from Bing Grounding responses
                for content in getattr(chunk, "contents", []):
                    annotations = getattr(content, "annotations", [])
                    if annotations:
                        citations.extend(annotations)
        
        except Exception as e:
            print(f"\nError during agent execution: {e}")
            import traceback
            traceback.print_exc()

    print()

    # Display collected citations
    if citations:
        print("\n\nCitations:")
        for i, citation in enumerate(citations, 1):
            print(f"[{i}] {citation['title']}: {citation.get('url')}")
    else:
        print("\nNo citations found in the response.")
    
    return citations

async def main() -> None:
    """Main function demonstrating Azure AI agent with Bing Grounding search."""
    # Get agent configuration from environment
    agent_name = os.getenv("FOUNDRY_AGENT_ID", "BingAgent")
    agent_version = os.getenv("FOUNDRY_AGENT_VERSION")  # Optional: specific version
    
    print(f"=== Azure AI Agent with Bing Grounding Search ===\n")
    print(f"Agent: {agent_name}")
    if agent_version:
        print(f"Version: {agent_version}")
    print()

    # Use AzureAIProjectAgentProvider to work with existing agents
    async with (
        AzureCliCredential() as credential,
        AzureAIProjectAgentProvider(credential=credential) as provider,
    ):
        # Get existing agent - two approaches:
        
        if agent_version:
            # Option 1: Get specific version using reference
            agent = await provider.get_agent(
                reference={"name": agent_name, "version": agent_version}
            )
        else:
            # Option 2: Get latest version using name only
            agent = await provider.get_agent(name=agent_name)
        
        # Run queries
        user_input = "How do i get a passport"
        print(f"User: {user_input}")
        citations = await run_agent_with_citations(agent, user_input)
        
        user_input = "How do i get a new drivers license"
        print(f"User: {user_input}")
        citations = await run_agent_with_citations(agent, user_input)
        print()


if __name__ == "__main__":
    asyncio.run(main())