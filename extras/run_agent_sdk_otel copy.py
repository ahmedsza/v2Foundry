# Before running the sample:
#    pip install --pre azure-ai-projects>=2.0.0b4

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
import os
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

load_dotenv()
resource = Resource.create({
    "service.name": os.getenv("OTEL_SERVICE_NAME", "openai-agents-app"),
})
provider = TracerProvider(resource=resource)

conn = os.getenv("AZURE_MONITOR_CONNECTION_STRING")
if conn:
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
    provider.add_span_processor(
        BatchSpanProcessor(AzureMonitorTraceExporter.from_connection_string(conn))
    )
else:
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
OpenAIAgentsInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
tracer = trace.get_tracer(__name__)
my_endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT")

project_client = AIProjectClient(
    endpoint=my_endpoint,
    credential=DefaultAzureCredential(),
)

my_agent = os.getenv("FOUNDRY_AGENT_ID", "BingAgent")
my_version = os.getenv("FOUNDRY_AGENT_VERSION", "2")
with tracer.start_as_current_span("agent_session[openai.agents]"):
    openai_client = project_client.get_openai_client()

    # Reference the agent to get a response
    response = openai_client.responses.create(
        input=[{"role": "user", "content": "How can I get a new passport?"}],
        extra_body={"agent_reference": {"name": my_agent, "version": my_version, "type": "agent_reference"}},
    )


    print(f"Response output: {response.output_text}")

    # Extract citations from annotations
    print("\n--- Citations ---")
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    for annotation in block.annotations:
                        if annotation.type == "url_citation":
                            print(f"Title: {annotation.title}")
                            print(f"URL:   {annotation.url}")
                            print()



