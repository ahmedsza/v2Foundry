# Before running the sample:
#    pip install --pre azure-ai-projects>=2.0.0b4

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
import os
import warnings
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

load_dotenv()
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
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
with tracer.start_as_current_span("agent_sdk_v2_trace"):
    openai_client = project_client.get_openai_client()

    # Reference the agent to get a streamed response
    annotations = []
    print("Response output: ", end="", flush=True)
    my_model = "gpt-4.1-mini"
    with openai_client.responses.stream(
        model=my_model,
        input=[{"role": "user", "content": "How can I get a new passport?"}],
        extra_body={"agent_reference": {"name": my_agent, "version": my_version, "type": "agent_reference"}},
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "response.completed":
                # Collect citations from the completed response
                for item in event.response.output:
                    if item.type == "message":
                        for block in item.content:
                            if block.type == "output_text":
                                for annotation in block.annotations:
                                    if annotation.type == "url_citation":
                                        annotations.append(annotation)
    print()

    # Print citations
    if annotations:
        print("\n--- Citations ---")
        for annotation in annotations:
            print(f"Title: {annotation.title}")
            print(f"URL:   {annotation.url}")
            print()



