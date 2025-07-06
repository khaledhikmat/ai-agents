from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.agent import Agent

from helpers.providers import get_llm_model

from service.config.envvars import EnvVarsConfigService
from service.graph.typex import IGraphService
from service.graph.neo4j import Neo4jGraphService

from agent.typex import AgentParameters

# agent dependecies
@dataclass
class InhAgentDeps:
    """Dependencies for the INH agent."""
    graphsvc: IGraphService

# global variable to hold the agent instance
inh_agent = Agent(
    get_llm_model(), # get the LLM model based on environment variables
    deps_type=InhAgentDeps,
    system_prompt="You are a helpful assistant that answers questions about system documentation based on the provided documentation. "
                    "Use the retrieve tool to get relevant information from the URL documentation before answering. "
                    "If the documentation doesn't contain the answer, clearly state that the information isn't available "
                    "in the current documentation and provide your best general knowledge response."
    )

# Called from the main app to initialize the agent parameters
def initialize_agent_params() -> AgentParameters:
    """Get the agent parameters."""
    # Initialize services
    cfg_svc = EnvVarsConfigService()
    graph_svc = Neo4jGraphService(cfg_svc)
    deps = InhAgentDeps(graphsvc=graph_svc)
    return AgentParameters(
        title="Inheritance Agent",
        description="An agent that answers questions about inheritance knowledge base.",
        deps=deps,
        agent=inh_agent
    )

# Called from the main app to finalize the agent parameters
async def finalize_agent_params(parameters: AgentParameters) -> None:
    """Finalize the agent dependencies."""
    await parameters.deps.graphsvc.finalize()

@inh_agent.tool
async def retrieve(context: RunContext[InhAgentDeps], search_query: str) -> str:
    """Retrieve relevant documents from LightRAG based on a search query.
    
    Args:
        context: The run context containing dependencies.
        search_query: The search query to find relevant documents.
        
    Returns:
        Formatted context information from the retrieved documents.
    """
    return await context.deps.ragsvc.retrieve(search_query)
