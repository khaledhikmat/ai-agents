from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.agent import Agent

from helpers.providers import get_llm_model

from service.config.envvars import EnvVarsConfigService
from service.graph.typex import IGraphService
from service.graph.neo4j import Neo4jGraphService

from agent.typex import AgentParameters
from .prompts import SYSTEM_PROMPT

# agent dependecies
@dataclass
class InhAgentDeps:
    """Dependencies for the INH agent."""
    graphsvc: IGraphService

# global variable to hold the agent instance
inh_agent = Agent(
    get_llm_model(), # get the LLM model based on environment variables
    deps_type=InhAgentDeps,
    system_prompt=SYSTEM_PROMPT
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
async def retrieve_persons(context: RunContext[InhAgentDeps]) -> List[Dict[str, Any]]:
    """Use this tool to retrieve max of 100 person nodes.

    Args:
        context: The run context containing dependencies.
        
    Returns:
        Formatted context information from the retrieved nodes.
    """
    # Query to get a graph result
    print("Retrieving persons...")
    query = """
        MATCH (p:Person) 
        RETURN p.name, p.profession, p.gender, p.education, p.photo, p.birth_certificate, p.death_certificate, p.inh_confinment, p.residence_country, p.residence_city, p.birth_country, p.birth_city, p.birth_day, p.birth_month, p.birth_year, p.death_city, p.death_country, p.death_day, p.death_month, p.death_year
        LIMIT 100;    
    """

    params = {
    }

    records = []

    try:
        records = await context.deps.graphsvc.query(
            query,
            params
        )
    except Exception as e:
        print(f"Error processing retrieve_persons query: {e}")

    # Convert results to dict for agent
    return [
        {
            "name": r["p.name"],
            "profession": r["p.profession"],
            "gender": r["p.gender"],
            "education": r["p.education"],
            "photo": r["p.photo"],
            "birth_certificate": r["p.birth_certificate"],
            "death_certificate": r["p.death_certificate"],
            "inheritance_confinement": r["p.inh_confinment"],
            "residence_country": r["p.residence_country"],
            "residence_city": r["p.residence_city"],
            "birth_country": r["p.birth_country"],
            "birth_city": r["p.birth_city"],
            "death_city": r["p.death_city"],
            "death_country": r["p.death_country"],
            "birth_day": r["p.birth_day"],
            "birth_month": r["p.birth_month"],
            "birth_year": r["p.birth_year"],
            "death_day": r["p.death_day"],
            "death_month": r["p.death_month"],
            "death_year": r["p.death_year"]
        }
        for r in records
    ]

@inh_agent.tool
async def retrieve_properties(context: RunContext[InhAgentDeps]) -> List[Dict[str, Any]]:
    """Use this tool to retrieve max of 100 property nodes.

    Args:
        context: The run context containing dependencies.
        
    Returns:
        Formatted context information from the retrieved nodes.
    """
    # Query to get a graph result
    print("Retrieving properties...")
    query = """
        MATCH (p:Property) 
        RETURN p.name, p.location, p.owner, p.area, p.area_unit, p.country, p.city, p.unsold, p.possessed, p.description, p.shares, p.lot, p.effects, p.organized 
        LIMIT 100;    
    """

    params = {
    }

    records = []

    try:
        records = await context.deps.graphsvc.query(
            query,
            params
        )
    except Exception as e:
        print(f"Error processing retrieve_properties query: {e}")

    # Convert results to dict for agent
    return [
        {
            "name": r["p.name"],
            "location": r["p.location"],
            "owner": r["p.owner"],
            "area": r["p.area"],
            "area_unit": r["p.area_unit"],
            "country": r["p.country"],
            "city": r["p.city"],
            "unsold": r["p.unsold"],
            "possessed": r["p.possessed"],
            "description": r["p.description"],
            "shares": r["p.shares"],
            "lot": r["p.lot"],
            "effects": r["p.effects"],
            "organized": r["p.organized"]
        }
        for r in records
    ]

@inh_agent.tool
async def retrieve_countries(context: RunContext[InhAgentDeps]) -> List[Dict[str, Any]]:
    """Use this tool to retrieve max of 100 country nodes.

    Args:
        context: The run context containing dependencies.
        
    Returns:
        Formatted context information from the retrieved nodes.
    """
    # Query to get a graph result
    print("Retrieving countries...")
    query = """
        MATCH (p:Country) 
        RETURN p.name 
        LIMIT 100;    
    """

    params = {
    }

    records = []

    try:
        records = await context.deps.graphsvc.query(
            query,
            params
        )
    except Exception as e:
        print(f"Error processing retrieve_countries query: {e}")

    # Convert results to dict for agent
    return [
        {
            "name": r["p.name"],
        }
        for r in records
    ]

@inh_agent.tool
async def retrieve_cities(context: RunContext[InhAgentDeps]) -> List[Dict[str, Any]]:
    """Use this tool to retrieve max of 100 city nodes.

    Args:
        context: The run context containing dependencies.

    Returns:
        Formatted context information from the retrieved nodes.
    """
    # Query to get a graph result
    print("Retrieving cities...")
    query = """
        MATCH (p:City) 
        RETURN p.name 
        LIMIT 100;    
    """

    params = {
    }

    records = []

    try:
        records = await context.deps.graphsvc.query(
            query,
            params
        )
    except Exception as e:
        print(f"Error processing retrieve_cities query: {e}")

    # Convert results to dict for agent
    return [
        {
            "name": r["p.name"],
        }
        for r in records
    ]

@inh_agent.tool
async def retrieve_properties_in_country(context: RunContext[InhAgentDeps], country: str) -> List[Dict[str, Any]]:
    """Use this tool to retrieve max of 100 property nodes in a specific country.

    Args:
        context: The run context containing dependencies.
        country: The country to filter properties by.

    Returns:
        Formatted context information from the retrieved nodes.
    """
    # Query to get a graph result
    print(f"Retrieving properties in country {country}...")
    query = """
        MATCH (p:Property) 
        WHERE p.country = $country
        RETURN p.name, p.location, p.owner, p.area, p.area_unit, p.country, p.city, p.unsold, p.possessed, p.description, p.shares, p.lot, p.effects, p.organized 
        LIMIT 100;    
    """

    params = {
        "country": country
    }

    records = []

    try:
        records = await context.deps.graphsvc.query(
            query,
            params
        )
    except Exception as e:
        print(f"Error processing retrieve_properties_in_country query: {e}")

    for r in records:
        print(f"Retrieved country {r}")

    # Convert results to dict for agent
    return [
        {
            "name": r["p.name"],
            "location": r["p.location"],
            "owner": r["p.owner"],
            "area": r["p.area"],
            "area_unit": r["p.area_unit"],
            "country": r["p.country"],
            "city": r["p.city"],
            "unsold": r["p.unsold"],
            "possessed": r["p.possessed"],
            "description": r["p.description"],
            "shares": r["p.shares"],
            "lot": r["p.lot"],
            "effects": r["p.effects"],
            "organized": r["p.organized"]
        }
        for r in records
    ]

@inh_agent.tool
async def retrieve_properties_in_city(context: RunContext[InhAgentDeps], city: str) -> List[Dict[str, Any]]:
    """Use this tool to retrieve max of 100 property nodes in a specific city.

    Args:
        context: The run context containing dependencies.
        city: The city to filter properties by.

    Returns:
        Formatted context information from the retrieved nodes.
    """
    # Query to get a graph result
    print(f"Retrieving properties in city {city}...")
    query = """
        MATCH (p:Property) 
        WHERE p.city = $city
        RETURN p.name, p.location, p.owner, p.area, p.area_unit, p.country, p.city, p.unsold, p.possessed, p.description, p.shares, p.lot, p.effects, p.organized 
        LIMIT 100;    
    """

    params = {
        "city": city
    }

    records = []

    try:
        records = await context.deps.graphsvc.query(
            query,
            params
        )
    except Exception as e:
        print(f"Error processing retrieve_properties_in_city query: {e}")

    # Convert results to dict for agent
    return [
        {
            "name": r["p.name"],
            "location": r["p.location"],
            "owner": r["p.owner"],
            "area": r["p.area"],
            "area_unit": r["p.area_unit"],
            "country": r["p.country"],
            "city": r["p.city"],
            "unsold": r["p.unsold"],
            "possessed": r["p.possessed"],
            "description": r["p.description"],
            "shares": r["p.shares"],
            "lot": r["p.lot"],
            "effects": r["p.effects"],
            "organized": r["p.organized"]
        }
        for r in records
    ]

@inh_agent.tool
async def retrieve_by_person(context: RunContext[InhAgentDeps], person: str) -> List[Dict[str, Any]]:
    """Use this tool to retrieve all relationships in the knowledge graph about a specific person node.

    Args:
        context: The run context containing dependencies.
        person: The name of the person to retrieve relationships for.

    Returns:
        Formatted context information from the retrieved relationships.
    """
    # Query to get a graph result
    print(f"Retrieving person {person} relationships...")
    query = """
        MATCH (p:Person {name: $person})-[r]-()
        RETURN p.name, type(r) AS relationship, r
        LIMIT 100;    
    """

    params = {
        "person": person
    }

    records = []

    try:
        records = await context.deps.graphsvc.query(
            query,
            params
        )
    except Exception as e:
        print(f"Error processing retrieve_by_person query: {e}")

    # Convert results to dict for agent
    return [
        {
            "name": r["p.name"],
            "relationship": r["relationship"],
            "details": dict(r)
        }
        for r in records
    ]

@inh_agent.tool
async def retrieve_person_children(context: RunContext[InhAgentDeps], person: str) -> List[Dict[str, Any]]:
    """Use this tool to retrieve all children of a specific person node.

    Args:
        context: The run context containing dependencies.
        person: The name of the person to retrieve children for.

    Returns:
        Formatted context information from the retrieved children relationships.
    """
    # Query to get a graph result
    print(f"Retrieving children of person {person}...")
    query = """
        MATCH (p:Person {name: $person})-[:PARENT_OF]->(c:Person)
        RETURN c.name AS child_name, p.name AS parent_name
        LIMIT 100;    
    """

    params = {
        "person": person
    }

    records = []

    try:
        records = await context.deps.graphsvc.query(
            query,
            params
        )
    except Exception as e:
        print(f"Error processing retrieve_person_children query: {e}")

    # Convert results to dict for agent
    return [
        {
            "child_name": r["child_name"],
            "parent_name": r["parent_name"]
        }
        for r in records
    ]

@inh_agent.tool
async def retrieve_person_grand_children(context: RunContext[InhAgentDeps], person: str) -> List[Dict[str, Any]]:
    """Use this tool to retrieve all grandchildren of a specific person node.

    Args:
        context: The run context containing dependencies.
        person: The name of the person to retrieve grandchildren for.

    Returns:
        Formatted context information from the retrieved grandchildren relationships.
    """
    # Query to get a graph result
    print(f"Retrieving grandchildren of person {person}...")
    query = """
        MATCH (p:Person {name: $person})-[:PARENT_OF]->(:Person)-[:PARENT_OF]->(gc:Person)
        RETURN gc.name AS grandchild_name, p.name AS grandparent_name
        LIMIT 100;    
    """

    params = {
        "person": person
    }

    records = []

    try:
        records = await context.deps.graphsvc.query(
            query,
            params
        )
    except Exception as e:
        print(f"Error processing retrieve_person_grand_children query: {e}")

    # Convert results to dict for agent
    return [
        {
            "grandchild_name": r["grandchild_name"],
            "grandparent_name": r["grandparent_name"]
        }
        for r in records
    ]

@inh_agent.tool
async def retrieve_person_inheritors(context: RunContext[InhAgentDeps], person: str) -> List[Dict[str, Any]]:
    """Use this tool to retrieve all inheritors which includes multi-generation childrenof a specific person node.

    Args:
        context: The run context containing dependencies.
        person: The name of the person to retrieve inheritors for.

    Returns:
        Formatted context information from the retrieved inheritors relationships.
    """
    # Query to get a graph result (this uses variable-length paths to find all descendants)
    print(f"Retrieving inheritors of person {person}...")
    query = """
        MATCH (p:Person {name: $person})-[:PARENT_OF*1..]->(descendant:Person)
        RETURN descendant.name AS inheritor_name, p.name AS root_ancestor
        LIMIT 100;
    """
    params = {
        "person": person
    }

    records = []

    try:
        records = await context.deps.graphsvc.query(
            query,
            params
        )
    except Exception as e:
        print(f"Error processing retrieve_person_inheritors query: {e}")

    # Convert results to dict for agent
    return [
        {
            "inheritor_name": r["inheritor_name"],
            "root_ancestor": r["root_ancestor"]
        }
        for r in records
    ]

@inh_agent.tool
async def retrieve_by_property(context: RunContext[InhAgentDeps], property: str) -> List[Dict[str, Any]]:
    """Use this tool to retrieve all relationships in the knowledge graph about a specific property node.

    Args:
        context: The run context containing dependencies.
        property: The name of the property to retrieve relationships for.

    Returns:
        Formatted context information from the retrieved relationships.
    """
    # Query to get a graph result
    print(f"Retrieving property {property} relationships...")
    query = """
        MATCH (p:Property {name: $property})-[r]-()
        RETURN p.name, type(r) AS relationship, r
        LIMIT 100;    
    """

    params = {
        "property": property
    }

    records = []

    try:
        records = await context.deps.graphsvc.query(
            query,
            params
        )
    except Exception as e:
        print(f"Error processing retrieve_by_property query: {e}")

    # Convert results to dict for agent
    return [
        {
            "name": r["p.name"],
            "relationship": r["relationship"],
            "details": dict(r)
        }
        for r in records
    ]

@inh_agent.tool
async def retrieve_by_property_n_country(context: RunContext[InhAgentDeps], property: str, country: str) -> List[Dict[str, Any]]:
    """Use this tool to retrieve all relationships in the knowledge graph about a specific property node in a specific country.

    Args:
        context: The run context containing dependencies.
        property: The name of the property to retrieve relationships for.
        country: The country to filter properties by.

    Returns:
        Formatted context information from the retrieved relationships.
    """
    # Query to get a graph result
    print(f"Retrieving property {property} and country {country} relationships...")
    query = """
        MATCH (p:Property {name: $property})-[r]-()
        WHERE p.country = $country
        RETURN p.name, type(r) AS relationship, r
        LIMIT 100;    
    """

    params = {
        "property": property,
        "country": country
    }

    records = []

    try:
        records = await context.deps.graphsvc.query(
            query,
            params
        )
    except Exception as e:
        print(f"Error processing retrieve_by_property_n_country query: {e}")

    # Convert results to dict for agent
    return [
        {
            "name": r["p.name"],
            "relationship": r["relationship"],
            "details": dict(r)
        }
        for r in records
    ]

@inh_agent.tool
async def retrieve_by_country(context: RunContext[InhAgentDeps], country: str) -> List[Dict[str, Any]]:
    """Use this tool to retrieve all relationships in the knowledge graph about a specific country node.

    Args:
        context: The run context containing dependencies.
        country: The name of the country to retrieve relationships for.

    Returns:
        Formatted context information from the retrieved relationships.
    """
    # Query to get a graph result
    print(f"Retrieving country {country} relationships...")
    query = """
        MATCH (c:Country {name: $country})-[r]-()
        RETURN c.name, type(r) AS relationship, r
        LIMIT 100;    
    """

    params = {
        "country": country
    }

    records = []

    try:
        records = await context.deps.graphsvc.query(
            query,
            params
        )
    except Exception as e:
        print(f"Error processing retrieve_by_country query: {e}")

    # Convert results to dict for agent
    return [
        {
            "name": r["c.name"],
            "relationship": r["relationship"],
            "details": dict(r)
        }
        for r in records
    ]

@inh_agent.tool
async def retrieve_by_city(context: RunContext[InhAgentDeps], city: str) -> List[Dict[str, Any]]:
    """Use this tool to retrieve all relationships in the knowledge graph about a specific city node.

    Args:
        context: The run context containing dependencies.
        city: The name of the city to retrieve relationships for.

    Returns:
        Formatted context information from the retrieved relationships.
    """
    # Query to get a graph result
    print(f"Retrieving city {city} relationships...")
    query = """
        MATCH (c:City {name: $city})-[r]-()
        RETURN c.name, type(r) AS relationship, r
        LIMIT 100;    
    """

    params = {
        "city": city
    }

    records = []

    try:
        records = await context.deps.graphsvc.query(
            query,
            params
        )
    except Exception as e:
        print(f"Error processing retrieve_by_city query: {e}")

    # Convert results to dict for agent
    return [
        {
            "name": r["c.name"],
            "relationship": r["relationship"],
            "details": dict(r)
        }
        for r in records
    ]
