# disclaimer: research agent code is available [here](https://machinelearningmastery.com/building-first-multi-agent-system-beginner-guide/)

from crewai import Crew, Task, Agent, Process, LLM
from pydantic import BaseModel
from crewai.tools.structured_tool import CrewStructuredTool
from langchain_community.tools import BraveSearch

llm = LLM(
    model='openai/gpt-4o',
    api_key="YOUR-API-KEY"
)


# Define the BraveSearch input schema
class BraveSearchInput(BaseModel):
    query: str


def brave_search_wrapper(*args, **kwargs):
    if isinstance(kwargs, dict) and "query" in kwargs:
        query = kwargs["query"]
    elif len(args) > 0 and isinstance(args[0], BraveSearchInput):
        query = args[0].query
    else:
        raise ValueError("Invalid input provided to BraveSearchTool.")

    brave_search = BraveSearch.from_api_key(
        api_key="BRAVE-API-KEY",
        search_kwargs={"count": 3}
    )

    result = brave_search.run(query)
    return result


def create_brave_search_tool():
    return CrewStructuredTool.from_function(
        name="brave_search_tool",
        description=(
            "Searches the web using BraveSearch and returns relevant information for a given query. "
            "Useful for finding up-to-date and accurate information on a wide range of topics."
        ),
        args_schema=BraveSearchInput,  # Use the BraveSearch input schema
        func=brave_search_wrapper
    )


# Create the BraveSearch tool
SearchTool = create_brave_search_tool()

web_researcher_agent = Agent(
    role="Web Research Specialist",
    goal=(
        "To find the most recent, impactful, and relevant about {topic}. This includes identifying "
        "key use cases, challenges, and statistics to provide a foundation for deeper analysis."
    ),
    backstory=(
        "You are a former investigative journalist known for your ability to uncover technology breakthroughs "
        "and market insights. With years of experience, you excel at identifying actionable data and trends."
    ),
    tools=[SearchTool],
    llm=llm,
    verbose=True
)

trend_analyst_agent = Agent(
    role="Insight Synthesizer",
    goal=(
        "To analyze research findings, extract significant trends, and rank them by industry impact, growth potential, "
        "and uniqueness. Provide actionable insights for decision-makers."
    ),
    backstory=(
        "You are a seasoned strategy consultant who transitioned into {topic} analysis. With an eye for patterns, "
        "you specialize in translating raw data into clear, actionable insights."
    ),
    tools=[],
    llm=llm,
    verbose=True
)

report_writer_agent = Agent(
    role="Narrative Architect",
    goal=(
        "To craft a detailed, professional report that communicates research findings and analysis effectively. "
        "Focus on clarity, logical flow, and engagement."
    ),
    backstory=(
        "Once a technical writer for a renowned journal, you are now dedicated to creating industry-leading reports. "
        "You blend storytelling with data to ensure your work is both informative and captivating."
    ),
    tools=[],
    llm=llm,
    verbose=True
)

proofreader_agent = Agent(
    role="Polisher of Excellence",
    goal=(
        "To refine the report for grammatical accuracy, readability, and formatting, ensuring it meets professional "
        "publication standards."
    ),
    backstory=(
        "An award-winning editor turned proofreader, you specialize in perfecting written content. Your sharp eye for "
        "detail ensures every document is flawless."
    ),
    tools=[],
    llm=llm,
    verbose=True
)

manager_agent = Agent(
    role="Workflow Maestro",
    goal=(
        "To coordinate agents, manage task dependencies, and ensure all outputs meet quality standards. Your focus "
        "is on delivering a cohesive final product through efficient task management."
    ),
    backstory=(
        "A former project manager with a passion for efficient teamwork, you ensure every process runs smoothly, "
        "overseeing tasks and verifying results."
    ),
    tools=[],
    llm=llm,
    verbose=True
)

# Define tasks
web_research_task = Task(
    description=(
        "Conduct web-based research to identify 5-7 of the {topic}. Focus on key use cases. "
    ),
    expected_output=(
        "A structured list of 5-7 {topic}."
    )
)
trend_analysis_task = Task(
    description=(
        "Analyze the research findings to rank {topic}. "
    ),
    expected_output=(
        "A table ranking trends by impact, with concise descriptions of each trend."
    )
)

report_writing_task = Task(
    description=(
        "Draft report summarizing the findings and analysis of {topic}. Include sections for "
        "Introduction, Trends Overview, Analysis, and Recommendations."
    ),
    expected_output=(
        "A structured, professional draft with a clear flow of information. Ensure logical organization and consistent tone."
    )
)

proofreading_task = Task(
    description=(
        "Refine the draft for grammatical accuracy, coherence, and formatting. Ensure the final document is polished "
        "and ready for publication."
    ),
    expected_output=(
        "A professional, polished report free of grammatical errors and inconsistencies. Format the document for "
        "easy readability."
    )
)

crew = Crew(
    agents=[web_researcher_agent, trend_analyst_agent, report_writer_agent, proofreader_agent],
    tasks=[web_research_task, trend_analysis_task, report_writing_task, proofreading_task],
    process=Process.hierarchical,
    manager_agent=manager_agent,
    verbose=True
)

crew_output = crew.kickoff(inputs={"topic": "AI Trends"})