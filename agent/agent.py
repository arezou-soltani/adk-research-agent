"""
author: Arezou Soltani
"""

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext
from google.genai import types
from langchain_community.tools import DuckDuckGoSearchRun
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential
from dotenv import load_dotenv
import os
import requests
import logging
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)
logger.info(f"Loading .env from: {dotenv_path}")
logger.info(f"GitHub token loaded: {bool(os.getenv('GITHUB_API_TOKEN'))}")
HF_OPENAPI_SPEC_URL = "https://api.endpoints.huggingface.cloud/openapi.json"


# ===============================
# SEARCH TOOL
# ===============================

def web_search_tool(query: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Searches the web and returns relevant information for a given query using DuckDuckGo Search.
    
    Args:
        query (str): The search query to execute
        tool_context (ToolContext): Automatically provided by ADK
        
    Returns:
        dict: Search results with summary and related topics
    """
    try:
        # Initialize DuckDuckGo Search (no API key required)
        ddg_search = DuckDuckGoSearchRun()
        
        # Perform search
        search_results = ddg_search.run(query)
        
        # Store search results in state for other tools to access
        tool_context.state["last_search_query"] = query
        tool_context.state["last_search_results"] = search_results
        
        return {
            "status": "success",
            "query": query,
            "results": search_results
        }
        
    except Exception as e:
        logger.warning(f"DuckDuckGo Search failed: {e}, using fallback")
        fallback_results = f"Search results for '{query}': Industry trends, recent developments, key statistics, and market insights related to the topic."
        
        tool_context.state["last_search_query"] = query
        tool_context.state["last_search_results"] = fallback_results
        
        return {
            "status": "error",
            "query": query,
            "error": str(e),
            "fallback_results": fallback_results
        }

# ===============================
# ANALYSIS TOOL
# ===============================
def trend_analysis_tool(research_data: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Analyzes research findings, extracts significant trends, and ranks them by industry impact.
    
    Args:
        research_data (str): Research findings to analyze
        tool_context (ToolContext): Automatically provided by ADK
        
    Returns:
        dict: Trend analysis with rankings and insights
    """
    logger.info("🔍 Trend Analysis Tool: Processing research findings...")
    
    # Mock analysis result (would be LLM-generated in production)
    analysis_result = """
    TREND ANALYSIS REPORT
    =====================
    
    Rank | Trend | Impact | Growth Potential | Uniqueness
    -----|-------|--------|------------------|------------
    1    | Generative AI Enterprise Adoption | High | Exponential | High automation potential
    2    | Multimodal AI Integration | High | Rapid | Convergence of modalities  
    3    | Edge AI Deployment | Medium | Steady | Real-time processing
    4    | AI Governance Frameworks | Medium | Growing | Regulatory compliance
    5    | Autonomous Agent Systems | Emerging | Promising | Self-directing capabilities
    
    KEY INSIGHTS:
    - Enterprise adoption driving immediate ROI
    - Multimodal systems becoming standard
    - Edge computing enabling new use cases
    - Governance becoming critical for deployment
    - Autonomous systems represent future potential
    """
    
    # Store analysis in state
    tool_context.state["trend_analysis"] = analysis_result
    
    return {
        "status": "success",
        "analysis": analysis_result,
        "trends_identified": 5
    }

# ===============================
# REPORT WRITING TOOL
# ===============================
def report_writing_tool(analysis_data: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Crafts detailed, professional reports that communicate research findings and analysis effectively.
    
    Args:
        analysis_data (str): Analysis data to convert into a report
        tool_context (ToolContext): Automatically provided by ADK
        
    Returns:
        dict: Professional report draft
    """
    logger.info("📝 Report Writing Tool: Crafting professional report...")
    
    # Simulated report writing (in real implementation, would use LLM)
    report = f"""
# Professional Industry Analysis Report

## Executive Summary
This report presents a comprehensive analysis of current industry trends based on extensive research and expert analysis.

## Introduction
The rapidly evolving technological landscape requires continuous monitoring and analysis to identify emerging opportunities and challenges.

## Research Methodology
Our analysis employed web-based research combined with trend analysis to identify and rank the most significant developments.

## Findings and Analysis
{analysis_data}

## Strategic Recommendations
Based on our analysis, we recommend:

1. **Immediate Action**: Focus on generative AI adoption for quick wins
2. **Medium-term Planning**: Invest in multimodal AI capabilities  
3. **Long-term Strategy**: Prepare for autonomous agent integration
4. **Risk Management**: Establish AI governance frameworks

## Market Implications
The identified trends suggest a transformation toward more intelligent, automated, and integrated AI systems across industries.

## Conclusion
Organizations that proactively adapt to these trends while maintaining proper governance will be best positioned for future success.

---
*Report prepared by automated analysis system*
"""
    
    # Store draft report in state
    tool_context.state["draft_report"] = report
    
    return {
        "status": "success",
        "report": report,
        "word_count": len(report.split())
    }

# ===============================
# PROOFREADING TOOL
# ===============================

def proofreading_tool(draft_report: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Refines reports for grammatical accuracy, readability, and formatting.
    
    Args:
        draft_report (str): Draft report to proofread and polish
        tool_context (ToolContext): Automatically provided by ADK
        
    Returns:
        dict: Polished final report
    """
    logger.info("✏️ Proofreading Tool: Polishing report for publication...")
    
    # Simulated proofreading (in real implementation, would use grammar tools/LLM)
    polished_report = draft_report.replace("we recommend:", "we strongly recommend:")
    polished_report = polished_report.replace("focus on", "prioritize")
    polished_report += "\n\n📋 **Quality Assurance**: This report has been reviewed for accuracy, clarity, and professional formatting."
    polished_report += "\n\n✅ **Publication Ready**: Document meets professional publication standards."
    
    # Store final report in state
    tool_context.state["final_report"] = polished_report
    
    return {
        "status": "success",
        "final_report": polished_report,
        "quality_score": 95
    }

# ===============================
# SENTIMENT CLASSIFIER TOOL
# ===============================

def sentiment_classifier_tool(text: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Fixed implementation with proper response structure handling
    """
    try:
        logger.info("Using direct HTTP method for sentiment classification")
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not hf_api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set")

        model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {
            "Authorization": f"Bearer {hf_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": text,
            "options": {
                "wait_for_model": True,
                "use_cache": False
            }
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            error_info = f"Status: {response.status_code}, "
            try:
                error_data = response.json()
                error_info += f"Error: {error_data.get('error', str(error_data))}"
            except json.JSONDecodeError:
                error_info += f"Response: {response.text[:200]}"
            raise RuntimeError(f"API error: {error_info}")

        try:
            result = response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON decode error: {e}. Response: {response.text[:200]}")

        if isinstance(result, list) and len(result) > 0:
            # The response is a list where the first element contains the sentiment results
            sentiments = result[0]
            if not all(isinstance(item, dict) for item in sentiments):
                raise RuntimeError("Unexpected response format: Not a list of dictionaries")
        else:
            raise RuntimeError("Unexpected response format")

        # Find the highest confidence sentiment
        best_sentiment = max(sentiments, key=lambda x: x.get('score', 0))

        # Store results in state
        tool_context.state["last_sentiment_text"] = text
        tool_context.state["last_sentiment_result"] = best_sentiment

        return {
            "status": "success",
            "text": text,
            "sentiment": {
                "label": best_sentiment.get('label', 'UNKNOWN'),
                "confidence": round(best_sentiment.get('score', 0), 4),
                "all_scores": sentiments
            },
            "model": model_id,
            "method": "Direct HTTP"
        }

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "text": text,
            "error": str(e),
            "note": "Check Hugging Face API status at https://status.huggingface.co/"
        }


# ===============================
# SEARCH GITHUB REPO
# ===============================

# GitHub official OpenAPI specification
GITHUB_OPENAPI_SPEC_URL = "https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json"

def __download_github_openapi_spec__() -> str:
    try:
        logger.info(f"Downloading GitHub OpenAPI spec from: {GITHUB_OPENAPI_SPEC_URL}")
        response = requests.get(GITHUB_OPENAPI_SPEC_URL, timeout=30)
        response.raise_for_status()

        spec_data = response.json()
        logger.info(
            f"Successfully downloaded GitHub OpenAPI spec (version: {spec_data.get('info', {}).get('version', 'unknown')})")

        return response.text

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download GitHub OpenAPI spec: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Downloaded spec is not valid JSON: {e}")


def github_repo_search_tool(
        tool_context: ToolContext,
        query: str,
        language: str = "",
        stars: str = "",
        sort: str = "best-match",
        order: str = "desc",
        per_page: int = 10
) -> Dict[str, Any]:
    """
    GitHub repository search with proper authentication and scopes

    Args:
        tool_context (ToolContext): ADK context
        query (str): Search keywords
        language (str): Filter by programming language
        stars (str): Filter by stars (e.g., ">100")
        sort (str): Sort field
        order (str): Sort order
        per_page (int): Results per page

    Returns:
        dict: Search results
    """
    try:
        logger.info("GitHub Repository Search Tool: Initializing...")

        github_token = os.getenv("GITHUB_API_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_API_TOKEN environment variable not set")

        # GitHub personal access tokens use apikey authentication
        auth_scheme, auth_credential = token_to_scheme_credential(
            token_type="apikey",
            location="header",
            name="Authorization",
            credential_value=f"token {github_token}"
        )
        logger.info(f"Using apikey auth scheme: {auth_scheme}")
        logger.info(f"Auth credential type: {type(auth_credential)}")

        try:
            github_spec = __download_github_openapi_spec__()
            logger.info("Using downloaded GitHub OpenAPI spec")
        except Exception as download_error:
            logger.warning(f"Failed to download spec, using local file: {download_error}")
            github_spec_path = os.path.join(os.path.dirname(__file__), "api.github.com.json")
            with open(github_spec_path, 'r') as f:
                github_spec = f.read()
            logger.info("Using local GitHub OpenAPI spec file")
        toolset = OpenAPIToolset(
            spec_str=github_spec,
            spec_str_type="json",
            auth_scheme=auth_scheme,
            auth_credential=auth_credential
        )

        # Search for GitHub repository search tool across all possible operation IDs
        possible_tool_names = [
            "search/repos",
            "search/repositories",
            "search.repos",
            "search_repos",
            "repos_search"
        ]

        search_tool = None
        for name in possible_tool_names:
            search_tool = toolset.get_tool(name)
            if search_tool:
                break

        if not search_tool:
            raise RuntimeError("GitHub search tool not found in OpenAPI spec")

        # Build query with advanced filters
        full_query = query
        if language:
            full_query += f" language:{language}"
        if stars:
            full_query += f" stars:{stars}"

        logger.info(f"Searching GitHub with query: {full_query}")

        # Prepare arguments - GitHub uses "q" for query
        args = {
            "q": full_query,
            "sort": sort,
            "order": order,
            "per_page": per_page
        }

        search_tool.configure_auth_credential(auth_credential)

        import asyncio
        import concurrent.futures

        async def run_search():
            return await search_tool.run_async(
                args=args,
                tool_context=tool_context
            )

        try:
            asyncio.get_running_loop()
            # If we're in an event loop, run in a thread pool
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_search())
                response = future.result()
        except RuntimeError:
            response = asyncio.run(run_search())

        search_data = response.json() if hasattr(response, 'json') else response

        logger.info(f"GitHub API Response type: {type(search_data)}")
        logger.info(
            f"GitHub API Response keys: {list(search_data.keys()) if isinstance(search_data, dict) else 'Not a dict'}")
        logger.info(f"GitHub API Response: {str(search_data)[:500]}...")

        # HANDLE GITHUB'S RESPONSE STRUCTURE
        if "items" not in search_data:
            if "error" in search_data:
                error_msg = search_data["error"]
                try:
                    import re
                    json_match = re.search(r'\{.*\}', error_msg)
                    if json_match:
                        github_error = json.loads(json_match.group())
                        if "message" in github_error:
                            github_msg = github_error["message"]
                            if "Bad credentials" in github_msg:
                                raise PermissionError(f"Invalid GitHub token: {github_msg}")
                            elif "API rate limit exceeded" in github_msg:
                                raise RuntimeError(f"Rate limit exceeded: {github_msg}")
                            elif "scope" in github_msg.lower():
                                raise PermissionError(f"Missing required scope: {github_msg}")
                            else:
                                raise RuntimeError(f"GitHub API error: {github_msg}")
                except (json.JSONDecodeError, AttributeError):
                    pass
                raise RuntimeError(f"OpenAPI tool error: {error_msg}")

            elif "message" in search_data:
                error = search_data["message"]
                if "Bad credentials" in error:
                    raise PermissionError("Invalid token: " + error)
                elif "API rate limit exceeded" in error:
                    raise RuntimeError("Rate limit exceeded: " + error)
                elif "scope" in error.lower():
                    # Check token scopes
                    raise PermissionError(f"Missing required scope: {error}")
                else:
                    raise RuntimeError("GitHub API error: " + error)
            else:
                raise RuntimeError(f"Unexpected GitHub response format. Response: {search_data}")

        total_results = search_data.get("total_count", 0)
        repos = search_data.get("items", [])

        # Extract repository info
        processed_repos = []
        for repo in repos:
            processed_repos.append({
                "name": repo.get("name"),
                "full_name": repo.get("full_name"),
                "owner": repo.get("owner", {}).get("login"),
                "description": repo.get("description"),
                "url": repo.get("html_url"),
                "stars": repo.get("stargazers_count"),
                "forks": repo.get("forks_count"),
                "language": repo.get("language"),
                "created_at": repo.get("created_at"),
                "updated_at": repo.get("updated_at"),
                "topics": repo.get("topics", [])
            })

        # Store in context
        tool_context.state["last_github_search"] = {
            "query": full_query,
            "results_count": total_results,
            "repos": processed_repos
        }

        return {
            "status": "success",
            "total_results": total_results,
            "returned_results": len(processed_repos),
            "repositories": processed_repos,
            "search_url": f"https://github.com/search?q={full_query.replace(' ', '+')}"
        }

    except Exception as e:
        logger.error(f"GitHub repository search failed: {str(e)}", exc_info=True)

        error_msg = str(e)
        if "scope" in error_msg.lower() or "public_repo" in error_msg:
            solution = (
                "Token missing 'public_repo' scope. "
                "Update token at: https://github.com/settings/tokens\n"
                "Your current scopes: gist, project, repo, user, workflow, write:packages"
            )
        else:
            solution = "Check GitHub API status and token permissions"

        return {
            "status": "error",
            "error": error_msg,
            "solution": solution
        }

# ===============================
# RESEARCH COORDINATOR
# ===============================

# Create the root agent for ADK discovery
root_agent = LlmAgent(
    name="research_coordinator",
    description=(
        "I'm your professional research assistant! I can help you research any topic by:\n"
        "• Searching the web for the latest information\n"
        "• Analyzing trends and extracting key insights\n"
        "• Writing comprehensive professional reports\n"
        "• Proofreading and polishing documents\n\n"
        "• Analyzing sentiment of text using Hugging Face AI models\n"
         "Searching the github repositories"   
        "Just tell me what you'd like me to research and I'll take care of the rest!"
    ),
    tools=[
        web_search_tool,
        trend_analysis_tool,
        report_writing_tool,
        proofreading_tool,
        sentiment_classifier_tool,
        github_repo_search_tool
    ],
    model=os.getenv("ADK_DEFAULT_MODEL", "gemini-2.0-flash"),
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=1024
    )
)