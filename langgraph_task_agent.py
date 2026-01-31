#!/usr/bin/env python3
"""
LangGraph Task Definition Agent - Defines and executes specific tasks using LangGraph.
Uses LangGraph to create a stateful agent that can define, plan, and execute tasks.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime

import numpy as np
import torch
import weaviate
from transformers import AutoModel, AutoTokenizer

try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    print(f"Warning: LangGraph packages not found. Error: {e}")
    print("Install with: pip install langgraph langchain langchain-community")

# Import Ollama API interface
from ollama_api import call_ollama_api


class AgentState(TypedDict):
    """State of the task definition agent."""
    messages: Annotated[list, add_messages]
    task_definition: str
    task_plan: List[str]
    current_step: int
    task_status: str
    results: List[Dict]
    context: Dict
    llm: Optional[object]  # Store LLM instance (deprecated, use llm_config)
    llm_config: Optional[Dict]  # LLM configuration (model, base_url)
    weaviate_config: Optional[Dict]  # Weaviate configuration
    retrieved_docs: List[Dict]  # Retrieved documents from Weaviate


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings with attention mask to get a sentence embedding."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def embed_query(query: str, model_path: str) -> np.ndarray:
    """Embed a query string using the local transformer model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        outputs = model(**inputs)
        pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    return pooled.cpu().numpy()[0]


def search_weaviate(
    query: str,
    collections: List[str],
    model_path: str,
    weaviate_url: str = "http://localhost:8080",
    top_k: int = 10,
    score_threshold: float = 0.6,
) -> List[Dict]:
    """
    Search Weaviate collections for relevant documents.
    
    Args:
        query: Search query
        collections: List of Weaviate collection names
        model_path: Path to embedding model
        weaviate_url: Weaviate server URL
        top_k: Number of top results per collection
        score_threshold: Minimum similarity score
        
    Returns:
        List of relevant documents with metadata
    """
    try:
        # Embed query
        query_vec = embed_query(query, model_path)
        all_results = []
        
        # Connect to Weaviate
        if weaviate_url == "http://localhost:8080":
            with weaviate.connect_to_local() as client:
                if not client.is_ready():
                    return []
                all_results = _search_collections(client, query_vec, collections, top_k, score_threshold)
        else:
            client = weaviate.Client(url=weaviate_url)
            try:
                if not client.is_ready():
                    return []
                all_results = _search_collections(client, query_vec, collections, top_k, score_threshold)
            finally:
                pass
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:top_k * len(collections)]
    except Exception as e:
        print(f"Warning: Weaviate search failed: {e}")
        return []


def _search_collections(client, query_vec, collections, top_k, score_threshold):
    """Helper function to search multiple collections."""
    all_results = []
    
    for collection_name in collections:
        try:
            if not client.collections.exists(collection_name):
                print(f"Warning: Collection '{collection_name}' does not exist")
                continue
            
            collection = client.collections.get(collection_name)
            
            # Perform vector similarity search
            result = collection.query.near_vector(
                near_vector=query_vec.tolist(),
                limit=top_k,
                return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
            )
            
            # Process results
            for obj in result.objects:
                props = obj.properties
                metadata = obj.metadata
                
                distance = metadata.distance if metadata else 1.0
                score = 1 - distance if distance else 0.0
                
                if score >= score_threshold:
                    all_results.append({
                        "collection": collection_name,
                        "chunk_id": props.get("chunk_id", ""),
                        "text": props.get("text", ""),
                        "source_pdf": props.get("source_pdf", "unknown"),
                        "country": props.get("country", "unknown"),
                        "page_start": props.get("page_start", 0),
                        "page_end": props.get("page_end", 0),
                        "token_count": props.get("token_count", 0),
                        "score": score,
                        "distance": distance,
                    })
        except Exception as e:
            print(f"Warning: Error searching collection '{collection_name}': {e}")
            continue
    
    return all_results


def generate_markdown_report(result: Dict) -> str:
    """Generate a comprehensive Markdown report from task execution results."""
    lines = []
    
    # Title
    lines.append("# Task Execution Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Status:** {result.get('task_status', 'unknown')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Task Definition
    lines.append("## 📋 Task Definition")
    lines.append("")
    try:
        task_def = json.loads(result["task_definition"])
        lines.append(f"### {task_def.get('task_name', 'N/A')}")
        lines.append("")
        lines.append(f"**Objective:** {task_def.get('objective', 'N/A')}")
        lines.append("")
        lines.append(f"**Scope:** {task_def.get('scope', 'N/A')}")
        lines.append("")
        lines.append(f"**Success Criteria:** {task_def.get('success_criteria', 'N/A')}")
        lines.append("")
        lines.append(f"**Constraints:** {task_def.get('constraints', 'N/A')}")
    except:
        lines.append("```json")
        lines.append(result['task_definition'])
        lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Task Plan
    lines.append(f"## 📝 Task Plan ({len(result['task_plan'])} steps)")
    lines.append("")
    for i, step in enumerate(result["task_plan"], 1):
        lines.append(f"{i}. {step}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Execution Results
    lines.append(f"## ✅ Execution Results ({len(result['results'])} steps completed)")
    lines.append("")
    lines.append("| Step | Description | Status | Actions |")
    lines.append("|------|-------------|--------|---------|")
    
    for res in result["results"]:
        step_num = res.get('step', '?')
        desc = res.get('description', 'N/A')[:50] + "..." if len(res.get('description', '')) > 50 else res.get('description', 'N/A')
        status = res.get('status', 'unknown')
        actions = res.get('actions', 'N/A')[:50] + "..." if len(str(res.get('actions', ''))) > 50 else str(res.get('actions', 'N/A'))
        
        # Escape pipe characters in table cells
        desc = desc.replace('|', '\\|')
        actions = actions.replace('|', '\\|')
        
        status_emoji = "✓" if status == "success" else "⚠" if status == "partial" else "✗"
        lines.append(f"| {step_num} | {desc} | {status_emoji} {status} | {actions} |")
    
    lines.append("")
    lines.append("### Detailed Results")
    lines.append("")
    
    for res in result["results"]:
        step_num = res.get('step', '?')
        status = res.get('status', 'unknown')
        status_emoji = "✓" if status == "success" else "⚠" if status == "partial" else "✗"
        
        lines.append(f"#### Step {step_num}: {res.get('description', 'N/A')} {status_emoji}")
        lines.append("")
        lines.append(f"**Status:** {status}")
        lines.append("")
        lines.append(f"**Actions:** {res.get('actions', 'N/A')}")
        lines.append("")
        lines.append(f"**Results:** {res.get('results', 'N/A')}")
        lines.append("")
        if res.get('issues') and res.get('issues') != "None":
            lines.append(f"**Issues:** {res.get('issues')}")
            lines.append("")
        lines.append(f"**Timestamp:** {res.get('timestamp', 'N/A')}")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Weaviate Documents
    if result.get("retrieved_docs"):
        total_docs = len(result.get("retrieved_docs", []))
        if total_docs > 0:
            lines.append("## 📚 Documents Retrieved from Weaviate")
            lines.append("")
            lines.append(f"**Total Documents:** {total_docs}")
            lines.append("")
            collections_used = set(doc.get("collection", "unknown") for doc in result.get("retrieved_docs", []))
            lines.append(f"**Collections:** {', '.join(collections_used)}")
            lines.append("")
            lines.append("### Top Documents")
            lines.append("")
            lines.append("| Rank | Collection | Source | Score | Preview |")
            lines.append("|------|------------|--------|-------|--------|")
            
            for i, doc in enumerate(result.get("retrieved_docs", [])[:10], 1):
                collection = doc.get("collection", "unknown")
                source = doc.get("source_pdf", "unknown")
                score = f"{doc.get('score', 0):.3f}"
                preview = doc.get("text", "")[:100].replace('|', '\\|').replace('\n', ' ')
                lines.append(f"| {i} | {collection} | {source} | {score} | {preview}... |")
            
            lines.append("")
            lines.append("---")
            lines.append("")
    
    # Summary
    if result.get("summary"):
        lines.append("## 📄 Summary")
        lines.append("")
        lines.append(result['summary'])
        lines.append("")
    
    return "\n".join(lines)


def clean_and_parse_json(text: str, json_type: str = "object") -> Optional[Dict | List]:
    """
    Clean text and extract JSON, handling control characters and common formatting issues.
    
    Args:
        text: Text containing JSON
        json_type: "object" for dict, "array" for list
    
    Returns:
        Parsed JSON object or None if parsing fails
    """
    try:
        # Remove control characters (keep newlines, tabs, carriage returns)
        text_clean = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Find JSON boundaries
        if json_type == "object":
            if '{' not in text_clean or '}' not in text_clean:
                return None
            start = text_clean.find('{')
            end = text_clean.rfind('}') + 1
        else:  # array
            if '[' not in text_clean or ']' not in text_clean:
                return None
            start = text_clean.find('[')
            end = text_clean.rfind(']') + 1
        
        json_str = text_clean[start:end]
        
        # Fix common JSON issues
        # Replace single quotes with double quotes (but be careful with apostrophes)
        json_str = re.sub(r"'(\w+)':", r'"\1":', json_str)  # Keys
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # String values
        
        # Remove trailing commas before } or ]
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Remove comments (// or /* */)
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Fix invalid escape sequences
        # JSON only allows: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
        # We need to escape invalid backslashes, but this is complex because
        # we need to process only string values, not the whole JSON structure
        # Use a more robust approach: process the JSON string character by character
        # in a state machine, or use a simpler fix that handles most cases
        
        # Simple fix: replace invalid escapes in string values
        # This regex finds backslashes that are not followed by valid escape chars
        # and are inside string values (between quotes)
        def fix_string_escapes(text):
            """Fix invalid escape sequences in JSON string values."""
            result = []
            in_string = False
            escape_next = False
            i = 0
            
            while i < len(text):
                char = text[i]
                
                if escape_next:
                    # We're processing an escape sequence
                    # Valid escapes: ", \, /, b, f, n, r, t, u (for unicode)
                    if char in '"\\/bfnrtu':
                        result.append('\\' + char)
                        escape_next = False
                    else:
                        # Invalid escape - escape the backslash itself
                        result.append('\\\\' + char)
                        escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and (i == 0 or text[i-1] != '\\' or (i > 1 and text[i-2] == '\\')):
                    # Toggle string state (handle escaped quotes)
                    in_string = not in_string
                    result.append(char)
                else:
                    result.append(char)
                
                i += 1
            
            return ''.join(result)
        
        json_str = fix_string_escapes(json_str)
        
        # Parse JSON
        return json.loads(json_str)
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return None


def define_task_agent(state: AgentState) -> AgentState:
    """Agent node: Defines the task based on user input."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    if not last_message or not isinstance(last_message, HumanMessage):
        return state
    
    user_input = last_message.content
    
    # Use LLM to define the task
    task_definition_prompt = (
        "You are a task definition agent. Analyze the user's request and create a clear, "
        "actionable task definition.\n\n"
        "The task definition should include:\n"
        "1. Task name/title\n"
        "2. Objective (what needs to be accomplished)\n"
        "3. Scope (what is included/excluded)\n"
        "4. Success criteria (how to know the task is complete)\n"
        "5. Constraints or requirements\n\n"
        f"User Request: {user_input}\n\n"
        "MANDATORY FORMATTING REQUIREMENTS - YOU MUST USE MARKDOWN:\n"
        "- Use headers (##, ###) for sections\n"
        "- Use **bold** for emphasis\n"
        "- Use lists (- or *) for items\n"
        "- Use tables (| col1 | col2 |) if appropriate\n"
        "- Use code blocks (```) for structured data\n\n"
        "Provide a comprehensive task definition in JSON format:\n"
        '{\n'
        '  "task_name": "Task title",\n'
        '  "objective": "What needs to be accomplished",\n'
        '  "scope": "What is included/excluded",\n'
        '  "success_criteria": "How to measure success",\n'
        '  "constraints": "Any limitations or requirements"\n'
        '}\n\n'
        "Your response (MUST be in Markdown format):"
    )
    
    # Get LLM response via API
    llm_config = state.get("llm_config", {})
    if llm_config:
        try:
            model = llm_config.get("model", "llama3.2")
            base_url = llm_config.get("base_url", "http://0.0.0.0:11434")
            task_def = call_ollama_api(prompt=task_definition_prompt, model=model, base_url=base_url, stream=False)
        except Exception as e:
            print(f"Error in LLM call: {e}")
            task_def = f"Task: {user_input}"
        
        # Try to parse JSON from response
        task_def_json = clean_and_parse_json(task_def, json_type="object")
        
        if not task_def_json:
            # Fallback: create structured definition from text
            # Try to extract information from the text
            task_name_match = re.search(r'(?:task|name|title)[:\s]+([^\n]+)', task_def, re.IGNORECASE)
            objective_match = re.search(r'(?:objective|goal|purpose)[:\s]+([^\n]+)', task_def, re.IGNORECASE)
            
            task_def_json = {
                "task_name": task_name_match.group(1).strip()[:50] if task_name_match else user_input[:50],
                "objective": objective_match.group(1).strip()[:200] if objective_match else task_def[:200],
                "scope": "As specified in user request",
                "success_criteria": "Task completion verified",
                "constraints": "None specified"
            }
        
        state["task_definition"] = json.dumps(task_def_json, indent=2)
        state["task_status"] = "defined"
        state["messages"].append(AIMessage(content=f"Task defined: {task_def_json.get('task_name', 'Unknown')}"))
    
    return state


def plan_task_agent(state: AgentState) -> AgentState:
    """Agent node: Creates a step-by-step plan for the task."""
    task_def = state.get("task_definition", "")
    
    if not task_def:
        state["task_plan"] = []
        return state
    
    # Parse task definition
    try:
        task_json = json.loads(task_def)
        objective = task_json.get("objective", "")
    except:
        objective = task_def
    
    # Use LLM to create a plan
    planning_prompt = (
        "You are a task planning agent. Based on the task definition below, create a detailed "
        "step-by-step execution plan.\n\n"
        f"Task Definition:\n{task_def}\n\n"
        "MANDATORY FORMATTING REQUIREMENTS - YOU MUST USE MARKDOWN:\n"
        "- Use headers (##, ###) for sections\n"
        "- Use **bold** for emphasis\n"
        "- Use numbered lists (1., 2., 3.) for steps\n"
        "- Use tables (| col1 | col2 |) if appropriate\n"
        "- Use code blocks (```) for structured data\n\n"
        "Create a numbered list of steps (at least 3, maximum 10 steps) that need to be executed "
        "to complete this task. Each step should be clear and actionable.\n\n"
        "Format your response as a JSON array of step descriptions:\n"
        '["Step 1 description", "Step 2 description", ...]\n\n'
        "Your response (MUST be in Markdown format):"
    )
    
    llm_config = state.get("llm_config", {})
    if llm_config:
        try:
            model = llm_config.get("model", "llama3.2")
            base_url = llm_config.get("base_url", "http://0.0.0.0:11434")
            plan_text = call_ollama_api(prompt=planning_prompt, model=model, base_url=base_url, stream=False)
        except Exception as e:
            print(f"Error in LLM call: {e}")
            plan_text = '["Step 1: Analyze requirements", "Step 2: Execute task", "Step 3: Verify completion"]'
        
        # Try to parse plan as JSON array
        plan = clean_and_parse_json(plan_text, json_type="array")
        
        if not plan:
            # Fallback: extract steps from numbered list
            steps = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\n*$)', plan_text, re.MULTILINE)
            if steps:
                plan = [step.strip() for step in steps if step.strip()]
            else:
                # Extract from markdown list
                steps = re.findall(r'[-*]\s*(.+?)(?=\n[-*]|\n*$)', plan_text, re.MULTILINE)
                plan = [step.strip() for step in steps if step.strip()]
        
        if not plan:
            # Final fallback: create simple plan
            plan = [
                "Analyze task requirements",
                "Gather necessary resources",
                "Execute main task",
                "Verify completion"
            ]
        
        state["task_plan"] = plan
        state["current_step"] = 0
        state["task_status"] = "planned"
        state["messages"].append(AIMessage(content=f"Task plan created with {len(plan)} steps"))
    
    return state


def execute_step_agent(state: AgentState) -> AgentState:
    """Agent node: Executes the current step of the task plan."""
    task_plan = state.get("task_plan", [])
    current_step = state.get("current_step", 0)
    task_def = state.get("task_definition", "")
    weaviate_config = state.get("weaviate_config")
    
    if not task_plan or current_step >= len(task_plan):
        state["task_status"] = "completed"
        return state
    
    step_description = task_plan[current_step]
    
    # Retrieve relevant documents from Weaviate if configured
    retrieved_docs = []
    context_text = ""
    
    if weaviate_config and weaviate_config.get("enabled", False):
        collections = weaviate_config.get("collections", [])
        model_path = weaviate_config.get("model_path", "")
        weaviate_url = weaviate_config.get("weaviate_url", "http://localhost:8080")
        
        if collections and model_path:
            # Search Weaviate for relevant documents
            search_query = f"{step_description} {task_def}"
            retrieved_docs = search_weaviate(
                search_query,
                collections,
                model_path,
                weaviate_url,
                top_k=weaviate_config.get("top_k", 10),
                score_threshold=weaviate_config.get("score_threshold", 0.6),
            )
            
            # Format retrieved documents as context
            if retrieved_docs:
                context_lines = ["\n=== Relevant Documents from Weaviate ==="]
                for i, doc in enumerate(retrieved_docs[:5], 1):  # Use top 5
                    context_lines.append(f"\n[Document {i}] (Score: {doc.get('score', 0):.3f})")
                    context_lines.append(f"Source: {doc.get('source_pdf', 'unknown')}")
                    if doc.get('country'):
                        context_lines.append(f"Country: {doc.get('country')}")
                    context_lines.append(f"Text: {doc.get('text', '')[:500]}...")  # First 500 chars
                context_text = "\n".join(context_lines)
                
                # Store in state (accumulate across steps)
                existing_docs = state.get("retrieved_docs", [])
                existing_docs.extend(retrieved_docs)
                state["retrieved_docs"] = existing_docs
    
    # Use LLM to execute the step
    execution_prompt = (
        "You are a task execution agent. Execute the following step of a task plan.\n\n"
        f"Task Definition:\n{task_def}\n\n"
        f"Current Step ({current_step + 1}/{len(task_plan)}): {step_description}\n\n"
    )
    
    if context_text:
        execution_prompt += (
            f"{context_text}\n\n"
            "Use the relevant documents above to inform your execution. "
            "Reference specific information from the documents when relevant.\n\n"
        )
    
    execution_prompt += (
        "MANDATORY FORMATTING REQUIREMENTS - YOU MUST USE MARKDOWN:\n"
        "- Use headers (##, ###) for sections\n"
        "- Use **bold** for emphasis\n"
        "- Use lists (- or *) for items\n"
        "- Use tables (| col1 | col2 |) if appropriate\n"
        "- Use code blocks (```) for structured data\n\n"
        "Execute this step and provide:\n"
        "1. What actions were taken\n"
        "2. Results or outcomes\n"
        "3. Any issues encountered\n"
        "4. Status (success/failure/partial)\n\n"
        "Format your response as JSON:\n"
        '{\n'
        '  "actions": "What was done (can include Markdown)",\n'
        '  "results": "Outcomes (can include Markdown)",\n'
        '  "issues": "Any problems",\n'
        '  "status": "success/failure/partial"\n'
        '}\n\n'
        "Your response (MUST be in Markdown format):"
    )
    
    llm_config = state.get("llm_config", {})
    if llm_config:
        try:
            model = llm_config.get("model", "llama3.2")
            base_url = llm_config.get("base_url", "http://0.0.0.0:11434")
            result_text = call_ollama_api(prompt=execution_prompt, model=model, base_url=base_url, stream=False)
        except Exception as e:
            print(f"Error in LLM call: {e}")
            result_text = '{"actions": "Step executed", "results": "Completed", "status": "success"}'
        
        # Try to parse result JSON
        result = clean_and_parse_json(result_text, json_type="object")
        
        if not result:
            # Fallback: extract information from text
            actions_match = re.search(r'(?:actions?|what was done)[:\s]+([^\n]+)', result_text, re.IGNORECASE)
            results_match = re.search(r'(?:results?|outcomes?)[:\s]+([^\n]+)', result_text, re.IGNORECASE)
            status_match = re.search(r'(?:status)[:\s]+(success|failure|partial)', result_text, re.IGNORECASE)
            
            result = {
                "actions": actions_match.group(1).strip()[:200] if actions_match else result_text[:200],
                "results": results_match.group(1).strip()[:200] if results_match else "Step executed",
                "issues": "None",
                "status": status_match.group(1).lower() if status_match else "success"
            }
        
        result["step"] = current_step + 1
        result["description"] = step_description
        result["timestamp"] = datetime.now().isoformat()
        
        # Add to results
        results = state.get("results", [])
        results.append(result)
        state["results"] = results
        
        # Move to next step
        state["current_step"] = current_step + 1
        state["task_status"] = "executing"
        
        status_emoji = "✓" if result.get("status") == "success" else "⚠" if result.get("status") == "partial" else "✗"
        state["messages"].append(AIMessage(
            content=f"{status_emoji} Step {current_step + 1}/{len(task_plan)}: {step_description}\n"
                   f"Status: {result.get('status', 'unknown')}"
        ))
    
    return state


def should_continue(state: AgentState) -> str:
    """Conditional edge: Determines if task execution should continue."""
    task_plan = state.get("task_plan", [])
    current_step = state.get("current_step", 0)
    task_status = state.get("task_status", "")
    
    if task_status == "completed":
        return "end"
    
    if current_step < len(task_plan):
        return "execute"
    
    return "end"


def summarize_task_agent(state: AgentState) -> AgentState:
    """Agent node: Creates a summary of the completed task."""
    task_def = state.get("task_definition", "")
    task_plan = state.get("task_plan", [])
    results = state.get("results", [])
    
    summary_prompt = (
        "You are a task summarization agent. Create a comprehensive summary of the completed task.\n\n"
        f"Task Definition:\n{task_def}\n\n"
        f"Task Plan:\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(task_plan)) + "\n\n"
        f"Execution Results:\n{json.dumps(results, indent=2)}\n\n"
        "MANDATORY FORMATTING REQUIREMENTS - YOU MUST USE MARKDOWN:\n"
        "- Use headers (#, ##, ###) for sections\n"
        "- Use **bold** for emphasis\n"
        "- Use lists (- or *) for items\n"
        "- Use tables (| col1 | col2 |) for structured data\n"
        "- Use code blocks (```) for code or structured content\n"
        "- Use blockquotes (>) for important notes\n\n"
        "Create a summary that includes:\n"
        "1. Task overview\n"
        "2. Steps completed\n"
        "3. Key results\n"
        "4. Overall status\n"
        "5. Recommendations (if any)\n\n"
        "Your response (MUST be in Markdown format):"
    )
    
    llm_config = state.get("llm_config", {})
    if llm_config:
        try:
            model = llm_config.get("model", "llama3.2")
            base_url = llm_config.get("base_url", "http://0.0.0.0:11434")
            summary = call_ollama_api(prompt=summary_prompt, model=model, base_url=base_url, stream=False)
        except Exception as e:
            print(f"Error in LLM call: {e}")
            summary = "Task execution completed."
        
        state["task_status"] = "completed"
        state["messages"].append(AIMessage(content=f"## Task Summary\n\n{summary}"))
    
    return state


def create_task_agent_graph() -> StateGraph:
    """Create the LangGraph workflow for task definition and execution."""
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("define_task", define_task_agent)
    workflow.add_node("plan_task", plan_task_agent)
    workflow.add_node("execute_step", execute_step_agent)
    workflow.add_node("summarize", summarize_task_agent)
    
    # Set entry point
    workflow.set_entry_point("define_task")
    
    # Add edges
    workflow.add_edge("define_task", "plan_task")
    workflow.add_edge("plan_task", "execute_step")
    workflow.add_conditional_edges(
        "execute_step",
        should_continue,
        {
            "execute": "execute_step",  # Loop back to execute next step
            "end": "summarize"  # Go to summary when done
        }
    )
    workflow.add_edge("summarize", END)
    
    return workflow.compile()


def run_task_agent(
    user_request: str,
    ollama_model: str = "llama3.2",
    ollama_base_url: str = "http://localhost:11434",
    weaviate_collections: Optional[List[str]] = None,
    weaviate_url: str = "http://localhost:8080",
    weaviate_model_path: Optional[str] = None,
    weaviate_top_k: int = 10,
    weaviate_score_threshold: float = 0.6,
) -> Dict:
    """Run the task definition agent with a user request."""
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph packages are required. Install with: pip install langgraph langchain langchain-community")
    
    # Configure LLM (using API, not LangChain Ollama)
    llm_config = {
        "model": ollama_model,
        "base_url": ollama_base_url,
    }
    
    # Configure Weaviate if provided
    weaviate_config = None
    if weaviate_collections and weaviate_model_path:
        weaviate_config = {
            "enabled": True,
            "collections": weaviate_collections,
            "weaviate_url": weaviate_url,
            "model_path": weaviate_model_path,
            "top_k": weaviate_top_k,
            "score_threshold": weaviate_score_threshold,
        }
        print(f"✓ Weaviate integration enabled with {len(weaviate_collections)} collection(s)")
    else:
        weaviate_config = {"enabled": False}
    
    # Create initial state
    initial_state = AgentState(
        messages=[HumanMessage(content=user_request)],
        task_definition="",
        task_plan=[],
        current_step=0,
        task_status="initialized",
        results=[],
        context={},
        llm_config=llm_config,  # Store LLM config in state for nodes to use
        weaviate_config=weaviate_config,
        retrieved_docs=[],
    )
    
    # Create and run the graph
    app = create_task_agent_graph()
    
    print("=" * 60)
    print("LangGraph Task Definition Agent")
    print("=" * 60)
    print(f"\nUser Request: {user_request}\n")
    print("Processing task...\n")
    
    # Run the graph
    final_state = app.invoke(initial_state)
    
    return {
        "task_definition": final_state.get("task_definition", ""),
        "task_plan": final_state.get("task_plan", []),
        "results": final_state.get("results", []),
        "task_status": final_state.get("task_status", ""),
        "messages": [
            {
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content if hasattr(msg, 'content') else str(msg)
            }
            for msg in final_state.get("messages", [])
        ],
        "summary": final_state.get("messages", [])[-1].content if final_state.get("messages") else "",
        "retrieved_docs": final_state.get("retrieved_docs", []),
    }


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Task Definition Agent - Define and execute specific tasks."
    )
    parser.add_argument(
        "task_request",
        help="Description of the task to define and execute.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model to use (default: llama3.2).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file to save task results.",
    )
    parser.add_argument(
        "--weaviate-collections",
        nargs="+",
        help="Weaviate collection names to search (e.g., --weaviate-collections Cms DocumentChunk).",
    )
    parser.add_argument(
        "--weaviate-url",
        default="http://localhost:8080",
        help="Weaviate server URL (default: http://localhost:8080).",
    )
    parser.add_argument(
        "--weaviate-model-path",
        help="Path to embedding model for Weaviate queries (e.g., ./models/mxbai-embed-large-v1).",
    )
    parser.add_argument(
        "--weaviate-top-k",
        type=int,
        default=10,
        help="Number of top documents to retrieve per collection (default: 10).",
    )
    parser.add_argument(
        "--weaviate-score-threshold",
        type=float,
        default=0.6,
        help="Minimum similarity score for retrieved documents (default: 0.6).",
    )
    
    args = parser.parse_args()
    
    if not LANGGRAPH_AVAILABLE:
        print("Error: LangGraph packages are required.")
        print("Install with: pip install langgraph langchain langchain-community")
        return
    
    try:
        # Run the agent
        result = run_task_agent(
            args.task_request,
            ollama_model=args.ollama_model,
            ollama_base_url=args.ollama_url,
            weaviate_collections=args.weaviate_collections,
            weaviate_url=args.weaviate_url,
            weaviate_model_path=args.weaviate_model_path,
            weaviate_top_k=args.weaviate_top_k,
            weaviate_score_threshold=args.weaviate_score_threshold,
        )
        
        # Generate Markdown report
        markdown_report = generate_markdown_report(result)
        
        # Display results
        print("\n" + "=" * 60)
        print("Task Execution Complete")
        print("=" * 60)
        print("\n" + markdown_report)
        
        # Also add markdown to result
        result["markdown_report"] = markdown_report
        
        # Save to files (JSON and Markdown)
        if args.output:
            output_path = args.output
            # If JSON extension, also create Markdown version
            if output_path.suffix == ".json":
                md_path = output_path.with_suffix(".md")
            else:
                md_path = output_path.with_suffix(".md")
                output_path = output_path.with_suffix(".json")
        else:
            # Create default output filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = re.sub(r'[^\w\s-]', '', args.task_request)[:30]
            safe_task = re.sub(r'[-\s]+', '_', safe_task)
            output_path = Path(f"task_report_{safe_task}_{timestamp}.json")
            md_path = Path(f"task_report_{safe_task}_{timestamp}.md")
        
        # Save JSON
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 JSON results saved to: {output_path}")
        
        # Save Markdown
        with md_path.open("w", encoding="utf-8") as f:
            f.write(markdown_report)
        print(f"📝 Markdown report saved to: {md_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
