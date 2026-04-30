"""
LangGraph workflow for career advice generation.
Based on the README: 4-node workflow with Graph Retrieval and LLM Generation.
"""
from typing import Dict, List, Optional, Any, TypedDict, Annotated
import json
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Try to import LangGraph, but provide fallback for demo/development
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Minimal fallback: no real state machine, sequential execution
    class StateGraph:
        def __init__(self, state_schema=None):
            self._nodes = {}
            self._edges = []
            self._entry = None
        def add_node(self, name, fn):
            self._nodes[name] = fn
        def add_edge(self, src, dst):
            self._edges.append((src, dst))
        def set_entry_point(self, name):
            self._entry = name
        def compile(self):
            return self
        def invoke(self, initial_state):
            """Execute nodes sequentially following edge order."""
            state = initial_state
            # Build adjacency from edges
            next_map = {}
            for s, d in self._edges:
                next_map[s] = d
            current = self._entry
            while current and current != END:
                if current in self._nodes:
                    state = self._nodes[current](state)
                current = next_map.get(current)
            return state
    END = "END"


class WorkflowStep(str, Enum):
    """Workflow step identifiers."""
    INIT = "init"
    GRAPH_RETRIEVAL = "graph_retrieval"
    PROMPT_CONSTRUCTION = "prompt_construction"
    LLM_GENERATION = "llm_generation"
    COMPLETE = "complete"


@dataclass
class WorkflowState:
    """State for the career advice workflow."""

    # Input
    user_id: str
    job_id: str

    # Step outputs
    current_step: WorkflowStep = WorkflowStep.INIT
    user_skills: Dict[str, str] = field(default_factory=dict)  # skill_id -> level
    job_required_skills: Dict[str, str] = field(default_factory=dict)  # skill_id -> min_level
    job_preferred_skills: Dict[str, str] = field(default_factory=dict)  # skill_id -> min_level

    # Graph retrieval results
    skill_gap: Dict[str, Dict[str, Optional[str]]] = field(default_factory=dict)  # skill_id -> {user_level, required_level}
    shortest_paths: List[List[str]] = field(default_factory=list)  # List of skill paths
    skill_coverage: float = 0.0

    # Prompt construction
    prompt: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    # LLM generation
    llm_response: Dict[str, Any] = field(default_factory=dict)
    career_advice: str = ""
    learning_path: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    errors: List[str] = field(default_factory=list)
    execution_time_ms: int = 0


class CareerAdvisorWorkflow:
    """Career advisor workflow using GraphRAG pattern."""

    def __init__(self, graph_loader: Any, llm_simulator: Any):
        """
        Initialize workflow.

        Args:
            graph_loader: GraphLoader instance for skill gap analysis
            llm_simulator: LLM simulator for generating advice
        """
        self.graph_loader = graph_loader
        self.llm_simulator = llm_simulator

        # Build workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow."""
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node(WorkflowStep.INIT, self._init_node)
        workflow.add_node(WorkflowStep.GRAPH_RETRIEVAL, self._graph_retrieval_node)
        workflow.add_node(WorkflowStep.PROMPT_CONSTRUCTION, self._prompt_construction_node)
        workflow.add_node(WorkflowStep.LLM_GENERATION, self._llm_generation_node)

        # Add edges (sequential flow)
        workflow.add_edge(WorkflowStep.INIT, WorkflowStep.GRAPH_RETRIEVAL)
        workflow.add_edge(WorkflowStep.GRAPH_RETRIEVAL, WorkflowStep.PROMPT_CONSTRUCTION)
        workflow.add_edge(WorkflowStep.PROMPT_CONSTRUCTION, WorkflowStep.LLM_GENERATION)
        workflow.add_edge(WorkflowStep.LLM_GENERATION, END)

        # Set entry point
        workflow.set_entry_point(WorkflowStep.INIT)

        return workflow.compile()

    def _init_node(self, state: WorkflowState) -> WorkflowState:
        """Initialize workflow state."""
        state.current_step = WorkflowStep.INIT
        print(f"[{state.current_step}] Initializing workflow for user {state.user_id}, job {state.job_id}")
        return state

    def _graph_retrieval_node(self, state: WorkflowState) -> WorkflowState:
        """Retrieve skill information from graph."""
        state.current_step = WorkflowStep.GRAPH_RETRIEVAL

        try:
            # Get user skills
            state.user_skills = self.graph_loader.get_user_skills(state.user_id)

            # Get job skills
            required, preferred = self.graph_loader.get_job_skills(state.job_id)
            state.job_required_skills = required
            state.job_preferred_skills = preferred

            # Calculate skill gap
            skill_gap_raw = self.graph_loader.get_skill_gap(state.user_id, state.job_id)
            state.skill_gap = {
                skill_id: {
                    "user_level": user_level,
                    "required_level": required_level
                }
                for skill_id, (user_level, required_level) in skill_gap_raw.items()
            }

            # Find shortest paths
            state.shortest_paths = self.graph_loader.find_shortest_paths(
                state.user_id, state.job_id, max_path_length=3
            )

            # Calculate skill coverage
            coverage_result = self.graph_loader.get_recommended_learning_path(state.user_id, state.job_id)
            state.skill_coverage = coverage_result.get("skill_coverage", 0.0)

            print(f"[{state.current_step}] Retrieved skill gap: {len(state.skill_gap)} skills")
            print(f"[{state.current_step}] Skill coverage: {state.skill_coverage:.2f}")

        except Exception as e:
            state.errors.append(f"Graph retrieval error: {str(e)}")
            print(f"[{state.current_step}] Error: {e}")

        return state

    def _prompt_construction_node(self, state: WorkflowState) -> WorkflowState:
        """Construct prompt for LLM."""
        state.current_step = WorkflowStep.PROMPT_CONSTRUCTION

        try:
            # Build context
            context = {
                "user_id": state.user_id,
                "job_id": state.job_id,
                "user_skill_count": len(state.user_skills),
                "required_skill_count": len(state.job_required_skills),
                "preferred_skill_count": len(state.job_preferred_skills),
                "skill_gap_count": len(state.skill_gap),
                "skill_coverage": state.skill_coverage,
                "missing_skills": list(state.skill_gap.keys()),
                "shortest_paths": state.shortest_paths[:3]  # Top 3 paths
            }
            state.context = context

            # Construct CoT (Chain of Thought) prompt
            prompt = self._build_cot_prompt(state)
            state.prompt = prompt

            print(f"[{state.current_step}] Constructed prompt with {len(state.skill_gap)} skill gaps")

        except Exception as e:
            state.errors.append(f"Prompt construction error: {str(e)}")
            print(f"[{state.current_step}] Error: {e}")

        return state

    def _build_cot_prompt(self, state: WorkflowState) -> str:
        """Build Chain of Thought prompt."""
        # Format skill information
        user_skills_str = ", ".join([f"{skill_id} ({level})"
                                   for skill_id, level in state.user_skills.items()][:10])
        required_skills_str = ", ".join([f"{skill_id} ({level})"
                                       for skill_id, level in state.job_required_skills.items()])

        # Format skill gap
        skill_gap_items = []
        for skill_id, levels in state.skill_gap.items():
            user_level = levels.get("user_level", "None")
            required_level = levels.get("required_level", "Unknown")
            skill_gap_items.append(f"- {skill_id}: Current={user_level}, Required={required_level}")

        skill_gap_str = "\n".join(skill_gap_items)

        # Build CoT prompt
        prompt = f"""You are a career advisor helping a user prepare for a job application.

## Context
User ID: {state.user_id}
Target Job ID: {state.job_id}

## Current Skills
The user currently has these skills: {user_skills_str}

## Job Requirements
The target job requires these skills: {required_skills_str}

## Skill Gap Analysis
The user has the following skill gaps:
{skill_gap_str}

Skill Coverage: {state.skill_coverage:.1%}

## Available Learning Paths
Potential skill development paths (from current to required):
{json.dumps(state.shortest_paths[:3], indent=2)}

## Task
Provide a personalized career development plan to help the user bridge the skill gaps.
Consider:
1. Which skills are most critical to learn first?
2. What resources or learning methods would be most effective?
3. What is a realistic timeline for skill development?
4. How can the user leverage existing skills to learn new ones?

## Output Format
Respond with a JSON object containing:
{{
  "summary": "Brief summary of the main recommendations",
  "critical_skills": ["list", "of", "most", "critical", "skills"],
  "learning_path": [
    {{
      "skill_id": "skill_name",
      "current_level": "beginner/intermediate/advanced/expert",
      "target_level": "beginner/intermediate/advanced/expert",
      "resources": ["resource1", "resource2"],
      "estimated_time": "time_estimate",
      "priority": "high/medium/low"
    }}
  ],
  "timeline_months": estimated_timeline_in_months,
  "confidence_score": 0.0_to_1.0
}}

## Instructions
Think step by step about the skill gaps and learning paths.
Prioritize based on job requirements and existing skills.
Be realistic about time estimates.
Use the skill paths above as guidance."""

        return prompt

    def _llm_generation_node(self, state: WorkflowState) -> WorkflowState:
        """Generate career advice using LLM."""
        state.current_step = WorkflowStep.LLM_GENERATION

        try:
            # Call LLM simulator
            response = self.llm_simulator.generate(
                prompt=state.prompt,
                temperature=0.3,  # Low temperature for consistent output
                max_tokens=1000
            )

            state.llm_response = response

            # Parse response
            if "response" in response:
                llm_output = response["response"]

                # Try to parse JSON
                try:
                    advice_data = json.loads(llm_output)
                    state.career_advice = advice_data.get("summary", "")
                    state.learning_path = advice_data.get("learning_path", [])
                    state.confidence_score = advice_data.get("confidence_score", 0.5)
                except json.JSONDecodeError:
                    # Fallback: store raw text
                    state.career_advice = llm_output
                    state.confidence_score = 0.3
            else:
                state.career_advice = "Unable to generate advice at this time."
                state.confidence_score = 0.1

            print(f"[{state.current_step}] Generated advice with confidence {state.confidence_score:.2f}")

        except Exception as e:
            state.errors.append(f"LLM generation error: {str(e)}")
            state.career_advice = f"Error generating advice: {str(e)}"
            state.confidence_score = 0.0
            print(f"[{state.current_step}] Error: {e}")

        return state

    def run(self, user_id: str, job_id: str) -> WorkflowState:
        """
        Run the workflow for a user and job.

        Args:
            user_id: User ID
            job_id: Job ID

        Returns:
            WorkflowState with results
        """
        start_time = datetime.now()

        # Create initial state
        initial_state = WorkflowState(user_id=user_id, job_id=job_id)

        # Run workflow
        if LANGGRAPH_AVAILABLE:
            final_state = self.workflow.invoke(initial_state)
        else:
            # Manual execution
            final_state = initial_state
            final_state = self._init_node(final_state)
            final_state = self._graph_retrieval_node(final_state)
            final_state = self._prompt_construction_node(final_state)
            final_state = self._llm_generation_node(final_state)

        # Calculate execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        final_state.execution_time_ms = int(execution_time)

        return final_state

    def get_workflow_summary(self, state: WorkflowState) -> Dict[str, Any]:
        """Get summary of workflow results."""
        return {
            "user_id": state.user_id,
            "job_id": state.job_id,
            "skill_gap_count": len(state.skill_gap),
            "skill_coverage": state.skill_coverage,
            "career_advice_length": len(state.career_advice),
            "learning_path_items": len(state.learning_path),
            "confidence_score": state.confidence_score,
            "execution_time_ms": state.execution_time_ms,
            "error_count": len(state.errors),
            "steps_completed": state.current_step.value
        }


def create_default_workflow(graph_loader: Any, llm_simulator: Any) -> CareerAdvisorWorkflow:
    """Create a default workflow instance."""
    return CareerAdvisorWorkflow(graph_loader, llm_simulator)