from imports import *
from manager import AgentState, AgentMemory , Task
from tools.learning_tools import *
from tools.web_tools import *
from tools.system_tools import *
from tools.travel_tools import *
from tools.planning_tools import *
from tools.learning_tools import *

class PlanningTools:
    """
    A class that groups all tools related to learning, planning, and goal setting.
    """
    def __init__(self, memory: AgentMemory, state: AgentState, llm, print_func):
        """Initializes the class with dependencies from the main agent."""
        self.memory = memory
        self.state = state
        self.llm = llm
        self.print_letter_by_letter = print_func  # A function for special printing

    def proactive_suggestion(self, context: str) -> str:
        """Generate proactive suggestions based on user context."""
        try:
            user_prefs = self.state.user_preferences
            recent_convs = self.memory.get_recent_conversations(self.state.session_id, 5)

            suggestion_prompt = f"""
            The user is asking for suggestions. Based on the context, provide 3 helpful suggestions.
            User's request: {context}
            User preferences: {json.dumps(user_prefs)}
            Recent activity: {json.dumps([c['user_input'] for c in recent_convs])}
            Format as a numbered list.
            """
            
            if self.llm is None:
                raise RuntimeError("LLM is not initialized.")
                
            suggestions = self.llm.invoke(suggestion_prompt).content
            
            # The tool should return the text, and the main agent can decide how to print it.
            # This is a good practice for separating logic from UI.
            console.print("\n[yellow]💡 Proactive suggestions:[/yellow]")
            self.print_letter_by_letter(str(suggestions))
            
            return f"💡 Proactive suggestions provided:\n{suggestions}"

        except Exception as e:
            return f"Suggestion generation error: {e}"

    def plan_tasks(self, goal: str) -> str:
        """Break down a goal into actionable tasks using the LLM."""
        try:
            planning_prompt = f"""
            Break down this goal into a JSON object of sequential and parallel tasks:
            Goal: {goal}
            Respond only with the JSON object.
            """
            if self.llm is None:
                raise RuntimeError("LLM is not initialized.")
            
            plan_str = self.llm.invoke(planning_prompt).content
            plan = json.loads(plan_str)
            
            return f"Task plan created for goal '{goal}':\n{json.dumps(plan, indent=2)}"

        except Exception as e:
            return f"Task planning error: {e}"

    def create_goal(self, goal_description: str) -> str:
        """Create a new goal for the agent to work towards."""
        try:
            goal_id = str(uuid.uuid4())

            goal_analysis_prompt = f"""
            Analyze this goal and provide a priority (1-10) in JSON format:
            Goal: {goal_description}
            Respond only with a JSON object like {{"priority": 7}}.
            """
            if self.llm is None:
                raise RuntimeError("LLM is not initialized.")
            
            analysis_str = self.llm.invoke(goal_analysis_prompt).content
            analysis = json.loads(analysis_str)
            priority = analysis.get("priority", 5)

            self.state.current_goal = goal_description
            if self.state.active_tasks is None:
                self.state.active_tasks = []
            self.state.active_tasks.append(goal_id)

            # Store goal in the database using the memory object's db_path
            with sqlite3.connect(self.memory.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO goals (id, description, status, priority, created_at) VALUES (?, ?, ?, ?, ?)',
                    (goal_id, goal_description, 'pending', priority, datetime.now())
                )
                conn.commit()

            return f"Goal created: {goal_description} (ID: {goal_id[:8]}, Priority: {priority})"

        except Exception as e:
            return f"Goal creation error: {e}"

