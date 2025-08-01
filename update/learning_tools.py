import json
import logging
from datetime import datetime
from typing import Any, Dict, Callable
from manager import AgentMemory , AgentState , Task
from imports import *
from tools.planning_tools import *
from tools.web_tools import *
from tools.system_tools import *
from tools.travel_tools import *

console = Console() # Initialize console for rich prints
logger = logging.getLogger(__name__)


class LearningTools:
    """
    A dedicated class that groups tools related to the agent's learning,
    context retrieval, and memory inspection.
    """
    def __init__(self, memory: AgentMemory, state: AgentState, llm: Any , logger_instance: logging.Logger):
        """
        Initializes the LearningTools class with dependencies from the main agent.
        
        Args:
            memory (AgentMemory): The agent's memory manager.
            state (AgentState): The agent's current state.
            llm (Any): The large language model instance.
        """
        self.memory = memory
        self.state = state
        self.llm = llm
        self.logger = logger_instance


    def learn_from_interaction(self, interaction_data: str) -> str:
        """Learn and store user preferences from interactions."""
        try:
            learning_prompt = f"""
            Analyze this interaction and extract user preferences and personal information:
            {interaction_data}

            Return preferences in JSON format:
            {{
                "personal_info": {{
                    "name": "user's name if mentioned",
                    "age": "age if mentioned",
                    "location": "location if mentioned",
                    "profession": "job/profession if mentioned"
                }},
                "communication_style": "preferred style",
                "interests": ["interest1", "interest2"],
                "work_patterns": "observed patterns",
                "priority_areas": ["area1", "area2"]
            }}
            """
            if self.llm is None:
                raise RuntimeError("LLM is not initialized. Cannot perform this operation.")
            
            # Ensure user_preferences is initialized in agent_state if it's not already
            if self.state.user_preferences is None:
                self.state.user_preferences = {}
                
            response = self.llm.invoke(learning_prompt)
            preferences = json.loads(str(response.content))
            
            # Update user preferences with new information
            self.state.user_preferences.update(preferences)
            
            # Store each piece of personal info separately for easier retrieval
            if "personal_info" in preferences:
                for key, value in preferences["personal_info"].items():
                    if value and value != "not mentioned":
                        self.memory.update_user_profile(f"personal_{key}", value)
            
            # Store the full preference analysis
            self.memory.update_user_profile("interaction_learning", {
                "raw_analysis": response.content,
                "preferences": preferences,
                "timestamp": datetime.now().isoformat()
            })

            console.print("[green]🧠 Preferences learned and updated in user profile.[/green]")
            return "Preferences learned and updated in user profile."

        except Exception as e:
            logger.error(f"Learning error: {str(e)}", exc_info=True)
            return f"Learning error: {str(e)}"

    def get_user_context(self) -> Dict[str, Any]:
        """Get comprehensive user context including personal info and conversation history."""
        try:
            # Get stored user profile
            user_profile = self.memory.get_user_profile()
            
            # Get recent conversations
            recent_convs = self.memory.get_recent_conversations(self.state.session_id, 5)
            
            # Filter conversations for relevance (basic keyword matching)
            filtered_convs = []
            for conv in recent_convs:
                # Simple keyword overlap check
                conv_keywords = set(conv['user_input'].lower().split())
                # Add conversation if it has some relevance (e.g., >2 overlapping words)
                if len(conv_keywords) > 2:  # Only include if conversation has meaningful content
                    filtered_convs.append(conv)
                if len(filtered_convs) >= 5:  # Limit to 5 relevant conversations
                    break
            
            # Extract personal information
            personal_info = {}
            for key, value in user_profile.items():
                if key.startswith('personal_'):
                    personal_info[key.replace('personal_', '')] = value
            
            return {
                "personal_info": personal_info,
                "preferences": self.state.user_preferences,
                "recent_conversations": filtered_convs,
                "profile_data": user_profile
            }
            
        except Exception as e:
            logger.error(f"Context retrieval error: {e}", exc_info=True)
            return {}

    def check_memory(self) -> str:
        """Debug function to check what's stored in memory."""
        try:
            user_profile = self.memory.get_user_profile()
            recent_convs = self.memory.get_recent_conversations(self.state.session_id, 5)
            
            memory_report = "📋 Memory Report:\n\n"
            memory_report += "🔹 User Profile Data:\n"
            if not user_profile:
                memory_report += "  No user profile data found.\n"
            else:
                for key, value in user_profile.items():
                    memory_report += f"  • {key}: {value}\n"
            
            memory_report += "\n🔹 Recent Conversations:\n"
            if not recent_convs:
                memory_report += "  No recent conversations found.\n"
            else:
                for i, conv in enumerate(recent_convs):
                    memory_report += f"  {i+1}. User: {conv['user_input'][:50]}...\n"
                    memory_report += f"    AIA: {conv['agent_response'][:50]}...\n"
            
            return memory_report
        except Exception as e:
            logger.error(f"Error checking memory: {e}", exc_info=True)
            return f"Error checking memory: {e}"

    def auto_learn_from_conversation(self, user_input: str, agent_response: str) -> str:
        """Automatically analyze each conversation for learning opportunities."""
        try:
            # Check if the conversation contains personal information or preferences
            learning_indicators = [
                'my name is', 'i am', 'i like', 'i prefer', 'i work', 'i live',
                'call me', 'i hate', 'i love', 'my favorite', 'i usually'
            ]
            
            if any(indicator in user_input.lower() for indicator in learning_indicators):
                self.logger.info("Learning indicators found, triggering learn_from_interaction.")
                # This call now works perfectly because learn_from_interaction is also a method of this class
                return self.learn_from_interaction(f"User: {user_input}\nAgent: {agent_response}")
            else:
                self.logger.debug(f"No learning indicators found in input: {user_input}")
                return "No new preferences detected in this interaction."
        except Exception as e:
            self.logger.error(f"Auto-learning error: {e}", exc_info=True)
            return f"An error occurred during the auto-learning process: {e}"

