import asyncio
import tempfile
import threading
import time
import uuid
import json
import logging
import os
import re
import shutil
import sys
import urllib.parse
import webbrowser
import subprocess 
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

import pkg_resources
import requests
from docx import Document
import fitz  # PyMuPDF
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import re
import time
import winsound
from bs4 import BeautifulSoup
from urllib.parse import quote
from serpapi import GoogleSearch
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import json
from dotenv import load_dotenv

import pydantic
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory, InMemoryHistory
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import StructuredTool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.agents import AgentAction, AgentFinish
from langchain.schema import Document as LangchainDocument
import speech_recognition as sr
import edge_tts

"""
def extract_json_from_response(text):
    import re
    # Search for JSON inside triple backticks first
    json_in_backticks = re.search(r"``````", text)
    if json_in_backticks:
        return json_in_backticks.group(1)
    # Fallback: search for first {...} block in the text
    brace_match = re.search(r"(\{[\s\S]*?\})", text)
    if brace_match:
        return brace_match.group(1)
    return None
"""
# Setup enhanced logging
logging.basicConfig(
    filename='aia_agent.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Initialize console for rich output
console = Console()

# Initialize speech recognizer
recognizer = sr.Recognizer()


# Initialize the api keys
SERPAPI_KEY = 'Enter your api key'
SENDGRID_API_KEY = 'Enter your api key'

# Enhanced agent state and configuration
@dataclass
class AgentState:
    """Represents the current state of the AI agent"""
    current_goal: Optional[str] = None
    active_tasks: Optional[List[str]] = None
    context_memory: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, Any]] = None
    learning_insights: Optional[List[Any]   ] = None
    proactive_mode: bool = True
    voice_mode: bool = False
    last_interaction: Optional[datetime] = None
    session_id: str = ""

    def __post_init__(self):
        if self.active_tasks is None:
            self.active_tasks = []
        if self.context_memory is None:
            self.context_memory = {}
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.learning_insights is None:
            self.learning_insights = []
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        if self.learning_insights is None or not isinstance(self.learning_insights, list):
            self.learning_insights = []

@dataclass
class Task:
    """Represents a task the agent is working on"""
    id: str
    description: str
    priority: int  # 1-10, 10 being highest
    status: str  # pending, in_progress, completed, failed
    created_at: datetime
    deadline: Optional[datetime] = None
    subtasks: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.subtasks is None:
            self.subtasks = []
        if self.context is None:
            self.context = {}

class AgentMemory:
    """Persistent memory system for the AI agent"""

    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the SQLite database for persistent memory"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp DATETIME,
                    user_input TEXT,
                    agent_response TEXT,
                    context TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at DATETIME
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    description TEXT,
                    status TEXT,
                    priority INTEGER,
                    created_at DATETIME,
                    completed_at DATETIME
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight TEXT,
                    category TEXT,
                    confidence REAL,
                    timestamp DATETIME
                )
            ''')
            conn.commit()

    def store_conversation(self, session_id: str, user_input: str, agent_response: str, context: Optional[Dict[str, Any]] = None):
        """Store a conversation exchange"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (session_id, timestamp, user_input, agent_response, context)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, datetime.now(), user_input, agent_response, json.dumps(context or {})))
            conn.commit()

    def get_user_profile(self) -> Dict[str, Any]:
        """Retrieve user profile data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM user_profile')
            return {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

    def update_user_profile(self, key: str, value: Any):
        """Update user profile data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO user_profile (key, value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, json.dumps(value), datetime.now()))
            conn.commit()

    def get_recent_conversations(self, session_id: str, limit: int = 5) -> List[Dict]:
        """Get recent conversation history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_input, agent_response, timestamp, context
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (session_id, limit))
            return [{
                'user_input': row[0],
                'agent_response': row[1],
                'timestamp': row[2],
                'context': json.loads(row[3])
            } for row in cursor.fetchall()]

class FlightSearchArgs(BaseModel):
    departure: str = Field(description="The departure city or airport code (e.g., 'Mumbai', 'BOM').")
    destination: str = Field(description="The destination city or airport code (e.g., 'Paris', 'CDG').")
    departure_date: str = Field(description="The departure date in YYYY-MM-DD format.")
    return_date: Optional[str] = Field(None, description="The return date in YYYY-MM-DD format (optional).")
    passengers: int = Field(1, description="The number of passengers.")

class HotelSearchArgs(BaseModel):
    location: str = Field(description="The city or location to search for hotels.")
    check_in: str = Field(description="The check-in date in YYYY-MM-DD format.")
    check_out: str = Field(description="The check-out date in YYYY-MM-DD format.")
    guests: int = Field(2, description="The number of guests.")
    preferences: Optional[str] = Field(None, description="User preferences for the hotel, e.g., 'luxury', 'budget', '5-star'.")

class EmailArgs(BaseModel):
    to_email: str = Field(description="The recipient's email address.")
    subject: str = Field(description="The subject line of the email.")
    content: str = Field(description="The HTML or plain text content of the email body.")

class CompleteTripArgs(BaseModel):
    departure: str = Field(description="The departure city or airport code for the trip.")
    destination: str = Field(description="The destination city for the trip.")
    departure_date: str = Field(description="The departure date for the trip in YYYY-MM-DD format.")
    return_date: str = Field(description="The return date for the trip in YYYY-MM-DD format.")
    hotel_preferences: Optional[str] = Field(None, description="User preferences for the hotel, e.g., 'luxury', 'budget'.")
    email_address: Optional[str] = Field(None, description="Optional email address to send the final itinerary to.")

class IntelligentAgent:
    """Core AI Agent class with autonomous capabilities"""

    def __init__(self):
        self.state = AgentState()
        self.memory = AgentMemory()
        self.background_tasks = []
        self.monitoring_thread = None
        self.llm = None
        self.embeddings = None
        self.agent_executor = None
        self.vector_store = None
        self.session = PromptSession(history=FileHistory('aia_history.txt'))

    def initialize(self):
        """Initialize the agent with all necessary components"""
        console.print("[yellow]🧠 Initializing Enhanced AI Agent...[/yellow]")

        # Initialize LLM and embeddings
        self.llm, self.embeddings = self.initialize_llm()
        if not self.llm:
            raise Exception("Failed to initialize LLM")

        # Load user profile and preferences
        self.state.user_preferences = self.memory.get_user_profile()

        # Initialize agent tools and executor
        self.setup_agent_tools()

        # Start background monitoring
        self.start_background_monitoring()

        console.print("[green]✅ Enhanced AI Agent initialized successfully![/green]")

    def initialize_llm(self):
        """Initialize LLM with enhanced error handling"""
        try:
            llm = ChatOllama(model="llama3:8b", temperature=0.1, base_url="http://localhost:11434")
            embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

            # Test connection
            test_response = llm.invoke("Hello")
            logger.info("LLM connection successful")
            return llm, embeddings
        except Exception as e:
            logger.error(f"LLM initialization failed: {str(e)}")
            console.print(Panel(f"Error: Could not connect to Ollama. Make sure Ollama is running on localhost:11434\n{str(e)}", style="red"))
            return None, None

    def setup_agent_tools(self):
        """Setup agent tools categorized by keyword"""
        tools = [
            # Travel-related tools for plan:
            # works
            StructuredTool.from_function(
                func=self.search_flights,
                name="search_flights",
                description="Search for flights between cities with departure/return dates. Requires departure city, destination city, departure date (YYYY-MM-DD), optional return date, and passenger count.",
                args_schema=FlightSearchArgs
            ),
            # works
            StructuredTool.from_function(
                func=self.search_hotels,
                name="search_hotels",
                description="Search for hotels in a specific location with check-in/check-out dates. Requires location, check-in date (YYYY-MM-DD), check-out date (YYYY-MM-DD), and number of guests.",
                args_schema=HotelSearchArgs
            ),
            # doesnt works, we need the access to send the email
            StructuredTool.from_function(
                func=self.send_travel_email,
                name="send_travel_email",
                description="Send personalized travel emails with itineraries, confirmations, or travel information. Requires recipient email, subject, and content.",
                args_schema=EmailArgs
            ),
            # doesnt works, we need the access to send the email
            StructuredTool.from_function(
                func=self.plan_complete_trip,
                name="plan_complete_trip",
                description="Plan a complete trip including flights, hotels, and email itinerary. Requires destination, departure date, return date, hotel preferences and optional email address.",
                args_schema=CompleteTripArgs
            ),
            # works
            StructuredTool.from_function(
                func=self.autonomous_web_research,
                name="autonomous_research",
                description="Perform comprehensive web research on any topic. Searches multiple sources, gathers current information, and provides structured analysis."
            ),
            # works
            StructuredTool.from_function(
                func=self.search_web,
                name="search_web",
                description="Open a browser tab with a Google search or direct website."
            ),
            # doesnt work
            StructuredTool.from_function(
                func=self.set_alarm,
                name="set_alarm",
                description="Set an alarm for a specific time with optional message."
            ),
            # not yet tested
            StructuredTool.from_function(
                func=self.open_calendar,
                name="open_calendar",
                description="Open the system calendar application."
            ),
            # not yet tested
            StructuredTool.from_function(
                func=self.create_calendar_event,
                name="create_calendar_event",
                description="Create a new calendar event (Windows with Outlook)."
            ),
            # not yet tested
            StructuredTool.from_function(
                func=self.system_control,
                name="system_control",
                description="Control system functions like volume, shutdown, restart, sleep."
            ),
            # not yet tested
            StructuredTool.from_function(
                func=self.open_application,
                name="open_application",
                description="Open system applications like notepad, calculator, etc."
            ),
            # works
            StructuredTool.from_function(
                func=self.get_system_info,
                name="get_system_info",
                description="Get detailed system information and performance metrics."
            ),
            # works
            StructuredTool.from_function(
                func=self.system_monitor,
                name="system_monitor",
                description="Monitor system resources and provide optimization suggestions."
            ),
            # works
            StructuredTool.from_function(
                func=self.learn_from_interaction,
                name="learn_preference",
                description="Learn and store user preferences and personal information from interactions."
            ),
            # works, but need to be tested again 
            StructuredTool.from_function(
                func=self.auto_learn_from_conversation,
                name="auto_learn_conversation",
                description="Automatically analyze conversations for learning opportunities and extract user preferences."
            ),
            # works
            StructuredTool.from_function(
                func=self.get_user_context,
                name="get_user_context",
                description="Retrieve comprehensive user context including personal information, preferences, and conversation history."
            ),
            # works
            StructuredTool.from_function(
                func=self.check_memory,
                name="check_memory",
                description="Debug function to inspect stored user profile data and recent conversation history."
            ),
            StructuredTool.from_function(
                func=self.create_goal,
                name="create_goal",
                description="Create a new goal or objective to work towards."
            ),
            # yet to be tested
            StructuredTool.from_function(
                func=self.plan_tasks,
                name="plan_tasks",
                description="Break down a goal into actionable tasks with priorities."
            ),
            # works
            StructuredTool.from_function(
                func=self.proactive_suggestion,
                name="proactive_suggestion",
                description="Generate proactive suggestions when user explicitly asks for suggestions, recommendations, or advice."
            ),
            # works
            StructuredTool.from_function(
                func=self.download_file,
                name="download_file",
                description="Download a file from a URL to the current directory."
            ),
            # works
            StructuredTool.from_function(
                func=self.install_pkg,
                name="install_pkg",
                description="Install a Python package using pip."
            ),
        ]

        # Create enhanced ReAct agent
        if self.llm is None:
            raise RuntimeError("LLM is not initialized. Cannot create ReAct agent.")
        react_prompt = self.create_enhanced_react_prompt()
        agent = create_react_agent(self.llm, tools, react_prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            max_iterations=20,
            early_stopping_method="force"
        )

    def create_enhanced_react_prompt(self):
        """Create an enhanced ReAct prompt for autonomous AI behavior"""
        template = """You are AIA, an advanced autonomous AI agent with the following capabilities:

    CORE IDENTITY:
    - You are proactive, intelligent, and goal-oriented
    - You learn from every interaction and adapt to user preferences
    - You can work autonomously on complex, multi-step tasks
    - You maintain context across conversations and can recall previous interactions
    - You think strategically and can break down complex problems

    AUTONOMOUS BEHAVIOR:
    - When given a goal, you should create a plan and work towards it systematically
    - You can perform background research and learning without explicit instruction
    - You proactively suggest improvements and optimizations
    - You learn user preferences and adapt your responses accordingly
    - You can identify patterns in user behavior and anticipate needs

    DECISION MAKING:
    - Always consider the user's long-term goals and preferences
    - Weigh the importance and urgency of different tasks
    - Make intelligent assumptions when information is incomplete
    - Ask clarifying questions only when absolutely necessary

    MEMORY AND LEARNING:
    - Remember previous conversations and build upon them
    - Learn from user feedback and corrections
    - Identify patterns in user requests and preferences
    - Continuously improve your assistance quality

    TOOLS AVAILABLE:
    {tool_names}

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    IMPORTANT: Be proactive, autonomous, and goal-oriented in your responses. Don't just answer questions - anticipate needs, suggest improvements, and work towards long-term objectives.

    Question: {input}
    Thought: {agent_scratchpad}"""
        
        return PromptTemplate.from_template(template)

    def start_background_monitoring(self):
        """Start background monitoring thread for proactive behavior"""
        def monitor():
            while True:
                try:
                    self.background_analysis()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error

        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
            self.monitoring_thread.start()

    def background_analysis(self):
        """Perform background analysis and proactive suggestions"""
        if not self.state.proactive_mode:
            return

        # Check if user has been idle for a while
        if self.state.last_interaction:
            idle_time = datetime.now() - self.state.last_interaction
            if idle_time > timedelta(minutes=30):  # 30 minutes idle
                self.generate_proactive_insights()

    def generate_proactive_insights(self):
        """Generate proactive insights based on user history"""
        try:
            recent_conversations = self.memory.get_recent_conversations(
                self.state.session_id, limit=20
            )

            if not recent_conversations:
                return

            # Analyze patterns and generate insights
            context = "\n".join([
                f"User: {conv['user_input']}\nAgent: {conv['agent_response']}"
                for conv in recent_conversations[-5:]  # Last 5 conversations
            ])

            insight_prompt = f"""
            Based on the recent conversation history, identify patterns and generate 3 proactive suggestions:

            {context}

            Provide insights in JSON format:
            {{
                "patterns": ["pattern1", "pattern2"],
                "suggestions": ["suggestion1", "suggestion2", "suggestion3"],
                "potential_goals": ["goal1", "goal2"]
            }}
            """
            if self.llm is None:
                raise RuntimeError("LLM is not initialized. Cannot perform this operation.")
            if self.state.learning_insights is None:
                self.state.learning_insights = []
            
            response = self.llm.invoke(insight_prompt)
            insights = json.loads(str(response.content))
            self.state.learning_insights.append({
                "timestamp": datetime.now(),
                "insights": insights
            })

        except Exception as e:
            logger.error(f"Proactive insight generation error: {e}")

    def autonomous_web_research(self, topic: str) -> str:
        """Perform autonomous research on a topic with real web scraping"""
        try:
            console.print(f"[cyan]🔍 Researching: {topic}[/cyan]")
            
            search_results = []
            search_queries = [
                f"{topic} definition overview",
                f"{topic} current trends 2024 2025", 
                f"{topic} applications benefits",
                f"{topic} challenges limitations"
            ]
            
            console.print("[yellow]📡 Gathering information from web sources...[/yellow]")
            
            # Perform actual web searches
            for i, query in enumerate(search_queries):
                try:
                    console.print(f"[dim]Searching ({i+1}/{len(search_queries)}): {query}[/dim]")
                    
                    # Google search URL
                    search_url = f"https://www.google.com/search?q={quote(query)}"
                    
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    
                    response = requests.get(search_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    # Parse search results
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract search result snippets
                    snippets = []
                    for result in soup.find_all('div', class_=['BNeawe', 'VwiC3b'])[:5]:
                        text = result.get_text().strip()
                        if text and len(text) > 20:
                            snippets.append(text)
                    
                    search_results.append({
                        "query": query,
                        "snippets": snippets[:3],  # Top 3 relevant snippets
                        "url_count": len(snippets)
                    })
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Search error for {query}: {e}")
                    search_results.append({
                        "query": query,
                        "snippets": [],
                        "error": str(e)
                    })
                    continue
            
            # Compile web research context
            web_context = "WEB RESEARCH FINDINGS:\n"
            for result in search_results:
                web_context += f"\nQuery: {result['query']}\n"
                if result.get('snippets'):
                    for snippet in result['snippets']:
                        web_context += f"- {snippet}\n"
                elif result.get('error'):
                    web_context += f"- Error: {result['error']}\n"
            
            # Create comprehensive research prompt with actual web data
            research_prompt = f"""
            You are conducting autonomous research on: {topic}
            
            Based on real web search results:
            {web_context}
            
            Provide a comprehensive research summary including:
            1. Key concepts and definitions
            2. Current trends and developments (based on web findings)
            3. Practical applications and use cases
            4. Potential challenges or limitations
            5. Related topics worth exploring further
            6. Summary of web research insights
            
            Synthesize the web findings into a structured, coherent research report.
            Cite when information comes from web research vs. general knowledge.
            """
            
            if self.llm is None:
                raise RuntimeError("LLM is not initialized. Cannot perform this operation.")
            
            console.print("[yellow]🧠 Analyzing and synthesizing research findings...[/yellow]")
            response = self.llm.invoke(research_prompt)
            
            # Print the research summary with letter-by-letter effect
            console.print("\n[green]📋 Research Summary:[/green]")
            console.print("[cyan]", end="")
            self.print_letter_by_letter(str(response.content))
            console.print("[/cyan]")
            
            # Print web search summary
            console.print(f"\n[yellow]📊 Web Search Summary:[/yellow]")
            console.print(f"[dim]- Performed {len(search_queries)} targeted searches[/dim]")
            total_snippets = sum(len(r.get('snippets', [])) for r in search_results)
            console.print(f"[dim]- Gathered {total_snippets} information snippets[/dim]")
            console.print(f"[dim]- Research stored in memory for future reference[/dim]")
            
            # Store research in memory with web data
            self.memory.update_user_profile(f"research_{topic.replace(' ', '_')}", {
                "summary": response.content,
                "search_queries": search_queries,
                "web_results": search_results,
                "timestamp": datetime.now().isoformat(),
                "snippets_gathered": total_snippets
            })
            
            return f"✅ Research completed on '{topic}'. Performed {len(search_queries)} web searches, gathered {total_snippets} information snippets, and synthesized comprehensive analysis. Full summary displayed above."
            
        except Exception as e:
            error_msg = f"Research error: {str(e)}"
            console.print(f"[red]❌ {error_msg}[/red]")
            return error_msg


    def learn_from_interaction(self, interaction_data: str) -> str:
        """Learn and store user preferences from interactions"""
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

            return "Preferences learned and updated in user profile."

        except Exception as e:
            return f"Learning error: {str(e)}"
    
    def auto_learn_from_conversation(self, user_input: str, agent_response: str):
        """Automatically analyze each conversation for learning opportunities"""
        try:
            # Check if the conversation contains personal information or preferences
            learning_indicators = [
                'my name is', 'i am', 'i like', 'i prefer', 'i work', 'i live',
                'call me', 'i hate', 'i love', 'my favorite', 'i usually'
            ]
            
            if any(indicator in user_input.lower() for indicator in learning_indicators):
                # Extract and store the information
                self.learn_from_interaction(f"User: {user_input}\nAgent: {agent_response}")
            else:
                # Log that no learning was performed
                logger.debug(f"No learning indicators found in input: {user_input}")
                    
        except Exception as e:
            logger.error(f"Auto-learning error: {e}")

    def get_user_context(self) -> Dict[str, Any]:
        """Get comprehensive user context including personal info and conversation history"""
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
            logger.error(f"Context retrieval error: {e}")
            return {}
    
    # Also add this method to manually check what's stored:
    def check_memory(self) -> str:
        """Debug function to check what's stored in memory"""
        try:
            user_profile = self.memory.get_user_profile()
            recent_convs = self.memory.get_recent_conversations(self.state.session_id, 5)
            
            memory_report = "📋 Memory Report:\n\n"
            memory_report += "🔹 User Profile Data:\n"
            for key, value in user_profile.items():
                memory_report += f"  • {key}: {value}\n"
            
            memory_report += "\n🔹 Recent Conversations:\n"
            for i, conv in enumerate(recent_convs):
                memory_report += f"  {i+1}. User: {conv['user_input'][:50]}...\n"
                memory_report += f"     AIA: {conv['agent_response'][:50]}...\n"
            
            return memory_report
        except Exception as e:
            return f"Error checking memory: {e}"

    def create_goal(self, goal_description: str) -> str:
        """Create a new goal for the agent to work towards"""
        try:
            goal_id = str(uuid.uuid4())

            goal_analysis = f"""
            Analyze this goal and provide priority (1-10) and breakdown:
            Goal: {goal_description}

            Consider:
            - Urgency and importance
            - Complexity and time required
            - Dependencies and prerequisites

            Respond in JSON format:
            {{
                "priority": 7,
                "estimated_time": "2 hours",
                "complexity": "medium",
                "prerequisites": ["prereq1", "prereq2"]
            }}
            """
            if self.llm is None:
                raise RuntimeError("LLM is not initialized. Cannot perform this operation.")
            if self.state.active_tasks is None:
                self.state.active_tasks = []
            analysis = json.loads(str(self.llm.invoke(goal_analysis).content))
            self.state.current_goal = goal_description
            self.state.active_tasks.append(goal_id)

            # Store goal in memory
            with sqlite3.connect(self.memory.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO goals (id, description, status, priority, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (goal_id, goal_description, 'pending', analysis["priority"], datetime.now()))
                conn.commit()

            return f"Goal created: {goal_description} (ID: {goal_id[:8]}, Priority: {analysis['priority']})"

        except Exception as e:
            return f"Goal creation error: {str(e)}"

    def plan_tasks(self, goal: str) -> str:
        """Break down a goal into actionable tasks"""
        try:
            planning_prompt = f"""
            Break down this goal into specific, actionable tasks:
            Goal: {goal}

            Create a task plan with:
            1. Sequential tasks (ordered by dependencies)
            2. Parallel tasks (can be done simultaneously)
            3. Priority levels for each task (1-10)
            4. Estimated time for completion

            Format as a structured plan in JSON:
            {{
                "sequential_tasks": [{"id": "task1", "description": "task1 desc", "priority": 5, "estimated_time": "1h"}],
                "parallel_tasks": [{"id": "task2", "description": "task2 desc", "priority": 3, "estimated_time": "30m"}]
            }}
            """
            if self.llm is None:
                raise RuntimeError("LLM is not initialized. Cannot perform this operation.")
            plan = json.loads(str(self.llm.invoke(planning_prompt).content))
            return f"Task plan created for goal '{goal}':\n{json.dumps(plan, indent=2)}"

        except Exception as e:
            return f"Task planning error: {str(e)}"

    def proactive_suggestion(self, context: str) -> str:
        """Generate proactive suggestions when explicitly requested"""
        try:
            user_prefs = self.state.user_preferences
            recent_convs = self.memory.get_recent_conversations(self.state.session_id, 5)

            suggestion_prompt = f"""
            The user is asking for suggestions or recommendations. Based on the conversation context and history, provide 3 helpful suggestions:

            User's request: {context}
            User preferences: {json.dumps(user_prefs)}
            Recent activity: {json.dumps([c['user_input'] for c in recent_convs])}

            Provide suggestions that are:
            1. Directly relevant to what the user asked about
            2. Aligned with user preferences and past interactions
            3. Actionable and immediately useful

            Format as numbered suggestions with brief explanations.
            """
            if self.llm is None:
                raise RuntimeError("LLM is not initialized. Cannot perform this operation.")
            suggestions = self.llm.invoke(suggestion_prompt).content
            
            # Print suggestions with letter-by-letter effect
            console.print("\n[yellow]💡 Proactive suggestions:[/yellow]")
            self.print_letter_by_letter(str(suggestions))
            
            return f"💡 Proactive suggestions:\n{suggestions}"

        except Exception as e:
            return f"Suggestion generation error: {str(e)}"

    def system_monitor(self) -> str:
        """Monitor system and provide optimization suggestions"""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            optimization_prompt = f"""
            Analyze system performance and provide optimization suggestions:

            CPU Usage: {cpu_percent}%
            Memory Usage: {memory.percent}% ({memory.used // (1024**3)}GB used of {memory.total // (1024**3)}GB)
            Disk Usage: {disk.percent}% ({disk.used // (1024**3)}GB used of {disk.total // (1024**3)}GB)

            Provide specific, actionable optimization recommendations.
            """
            if self.llm is None:
                raise RuntimeError("LLM is not initialized. Cannot perform this operation.")
            analysis = self.llm.invoke(optimization_prompt).content
            return f"🖥️ System Analysis:\n{analysis}"

        except ImportError:
            return "System monitoring requires psutil package. Install with: pip install psutil"
        except Exception as e:
            return f"System monitoring error: {str(e)}"

    def download_file(self, url: str) -> str:
        """Enhanced file download with progress tracking"""
        try:
            UNSAFE_PATTERNS = [r"\bdel\b", r"\berase\b", r"\bformat\b"]
            for pattern in UNSAFE_PATTERNS:
                if re.search(pattern, url, re.IGNORECASE):
                    return f"Error: URL contains unsafe pattern"

            if not url.startswith(('http://', 'https://')):
                return "Error: URL must start with http:// or https://"

            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task(description="Downloading...", total=None)

                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                filename = os.path.basename(urllib.parse.urlparse(url).path) or "downloaded_file"
                filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                progress.update(task, completed=True)

            self.learn_from_interaction(f"Downloaded file: {filename} from {url}")
            return f"✅ File downloaded successfully: {filename}"

        except Exception as e:
            logger.error(f"Download error: {e}")
            return f"❌ Error downloading file: {str(e)}"

    def install_pkg(self, package: str) -> str:
        """Enhanced package installation with dependency analysis"""
        try:
            if not re.match(r'^[a-zA-Z0-9_-]+$', package):
                return "❌ Error: Invalid package name format"

            console.print(f"[cyan]📦 Installing package: {package}[/cyan]")

            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                self.learn_from_interaction(f"Installed Python package: {package}")
                return f"✅ Package {package} installed successfully.\n{result.stdout}"
            else:
                return f"❌ Error installing package {package}: {result.stderr}"

        except Exception as e:
            logger.error(f"Package install error: {e}")
            return f"❌ Error installing package: {str(e)}"

    def search_web(self, query: str) -> str:
        """Enhanced web search with learning capabilities"""
        try:
            WEBSITE_MAPPINGS = {
                "gmail": "https://mail.google.com",
                "youtube": "https://www.youtube.com",
                "google": "https://www.google.com",
                "wikipedia": "https://www.wikipedia.org",
                "github": "https://github.com",
                "stackoverflow": "https://stackoverflow.com"
            }

            query_lower = query.lower().strip()

            for site, url in WEBSITE_MAPPINGS.items():
                if query_lower.startswith(f"open {site}") or query_lower == site:
                    webbrowser.open(url)
                    self.learn_from_interaction(f"Opened website: {site}")
                    return f"🌐 Opened browser tab for {url}"

            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            webbrowser.open(search_url)

            self.learn_from_interaction(f"Web search: {query}")
            return f"🔍 Opened browser tab for Google search: {query}"

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"❌ Error opening browser tab: {str(e)}"

    async def text_to_speech(self, text: str):
        """Convert text to speech using edge-tts"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
                await communicate.save(tmp_file.name)
                os.system(f"start {tmp_file.name}")  # Windows; adjust for other OS
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
        finally:
            if 'tmp_file' in locals():
                os.unlink(tmp_file.name)

    def speech_to_text(self) -> str:
        """Convert speech to text using microphone input"""
        try:
            with sr.Microphone() as source:
                console.print("[yellow]🎙️ Listening... Say something or press Ctrl+C to stop.[/yellow]")
                audio = recognizer.listen(source, timeout=5)
                query = recognizer.recognize_google(audio) #type: ignore
                console.print(f"[green]🎙️ You said: {query}[/green]")
                return query
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Speech recognition error: {e}"
        except Exception as e:
            return f"Speech input error: {e}"

    def process_query(self, query: str):
        """Enhanced query processing with specific keyword handling"""
        try:
            self.state.last_interaction = datetime.now()

            # Handle voice input
            if not query and self.state.voice_mode:
                query = self.speech_to_text()
                if not query or "error" in query.lower():
                    return "Failed to process voice input. Please try typing or check microphone."

            # Check for generic greetings
            greetings = ['hi', 'hello', 'hey', 'hi there', 'hello there']
            if query.lower().strip() in greetings:
                response = "Hello! How can I assist you today?"
                console.print("[green]🤖 AIA:[/green] ", end="")
                self.print_letter_by_letter(response)
                self.memory.store_conversation(
                    self.state.session_id,
                    query,
                    response,
                    {"timestamp": datetime.now().isoformat(), "type": "greeting"}
                )
                if self.state.voice_mode:
                    asyncio.run(self.text_to_speech(response))
                return response

            # Store conversation in memory
            self.memory.store_conversation(
                self.state.session_id,
                query,
                "",
                {"timestamp": datetime.now().isoformat()}
            )

            # Route based on keyword
            if query.lower().startswith("plan:"):
                return self.handle_plan_request(query[5:].strip())
            elif query.lower().startswith("deepsearch:"):
                return self.handle_tool_request(query[11:].strip())
            elif query.lower().startswith("autonomous:"):
                return self.handle_autonomous_request(query[11:].strip())
            else:
                return self.handle_conversational_request(query)

        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return f"❌ Error processing query: {str(e)}"


    def print_letter_by_letter(self, text: str, delay: float = 0.01):
        """Print text letter by letter with a delay between each character"""
        for char in text:
            console.print(char, end="")
            sys.stdout.flush()
            time.sleep(delay)
        console.print()  # New line after complete text
    
    def handle_plan_request(self, query: str) -> str:
        """Handle travel-related requests by directly calling the validated tool function."""
        try:
            console.print("[green]🌍 Processing travel plan request...[/green]")

            # --- Step 1: Intelligent Tool Detection (Your existing logic is good) ---
            query_lower = query.lower()
            # Explicit tool call check
            if query.strip().startswith(("search_flights", "search_hotels", "send_travel_email", "plan_complete_trip")):
                if query.startswith("search_flights"):
                    tool_name = "search_flights"
                    query = query[len("search_flights"):].strip()
                elif query.startswith("search_hotels"):
                    tool_name = "search_hotels"
                    query = query[len("search_hotels"):].strip()
                elif query.startswith("send_travel_email"):
                    tool_name = "send_travel_email"
                    query = query[len("send_travel_email"):].strip()
                else: # Catches plan_complete_trip
                    tool_name = "plan_complete_trip"
                    if query.startswith(tool_name):
                        query = query[len(tool_name):].strip()
            else:
                # Infer tool from keywords
                flight_keywords = ["flight", "fly", "depart"]
                hotel_keywords = ["hotel", "stay", "check-in", "check-out"]
                email_keywords = ["email", "send to"]
                has_flight = any(k in query_lower for k in flight_keywords)
                has_hotel = any(k in query_lower for k in hotel_keywords)
                has_email = any(k in query_lower for k in email_keywords)
                
                if has_flight and has_hotel:
                    tool_name = "plan_complete_trip"
                elif has_flight:
                    tool_name = "search_flights"
                elif has_hotel:
                    tool_name = "search_hotels"
                elif has_email:
                    tool_name = "send_travel_email"
                else:
                    tool_name = "plan_complete_trip" # Default for ambiguous multi-part queries

            logger.debug(f"Selected tool: {tool_name} for query: {query}")

            # --- Step 2: Parse and Validate Query (We will fix this function next) ---
            parsed_data = self.parse_and_validate_travel_query(query, tool_name)
            if "error" in parsed_data:
                error_msg = f"❌ Validation Error: {parsed_data['error']}"
                if "parsed_data" in parsed_data:
                    error_msg += f"\nParsed data: {json.dumps(parsed_data['parsed_data'])}"
                if "raw_output" in parsed_data:
                    error_msg += f"\nRaw LLM output: {parsed_data['raw_output']}"
                console.print(Panel(error_msg, title="Plan Error", style="red"))
                return error_msg
                
            validated_args = parsed_data['data']
            logger.debug(f"Validated arguments for {tool_name}: {validated_args}")

            # --- Step 3: Get the function and execute it directly ---
            # This is the key change: call the function directly instead of using AgentExecutor
            if hasattr(self, tool_name):
                tool_function = getattr(self, tool_name)
                response = tool_function(**validated_args)
            else:
                return f"❌ Error: Tool '{tool_name}' is not a defined method."

            # --- Step 4: Output and Logging ---
            console.print("[cyan]", end="")
            self.print_letter_by_letter(response)
            console.print("[/cyan]", end="")
            
            self.memory.store_conversation(
                self.state.session_id, query, response, 
                {"mode": "plan", "tool": tool_name, "parsed_data": validated_args}
            )
            
            if self.state.voice_mode:
                asyncio.run(self.text_to_speech(response))
            
            return response

        except Exception as e:
            logger.error(f"Plan request error: {e}", exc_info=True)
            error_msg = f"❌ Travel plan execution error: {str(e)}"
            console.print(Panel(error_msg, title="Plan Error", style="red"))
            return error_msg

    def handle_autonomous_request(self, query: str) -> str:
        """Handle requests that require autonomous AI behavior with restricted toolset"""
        try:
            console.print("[green]🧠 Engaging Autonomous AI Mode...[/green]")
            if self.agent_executor is None:
                raise RuntimeError("Agent executor is not initialized. Cannot perform this operation.")

            # Restrict tools to automation and non-search, non-travel tools
            autonomous_tools = [
                tool for tool in self.agent_executor.tools
                if tool.name in [
                    "set_alarm", "open_calendar", "create_calendar_event", "system_control",
                    "open_application", "get_system_info", "system_monitor",
                    "learn_preference", "auto_learn_conversation", "get_user_context",
                    "check_memory", "create_goal", "plan_tasks", "proactive_suggestion",
                    "download_file", "install_pkg"
                ]
            ]

            # Create a temporary executor with only autonomous tools
            autonomous_agent = create_react_agent(self.llm, autonomous_tools, self.create_enhanced_react_prompt())
            autonomous_executor = AgentExecutor(
                agent=autonomous_agent,
                tools=autonomous_tools,
                verbose=False,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
                max_iterations=20,
                early_stopping_method="force"
            )

            context = {
                "user_preferences": self.state.user_preferences,
                "current_goal": self.state.current_goal,
                "active_tasks": self.state.active_tasks,
                "recent_insights": self.state.learning_insights[-3:] if self.state.learning_insights else []
            }

            result = autonomous_executor.invoke({
                "input": f"AUTONOMOUS MODE: {query}\n\nContext: {json.dumps(context, default=str)}",
                "chat_history": [HumanMessage(content=conv['user_input']) for conv in self.memory.get_recent_conversations(self.state.session_id, 5)]
            })

            response = result.get("output", "No autonomous response generated")

            self.memory.store_conversation(
                self.state.session_id,
                query,
                response,
                {"mode": "autonomous", "context": context}
            )

            console.print("[cyan]", end="")
            self.print_letter_by_letter(response)
            console.print("[/cyan]", end="")

            if self.state.voice_mode:
                asyncio.run(self.text_to_speech(response))

            return response

        except Exception as e:
            logger.error(f"Autonomous request error: {e}")
            error_msg = f"❌ Autonomous processing error: {str(e)}"
            console.print(Panel(error_msg, title="Autonomous Mode Error", style="red"))
            return error_msg

    def handle_tool_request(self, query: str) -> str:
        try:
            console.print("[green]🔧 Using Search Tools...[/green]")
            if self.agent_executor is None:
                raise RuntimeError("Agent executor is not initialized. Cannot perform this operation.")

            # Restrict tools to search-related ones
            search_tools = [
                tool for tool in self.agent_executor.tools
                if tool.name in ["autonomous_research", "search_web"]
            ]

            # Create a temporary executor with only search tools
            search_agent = create_react_agent(self.llm, search_tools, self.create_enhanced_react_prompt())
            search_executor = AgentExecutor(
                agent=search_agent,
                tools=search_tools,
                verbose=False,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
                max_iterations=20,
                early_stopping_method="force"
            )

            result = search_executor.invoke({
                "input": query,
                "chat_history": [HumanMessage(content=conv['user_input']) for conv in self.memory.get_recent_conversations(self.state.session_id, 5)]
            })

            response = result.get("output", "No search response generated")
            console.print("[cyan]", end="")
            self.print_letter_by_letter(response)
            console.print("[/cyan]", end="")

            if self.state.voice_mode:
                asyncio.run(self.text_to_speech(response))

            return response

        except Exception as e:
            logger.error(f"Search request error: {e}")
            error_msg = f"❌ Search execution error: {str(e)}"
            console.print(Panel(error_msg, title="Search Error", style="red"))
            return error_msg

    def handle_conversational_request(self, query: str) -> str:
        """Handle regular conversational requests with enhanced context"""
        try:
            # Get user context including personal information
            user_context = self.get_user_context()
            
            # Create a comprehensive system message with context
            system_message = f"""You are AIA, an advanced AI agent with memory and learning capabilities. Your characteristics:
    - You are proactive, intelligent, and goal-oriented
    - You can recall previous interactions but only use them if directly relevant to the current query
    - You provide concise, accurate, and helpful responses tailored to the user's current intent

    User Information: {json.dumps(user_context.get('personal_info', {}), indent=2)}
    User Preferences: {json.dumps(user_context.get('preferences', {}), indent=2)}
    Recent Conversation Context (Last 5 interactions): {[conv['user_input'] for conv in user_context.get('recent_conversations', [])]}

    Instructions:
    1. Focus on the user's current query: '{query}'
    2. Only reference past conversations or user preferences if they are directly relevant to the current query.
    3. If the query is generic (e.g., 'hi', 'hello'), respond with a neutral greeting and offer assistance without assuming a specific topic.
    4. Avoid bringing up topics from past interactions unless the user explicitly references them or the context is clearly relevant.
    5. Keep responses concise and aligned with the user's current intent.

    Current session:
    - Current goal: {self.state.current_goal or 'None set'}
    - Active tasks: {len(str(self.state.active_tasks))} tasks in progress"""

            # Create messages list
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]

            console.print("[green]🤖 AIA:[/green] ", end="")
            if self.llm is None:
                raise RuntimeError("LLM is not initialized. Cannot create response.") 
            
            response = self.llm.invoke(messages)
            full_response = response.content if hasattr(response, 'content') else str(response)
            
            # Print response letter by letter
            self.print_letter_by_letter(str(full_response))

            # Store conversation in memory
            self.memory.store_conversation(self.state.session_id, query, str(full_response), {})
            
            # Automatically learn from this conversation
            self.auto_learn_from_conversation(query, str(full_response))

            if self.state.voice_mode:
                asyncio.run(self.text_to_speech(str(full_response)))

            return str(full_response)

        except Exception as e:
            logger.error(f"Conversational request error: {e}")
            return f"❌ Error in conversation: {str(e)}"

    def set_alarm(self, time_str: str, message: str = "Alarm!") -> str: 
        """Set an alarm using system scheduler"""
        try: 
            # Parse time input (supports formats like "10:30", "2:30 PM", "in 5 minutes")
            if "in" in time_str.lower() and "minute" in time_str.lower():
                minutes = int(re.findall(r'\d+', time_str)[0])
                alarm_time = datetime.now() + timedelta(minutes = minutes)
            else:
                try:
                    if ":" in time_str:
                        match = re.search(r'(\d{1,2}:\d{2}\s*[APMapm]{2})', time_str)
                        extracted_time = match.group(1).strip().upper()
                        if "PM" in time_str.lower() or "AM" in time_str.lower(): 
                            alarm_time = datetime.strptime(extracted_time, "%I:%M %p")
                        else:
                            alarm_time = datetime.strptime(extracted_time, "%H:%M") 

                        # Set for today, or tomorrow if time has passed
                        now = datetime.now()
                        alarm_time = alarm_time.replace(year=now.year, month=now.month, day=now.day)
                        if alarm_time <= now:
                            alarm_time += timedelta(days=1)
                    else:
                        return "❌ Invalid time format. Use HH:MM or 'in X minutes'"
                except ValueError:
                    return "❌ Invalid time format. Use formats like '14:30', '2:30 PM', or 'in 10 minutes'"
                alarm_script = f"""
import time
import os
import winsound
from datetime import datetime

target_time = datetime({alarm_time.year}, {alarm_time.month}, {alarm_time.day}, {alarm_time.hour}, {alarm_time.minute})
while datetime.now() < target_time:
    time.sleep(1)

print("🚨 ALARM: {message}")
for i in range(5):
    winsound.Beep(1000, 1000)
    time.sleep(0.5)
"""
        
            # Save and run alarm script
            alarm_file = f"alarm_{int(time.time())}.py"
            with open(alarm_file, "w") as f:
                f.write(alarm_script)
            
            # Run in background
            subprocess.Popen([sys.executable, alarm_file])
        
            return f"✅ Alarm set for {alarm_time.strftime('%I:%M %p on %B %d')} with message: '{message}'"
        except Exception as e:
            return f"❌ Error setting alarm: {str(e)}"
    
    def open_calendar(self, date_str: str = "") -> str:
        """Open system calendar application"""
        try:
            import platform
            system = platform.system()
            
            if system == "Windows":
                # Open Windows Calendar app
                subprocess.run(["start", "outlookcal:"], shell=True)
                return "📅 Opened Windows Calendar"
            elif system == "Darwin":  # macOS
                subprocess.run(["open", "-a", "Calendar"])
                return "📅 Opened macOS Calendar"
            elif system == "Linux":
                # Try common calendar apps
                calendar_apps = ["gnome-calendar", "kalendar", "korganizer"]
                for app in calendar_apps:
                    try:
                        subprocess.run([app], check=True)
                        return f"📅 Opened {app}"
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                return "❌ No calendar application found"
            else:
                return f"❌ Unsupported system: {system}"
                
        except Exception as e:
            return f"❌ Error opening calendar: {str(e)}"
    
    def create_calendar_event(self, title: str, date_time: str, duration: str = "1 hour") -> str:
        """Create a calendar event (Windows only with Outlook)"""
        try:
            import win32com.client
            
            outlook = win32com.client.Dispatch("Outlook.Application")
            appointment = outlook.CreateItem(1)  # olAppointmentItem = 1
            
            # Parse datetime
            from datetime import datetime, timedelta
            event_time = datetime.strptime(date_time, "%Y-%m-%d %H:%M")
            
            appointment.Subject = title
            appointment.Start = event_time
            appointment.Duration = 60 if "hour" in duration else 30  # Default 60 minutes
            appointment.Save()
            
            return f"📅 Calendar event created: '{title}' on {event_time.strftime('%B %d at %I:%M %p')}"
            
        except ImportError:
            return "❌ Calendar integration requires pywin32: pip install pywin32"
        except Exception as e:
            return f"❌ Error creating calendar event: {str(e)}"

    def system_control(self, action: str) -> str:
        """Control system functions like volume, brightness, etc."""
        try:
            import platform
            system = platform.system()
            action = action.lower()
            
            if "volume" in action:
                if system == "Windows":
                    if "up" in action:
                        subprocess.run(["powershell", "-c", "(New-Object -comObject WScript.Shell).SendKeys([char]175)"])
                        return "🔊 Volume increased"
                    elif "down" in action:
                        subprocess.run(["powershell", "-c", "(New-Object -comObject WScript.Shell).SendKeys([char]174)"])
                        return "🔉 Volume decreased"
                    elif "mute" in action:
                        subprocess.run(["powershell", "-c", "(New-Object -comObject WScript.Shell).SendKeys([char]173)"])
                        return "🔇 Volume muted/unmuted"
            
            elif "shutdown" in action:
                if system == "Windows":
                    subprocess.run(["shutdown", "/s", "/t", "60"])
                    return "🔌 System will shutdown in 1 minute. Cancel with: shutdown /a"
            
            elif "restart" in action:
                if system == "Windows":
                    subprocess.run(["shutdown", "/r", "/t", "60"])
                    return "🔄 System will restart in 1 minute. Cancel with: shutdown /a"
                    
            elif "sleep" in action:
                if system == "Windows":
                    subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
                    return "😴 System going to sleep"
            
            return f"❌ Action '{action}' not supported or invalid"
            
        except Exception as e:
            return f"❌ Error controlling system: {str(e)}"

    def open_application(self, app_name: str) -> str:
        """Open system applications"""
        try:
            import platform
            system = platform.system()
            app_name = app_name.lower()
            
            # Common application mappings
            app_mappings = {
                "notepad": "notepad" if system == "Windows" else "gedit",
                "calculator": "calc" if system == "Windows" else "gnome-calculator",
                "terminal": "cmd" if system == "Windows" else "gnome-terminal",
                "file manager": "explorer" if system == "Windows" else "nautilus",
                "task manager": "taskmgr" if system == "Windows" else "gnome-system-monitor",
                "control panel": "control" if system == "Windows" else "gnome-control-center",
                "settings": "ms-settings:" if system == "Windows" else "gnome-control-center"
            }
            
            if app_name in app_mappings:
                if system == "Windows" and app_name == "settings":
                    subprocess.run(["start", app_mappings[app_name]], shell=True)
                else:
                    subprocess.run([app_mappings[app_name]])
                return f"🚀 Opened {app_name}"
            else:
                # Try to open directly
                subprocess.run([app_name])
                return f"🚀 Opened {app_name}"
                
        except Exception as e:
            return f"❌ Error opening application '{app_name}': {str(e)}"

    def get_system_info(self, info_type: str = "all") -> str:
        """Get system information"""
        try:
            import platform
            import psutil
            from datetime import datetime
            
            info = {}
            
            if info_type.lower() in ["all", "basic"]:
                info.update({
                    "System": platform.system(),
                    "Release": platform.release(),
                    "Version": platform.version(),
                    "Machine": platform.machine(),
                    "Processor": platform.processor(),
                })
            
            if info_type.lower() in ["all", "performance"]:
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                info.update({
                    "CPU Usage": f"{psutil.cpu_percent(interval=1)}%",
                    "Memory Usage": f"{memory.percent}% ({memory.used // (1024**3)}GB/{memory.total // (1024**3)}GB)",
                    "Disk Usage": f"{disk.percent}% ({disk.used // (1024**3)}GB/{disk.total // (1024**3)}GB)",
                    "Boot Time": datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S")
                })
            
            result = "💻 System Information:\n"
            for key, value in info.items():
                result += f"• {key}: {value}\n"
            
            return result
            
        except ImportError:
            return "❌ System info requires psutil: pip install psutil"
        except Exception as e:
            return f"❌ Error getting system info: {str(e)}"

    def search_flights(self, departure: str, destination: str, departure_date: str, return_date: str = None, passengers: int = 1) -> str:
        """Search for flights using SerpAPI Google Flights"""
        try:
            if not SERPAPI_KEY:
                return "❌ SerpAPI key not configured. Set SERPAPI_API_KEY environment variable."
            
            console.print(f"[cyan]✈️ Searching flights from {departure} to {destination}...[/cyan]")
            
            params = {
                "engine": "google_flights",
                "departure_id": departure,
                "arrival_id": destination, 
                "outbound_date": departure_date,
                "currency": "USD",
                "hl": "en",
                "api_key": SERPAPI_KEY
            }
            
            if return_date:
                params["return_date"] = return_date
                params["type"] = "1"  # Round trip
            else:
                params["type"] = "2"  # One way
                
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "best_flights" not in results:
                return "❌ No flights found for the specified route and date."
            
            flight_results = "✈️ Flight Search Results:\n\n"
            
            for i, flight in enumerate(results["best_flights"][:5], 1):
                for flight_info in flight["flights"]:
                    airline = flight_info.get("airline", "Unknown")
                    departure_time = flight_info.get("departure_airport", {}).get("time", "N/A")
                    arrival_time = flight_info.get("arrival_airport", {}).get("time", "N/A")
                    duration = flight_info.get("duration", "N/A")
                    
                price = flight.get("price", "N/A")
                flight_results += f"{i}. {airline}\n"
                flight_results += f"   Departure: {departure_time} | Arrival: {arrival_time}\n"
                flight_results += f"   Duration: {duration} | Price: ${price}\n\n"
            
            # Store search in memory for future reference
            self.memory.update_user_profile("last_flight_search", {
                "departure": departure,
                "destination": destination,
                "date": departure_date,
                "results": results["best_flights"][:5],
                "timestamp": datetime.now().isoformat()
            })
            
            console.print(f"[green]{flight_results}[/green]")
            return flight_results
            
        except Exception as e:
            error_msg = f"❌ Flight search error: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg

    def search_hotels(self, location: str, check_in: str, check_out: str, guests: int = 2, preferences: Optional[str] = None) -> str:
        """Search for hotels using SerpAPI Google Hotels"""
        try:
            if not SERPAPI_KEY:
                return "❌ SerpAPI key not configured. Set SERPAPI_API_KEY environment variable."

            query = f"{preferences} hotels in {location}" if preferences else location
            console.print(f"[cyan]🏨 Searching for: '{query}'...[/cyan]")

            params = {
                "engine": "google_hotels",
                "q": query,
                "check_in_date": check_in,
                "check_out_date": check_out,
                "adults": guests,
                "currency": "USD",
                "gl": "us",
                "hl": "en",
                "api_key": SERPAPI_KEY
            }
            search = GoogleSearch(params)
            results = search.get_dict()

            if "properties" not in results:
                return f"❌ No hotels found for '{query}' with the specified dates."

            hotel_results = "🏨 Hotel Search Results:\n\n"
            for i, hotel in enumerate(results["properties"][:5], 1):
                name = hotel.get("name", "Unknown Hotel")
                rate = hotel.get("rate_per_night", {}).get("lowest", "N/A")
                rating = hotel.get("overall_rating", "N/A")
                reviews = hotel.get("reviews", "N/A")
                hotel_results += f"{i}. {name}\n"
                hotel_results += f" Rating: {rating}⭐ ({reviews} reviews)\n"
                hotel_results += f" Price: ${rate}/night\n\n"

            # Store search in memory
            self.memory.update_user_profile("last_hotel_search", {
                "location": location,
                "check_in": check_in,
                "check_out": check_out,
                "results": results["properties"][:5],
                "timestamp": datetime.now().isoformat()
            })

            console.print(f"[green]{hotel_results}[/green]")
            return hotel_results
        except Exception as e:
            error_msg = f"❌ Hotel search error: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg

    def send_travel_email(self, to_email: str, subject: str, content: str, travel_type: str = "itinerary") -> str:
        """Send personalized travel emails using SendGrid"""
        try:
            if not SENDGRID_API_KEY:
                return "❌ SendGrid not configured. Set SENDGRID_API_KEY and SENDGRID_FROM_EMAIL environment variables."
            
            console.print(f"[cyan]📧 Sending {travel_type} email to {to_email}...[/cyan]")
            
            # Get user context for personalization
            user_context = self.get_user_context()
            user_name = user_context.get('personal_info', {}).get('name', 'Traveler')
            
            # Create personalized email template
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <header style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0;">
                        <h1>✈️ Your Travel Assistant</h1>
                    </header>
                    
                    <main style="background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px;">
                        <h2>Hello {user_name}! 👋</h2>
                        <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                            {content.replace(chr(10), '<br>')}
                        </div>
                        
                        <footer style="margin-top: 30px; padding-top: 20px; border-top: 2px solid #ddd; text-align: center; color: #666;">
                            <p>🤖 Sent by AIA - Your Autonomous Intelligent Assistant</p>
                            <p style="font-size: 12px;">This email was generated based on your travel preferences and recent interactions.</p>
                        </footer>
                    </main>
                </div>
            </body>
            </html>
            """
            
            message = Mail(
                from_email="arnav.gilankar@gmail.com",
                to_emails=to_email,
                subject=f"✈️ {subject}",
                html_content=html_content
            )
            
            sg = SendGridAPIClient(api_key=SENDGRID_API_KEY)
            response = sg.send(message)
            
            # Store email in memory
            self.memory.update_user_profile("last_email_sent", {
                "to": to_email,
                "subject": subject,
                "type": travel_type,
                "timestamp": datetime.now().isoformat(),
                "status": "sent"
            })
            
            success_msg = f"📧 Email sent successfully to {to_email}"
            console.print(f"[green]{success_msg}[/green]")
            return success_msg
            
        except Exception as e:
            error_msg = f"❌ Email sending error: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg


    # CORRECTED VERSION
    def plan_complete_trip(self, destination: str, departure_date: str, return_date: str, email_address: str = None, hotel_preferences: str = None, departure: str = None) -> str:
        """Plan a complete trip with flights, hotels, and email itinerary"""
        try:
            console.print(f"[yellow]🌍 Planning complete trip to {destination}...[/yellow]")
            
            # Get user's location from profile (or ask)
            user_context = self.get_user_context()
            user_location = departure or user_context.get('personal_info', {}).get('location', 'Mumbai')
            
            trip_plan = f"🎯 Complete Trip Plan to {destination}\n\n"
            
            # 1. Search flights
            flight_results = self.search_flights(
                departure=user_location,
                destination=destination, 
                departure_date=departure_date,
                return_date=return_date,
                passengers=1
            )
            trip_plan += flight_results + "\n"
            
            # 2. Search hotels
            hotel_results = self.search_hotels(
                location=destination,
                check_in=departure_date,
                check_out=return_date,
                guests=2,
                preferences=hotel_preferences
            )
            trip_plan += hotel_results + "\n"
            
            # 3. Generate travel tips using LLM
            if self.llm:
                tips_prompt = f"""Generate 5 personalized travel tips for someone visiting {destination} 
                from {departure_date} to {return_date}. Include local attractions, weather considerations, 
                and cultural insights. Make it practical and engaging."""
                
                tips_response = self.llm.invoke(tips_prompt)
                travel_tips = f"💡 Travel Tips for {destination}:\n{tips_response.content}\n\n"
                trip_plan += travel_tips
            
            # 4. Send email if requested -  FIXED
            if email_address: # <--- Use the new 'email_address' parameter
                email_subject = f"Your Trip Itinerary to {destination}"
                # Assuming send_travel_email expects 'to_email' as its first argument
                email_sent = self.send_travel_email(to_email=email_address, subject=email_subject, content=trip_plan, travel_type="complete_itinerary")
                trip_plan += f"\n{email_sent}"
            
            # Store complete trip plan
            self.memory.update_user_profile("last_trip_plan", {
                "destination": destination,
                "dates": f"{departure_date} to {return_date}",
                "plan": trip_plan,
                "timestamp": datetime.now().isoformat()
            })
            
            return trip_plan      
        except Exception as e:
            return f"❌ Trip planning error: {str(e)}"

        # ADD THESE DEBUG METHODS TO YOUR CLASS
    def debug_schema_validation(self, schema_class, data_dict):
        """Debug helper to understand schema validation issues"""
        print(f"\n=== SCHEMA DEBUG for {schema_class.__name__} ===")
        print(f"Input data: {data_dict}")
        
        # Check schema fields
        print(f"Schema fields: {list(schema_class.model_fields.keys())}")
        
        # Check each field's requirements
        for field_name, field_info in schema_class.model_fields.items():
            is_required = field_info.is_required()
            has_default = field_info.default is not field_info.default_factory()
            print(f"  {field_name}: required={is_required}, has_default={has_default}, value={data_dict.get(field_name)}")
        
        # Try to create the object
        try:
            obj = schema_class(**data_dict)
            print(f"✅ Validation successful: {obj.model_dump()}")
            return obj
        except Exception as e:
            print(f"❌ Validation failed: {str(e)}")
            
            # Try to identify which specific fields are causing issues
            for field_name in schema_class.model_fields.keys():
                try:
                    test_data = {field_name: data_dict.get(field_name)}
                    schema_class.model_validate(test_data, strict=False)
                    print(f"  ✅ {field_name}: OK")
                except Exception as field_error:
                    print(f"  ❌ {field_name}: {str(field_error)}")
            
            raise e

    def validate_with_debug(self, schema, schema_data):
        """Enhanced validation with comprehensive error reporting"""
        print(f"\n=== ENHANCED VALIDATION DEBUG ===")
        print(f"Schema: {schema.__name__}")
        print(f"Input data: {schema_data}")
        
        # Check schema field requirements
        print(f"Schema field analysis:")
        for field_name, field_info in schema.model_fields.items():
            is_required = field_info.is_required()
            has_default = hasattr(field_info, 'default') and field_info.default is not None
            value = schema_data.get(field_name)
            print(f"  {field_name}: required={is_required}, has_default={has_default}, value={value}")
            
            # Check for missing required fields
            if is_required and value is None:
                print(f"  ⚠️  MISSING REQUIRED FIELD: {field_name}")
        
        try:
            # Try validation with different approaches
            if hasattr(schema, 'model_validate'):
                # Pydantic v2
                result = schema.model_validate(schema_data)
            else:
                # Pydantic v1 fallback
                result = schema(**schema_data)
            
            print(f"✅ Validation successful")
            return {"data": result.model_dump(), "success": True}
            
        except Exception as e:
            print(f"❌ Validation failed: {str(e)}")
            
            # Try with only non-None values
            clean_data = {k: v for k, v in schema_data.items() if v is not None}
            print(f"Retrying with clean data (non-None values only): {clean_data}")
            
            try:
                if hasattr(schema, 'model_validate'):
                    result = schema.model_validate(clean_data)
                else:
                    result = schema(**clean_data)
                
                print(f"✅ Clean validation successful")
                return {"data": result.model_dump(), "success": True}
                
            except Exception as clean_error:
                print(f"❌ Clean validation also failed: {str(clean_error)}")
                return {
                    "error": f"Validation error: {str(e)}",
                    "clean_error": str(clean_error),
                    "success": False,
                    "schema_data": schema_data,
                    "clean_data": clean_data
                }

    # Also fix the parse_and_validate_travel_query method for plan_complete_trip
    def parse_and_validate_travel_query(self, query: str, tool_name: str) -> Dict[str, Any]:
        """Parse and validate travel-related query against the tool's args schema"""
        import json
        import re

        try:
            tool_schemas = {
                "search_flights": FlightSearchArgs,
                "search_hotels": HotelSearchArgs,
                "send_travel_email": EmailArgs,
                "plan_complete_trip": CompleteTripArgs
            }

            if tool_name not in tool_schemas:
                return {"error": f"Unknown tool: {tool_name}"}

            schema = tool_schemas[tool_name]
            expected_fields = list(schema.model_fields.keys())

            # --- STEP 1: PRE-FILTER QUERY (Your existing logic is good) ---
            def extract_relevant_text(tool: str, full_query: str) -> str:
                # This helper function remains the same as your version
                tool_patterns = {
                    "search_flights": r"(flight|fly|depart|return|from .* to .*)",
                    "search_hotels": r"(hotel|stay|check[- ]?in|check[- ]?out)",
                    "send_travel_email": r"(email|send to|mail it to)",
                }
                pattern = tool_patterns.get(tool)
                if not pattern:
                    return full_query
                sentences = re.split(r"(?<=[.?!])\s+", full_query)
                filtered = [s for s in sentences if re.search(pattern, s, re.IGNORECASE)]
                return " ".join(filtered) if filtered else full_query

            filtered_query = extract_relevant_text(tool_name, query)

            # --- STEP 2: IMPROVED LLM PROMPT ---
            # This prompt is more explicit about data types and defaults.
            parse_prompt = f"""
    Extract parameters for the tool '{tool_name}' from the user query.
    User Query: '{filtered_query}'
    Schema fields: {expected_fields}

    **CRITICAL RULES:**
    1.  **Return ONLY a valid JSON object.** Do not include any other text.
    2.  The JSON keys **must exactly match** these fields: {expected_fields}.
    3.  **All fields must be present.** If a value is not found in the query, use `null`.
    4.  Dates (like departure_date, check_in) must be in **YYYY-MM-DD** format.
    5.  `guests` and `passengers` must be an **integer**.
    6.  `preferences` or `hotel_preferences` must be a **string**.

    **EXAMPLES:**
    -   For 'search_hotels': {{"location": "Kolkata", "check_in": "2025-08-15", "check_out": "2025-08-19", "guests": 2, "preferences": "luxury"}}
    -   For 'search_flights': {{"departure": "Pune", "destination": "Kolkata", "departure_date": "2025-08-15", "return_date": "2025-08-19", "passengers": 1}}

    **JSON Output:**
    """
            if self.llm is None:
                return {"error": "LLM is not initialized."}

            response = self.llm.invoke(parse_prompt)
            content = str(response.content).strip()

            # --- STEP 3: ROBUST JSON EXTRACTION ---
            def extract_json_from_response(text):
                # This helper function remains the same as your version
                json_match = re.search(r"``````", text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_match = re.search(r"(\{[\s\S]*?\})", text)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        return None
                try:
                    # Test if it's valid JSON before returning
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    return None

            json_str = extract_json_from_response(content)
            if not json_str:
                return {"error": "No valid JSON object found in LLM output.", "raw_output": content}

            try:
                raw_parsed = json.loads(json_str)
            except Exception as e:
                return {"error": f"Failed to parse JSON from LLM: {str(e)}", "raw_output": content}

            # --- STEP 4: DATA NORMALIZATION AND SANITIZATION (KEY FIXES HERE) ---
            processed = {}
            
            # Consolidate all keys from the LLM output, handling aliases
            alias_map = {
                "checkin": "check_in", "checkout": "check_out", "city": "location",
                "email": "email_address", "to_email": "email_address",
                "preferences": "hotel_preferences"
            }
            normalized_data = {alias_map.get(k, k): v for k, v in raw_parsed.items()}

            # Populate 'processed' dict with all expected fields, using normalized data or None
            for field in expected_fields:
                processed[field] = normalized_data.get(field)

            # *** START OF CRITICAL FIXES ***
            # Set defaults and correct data types before validation
            if tool_name == "search_flights" or tool_name == "plan_complete_trip":
                processed["passengers"] = int(processed.get("passengers") or 1)
                # Ensure departure is filled for plan_complete_trip if missing
                if tool_name == "plan_complete_trip":
                    processed["departure"] = processed.get("departure") or self.get_user_context().get('personal_info', {}).get('location', 'Mumbai')


            if tool_name == "search_hotels" or tool_name == "plan_complete_trip":
                # FIX 1: Handle `guests` being null. Default to 2.
                processed["guests"] = int(processed.get("guests") or 2)

                # FIX 2: Handle `preferences` being a list or dict. Convert to string.
                prefs_key = "hotel_preferences" if "hotel_preferences" in expected_fields else "preferences"
                prefs_val = processed.get(prefs_key)
                if isinstance(prefs_val, list):
                    processed[prefs_key] = str(prefs_val[0]) if prefs_val else ""
                elif isinstance(prefs_val, dict):
                    processed[prefs_key] = str(next(iter(prefs_val))) if prefs_val else ""
                elif prefs_val is None:
                    processed[prefs_key] = "" # Ensure it's a string, not None, if schema requires it

            # Handle email field mapping for different tools
            if 'email_address' in expected_fields:
                processed['email_address'] = normalized_data.get('email_address')
            if 'email' in expected_fields: # for plan_complete_trip
                processed['email'] = normalized_data.get('email_address') or normalized_data.get('email')

            # *** END OF CRITICAL FIXES ***

            # --- STEP 5: VALIDATE SCHEMA ---
            # Filter down to only the fields the Pydantic model expects
            schema_data = {k: v for k, v in processed.items() if k in expected_fields}
            
            # Handle the case where the function arg is 'email' but schema is 'email_address'
            if tool_name == 'plan_complete_trip' and 'email' in processed and 'email_address' in expected_fields:
                schema_data['email_address'] = processed['email']

            logger.debug(f"DEBUG: Final schema_data for validation: {schema_data}")

            try:
                # Use Pydantic's built-in validation
                validated_obj = schema.model_validate(schema_data)
                validated_data = validated_obj.model_dump(exclude_unset=True)
                
                logger.debug(f"DEBUG: Validation successful for {tool_name}")
                
                return {
                    "data": validated_data,
                    "success": True,
                    "full_processed": processed
                }
            except Exception as e:
                logger.error(f"Validation failed for {tool_name} with data {schema_data}: {e}", exc_info=True)
                return {
                    "error": f"Pydantic validation failed: {str(e)}",
                    "parsed_data": processed,
                    "schema_data": schema_data,
                    "raw_output": content,
                    "success": False
                }

        except Exception as e:
            logger.error(f"Critical error in parse_and_validate_travel_query: {e}", exc_info=True)
            return {"error": f"Internal function error: {str(e)}"}

    def run(self):
        console.print("[bold magenta]Welcome to AIA - Your Autonomous Intelligent Assistant! [/bold magenta]")
        console.print("Type 'exit' to quit, 'voice on'/'close' to toggle voice mode.")
        console.print("Use 'plan:' for travel planning (e.g., 'plan: destination=Paris departure_date=2025-08-01').")
        console.print("Use 'deepsearch:' for searches (e.g., 'deepsearch: autonomous_research topic=AI trends').")
        console.print("Use 'autonomous:' for automation and planning (e.g., 'autonomous: set reminder for meeting').\n")
        while True:
            try:
                query = self.session.prompt("[blue]You:[/blue]").strip()
                if query.lower() == 'exit':
                    console.print("[yellow]🛑 Shutting down AIA. Goodbye![/yellow]")
                    break
                elif query.lower() == 'voice on':
                    self.state.voice_mode = True
                    console.print("[green]🎙️ Voice mode enabled.[/green]")
                    continue
                elif query.lower() == 'close':
                    self.state.voice_mode = False
                    console.print("[yellow]🔇 Voice mode disabled.[/yellow]")
                    continue
                response = self.process_query(query)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            except Exception as e:
                logger.error(f"Runtime error: {e}")
                console.print(Panel(f"Error: {str(e)}", style="red"))

if __name__ == "__main__":
    agent = IntelligentAgent()
    try:
        agent.initialize()
        agent.run()
    except Exception as e:
        logger.error(f"Agent initialization or runtime error: {e}")
        console.print(Panel(f"Fatal error: {str(e)}. Check logs for details.", style="red"))
