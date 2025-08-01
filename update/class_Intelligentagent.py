from imports import *
from manager import AgentState, AgentMemory , Task
from tools.planning_tools import *
from tools.learning_tools import *
from tools.travel_tools import *
from tools.web_tools import *
from tools.system_tools import *
from schemas import FlightSearchArgs, HotelSearchArgs, EmailArgs, CompleteTripArgs 
from config import SERPAPI_KEY, SENDGRID_API_KEY, SENDGRID_FROM_EMAIL


console = Console()
recognizer = sr.Recognizer()


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
        self.planning_tools = None
        self.travel_tools = None
        self.web_tools = None
        self.learning_tools = None
        self.system_tools = None

    def initialize(self):
        """Initialize the agent with all necessary components"""
        print("--- Inside IntelligentAgent.initialize() ---") 
        console.print("[yellow]🧠 Initializing Enhanced AI Agent...[/yellow]")

        # Initialize LLM and embeddings
        self.llm, self.embeddings = self.initialize_llm()

        if not self.llm:
            raise Exception("Failed to initialize LLM")

        # Load user profile and preferences
        self.state.user_preferences = self.memory.get_user_profile()

        self.learning_tools = LearningTools(
            memory=self.memory,
            state=self.state,
            llm=self.llm,
            logger_instance=logger
        
        )


        # Initialize travel tools
        self.travel_tools = TravelTools(
            memory=self.memory,
            llm=self.llm,
            get_user_context_func=self.learning_tools.get_user_context, # Pass the method from IntelligentAgent
            serpapi_key='c6f0d0e23fdec28e86008c1f5df5fa861504378032466af0fb0eee32ceb0b642', # Your actual key
            sendgrid_api_key='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', # Your actual key
            sendgrid_from_email='arnav.gilankar@gmail.com' # Your actual email
        )

        self.web_tools = WebTools(
            llm=self.llm,
            memory=self.memory,
            print_letter_by_letter_func=self.print_letter_by_letter,
            get_user_context_func=self.learning_tools.get_user_context, # Use get_user_context from LearningTools
            logger_instance=logger # Pass the global logger from IntelligentAgent.py
        )

        # Initialize planning tools
        self.planning_tools = PlanningTools(
            memory=self.memory,
            state=self.state,
            llm=self.llm,
            print_func=self.print_letter_by_letter # IntelligentAgent's method
        )

        self.system_tools = SystemTools(
            llm=self.llm,
            memory=self.memory,
            state=self.state,
            logger_instance=logger, # Pass the global logger from IntelligentAgent.py
            print_letter_by_letter_func=self.print_letter_by_letter,
            learn_from_interaction_func=self.learning_tools.learn_from_interaction # Pass learn_from_interaction from LearningTools
        )

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
        self.tools = [
            # Travel-related tools for plan:
            # works
            StructuredTool.from_function(
                func=self.travel_tools.search_flights,
                name="search_flights",
                description="Search for flights between cities with departure/return dates. Requires departure city, destination city, departure date (YYYY-MM-DD), optional return date, and passenger count.",
                args_schema=FlightSearchArgs
            ),
            # works
            StructuredTool.from_function(
                func=self.travel_tools.search_hotels,
                name="search_hotels",
                description="Search for hotels in a specific location with check-in/check-out dates. Requires location, check-in date (YYYY-MM-DD), check-out date (YYYY-MM-DD), and number of guests.",
                args_schema=HotelSearchArgs
            ),
            # doesnt works, we need the access to send the email
            StructuredTool.from_function(
                func=self.travel_tools.send_travel_email,
                name="send_travel_email",
                description="Send personalized travel emails with itineraries, confirmations, or travel information. Requires recipient email, subject, and content.",
                args_schema=EmailArgs
            ),
            # doesnt works, we need the access to send the email
            StructuredTool.from_function(
                func=self.travel_tools.plan_complete_trip,
                name="plan_complete_trip",
                description="Plan a complete trip including flights, hotels, and email itinerary. Requires destination, departure date, return date, hotel preferences and optional email address.",
                args_schema=CompleteTripArgs
            ),
            # works
            StructuredTool.from_function(
                func=self.web_tools.autonomous_web_research,
                name="autonomous_research",
                description="Perform comprehensive web research on any topic. Searches multiple sources, gathers current information, and provides structured analysis."
            ),
            # works
            StructuredTool.from_function(
                func=self.web_tools.search_web,
                name="search_web",
                description="Open a browser tab with a Google search or direct website."
            ),
            # doesnt work
            StructuredTool.from_function(
                func=self.system_tools.set_alarm,
                name="set_alarm",
                description="Set an alarm for a specific time with optional message."
            ),

            # works
            StructuredTool.from_function(
            func=SystemTools.open_calendar,  # Use the class, not the instance.
            name="open_calendar",
            description="Open the system calendar application."
            ),

            # not yet tested
            StructuredTool.from_function(
                func=self.system_tools.create_calendar_event,
                name="create_calendar_event",
                description="Create a new calendar event (Windows with Outlook)."
            ),
            # not yet tested
            StructuredTool.from_function(
                func=self.system_tools.system_control,
                name="system_control",
                description="Control system functions like volume, shutdown, restart, sleep."
            ),
            # not yet tested
            StructuredTool.from_function(
                func=self.system_tools.open_application,
                name="open_application",
                description="Open system applications like notepad, calculator, etc."
            ),
            # works
            StructuredTool.from_function(
                func=self.system_tools.get_system_info,
                name="get_system_info",
                description="Get detailed system information and performance metrics."
            ),
            # works
            StructuredTool.from_function(
                func=self.system_tools.system_monitor,
                name="system_monitor",
                description="Monitor system resources and provide optimization suggestions."
            ),
            # works
            StructuredTool.from_function(
                func=self.learning_tools.learn_from_interaction,
                name="learn_preference",
                description="Learn and store user preferences and personal information from interactions."
            ),
            # works, but need to be tested again 
            # StructuredTool.from_function(
                #func=self.learning_tools.auto_learn_from_conversation,
                #name="auto_learn_conversation",
                #description="Automatically analyze conversations for learning opportunities and extract user preferences."
           # ),
            # works
            StructuredTool.from_function(
                func=self.learning_tools.get_user_context,
                name="get_user_context",
                description="Retrieve comprehensive user context including personal information, preferences, and conversation history."
            ),
            # works
            StructuredTool.from_function(
                func=self.learning_tools.check_memory,
                name="check_memory",
                description="Debug function to inspect stored user profile data and recent conversation history."
            ),
            StructuredTool.from_function(
                func=self.planning_tools.create_goal,
                name="create_goal",
                description="Create a new goal or objective to work towards."
            ),
            # yet to be tested
            StructuredTool.from_function(
                func=self.planning_tools.plan_tasks,
                name="plan_tasks",
                description="Break down a goal into actionable tasks with priorities."
            ),
            # works
            StructuredTool.from_function(
                func=self.planning_tools.proactive_suggestion,
                name="proactive_suggestion",
                description="Generate proactive suggestions when user explicitly asks for suggestions, recommendations, or advice."
            ),
            # works
            StructuredTool.from_function(
                func=self.web_tools.download_file,
                name="download_file",
                description="Download a file from a URL to the current directory."
            ),
            # works
            StructuredTool.from_function(
                func=self.system_tools.install_pkg,
                name="install_pkg",
                description="Install a Python package using pip."
            ),
        ]

        # Create enhanced ReAct agent
        if self.llm is None:
            raise RuntimeError("LLM is not initialized. Cannot create ReAct agent.")

        react_prompt = self.create_enhanced_react_prompt()
        agent = create_react_agent(self.llm, self.tools, react_prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
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
            if hasattr(self.travel_tools, tool_name):
                tool_function = getattr(self.travel_tools, tool_name)
                response = tool_function(**validated_args)
            else:
                return f"❌ Error: Tool '{tool_name}' is not a defined method."

            # --- Step 4: Output and Logging ---
            colored_response = f"[cyan]{response}[/cyan]"
            self.print_letter_by_letter(colored_response)
            
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
            user_context = self.learning_tools.get_user_context()
            
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
            self.learning_tools.auto_learn_from_conversation(query, str(full_response))

            if self.state.voice_mode:
                asyncio.run(self.text_to_speech(str(full_response)))

            return str(full_response)

        except Exception as e:
            logger.error(f"Conversational request error: {e}")
            return f"❌ Error in conversation: {str(e)}"



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
                    processed["departure"] = processed.get("departure") or self.learning_tools.get_user_context().get('personal_info', {}).get('location', 'Mumbai')


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

    def parse_event_details(self, text: str):
        """
        Parses event details from user input text using regex and dateutil.
        Returns a tuple (title, date_time_str, location) or None if parsing fails.
        """
        if parse is None:
            # Handle the case where dateutil is not installed
            return None

        # Use regex to find the key-value pairs
        date_match = re.search(r'date\s*[:=]\s*([\w\s]+)', text, re.IGNORECASE)
        time_match = re.search(r'time\s*[:=]\s*([\w\s:AMPampm]+)', text, re.IGNORECASE)
        event_match = re.search(r'event\s*[:=]\s*([^,]+)', text, re.IGNORECASE)
        location_match = re.search(r'location\s*[:=]\s*([^,]+)', text, re.IGNORECASE)

        if not (date_match and time_match and event_match):
            self.logger.warning("Parsing failed: Missing date, time, or event.")
            return None

        # Extract the raw strings
        raw_date = date_match.group(1).strip()
        raw_time = time_match.group(1).strip()
        title = event_match.group(1).strip()
        location = location_match.group(1).strip() if location_match else ""

        # Combine date and time and parse robustly using dateutil
        try:
            # Construct a string that dateutil can easily parse
            combined_str = f"{raw_date} {datetime.now().year} {raw_time}"
            dt_object = parse(combined_str, fuzzy=True)
            # Format it correctly for your create_calendar_event method
            date_time_str = dt_object.strftime("%Y-%m-%d %H:%M")
        except Exception as e:
            self.logger.error(f"Could not parse date/time string: {combined_str} - Error: {e}")
            return None

        return title, date_time_str, location

    def run(self):
        console.print("[bold magenta]Welcome to AIA - Your Autonomous Intelligent Assistant![/bold magenta]")
        # ... other welcome prints ...
        while True:
            try:
                query = self.session.prompt("[blue]You:[/blue]").strip()
                q_lower = query.lower()

                if q_lower == "exit":
                    console.print("[yellow]🛑 Shutting down AIA. Goodbye![/yellow]")
                    break
                
                # --- START OF THE FIX ---
                # Check for the "create event" intent first
                elif "create event" in q_lower or "create an event" in q_lower or "event:" in q_lower:
                    parsed_details = self.parse_event_details(query)
                    if parsed_details:
                        title, date_time_str, location = parsed_details
                        # Call the ACTUAL tool and get the REAL response
                        response = self.system_tools.create_calendar_event(
                            title=title, 
                            date_time=date_time_str, 
                            location=location
                        )
                    else:
                        response = "❌ Sorry, I couldn't parse the event details. Please provide them clearly, for example: 'create event date: tomorrow, time: 3 PM, event: Team Meeting, location: Room 404'"

                # --- END OF THE FIX ---

                elif "open calendar" in q_lower or "open calender" in q_lower:
                    response = SystemTools.open_calendar()

                else:
                    # Fallback to your general LLM processing for all other queries
                    response = self.process_query(query)

                # Print the real response from the tool or LLM
                self.print_letter_by_letter(response)
                if self.state.voice_mode:
                    import asyncio
                    asyncio.run(self.text_to_speech(response))

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            except Exception as e:
                # Log the full error for debugging
                logger.error(f"Runtime error in main loop: {e}", exc_info=True)
                console.print(Panel(f"An unexpected error occurred: {str(e)}", style="bold red", title="Error"))

if __name__ == "__main__":
    agent = IntelligentAgent()
    try:
        agent.initialize()
        agent.run()
    except Exception as e:
        logger.error(f"Agent initialization or runtime error: {e}")
        console.print(Panel(f"Fatal error: {str(e)}. Check logs for details.", style="red"))
