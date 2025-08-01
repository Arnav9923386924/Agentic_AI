from imports import *

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