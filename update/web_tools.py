from imports import *
from manager import AgentMemory , AgentState , Task
from tools.learning_tools import *
from tools.system_tools import *
from tools.planning_tools import *
from tools.travel_tools import *

console = Console() 
logger = logging.getLogger(__name__)

class WebTools:
    """
    A dedicated class that groups all web-related tools for the agent,
    including web research, browsing, and downloading.
    """
    def __init__(self, llm: Any, memory: AgentMemory, print_letter_by_letter_func: Callable,
                 get_user_context_func: Callable[[], dict], logger_instance: logging.Logger):
        """
        Initializes the WebTools class with dependencies from the main agent.
        
        Args:
            llm (Any): The large language model instance.
            memory (AgentMemory): The agent's memory manager.
            print_letter_by_letter_func (Callable): A function to print text letter by letter.
            get_user_context_func (Callable[[], dict]): A function to retrieve user context.
            logger_instance (logging.Logger): The logger instance from the main agent.
        """
        self.llm = llm
        self.memory = memory
        self.print_letter_by_letter = print_letter_by_letter_func
        self.get_user_context = get_user_context_func
        self.logger = logger_instance

    def autonomous_web_research(self, topic: str) -> str:
        """Perform autonomous research on a topic with real web scraping."""
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
                    search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
                    
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
                    self.logger.error(f"Search error for {query}: {e}") # Use self.logger
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
            
            if self.llm is None: # Use self.llm
                raise RuntimeError("LLM is not initialized. Cannot perform this operation.")
            
            console.print("[yellow]🧠 Analyzing and synthesizing research findings...[/yellow]")
            response = self.llm.invoke(research_prompt) # Use self.llm
            
            # Print the research summary with letter-by-letter effect
            console.print("\n[green]📋 Research Summary:[/green]")
            console.print("[cyan]", end="")
            self.print_letter_by_letter(str(response.content)) # Use self.print_letter_by_letter
            console.print("[/cyan]")
            
            # Print web search summary
            console.print(f"\n[yellow]📊 Web Search Summary:[/yellow]")
            console.print(f"[dim]- Performed {len(search_queries)} targeted searches[/dim]")
            total_snippets = sum(len(r.get('snippets', [])) for r in search_results)
            console.print(f"[dim]- Gathered {total_snippets} information snippets[/dim]")
            console.print(f"[dim]- Research stored in memory for future reference[/dim]")
            
            # Store research in memory with web data using self.memory
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
            self.logger.error(f"Autonomous web research error: {e}", exc_info=True) # Use self.logger
            return error_msg

    def search_web(self, query: str) -> str:
        """Enhanced web search with learning capabilities. Opens a browser tab."""
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
                    # The learn_from_interaction tool is in LearningTools.
                    # It's better not to call other tools directly from within a tool.
                    # Instead, the agent's main logic should handle the learning aspect
                    # based on tool usage. For now, we'll keep the print.
                    # If you want to use learn_from_interaction directly, you'd need to
                    # pass self.learning_tools.learn_from_interaction as a dependency.
                    # For simplicity, let's assume learn_from_interaction is for agent's background learning.
                    self.logger.info(f"Opened website: {site}")
                    return f"🌐 Opened browser tab for {url}"

            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            webbrowser.open(search_url)

            self.logger.info(f"Web search: {query}") # Use self.logger
            return f"🔍 Opened browser tab for Google search: {query}"

        except Exception as e:
            self.logger.error(f"Web search error: {e}", exc_info=True) # Use self.logger
            return f"❌ Error opening browser tab: {str(e)}"

    def download_file(self, url: str) -> str:
        """Enhanced file download with progress tracking."""
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

            self.logger.info(f"Downloaded file: {filename} from {url}") # Use self.logger
            return f"✅ File downloaded successfully: {filename}"

        except Exception as e:
            self.logger.error(f"Download error: {e}", exc_info=True) # Use self.logger
            return f"❌ Error downloading file: {str(e)}"

