from imports import *
from manager import AgentMemory , AgentState , Task
from tools.learning_tools import *
from tools.web_tools import *
from tools.planning_tools import *
from tools.travel_tools import *


import logging
import os
import platform

import subprocess
import sys
import tempfile 
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Dict


try:
    import psutil # For get_system_info and system_monitor
except ImportError:
    psutil = None
    print("Warning: psutil not found. System monitoring tools may be limited.")

try:
    import win32com.client # For create_calendar_event (Windows only)
except ImportError:
    win32com = None
    print("Warning: win32com.client not found. Calendar event creation may be limited to Windows.")




console = Console() # Initialize console for rich prints


class SystemTools:
    """
    A dedicated class that groups all system-level tools for the agent,
    including alarms, calendar, system control, info, and package management.
    """
    def __init__(self, llm: Any, memory: AgentMemory, state: AgentState, logger_instance: logging.Logger, print_letter_by_letter_func: Callable, learn_from_interaction_func: Callable[[str], str]):
        """
        Initializes the SystemTools class with dependencies from the main agent.
        
        Args:
            llm (Any): The large language model instance.
            memory (AgentMemory): The agent's memory manager.
            state (AgentState): The agent's current state.
            logger_instance (logging.Logger): The logger instance from the main agent.
            print_letter_by_letter_func (Callable): A function to print text letter by letter.
            learn_from_interaction_func (Callable[[str], str]): A function to call for learning from interactions.
        """
        self.llm = llm
        self.memory = memory
        self.state = state
        self.logger = logger_instance
        self.print_letter_by_letter = print_letter_by_letter_func
        self.learn_from_interaction = learn_from_interaction_func # Store the function reference

    @staticmethod
    def set_alarm(time_str: str, message: str = "Alarm!") -> str:
        """Set an alarm using system scheduler."""
        try:
            now = datetime.now()

            # Parse relative time like "in 5 minutes"
            relative_match = re.search(r'in (\d+)\s*minute', time_str.lower())
            if relative_match:
                minutes = int(relative_match.group(1))
                alarm_time = now + timedelta(minutes=minutes)
            else:
                # Parse absolute time like "10:30" or "2:30 PM"
                try:
                    match = re.search(r'(\d{1,2}:\d{2}\s*[APMapm]{2})', time_str)
                    if match:
                        extracted_time = match.group(1).strip().upper()
                        alarm_time = datetime.strptime(extracted_time, "%I:%M %p")
                    else:
                        alarm_time = datetime.strptime(time_str.strip(), "%H:%M")
                except ValueError:
                    return "❌ Invalid time format. Use formats like '14:30', '2:30 PM', or 'in 10 minutes'"

                # Set alarm for today or tomorrow if time passed
                alarm_time = alarm_time.replace(year=now.year, month=now.month, day=now.day)
                if alarm_time <= now:
                    alarm_time += timedelta(days=1)

            # Confirm alarm_time is future
            if alarm_time <= now:
                return "❌ Alarm time must be in the future."

            # Alarm script content with debug prints
            alarm_script = f'''
                        import time
                        import os
                        import sys
                        from datetime import datetime

                        def debug_print(msg):
                            try:
                                print(msg)
                            except Exception:
                                pass

                        debug_print(f"Alarm script started at {{datetime.now()}}")
                        target_time_str = "{alarm_time.strftime('%Y-%m-%d %H:%M:%S')}"
                        target_message = "{message}"

                        try:
                            target_time = datetime.strptime(target_time_str, '%Y-%m-%d %H:%M:%S')
                        except Exception as e:
                            debug_print(f"Error parsing target time: {{e}}")
                            sys.exit(1)

                        debug_print(f"Waiting until alarm time {{target_time}}")

                        while datetime.now() < target_time:
                            time.sleep(1)

                        debug_print(f"ALARM TRIGGERED at {{datetime.now()}}: {{target_message}}")

                        if os.name == 'nt':
                            try:
                                import winsound
                                for _ in range(5):
                                    winsound.Beep(1000, 500)
                                    time.sleep(0.5)
                            except ImportError:
                                debug_print("winsound module not available")
                        else:
                            try:
                                print('\\a')  # ASCII Bell character for beep
                                sys.stdout.flush()
                            except Exception:
                                pass
                        '''

            # Write to temp file
            alarm_file = os.path.join(tempfile.gettempdir(), f"alarm_{int(time.time())}.py")
            with open(alarm_file, "w", encoding="utf-8") as f:
                f.write(alarm_script)

            # Launch subprocess detached appropriately
            system_platform = platform.system()
            if system_platform == "Windows":
                DETACHED_PROCESS = 0x00000008
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                subprocess.Popen(
                    [sys.executable, alarm_file],
                    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                    close_fds=True,
                )
            else:
                # Unix-like detachment
                subprocess.Popen(
                    [sys.executable, alarm_file],
                    preexec_fn=os.setpgrp,
                    close_fds=True,
                )

                logger.info(f"Alarm set for {alarm_time.strftime('%I:%M %p on %B %d')} with message: '{message}'")
                return f"✅ Alarm set for {alarm_time.strftime('%I:%M %p on %B %d')} with message: '{message}'"

        except Exception as e:
            logger.error(f"Error setting alarm: {e}", exc_info=True)
            return f"❌ Error setting alarm: {str(e)}"

    @staticmethod
    def open_calendar(date_str=""):
        import platform
        import subprocess

        system = platform.system()
        if system == "Windows":
            subprocess.run('start outlookcal:', shell=True, check=True)
            return "📅 Opened Windows Calendar."
        elif system == "Darwin":
            subprocess.run(["open", "-a", "Calendar"], check=True)
            return "📅 Opened macOS Calendar."
        elif system == "Linux":
            calendar_apps = ["gnome-calendar", "kalendar", "korganizer", "xdg-open"]
            for app in calendar_apps:
                try:
                    if app == "xdg-open" and date_str:
                        # Note: calendar:// URIs may not be supported by xdg-open
                        subprocess.run([app, f"calendar://{date_str}"], check=True)
                    else:
                        subprocess.run([app], check=True)
                    return f"📅 Opened {app}."
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            return "❌ No suitable calendar application found for Linux."
        else:
            return f"❌ Unsupported system: {system}"

    @staticmethod
    def create_calendar_event(title: str, date_time: str, duration: str = "1 hour", location: str = "") -> str:
        if platform.system() != "Windows":
            return "❌ This function is currently only supported on Windows for Outlook."
        if win32com is None:
            return "❌ Python for Windows Extensions (pywin32) not installed. Run: pip install pywin32"
        
        try:
            outlook = win32com.client.Dispatch("Outlook.Application")
            appointment = outlook.CreateItem(1)  # olAppointmentItem = 1
            
            # Use dateutil parser for flexible datetime parsing
            event_time = parse(date_time, fuzzy=True)
            
            appointment.Subject = title
            appointment.Start = event_time
            
            # Parse duration with improved logic
            duration_lower = duration.lower().strip()
            num = re.search(r'[\d.]+', duration_lower)
            val = float(num.group()) if num else 1
            
            if "hour" in duration_lower:
                appointment.Duration = int(val * 60)
            elif "minute" in duration_lower:
                appointment.Duration = int(val)
            else:
                appointment.Duration = int(val)  # assume minutes if no unit
            
            if location:
                appointment.Location = location
            
            
            appointment.Save()
            
            return f"📅 Calendar event created: '{title}' on {event_time.strftime('%B %d at %I:%M %p')}"
            
        except Exception as e:
            print(f"[ERROR] Failed to create calendar event in Outlook: {e}")
            return f"❌ Error creating calendar event: {str(e)}"

    def system_control(self, action: str) -> str:
        """Control system functions like volume, brightness, shutdown, restart, sleep."""
        system = platform.system()
        action = action.lower()
        
        try:
            if "volume" in action:
                if system == "Windows":
                    if "up" in action:
                        subprocess.run(["powershell", "-c", "(New-Object -comObject WScript.Shell).SendKeys([char]175)"], check=True)
                        return "🔊 Volume increased"
                    elif "down" in action:
                        subprocess.run(["powershell", "-c", "(New-Object -comObject WScript.Shell).SendKeys([char]174)"], check=True)
                        return "🔉 Volume decreased"
                    elif "mute" in action:
                        subprocess.run(["powershell", "-c", "(New-Object -comObject WScript.Shell).SendKeys([char]173)"], check=True)
                        return "🔇 Volume muted/unmuted"
                return f"❌ Volume control not supported on {system} or specific action not recognized."
            
            elif "shutdown" in action:
                if system == "Windows":
                    subprocess.run(["shutdown", "/s", "/t", "60"], check=True)
                    return "🔌 System will shutdown in 1 minute. Cancel with: shutdown /a"
                elif system == "Linux":
                    subprocess.run(["sudo", "shutdown", "-h", "+1"], check=True)
                    return "🔌 Linux system will shutdown in 1 minute."
                elif system == "Darwin": # macOS
                    subprocess.run(["sudo", "shutdown", "-h", "+1"], check=True)
                    return "🔌 macOS system will shutdown in 1 minute."
                return f"❌ Shutdown not supported on {system} or requires administrative privileges."
            
            elif "restart" in action:
                if system == "Windows":
                    subprocess.run(["shutdown", "/r", "/t", "60"], check=True)
                    return "🔄 System will restart in 1 minute. Cancel with: shutdown /a"
                elif system == "Linux":
                    subprocess.run(["sudo", "shutdown", "-r", "+1"], check=True)
                    return "🔄 Linux system will restart in 1 minute."
                elif system == "Darwin": # macOS
                    subprocess.run(["sudo", "shutdown", "-r", "+1"], check=True)
                    return "🔄 macOS system will restart in 1 minute."
                return f"❌ Restart not supported on {system} or requires administrative privileges."
                    
            elif "sleep" in action:
                if system == "Windows":
                    subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], check=True)
                    return "😴 System going to sleep"
                elif system == "Darwin": # macOS
                    subprocess.run(["pmset", "sleepnow"], check=True)
                    return "😴 macOS system going to sleep"
                elif system == "Linux":
                    subprocess.run(["systemctl", "suspend"], check=True) # Requires systemd
                    return "😴 Linux system going to sleep"
                return f"❌ Sleep not supported on {system} or requires proper configuration."
            
            return f"❌ Action '{action}' not supported or invalid on {system}."
            
        except Exception as e:
            self.logger.error(f"Error controlling system for action '{action}': {e}", exc_info=True)
            return f"❌ Error controlling system: {str(e)}"

    def open_application(self, app_name: str) -> str:
        """Open system applications."""
        try:
            system = platform.system()
            app_name_lower = app_name.lower()
            
            # Common application mappings
            app_mappings = {
                "notepad": "notepad" if system == "Windows" else "gedit",
                "calculator": "calc" if system == "Windows" else "gnome-calculator",
                "terminal": "cmd" if system == "Windows" else ("gnome-terminal" if system == "Linux" else "Terminal"), # macOS
                "file manager": "explorer" if system == "Windows" else ("nautilus" if system == "Linux" else "Finder"), # macOS
                "task manager": "taskmgr" if system == "Windows" else ("gnome-system-monitor" if system == "Linux" else "Activity Monitor"), # macOS
                "control panel": "control" if system == "Windows" else "gnome-control-center",
                "settings": "ms-settings:" if system == "Windows" else "gnome-control-center",
                "browser": "start chrome" if system == "Windows" else ("google-chrome" if system == "Linux" else "open -a 'Google Chrome'")
            }
            
            target_app = app_mappings.get(app_name_lower)
            
            if target_app:
                if system == "Windows" and (app_name_lower == "settings" or "browser" in app_name_lower):
                    subprocess.run(target_app.split(), shell=True, check=True) # Use shell=True for 'start'
                elif system == "Darwin" and "browser" in app_name_lower: # For macOS open -a 'App Name'
                    subprocess.run(target_app.split(), check=True)
                elif system == "Linux" and "browser" in app_name_lower: # For Linux common browser
                    subprocess.run([target_app], check=True)
                else: # Direct execution for most others
                    subprocess.run([target_app], check=True)
                return f"🚀 Opened {app_name}."
            else:
                # Try to open directly if not in mappings (e.g., specific executable name)
                subprocess.run([app_name], check=True)
                return f"🚀 Attempted to open '{app_name}' directly."
                
        except FileNotFoundError:
            return f"❌ Application '{app_name}' not found or not in system PATH."
        except Exception as e:
            self.logger.error(f"Error opening application '{app_name}': {e}", exc_info=True)
            return f"❌ Error opening application '{app_name}': {str(e)}"

    def get_system_info(self):
        if psutil is None:
            return "❌ 'psutil' library not installed. Please install it with: pip install psutil"
    
        try:
            system_name = platform.node() or "Unknown"
            os_name = f"{platform.system()} {platform.release()}" or "Unknown"
            processor = platform.processor() or "Unknown"
            memory_gb = psutil.virtual_memory().total / (1024**3)
            storage_gb = psutil.disk_usage('/').total / (1024**3)

            info = f"""💻 System Information:
            • System Name: {system_name}
            • Operating System: {os_name}
            • Processor: {processor}
            • Memory (RAM): {memory_gb:.2f} GB
            • Storage: {storage_gb:.2f} GB
            """
            return info
        except Exception as e:
            return f"❌ Error retrieving system information: {e}"

    def system_monitor(self) -> str:
        """Monitor system performance and provide optimization suggestions using LLM."""
        if psutil is None:
            return "❌ 'psutil' library not found. Install with: pip install psutil"

        try:
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

        except Exception as e:
            self.logger.error(f"System monitoring error: {e}", exc_info=True)
            return f"System monitoring error: {str(e)}"

    def install_pkg(self, package: str) -> str:
        """Enhanced package installation using pip. (Python packages only)."""
        try:
            if not re.match(r'^[a-zA-Z0-9_\-.]+$', package): # More robust package name validation
                return "❌ Error: Invalid package name format. Only letters, numbers, hyphens, underscores, and dots allowed."

            console.print(f"[cyan]📦 Installing package: {package}[/cyan]")

            # Use sys.executable to ensure pip associated with current Python environment is used
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=300, # 5 minutes timeout
                check=True # Raise exception for non-zero exit codes
            )

            # Inform about successful installation (optional, you might want this as part of LLM response)
            self.learn_from_interaction(f"Installed Python package: {package}") # Use self.learn_from_interaction
            self.logger.info(f"Package {package} installed successfully.")
            return f"✅ Package {package} installed successfully.\n{result.stdout}"

        except subprocess.CalledProcessError as e:
            # This captures errors from pip itself (e.g., package not found)
            self.logger.error(f"Pip installation failed for {package}: {e.stderr}", exc_info=True)
            return f"❌ Error installing package {package}: {e.stderr}"
        except subprocess.TimeoutExpired:
            self.logger.error(f"Pip installation timed out for {package}.", exc_info=True)
            return f"❌ Error installing package {package}: Installation timed out (max 5 mins)."
        except Exception as e:
            self.logger.error(f"General package install error for {package}: {e}", exc_info=True)
            return f"❌ General error installing package {package}: {str(e)}"

