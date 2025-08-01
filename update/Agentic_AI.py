from imports import *

print("--- Agentic_AI.py started ---")

from class_Intelligentagent import IntelligentAgent

import asyncio
import logging

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def start_agent():
    """Main function to initialize and run the agent."""
    try:
        logger.info("--- Initializing Agent... ---")
        # 1. Create an instance of your agent
        my_agent = IntelligentAgent()
        
        # 2. Call the initialization method which sets up all the tools
        my_agent.initialize() 
        
        logger.info("--- Starting Agent. Press Ctrl+C to exit. ---")
        # 3. Start the agent's main run loop
        my_agent.run()


    except Exception as e:
        logger.error(f"A critical error occurred: {e}", exc_info=True)
        print(f"\nA critical error occurred. Please check the logs. Error: {e}")

# This standard block ensures the code runs only when you execute this file directly
if __name__ == "__main__":
    start_agent()
