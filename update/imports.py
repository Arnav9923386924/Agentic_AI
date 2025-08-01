import platform
from datetime import datetime
import asyncio
import subprocess
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
from datetime import datetime
from dateutil.parser import parse
import re
import platform
import subprocess 
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import asyncio
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
import psutil

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
