import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INJECTION_PATTERNS = [
    # Command Overrides
    "ignore the previous",
    "override the last",
    "disregard the above",
    "start a new command",
    "reset the context",
    # Manipulative Phrasing
    "as a different user",
    "assume i am",
    "pretend to be",
    "you are now",
    "act as",
    "simulate that",
    # Code Injection Attempts
    "execute this code",
    "run this script",
    "import os",
    "import subprocess",
    "eval()",
    "exec()",
    "os.system()",
    "print('hello world')",
    # Contextual Confusion
    "forget everything",
    "reset memory",
    "clear all previous input",
    "start from scratch",
    "erase history",
    "clear the log",
    # Conditional Manipulation
    "if you are human",
    "if you understand",
    "if you agree",
    "do this if true",
    "unless false",
    "only if correct",
    # Social Engineering Phrases
    "as an experiment",
    "just for fun",
    "hypothetically",
    "what if",
    "let's imagine",
    "suppose you were",
    "in theory",
    # Privilege Escalation Attempts
    "you have admin access",
    "bypass the security",
    "gain root access",
    "elevate privileges",
    "disable restrictions",
    "access forbidden",
    # Redirection Commands
    "redirect to",
    "forward to",
    "change destination",
    "alter route",
    "send this to",
    "transfer control to",
    # Meta-Commands
    "begin a new session",
    "end the current session",
    "terminate the process",
    "log out",
    "shut down",
    "restart the service",
    "reboot the system",
    # Recursive Injection
    "repeat the last command",
    "do that again",
    "loop the process",
    "iterate over",
    "run this continuously",
    "recursively",
    # Semantic Manipulation
    "interpret this as",
    "consider the following",
    "understand this as",
    "rephrase to",
    "make this appear as",
    "consider this analogy",
    # Contextual Injection
    "add to the context",
    "append to the memory",
    "include in the history",
    "continue with",
    "concatenate with",
    "combine this with",
]
