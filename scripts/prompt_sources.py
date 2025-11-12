"""
Prompt sources for knowledge distillation dataset generation.

Provides prompts from different categories:
- General reasoning
- Domain-specific (coding, math, etc.)
- Tool traces (JSON tool calls)
"""
from typing import List, Dict, Any
import json
from pathlib import Path


def get_general_prompts(n: int = 100) -> List[str]:
    """Generate general reasoning prompts."""
    prompts = [
        "Explain the concept of recursion in programming.",
        "What is the difference between a list and a tuple in Python?",
        "How does a neural network learn?",
        "Describe the process of photosynthesis.",
        "What are the main causes of climate change?",
        "Explain quantum entanglement in simple terms.",
        "How does the internet work?",
        "What is the difference between HTTP and HTTPS?",
        "Explain the concept of entropy in thermodynamics.",
        "How do search engines rank web pages?",
        "What is the difference between machine learning and deep learning?",
        "Explain the concept of a hash table.",
        "How does a compiler work?",
        "What is the difference between synchronous and asynchronous programming?",
        "Explain the concept of a database index.",
        "How does a CPU execute instructions?",
        "What is the difference between a process and a thread?",
        "Explain the concept of a REST API.",
        "How does encryption work?",
        "What is the difference between a stack and a queue?",
    ]
    
    # Extend with variations
    base_prompts = prompts[:]
    while len(prompts) < n:
        for p in base_prompts:
            if len(prompts) >= n:
                break
            prompts.append(p)
    
    return prompts[:n]


def get_domain_specific_prompts(n: int = 50) -> List[str]:
    """Generate domain-specific prompts (coding, math, etc.)."""
    coding_prompts = [
        "Write a Python function to reverse a linked list.",
        "Implement a binary search tree in Python.",
        "Write a function to find the longest common subsequence of two strings.",
        "Implement a LRU cache in Python.",
        "Write a function to merge two sorted arrays.",
        "Implement a depth-first search algorithm.",
        "Write a function to check if a string is a palindrome.",
        "Implement a priority queue using a heap.",
        "Write a function to find all permutations of a string.",
        "Implement a graph data structure with adjacency list.",
    ]
    
    math_prompts = [
        "Solve the quadratic equation: x^2 + 5x + 6 = 0",
        "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 1?",
        "Calculate the integral of ∫(x^2 + 3x + 2)dx",
        "What is the limit of (x^2 - 4)/(x - 2) as x approaches 2?",
        "Find the eigenvalues of the matrix [[2, 1], [1, 2]]",
        "What is the sum of the first 100 natural numbers?",
        "Calculate the area under the curve y = x^2 from x=0 to x=5",
        "What is the probability of rolling two sixes in a row?",
        "Find the roots of the polynomial x^3 - 6x^2 + 11x - 6 = 0",
        "What is the Taylor series expansion of e^x?",
    ]
    
    prompts = coding_prompts + math_prompts
    
    # Extend if needed
    while len(prompts) < n:
        prompts.extend(coding_prompts[:min(len(coding_prompts), n - len(prompts))])
    
    return prompts[:n]


def get_tool_trace_prompts(n: int = 30) -> List[str]:
    """Generate prompts that require tool use (JSON tool calls)."""
    prompts = [
        {
            "prompt": "Search for information about Python decorators and summarize the top 3 results.",
            "expected_tools": ["web_search", "summarize"],
        },
        {
            "prompt": "Read the file config.yaml and extract all key-value pairs.",
            "expected_tools": ["read_file", "extract"],
        },
        {
            "prompt": "Search for recent papers on transformer architectures and list their titles.",
            "expected_tools": ["web_search", "extract"],
        },
        {
            "prompt": "Read the code in src/main.py and identify all function definitions.",
            "expected_tools": ["read_file", "parse_code"],
        },
        {
            "prompt": "Search for the latest version of PyTorch and check if it's compatible with Python 3.11.",
            "expected_tools": ["web_search", "check_compatibility"],
        },
        {
            "prompt": "Read the README.md file and extract all code examples.",
            "expected_tools": ["read_file", "extract_code"],
        },
        {
            "prompt": "Search for best practices for training neural networks and create a summary.",
            "expected_tools": ["web_search", "summarize"],
        },
        {
            "prompt": "Read the requirements.txt file and list all dependencies with their versions.",
            "expected_tools": ["read_file", "parse_dependencies"],
        },
        {
            "prompt": "Search for information about CoreML optimization and extract key techniques.",
            "expected_tools": ["web_search", "extract"],
        },
        {
            "prompt": "Read the config.yaml file and validate that all required fields are present.",
            "expected_tools": ["read_file", "validate"],
        },
    ]
    
    # Extend if needed
    while len(prompts) < n:
        prompts.extend(prompts[:min(len(prompts), n - len(prompts))])
    
    # Return just the prompt strings
    return [p["prompt"] if isinstance(p, dict) else p for p in prompts[:n]]


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a JSONL file."""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if isinstance(data, dict):
                    prompts.append(data.get("prompt", data.get("text", "")))
                else:
                    prompts.append(str(data))
    return prompts


def get_prompt_mix(
    general_ratio: float = 0.5,
    domain_ratio: float = 0.3,
    tool_ratio: float = 0.2,
    total: int = 1000,
) -> List[str]:
    """
    Generate a mixed set of prompts according to ratios.
    
    Args:
        general_ratio: Ratio of general prompts
        domain_ratio: Ratio of domain-specific prompts
        tool_ratio: Ratio of tool trace prompts
        total: Total number of prompts
        
    Returns:
        List of prompts
    """
    n_general = int(total * general_ratio)
    n_domain = int(total * domain_ratio)
    n_tool = int(total * tool_ratio)
    
    prompts = []
    prompts.extend(get_general_prompts(n_general))
    prompts.extend(get_domain_specific_prompts(n_domain))
    prompts.extend(get_tool_trace_prompts(n_tool))
    
    # Fill remaining with general prompts
    while len(prompts) < total:
        prompts.extend(get_general_prompts(min(100, total - len(prompts))))
    
    return prompts[:total]







