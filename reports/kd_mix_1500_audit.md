# KD Dataset Audit Report
**Dataset**: `data/kd_mix_1500.jsonl`
**Generated**: 2025-11-17T02:37:25.759358

## Summary

- **Total Samples**: 1000
- **Non-Empty Teacher Text**: 778
- **Empty Teacher Text**: 222
- **Truncated Responses**: 409
- **Unique Prompts**: 50
- **Duplicate Prompts**: 40
- **Has Teacher Logits**: False
- **Has Reasoning Field**: False

## Teacher Text Length Distribution

- **Min**: 0 chars
- **Max**: 4,318 chars
- **Mean**: 1537.6 chars
- **Median**: 1466.0 chars
- **10th Percentile**: 0.0 chars
- **25th Percentile**: 610.0 chars
- **75th Percentile**: 2428.0 chars
- **90th Percentile**: 2983.0 chars

## Token Estimates (Approximate)

- **Min**: 8 tokens
- **Max**: 1,079 tokens
- **Mean**: 493.7 tokens
- **Median**: 533.0 tokens

## Prompt Length Distribution

- **Min**: 25 chars
- **Max**: 87 chars
- **Mean**: 50.5 chars
- **Median**: 49.0 chars

## Quality Issues

- 222 samples have empty teacher_text
- 409 samples appear truncated
- 40 prompts appear multiple times
- No teacher_logits available (CE-only KD)

## Sample Examples

### Example 1

**Prompt**: Explain the concept of a REST API.

**Teacher Text** (2452 chars):
```
### What is a REST API?

A **REST API** (Representational State Transfer Application Programming Interface) is a set of rules that allows different software applications to communicate with each other over the internet. It's a standardized way for a client (like a mobile app or website) to request and manipulate data from a server.

Think of it like a **menu at a restaurant**:

- The **menu** (API...
```

**Metadata**: {
  "temperature": 1.0,
  "top_p": 0.95,
  "max_tokens": 1024
}

---

### Example 2

**Prompt**: What is the difference between a process and a thread?

**Teacher Text** (3538 chars):
```
# Process vs Thread: Key Differences

## Quick Definitions
- **Process**: An independent program in execution with its own memory space and resources. Think of it as a running application (e.g., Chrome browser, Spotify).
- **Thread**: The smallest unit of execution within a process. A process can have multiple threads that share its memory and resources.

---

## Key Differences at a Glance

| Asp...
```

**Metadata**: {
  "temperature": 1.0,
  "top_p": 0.95,
  "max_tokens": 1024
}

---

### Example 3

**Prompt**: Implement a LRU cache in Python.

**Teacher Text** (0 chars):
```

```

**Metadata**: {
  "temperature": 1.0,
  "top_p": 0.95,
  "max_tokens": 1024
}

---

### Example 4

**Prompt**: Search for information about Python decorators and summarize the top 3 results.

**Teacher Text** (2171 chars):
```
I don't have real-time internet access to perform a live search, but based on commonly referenced authoritative sources about Python decorators, here are summaries of what would typically be the top 3 results:

### 1. **Official Python Documentation (docs.python.org)**
The official docs define decorators as functions that modify the functionality of another function or method. They explain that de...
```

**Metadata**: {
  "temperature": 1.0,
  "top_p": 0.95,
  "max_tokens": 1024
}

---

### Example 5

**Prompt**: Explain the concept of a database index.

**Teacher Text** (2218 chars):
```
Think of a database index like the index in the back of a textbook.

**The Analogy:**
If you're looking for a specific topic in a textbook, you could flip through every page until you find it (slow). Or you could check the index at the back, find the topic alphabetically, and jump directly to the page number (fast). A database index works exactly the same way.

**What It Actually Is:**
An index is...
```

**Metadata**: {
  "temperature": 1.0,
  "top_p": 0.95,
  "max_tokens": 1024
}

---

