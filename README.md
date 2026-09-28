# ResolveAI

ResolveAI is an end-to-end AI customer support agent designed to
understand customer issues, retrieve relevant historical cases,
generate appropriate responses, and determine when a conversation
should be escalated to a human.

The project demonstrates how traditional ML, information retrieval,
LLMs, and decision logic can be composed into a practical AI system.

## Architecture

Customer Message
       ↓
Intent Classifier
       ↓
Historical Case Retrieval
       ↓
Response Generation
       ↓
Escalation Decision
       ↓
AI Response / Human Support

## Core Components

- Intent Classification
  - TF-IDF
  - Logistic Regression

- Historical Retrieval
  - TF-IDF similarity search
  - Sentence embeddings

- Response Generation
  - LLM-based response generation
  - Historical cases used as context

- Decision Layer
  - Determines whether the issue can be handled automatically
  - Escalates account-specific and sensitive issues to humans

- Evaluation
  - Intent classification metrics
  - Retrieval evaluation
  - End-to-end agent evaluation

## Example

Customer:

> My package says delivered but I never received it.

ResolveAI:

1. Detects `delivery_issue`
2. Retrieves similar historical cases
3. Generates a support response
4. Determines whether escalation is required

## Why this project?

The goal is not simply to call an LLM API.

ResolveAI explores how multiple AI/ML components can be combined
into an application-level system:

ML → Retrieval → LLM → Decision Engine → API → UI