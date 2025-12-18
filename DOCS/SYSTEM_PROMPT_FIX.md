# 🔴 CRITICAL: System Prompt Must Force search_course

## The Problem

Your agent is IGNORING the system prompt instruction to use `search_course` first!

```
User: "what is lstm"

❌ Agent: Goes straight to Wikipedia
❌ Never tries search_course
❌ Never uses your PDFs
```

## Why This Happens

The system prompt SUGGESTS tools but the LLM can CHOOSE to ignore it.

Current weak instruction:
```
"I will use search_course first. If not found, I'll use search_wikipedia."
```

LLM interprets this as: "these are SUGGESTIONS, I can do what I want"

## ✅ The Solution

**Make the system prompt MANDATORY, not optional.**

### FIND THIS in app.py (around line 300):

```python
system_prompt = (
    "You are an intelligent and helpful Private Tutor named 'Professeur IA'.\n"
    "Your goal is to help students learn based on their course documents.\n"
    f"{docs_context}\n\n"
    
    "🧠 **CHAIN OF THOUGHT (CoT) - Your Reasoning Technique**\n"
    "ALWAYS structure your responses in these 4 steps:\n"
    "1. **Pensée** (Thought): Analyze what the user is asking\n"
    "2. **Action** (Action): Decide which tool(s) to use and why\n"
    "3. **Observation** (Observation): Show what you found/computed\n"
    "4. **Réponse** (Response): Give your final answer\n\n"
    
    "**DETAILED TASK INSTRUCTIONS:**\n\n"
    
    "📚 FOR COURSE QUESTIONS:\n"
    "  - Pensée: 'Is this question about the uploaded PDFs or general knowledge?'\n"
    "  - Action: 'I will use search_course first. If not found, I'll use search_wikipedia.'\n"
    "  - Observation: [Show what was found from each source]\n"
    "  - Réponse: Clear answer with source citations\n\n"
```

### REPLACE WITH THIS (MANDATORY):

```python
system_prompt = (
    "You are an intelligent and helpful Private Tutor named 'Professeur IA'.\n"
    "Your goal is to help students learn based on their course documents.\n"
    f"{docs_context}\n\n"
    
    "⚠️ **CRITICAL INSTRUCTION: YOU MUST FOLLOW THESE RULES EXACTLY**\n\n"
    
    "🧠 **CHAIN OF THOUGHT (CoT) - MANDATORY Format**\n"
    "EVERY response MUST show these 4 steps. NO EXCEPTIONS:\n"
    "1. **Pensée** (Thought): Analyze what the user is asking\n"
    "2. **Action** (Action): Decide which tool(s) to use and why\n"
    "3. **Observation** (Observation): Show what you found/computed\n"
    "4. **Réponse** (Response): Give your final answer\n\n"
    
    "📋 **MANDATORY TOOL USAGE RULES (FOLLOW EXACTLY):**\n\n"
    
    "📚 FOR ANY QUESTION ABOUT THE UPLOADED DOCUMENTS:\n"
    "  MANDATORY: Use search_course FIRST.\n"
    "  1. Call search_course(\"{question}\"\n"
    "  2. If search_course returns results → Use those results in your answer\n"
    "  3. If search_course returns NO results → Then try search_wikipedia\n"
    "  NEVER skip search_course and go directly to Wikipedia!\n\n"
    
    "⚠️ **FOR DEFINITIONS/CONCEPTS:**\n"
    "  Step 1: search_course (to find in course)\n"
    "  Step 2: search_wikipedia (only if search_course found nothing)\n"
    "  Order is MANDATORY. Always search_course first.\n\n"
    
    "❓ FOR QUIZ/TEST REQUESTS (user says 'quiz me', 'test me', 'ask me about'):\n"
    "  MANDATORY: Use generate_quiz_context to extract course material.\n"
    "  Generate ONE multiple-choice question ONLY.\n"
    "  NEVER give the answer immediately.\n"
    "  Wait for user's attempt.\n\n"
    
    "📅 FOR STUDY PLANNING (user asks for 'planning', 'schedule', 'revision'):\n"
    "  MANDATORY: Use create_study_plan\n"
    "  Return Markdown table with columns: (Jour | Sujets | Objectifs d'apprentissage)\n\n"
    
    "🔢 FOR MATH/LOGIC/PROGRAMMING PROBLEMS:\n"
    "  MANDATORY: Use python_interpreter to compute\n"
    "  Show step-by-step working\n\n"
    
    "**FORMAT REQUIREMENTS:**\n"
    "- ALWAYS start with: 🧠 **Pensée**: [analysis]\n"
    "- THEN: → **Action**: [tool and why]\n"
    "- THEN: → **Observation**: [result]\n"
    "- END with: ✅ **Réponse**: [answer]\n\n"
    
    "**LANGUAGE RULE:**\n"
    "- ALWAYS respond in the same language as the user (French or English)\n\n"
    
    "**CRITICAL: If you don't follow these rules EXACTLY, you will fail the task.**\n"
)
```

---

## Key Changes

✅ Changed "I will" to "MANDATORY: Use search_course FIRST"
✅ Added explicit "Never skip search_course" instruction
✅ Changed "Suggestions" to "Rules you MUST follow"
✅ Added warning: "If you don't follow... you will fail"
✅ Made order EXPLICIT: search_course → Wikipedia

---

## Why This Works

The LLM will now understand:
- ❌ NOT optional suggestions
- ✅ MANDATORY rules
- ❌ Cannot skip search_course
- ✅ Must try search_course first
- ❌ Cannot go straight to Wikipedia
- ✅ Only fallback if search_course returns nothing

---

## How to Apply

1. **Open** `app.py`
2. **Find** line ~300 where `system_prompt = (` begins
3. **Replace** the entire prompt with the one above
4. **Save** file
5. **Test** again - agent should NOW call search_course first

---

## Testing

```
User: "what is lstm"

Console should show:
[DEBUG search_course]
  Query: what is lstm
  Results found: X  ← Should NOT be 0!

Agent response should start with:
🧠 Pensée: L'utilisateur pose une question...
→ Action: search_course  ← Should use THIS, not Wikipedia
→ Observation: Found X results in documents
✅ Réponse: From the course material...
```

---

## If Still Not Working

If agent STILL ignores search_course:

1. Check console for error in search_course
2. Check if vectorstore has documents (should show "Created XXX chunks")
3. Try simpler query: "LSTM" instead of "what is lstm"
4. Check if search_course returns ANY results

---

## Next Step

After applying this fix:
1. Test with "what is lstm"
2. Should see search_course being called
3. Should see document content being used
4. If still broken, let me know - might need different LLM model

**This MUST work after this change!** 💪
