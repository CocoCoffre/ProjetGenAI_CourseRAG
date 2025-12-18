# 🔧 Document Retrieval Bug - Complete Fix

## What Went Wrong

After implementing CoT in the system prompt, your tools stopped accessing documents properly.

## Root Cause

Two issues combined:

### Issue 1: Malformed output in `create_study_plan`
```python
context_str += f"**📖 {filename}**\n``````\n\n"
                                 ^^^^^^ This is BROKEN (6 backticks)
```
This creates malformed text that confuses the LLM.

### Issue 2: Special characters in tool outputs
Your tools return emoji and special formatting that the LLM sees as noise.

## The Fix

### Fix 1: `search_course` function (around line 68)

**FIND:**
```python
@tool
def search_course(query: str) -> str:
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "Aucun document n'est chargé."
    
    results = st.session_state.vectorstore.similarity_search(query, k=4)
    return "\n\n".join([doc.page_content for doc in results])
```

**REPLACE WITH:**
```python
@tool
def search_course(query: str) -> str:
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "Aucun document n'est charge."
    
    results = st.session_state.vectorstore.similarity_search(query, k=4)
    
    if not results:
        return f"Aucun resultat trouve pour '{query}' dans les documents."
    
    return "\n\n".join([doc.page_content for doc in results])
```

### Fix 2: `create_study_plan` function (around line 143)

**FIND:**
```python
@tool
def create_study_plan(days: int, focus: str = "All") -> str:
    previews = st.session_state.get("doc_previews", {})
    
    if not previews:
        return (
            "❌ Aucun document détecté..."
        )
    
    context_str = "📚 **Documents disponibles pour la révision :**\n\n"
    for filename, preview in previews.items():
        context_str += f"**📖 {filename}**\n``````\n\n"
    
    return (
        f"{context_str}\n"
        f"**INSTRUCTION POUR L'AGENT :**\n"
        f"Crée un planning..."
    )
```

**REPLACE WITH:**
```python
@tool
def create_study_plan(days: int, focus: str = "All") -> str:
    previews = st.session_state.get("doc_previews", {})
    
    if not previews:
        return (
            "Aucun document detecte. Verifiez que vous avez uploade et traite les PDFs."
        )
    
    context_str = "Documents disponibles pour la revision:\n\n"
    for filename, preview in previews.items():
        context_str += f"{filename}\nDebut: {preview[:400]}...\n\n"
    
    return (
        f"{context_str}\n"
        f"Cree un planning de revision detaille sur {days} jour(s) en citant les themes principaux "
        f"de chaque document. Format: tableau Markdown avec colonnes "
        f"(Jour | Sujets a reviser | Objectifs d'apprentissage)."
    )
```

## Testing After Fix

```
✅ Upload PDFs → See green "Mémoire chargée"
✅ Ask: "C'est quoi un LSTM?" → Should search_course, not Wikipedia
✅ Ask: "Fais moi un planning sur 3 jours" → Should use create_study_plan
✅ Ask: "Teste-moi sur les RNN" → Should generate quiz question
```

## Expected Result

CoT format with document access:
```
Pensée: [thinking]
Action: [tool choice]
Observation: [document result]
Réponse: [answer]
```

Instead of falling back to Wikipedia!
