# 🔧 EXACT CODE FIXES - Copy & Paste

## Fix 1: Update build_vector_store Function

**Location**: Around line 57 in app.py

### ❌ FIND THIS:
```python
def build_vector_store(documents):
    """Indexation."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore
```

### ✅ REPLACE WITH THIS:
```python
def build_vector_store(documents):
    """Indexation with optimized chunking."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # Smaller chunks for better retrieval
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    splits = text_splitter.split_documents(documents)
    print(f"DEBUG: Created {len(splits)} chunks from {len(documents)} documents")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    print(f"DEBUG: Vectorstore indexed with {vectorstore.index.ntotal} vectors")
    
    return vectorstore
```

**What changed:**
- Reduced chunk_size from 1000 to 500 (better granularity)
- Reduced chunk_overlap from 200 to 100 (less redundancy)
- Added separators for better splitting
- Added debug output

---

## Fix 2: Update search_course Function

**Location**: Around line 68 in app.py

### ❌ FIND THIS:
```python
@tool
def search_course(query: str) -> str:
    """
    Searches for information strictly within the uploaded PDF course documents.
    Use this tool to answer questions about the specific course content.
    """
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "Aucun document n'est charge."
    
    results = st.session_state.vectorstore.similarity_search(query, k=4)
    
    if not results:
        return f"Aucun resultat trouve pour '{query}' dans les documents."
    
    return "\n\n".join([doc.page_content for doc in results])
```

### ✅ REPLACE WITH THIS:
```python
@tool
def search_course(query: str) -> str:
    """
    Searches for information strictly within the uploaded PDF course documents.
    Use this tool to answer questions about the specific course content.
    """
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "ERROR: Vectorstore not initialized. No documents loaded."
    
    print(f"\n[DEBUG search_course]")
    print(f"  Query: {query}")
    print(f"  Vectorstore size: {st.session_state.vectorstore.index.ntotal}")
    
    try:
        results = st.session_state.vectorstore.similarity_search(query, k=4)
        print(f"  Results found: {len(results)}")
        
        if not results or len(results) == 0:
            print(f"  No matches for '{query}'")
            return f"No content found for '{query}' in documents. Try different keywords or check search_wikipedia."
        
        # Format results with source information
        formatted = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown source')
            print(f"  Match {i}: {source} - {doc.page_content[:50]}...")
            formatted.append(f"[From {source}]\n{doc.page_content}")
        
        return "\n---\n".join(formatted)
    
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return f"Error searching documents: {str(e)}"
```

**What changed:**
- Better error messages that don't confuse the agent
- Debug output to console so you can see what's happening
- Catches exceptions properly
- Formats results with source citations
- Distinguishes between "no vectorstore" vs "no results"

---

## Fix 3: Add Search Quality Check (Optional but Recommended)

**Add this new function after build_vector_store():**

```python
def check_vectorstore_quality(vectorstore, test_queries):
    """Check if vectorstore is working by running test queries."""
    print("\n[VECTORSTORE QUALITY CHECK]")
    print(f"Total vectors indexed: {vectorstore.index.ntotal}")
    
    for query in test_queries:
        results = vectorstore.similarity_search(query, k=2)
        print(f"\nQuery: '{query}'")
        print(f"  Matches: {len(results)}")
        if results:
            print(f"  Top match: {results[0].page_content[:100]}...")
```

Then in `main()`, after creating vectorstore, add:

```python
if process_btn and files:
    with st.spinner("Analyse..."):
        docs = process_documents(files)
        st.session_state.vectorstore = build_vector_store(docs)
        
        # Check vectorstore quality
        check_vectorstore_quality(
            st.session_state.vectorstore,
            ["LSTM", "gradient", "neurone", "apprentissage"]
        )
        
        st.success("Pret !")
```

---

## How to Apply

1. **Open** `app.py`
2. **Find** line ~57 (build_vector_store function)
3. **Replace** entire function with Fix 1
4. **Find** line ~68 (search_course function)
5. **Replace** entire function with Fix 2
6. **Save** file
7. **Test**: Upload PDFs again and watch console for debug output

---

## Testing

### Step 1: Upload & Process
```
1. Upload your 3 PDFs
2. Click "Traiter les documents"
3. Watch console for:
   [DEBUG: Created XXX chunks...]
   [DEBUG: Vectorstore indexed with XXX vectors]
   [VECTORSTORE QUALITY CHECK]
```

### Step 2: Search Test
```
In chat, ask: "C'est quoi un LSTM?"

Watch console for:
[DEBUG search_course]
  Query: C'est quoi un LSTM?
  Vectorstore size: 145
  Results found: 3  ← Should NOT be 0!
  Match 1: Lecture 03...
```

### Step 3: Verify Response
```
Agent should show:
🦯 Pensée: L'utilisateur pose une question sur les LSTMs...
→ Action: search_course('LSTM')
→ Observation: Found 3 relevant sections
✅ Réponse: Un LSTM est... [actual course content]
```

---

## If Still No Results

If console shows "Results found: 0":

1. Check PDF content is being loaded:
   ```python
   # In process_documents(), add after loader.load():
   print(f"Loaded {len(docs)} pages from {file.name}")
   if docs:
       print(f"First page: {docs[0].page_content[:200]}...")
   ```

2. Check chunks are created:
   ```python
   # The debug output will show chunk count
   # Should be > 10 for reasonable PDFs
   ```

3. Try simpler queries:
   ```
   Instead of: "C'est quoi un vanishing gradient?"
   Try: "LSTM"
   Or: "gradient"
   ```

4. Check similarity scores:
   ```python
   # Add this to search_course after similarity_search:
   results = vectorstore.similarity_search_with_scores(query, k=4)
   for doc, score in results:
       print(f"Score: {score:.3f} - {doc.page_content[:50]}...")
   ```

---

## Expected Outcome

After applying these fixes:

✅ Smaller chunks = better matching  
✅ Debug output = visibility into what's happening  
✅ Better error messages = clearer feedback  
✅ search_course finds documents = no Wikipedia fallback  
✅ CoT shows actual document content = meets project requirements  

**Result: Your RAG system actually works!** 🚀
