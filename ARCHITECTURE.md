Architecture: Local-First RAG SystemThis document details the architecture of the local-first Enterprise Knowledge Assistant, which is orchestrated by Docker Compose.Core PhilosophyThe system is built on a "Two-Model" RAG architecture:Embedding Model (The "Librarian"): A small, fast model (all-mpnet-base-v2) that excels at one thing: converting text into high-dimensional vectors (numeric "fingerprints"). It is used to index documents and find context.Language Model (The "Brain"): A large, powerful model (Mistral-7B-Instruct) that excels at reasoning and writing. It is used to read the context (found by the "Librarian") and write the final answer.This separation of concerns is efficient: the "Librarian" is cheap to run and can scan millions of documents, while the "Brain" is only invoked on a few, highly-relevant paragraphs.Component Overviewui (Streamlit): The user-facing web application. It takes a user's question and sends it to the api.api (FastAPI): The central "orchestrator." It implements the RAG pipeline, coordinates all other services, and enforces the "zero-hallucination" business logic.llama-cpp (LLM Server): A dedicated server that hosts the "Brain" (Mistral-7B-Instruct) and makes it available via an OpenAI-compatible API.qdrant (Vector Database): The specialized database that stores the vector "fingerprints" (embeddings) of all Confluence document chunks.ingest (Ingestion Service): A one-time job that populates the qdrant database. It contains the "Librarian" (all-mpnet-base-v2) and the Confluence sync logic.Architecture Diagram (Local Docker Network)This diagram shows how the services communicate inside the Docker network.graph TD
    subgraph "Your Local Machine (Host)"
        U(User) -- "http://localhost:8501" --> UI
    end

    subgraph "Docker Network (docker-compose)"
        UI(fa:fa-window-maximize UI - Streamlit) -- "POST /ask (http://api:8000)" --> API(fa:fa-microchip API - FastAPI)

        API -- "1. Embed Query" --> EM_API(Embedding Model<br>in API)
        EM_API -- "2. Vector Search" --> Q(fa:fa-database Qdrant DB)

        API -- "3. Build Prompt w/ Context" --> API
        API -- "4. Get Answer (http://llama-cpp:8080)" --> LLM(fa:fa-brain LLM Server<br>llama-cpp)

        LLM -- "5. Answer" --> API
        API -- "6. Final Response" --> UI

        subgraph "One-Time Job"
            I(fa:fa-download Ingest Service) -- "Fetch Pages" --> C(fa:fa-cloud Confluence API)
            I -- "Embed Chunks" --> EM_I(Embedding Model<br>in Ingest)
            EM_I -- "Write Vectors" --> Q
        end
    end

    style U fill:#fff,stroke:#333,stroke-width:2px
    style C fill:#fff,stroke:#333,stroke-width:2px
Data Flow 1: Ingestion (Populating the Database)This process runs once when you first start the system via the ingest service.Run: The ingest container starts.Initialize DB: src/init_db.py runs, connecting to qdrant and creating the confluence_kb collection with the correct 768-dimension vector size and indexes.Fetch: src/data_pipeline.py connects to the Confluence API (using your .env credentials) and fetches all pages from the specified CONFLUENCE_SPACES.Clean & Chunk: For each page, the raw HTML is cleaned. The text is split into 300-500 token chunks (using tiktoken) to ensure semantic meaning.Embed: The "Librarian" (all-mpnet-base-v2) reads each chunk and generates a 768-dimension vector embedding.Store: The service intelligently deletes old versions of the page from qdrant and upserts the new chunks, their text, and their metadata (URL, title, version) into the qdrant database.Exit: The ingest container finishes its job and stops.Data Flow 2: Query (Answering a Question)This is the main RAG pipeline, all orchestrated by the api service.Query: A user types "What is our WFH policy?" into the ui (Streamlit).API Call: The ui sends a POST request to http://api:8000/ask with the query.Embed: The api service's "Librarian" (all-mpnet-base-v2) converts the query "What is our WFH policy?" into a vector.Search: The api sends this query vector to qdrant and asks, "Find me the top 4 most similar vectors."Retrieve: qdrant instantly returns the top 4 matching chunks, which include the raw text and metadata (the "context").Zero-Hallucination Check:If no chunks are found (or they are below the similarity threshold), the api stops. It does not call the LLM. It returns the "I don't have enough information..." message.If context is found, the pipeline continues.Build Prompt: The api assembles a large, complex prompt for the "Brain" (LLM), which looks like this:System: You are an enterprise knowledge assistant...
User: Context:
[1] (v5) WFH Policy: ...employees must be available...
[2] (v2) Remote Work: ...all remote staff must use the VPN...

Question:
What is our WFH policy?

Response:
Generate: The api sends this complete prompt to the llama-cpp service.Reason: The llama-cpp server (running Mistral-7B-Instruct) reads the prompt and writes an answer based only on the context provided.Respond: The api gets the answer, packages it with the source metadata (links, titles), and sends it back to the ui.Display: The ui displays the answer and the clickable source links.