# ReAct Agent for PDF Data Extraction using LangChain with Ollama
# This implementation uses local Ollama models: gemma2:4b for LLM and bge-m3:latest for embeddings

import os
from typing import List, Dict, Any
import pandas as pd
from pydantic import BaseModel, Field

# Core LangChain imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Document processing imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool

# Agent imports - Modern LangGraph approach
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Alternative: Traditional LangChain approach
from langchain.agents import create_react_agent as create_legacy_react_agent
from langchain.agents import AgentExecutor
from langchain import hub


class PDFAnalysisResult(BaseModel):
    """Structured output for PDF analysis"""

    summary: str = Field(description="Brief summary of the analysis")
    extracted_data: Dict[str, Any] = Field(description="Key data points extracted")
    table_found: bool = Field(description="Whether tables were detected")
    confidence_score: float = Field(description="Confidence in extraction accuracy")


class PDFReActAgent:
    """
    A ReAct agent specialized for extracting specific data and tables from PDF files.
    Uses local Ollama models for complete offline operation.

    This agent combines reasoning and action-taking capabilities to:
    1. Load and process PDF documents
    2. Search for specific information using semantic search
    3. Extract structured data and tables
    4. Provide detailed analysis with confidence scores

    Requirements:
    - Ollama installed locally
    - gemma2:4b model pulled (ollama pull gemma2:4b)
    - bge-m3:latest model pulled (ollama pull bge-m3:latest)
    """

    def __init__(
        self,
        model_name: str = "gemma3:4b",
        embedding_model: str = "bge-m3:latest",
        temperature: float = 0.1,
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the PDF ReAct Agent with Ollama models

        Args:
            model_name: Ollama model to use for chat (gemma2:4b recommended)
            embedding_model: Ollama model for embeddings (bge-m3:latest recommended)
            temperature: Controls randomness in responses (low for precise extraction)
            ollama_base_url: Base URL for Ollama server
        """
        # Initialize the language model with Ollama
        self.llm = ChatOllama(
            model=model_name, temperature=temperature, base_url=ollama_base_url
        )

        # Initialize embeddings with Ollama
        self.embeddings = OllamaEmbeddings(
            model=embedding_model, base_url=ollama_base_url
        )

        # Initialize text splitter for chunking documents
        # Smaller chunks for local models to ensure better context handling
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller for local models
            chunk_overlap=150,  # Maintains continuity
            separators=["\n\n", "\n", ". ", " ", ""],  # PDF-specific separators
        )

        # Storage for processed documents and vectorstore
        self.vectorstore = None
        self.current_pdf_path = None
        self.memory = MemorySaver()  # For conversation persistence

        # Initialize the agent (will be set up after tools are created)
        self.agent = None

        print(f"Initialized agent with Ollama models:")
        print(f"  - LLM: {model_name}")
        print(f"  - Embeddings: {embedding_model}")
        print(f"  - Server: {ollama_base_url}")

    def verify_ollama_models(self) -> bool:
        """
        Verify that required Ollama models are available

        Returns:
            bool: True if models are available, False otherwise
        """
        try:
            # Test LLM
            test_response = self.llm.invoke("Hello")
            print("✓ LLM model verified")

            # Test embeddings
            test_embedding = self.embeddings.embed_query("test")
            print("✓ Embedding model verified")

            return True
        except Exception as e:
            print(f"✗ Model verification failed: {str(e)}")
            print("\nPlease ensure:")
            print("1. Ollama is running (ollama serve)")
            print("2. Required models are pulled:")
            print("   - ollama pull gemma2:4b")
            print("   - ollama pull bge-m3:latest")
            return False

    def load_and_process_pdf(self, pdf_path: str) -> bool:
        """
        Load and process a PDF file for analysis

        Args:
            pdf_path: Path to the PDF file

        Returns:
            bool: Success status of PDF processing
        """
        try:
            print(f"Loading PDF: {pdf_path}")

            # Use PyMuPDFLoader for fast, accurate PDF processing
            # Mode 'page' preserves page structure, useful for table extraction
            loader = PyMuPDFLoader(
                pdf_path,
                mode="page",  # Process page by page for better structure
                extract_images=False,  # Set to True if image extraction needed
                extract_tables="markdown",  # Extract tables in markdown format
            )

            # Load documents
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from PDF")

            # Split documents into smaller chunks for better retrieval
            texts = self.text_splitter.split_documents(documents)
            print(f"Split into {len(texts)} chunks")

            # Create vector store for semantic search with Ollama embeddings
            print("Creating embeddings (this may take a moment with local models)...")
            self.vectorstore = FAISS.from_documents(texts, self.embeddings)
            self.current_pdf_path = pdf_path

            print("PDF processing completed successfully")
            return True

        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return False

    def create_pdf_tools(self) -> List:
        """
        Create specialized tools for PDF data extraction

        Returns:
            List of tools for the ReAct agent
        """

        @tool
        def pdf_semantic_search(query: str) -> str:
            """
            Search through the PDF content using semantic similarity.

            Use this tool when you need to find specific information, data points,
            or sections related to a particular topic in the PDF.

            Args:
                query: The search query describing what information you're looking for

            Returns:
                Relevant content from the PDF
            """
            if not self.vectorstore:
                return "No PDF has been loaded. Please load a PDF first."

            try:
                # Perform similarity search with moderate k for local models
                docs = self.vectorstore.similarity_search(query, k=4)

                # Combine results with page information for context
                results = []
                for i, doc in enumerate(docs):
                    page_num = doc.metadata.get("page", "Unknown")
                    content = doc.page_content.strip()
                    # Truncate very long content for local model efficiency
                    if len(content) > 500:
                        content = content[:500] + "..."
                    results.append(f"Page {page_num}: {content}")

                return "\n\n".join(results)
            except Exception as e:
                return f"Search error: {str(e)}"

        @tool
        def extract_numerical_data(search_terms: str) -> str:
            """
            Extract numerical data, statistics, or financial figures from the PDF.

            Use this tool when you need to find specific numbers, percentages,
            dates, amounts, or quantitative data.

            Args:
                search_terms: Keywords related to the numerical data you're seeking

            Returns:
                Extracted numerical data with context
            """
            if not self.vectorstore:
                return "No PDF has been loaded. Please load a PDF first."

            try:
                # Search for content likely to contain numerical data
                numerical_queries = [
                    f"{search_terms} numbers data statistics",
                    f"{search_terms} amount value percentage",
                ]

                all_results = []
                for query in numerical_queries:
                    docs = self.vectorstore.similarity_search(query, k=3)
                    for doc in docs:
                        content = doc.page_content
                        # Look for patterns that might contain numerical data
                        import re

                        numbers = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b", content)
                        if numbers:
                            page_num = doc.metadata.get("page", "Unknown")
                            # Limit content length for local model efficiency
                            preview = (
                                content[:300] + "..." if len(content) > 300 else content
                            )
                            all_results.append(f"Page {page_num}: {preview}")

                return (
                    "\n\n".join(all_results)
                    if all_results
                    else "No numerical data found for the given search terms."
                )
            except Exception as e:
                return f"Extraction error: {str(e)}"

        @tool
        def find_tables_and_structured_data(description: str) -> str:
            """
            Find and extract tables or structured data from the PDF.

            Use this tool when looking for tabular information, lists,
            or any structured data presentation.

            Args:
                description: Description of the table or structured data you're looking for

            Returns:
                Found tables and structured data
            """
            if not self.vectorstore:
                return "No PDF has been loaded. Please load a PDF first."

            try:
                # Search for table-like content
                table_queries = [
                    f"{description} table data structure",
                    f"{description} rows columns organized",
                ]

                results = []
                for query in table_queries:
                    docs = self.vectorstore.similarity_search(query, k=3)
                    for doc in docs:
                        content = doc.page_content
                        # Look for table indicators
                        if any(
                            indicator in content.lower()
                            for indicator in [
                                "table",
                                "|",
                                "row",
                                "column",
                                "---",
                                "header",
                            ]
                        ):
                            page_num = doc.metadata.get("page", "Unknown")
                            # Limit content for local model processing
                            table_content = (
                                content[:400] + "..." if len(content) > 400 else content
                            )
                            results.append(
                                f"Table found on Page {page_num}:\n{table_content}"
                            )

                return (
                    "\n\n".join(results)
                    if results
                    else "No tables found matching the description."
                )
            except Exception as e:
                return f"Table search error: {str(e)}"

        @tool
        def analyze_pdf_structure(focus_area: str = "overall") -> str:
            """
            Analyze the overall structure and organization of the PDF.

            Use this tool to understand document layout, section organization,
            and to identify the best approach for extracting specific information.

            Args:
                focus_area: Specific area to focus analysis on (default: "overall")

            Returns:
                Analysis of PDF structure and organization
            """
            if not self.vectorstore:
                return "No PDF has been loaded. Please load a PDF first."

            try:
                # Get a sample of documents to analyze structure
                sample_docs = self.vectorstore.similarity_search(focus_area, k=8)

                # Analyze content patterns
                total_pages = len(
                    set(doc.metadata.get("page", 0) for doc in sample_docs)
                )
                content_types = []

                for doc in sample_docs:
                    content = doc.page_content.lower()
                    if any(word in content for word in ["table", "figure", "chart"]):
                        content_types.append("Visual/Tabular")
                    elif any(
                        word in content
                        for word in ["abstract", "introduction", "conclusion"]
                    ):
                        content_types.append("Structured Text")
                    elif len(content.split()) < 50:
                        content_types.append("Headers/Metadata")
                    else:
                        content_types.append("Main Content")

                # Simple analysis without pandas dependency issues
                content_counts = {}
                for ct in content_types:
                    content_counts[ct] = content_counts.get(ct, 0) + 1

                most_common = (
                    max(content_counts, key=content_counts.get)
                    if content_counts
                    else "Unknown"
                )

                structure_analysis = f"""
PDF Structure Analysis:
- Total pages analyzed: {total_pages}
- Content distribution: {content_counts}
- Document appears to be: {most_common} focused
- Recommended extraction approach: {"Table extraction" if "Visual/Tabular" in content_counts else "Text analysis"}
                """

                return structure_analysis.strip()
            except Exception as e:
                return f"Analysis error: {str(e)}"

        # Create retriever tool for general document search
        if self.vectorstore:
            retriever_tool = create_retriever_tool(
                self.vectorstore.as_retriever(search_kwargs={"k": 4}),
                "pdf_document_retriever",
                "Retrieve relevant sections from the loaded PDF document based on query similarity.",
            )

            return [
                pdf_semantic_search,
                extract_numerical_data,
                find_tables_and_structured_data,
                analyze_pdf_structure,
                retriever_tool,
            ]
        else:
            return [
                pdf_semantic_search,
                extract_numerical_data,
                find_tables_and_structured_data,
                analyze_pdf_structure,
            ]

    def setup_agent(self) -> None:
        """
        Set up the ReAct agent with PDF-specific tools and prompt
        """
        # Create tools for PDF processing
        tools = self.create_pdf_tools()

        # Custom prompt optimized for local models
        system_prompt = """You are a PDF analysis expert using local AI models. 

Your job is to analyze PDF documents and extract specific data or tables as requested. 

Follow these guidelines:
1. Think step-by-step about what information you need
2. Choose the most appropriate tool for each task
3. Be precise and thorough in your analysis
4. Always provide page references when possible
5. If you need more information, use multiple tools
6. Keep responses focused and relevant

Available tools help you search, extract numbers, find tables, and analyze document structure.

Work efficiently and provide clear, actionable results."""

        # Modern LangGraph approach (recommended)
        self.agent = create_react_agent(
            model=self.llm,
            tools=tools,
            checkpointer=self.memory,  # Enables conversation memory
            prompt=system_prompt,
        )

        print("ReAct Agent initialized successfully!")

    def analyze_pdf_request(self, user_request: str, thread_id: str = "default") -> str:
        """
        Process a user request to extract information from the PDF

        Args:
            user_request: What the user wants to extract from the PDF
            thread_id: Conversation thread identifier for memory persistence

        Returns:
            Agent's response with extracted information
        """
        if not self.agent:
            return "Agent not initialized. Please call setup_agent() first."

        if not self.vectorstore:
            return "No PDF loaded. Please load a PDF file first using load_and_process_pdf()."

        try:
            # Configure the conversation thread for memory persistence
            config = {"configurable": {"thread_id": thread_id}}

            # Invoke the agent with the user request
            response = self.agent.invoke(
                {"messages": [HumanMessage(content=user_request)]}, config=config
            )

            # Extract the final message from the response
            if hasattr(response, "messages") and response.messages:
                return response.messages[-1].content
            else:
                return str(response)
        except Exception as e:
            return f"Error processing request: {str(e)}"


# Example usage and demonstration
def main():
    """
    Demonstration of the PDF ReAct Agent with Ollama
    """
    print("Initializing PDF ReAct Agent with Ollama...")
    print("=" * 60)

    # Create agent instance
    agent = PDFReActAgent()

    # Verify models are available
    if not agent.verify_ollama_models():
        print("\nSetup incomplete. Please ensure Ollama models are available.")
        return

    # Example: Load a PDF file (replace with your PDF path)
    pdf_path = "sample_document.pdf"  # Replace with actual PDF path

    print(f"\nAttempting to load PDF: {pdf_path}")

    if agent.load_and_process_pdf(pdf_path):
        # Set up the agent with tools
        agent.setup_agent()

        # Example requests for different types of data extraction
        example_requests = [
            "Find all financial data and revenue figures in this document",
            "Extract any tables related to performance metrics or statistics",
            "What is the main topic of this document and what key data does it contain?",
            "Look for any dates, deadlines, or time-related information",
            "Find contact information, names, or organizational details",
        ]

        print("\n" + "=" * 60)
        print("PDF REACT AGENT - EXAMPLE DEMONSTRATIONS")
        print("=" * 60)

        for i, request in enumerate(example_requests, 1):
            print(f"\nExample {i}: {request}")
            print("-" * 50)
            try:
                response = agent.analyze_pdf_request(request)
                print(response)
            except Exception as e:
                print(f"Error: {str(e)}")
            print("\n")

    else:
        print("Failed to load PDF. Please check the file path and try again.")
        print("\nTo test with a sample PDF, you can:")
        print("1. Download any PDF file")
        print("2. Update the 'pdf_path' variable with the correct path")
        print("3. Run the script again")


# Interactive mode for testing
def interactive_mode():
    """
    Interactive mode for testing the agent
    """
    print("PDF ReAct Agent - Interactive Mode")
    print("=" * 40)

    agent = PDFReActAgent()

    if not agent.verify_ollama_models():
        return

    # Get PDF path from user
    pdf_path = input("Enter path to PDF file: ").strip()
    pdf_path = (
        pdf_path
        if pdf_path
        else "/home/nitish/Documents/github/RAGStudio/docs/ISLP_website.pdf"
    )  # Default sample path

    if agent.load_and_process_pdf(pdf_path):
        agent.setup_agent()

        print("\nAgent ready! Type 'quit' to exit.")
        print("Example queries:")
        print("- Find all tables in this document")
        print("- Extract financial data")
        print("- What are the key findings?")
        print("- Analyze the document structure")

        while True:
            try:
                query = input("\nYour query: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    break

                if query:
                    print("\nProcessing...")
                    response = agent.analyze_pdf_request(query)
                    print(f"\nResponse:\n{response}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Choose mode
    mode = input("Choose mode: (1) Demo or (2) Interactive [1]: ").strip()

    if mode == "2":
        interactive_mode()
    else:
        main()
