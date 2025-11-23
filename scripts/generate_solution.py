"""
Script to generate solution.json using the RAG system.

This script:
1. Loads the reference QA dataset from data/generated_qa_data_tum.json
2. Initializes a RAG system with document embeddings
3. Re-generates answers and evaluations for all questions using the RAG pipeline
4. Saves results to data/solution.json with the exact same data structure
5. Uses ONLY RAG-generated results (no reference answers/context)

The output maintains the same schema:
{index, question, answer, location_dependency_evaluator_target_answer, context, 
 groundedness_score, groundedness_eval, question_relevancy_score, question_relevancy_eval,
 faithfulness_score, faithfulness_eval}
"""

from ai_eval.resources.rag_prototype import (
    RAGPrototype,
    RAGConfig,
    RetrievalMode,
    EmbeddingModel,
    LLMModel,
)
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.docstore.document import Document

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def load_qa_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load QA dataset from JSON file."""
    import json
    logger.info(f"Loading QA dataset from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must contain a list of items")

    logger.info(f"Loaded {len(data)} QA pairs")
    return data


def extract_documents_from_dataset(dataset: List[Dict[str, Any]]) -> List[Document]:
    """Extract unique contexts as documents for the vectorstore."""
    logger.info("Extracting unique contexts as documents...")

    # Use a dict to deduplicate contexts
    unique_contexts = {}
    for item in dataset:
        context = item.get("context", "").strip()
        if context and context not in unique_contexts:
            unique_contexts[context] = Document(
                page_content=context,
                metadata={
                    "source": "generated_qa_data_tum",
                    "index": item.get("index", -1)
                }
            )

    documents = list(unique_contexts.values())
    logger.info(f"Extracted {len(documents)} unique documents")
    return documents


def initialize_rag_system(documents: List[Document]) -> RAGPrototype:
    """Initialize the RAG system with documents."""
    logger.info("Initializing RAG system...")

    # Check for required API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found in environment. "
            "Please set it in .env file."
        )

    if not os.getenv("JINA_API_KEY"):
        raise RuntimeError(
            "JINA_API_KEY not found in environment. "
            "Please set it in .env file for embeddings."
        )

    if not os.getenv("QDRANT_URL"):
        raise RuntimeError(
            "QDRANT_URL not found in environment. "
            "Please set it in .env file."
        )

    if not os.getenv("QDRANT_API_KEY"):
        raise RuntimeError(
            "QDRANT_API_KEY not found in environment. "
            "Please set it in .env file."
        )

    # Configure RAG system for text-only mode
    config = RAGConfig(
        retrieval_mode=RetrievalMode.TEXT_ONLY,
        embedding_model=EmbeddingModel.JINA_V4,
        llm_model=LLMModel.CLAUDE_HAIKU_4_5,
        top_k=3,  # Retrieve top 3 most relevant documents
        chunk_size=1024,
        chunk_overlap=200,
        collection_name="solution_generation",
        force_recreate=False,  # Reuse existing embeddings if collection exists
    )

    # Initialize RAG
    rag = RAGPrototype(config)

    # Deploy embeddings with the documents (only if not already deployed)
    if not rag.is_deployed:
        logger.info("Deploying embeddings...")
        rag.deploy_embeddings(documents)
    else:
        logger.info("✅ Using existing embeddings from collection: %s",
                    config.collection_name)

    logger.info("RAG system initialized successfully")
    return rag


def main():
    """Main execution function."""
    # Define paths
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "generated_qa_data_tum.json"
    output_path = project_root / "data" / "solution.json"

    try:
        logger.info("="*60)
        logger.info("STARTING SOLUTION GENERATION")
        logger.info("="*60)

        # Load dataset
        dataset = load_qa_dataset(dataset_path)

        # Extract documents for RAG
        documents = extract_documents_from_dataset(dataset)

        # Initialize RAG system
        rag = initialize_rag_system(documents)

        # Extract questions from dataset
        questions = [item.get("question", "") for item in dataset]
        logger.info(f"Extracted {len(questions)} questions from dataset")

        # Generate solution file using the new method
        logger.info("Generating solution file using RAG...")
        rag.generate_solution_file(
            reference_dataset_path=dataset_path,
            output_path=output_path,
            limit=None,  # Process all questions
            overwrite=True,
            show_progress=True,
        )

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("GENERATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
