"""
Script to generate RAG answers for all questions in the QA dataset.

This script:
1. Loads the QA dataset from data/generated_qa_data_tum.json
2. Initializes a RAG system with document embeddings
3. Generates answers for all questions using the RAG pipeline
4. Saves results to data/rag_results.json in the same format
"""

from ai_eval.resources.rag_prototype import (
    RAGPrototype,
    RAGConfig,
    RetrievalMode,
    EmbeddingModel,
    LLMModel,
)
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.docstore.document import Document
from tqdm import tqdm

# Add parent directory to path
import sys
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

    # Configure RAG system for text-only mode
    config = RAGConfig(
        retrieval_mode=RetrievalMode.TEXT_ONLY,
        embedding_model=EmbeddingModel.JINA_V3,
        llm_model=LLMModel.CLAUDE_HAIKU_4_5,
        top_k=3,  # Retrieve top 3 most relevant documents
        chunk_size=512,
        chunk_overlap=50,
        use_qdrant=False,  # Use in-memory vectorstore for speed
        collection_name=f"qa_eval_{os.getpid()}",
    )

    # Initialize RAG
    rag = RAGPrototype(config)

    # Deploy embeddings with the documents
    logger.info("Deploying embeddings...")
    rag.deploy_embeddings(documents)

    logger.info("RAG system initialized successfully")
    return rag


def generate_answers(
    dataset: List[Dict[str, Any]],
    rag: RAGPrototype,
    output_path: Path,
) -> List[Dict[str, Any]]:
    """Generate answers for all questions using RAG."""
    logger.info(f"Generating answers for {len(dataset)} questions...")

    results = []

    for item in tqdm(dataset, desc="Generating answers"):
        question = item.get("question", "")

        if not question:
            logger.warning(
                f"Skipping item {item.get('index', '?')} - no question found")
            # Keep original item unchanged
            results.append(item.copy())
            continue

        try:
            # Generate answer using RAG
            generated_answer, relevant_docs = rag.answer(question)

            # Create result item with generated answer
            result_item = item.copy()
            result_item["generated_answer"] = generated_answer
            result_item["generated_context"] = "\n\n".join([
                doc.page_content for doc in relevant_docs
            ])
            result_item["num_retrieved_docs"] = len(relevant_docs)

            results.append(result_item)

        except Exception as e:
            logger.error(
                f"Error generating answer for item {item.get('index', '?')}: {e}")
            # Keep original item with error info
            result_item = item.copy()
            result_item["generated_answer"] = f"ERROR: {str(e)}"
            result_item["generated_context"] = ""
            result_item["num_retrieved_docs"] = 0
            results.append(result_item)

    logger.info(f"Generated {len(results)} results")
    return results


def save_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save results to JSON file."""
    logger.info(f"Saving results to {output_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with pretty formatting
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved successfully to {output_path}")


def main():
    """Main execution function."""
    # Define paths
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "generated_qa_data_tum.json"
    output_path = project_root / "data" / "rag_results.json"

    try:
        # Load dataset
        dataset = load_qa_dataset(dataset_path)

        # Extract documents for RAG
        documents = extract_documents_from_dataset(dataset)

        # Initialize RAG system
        rag = initialize_rag_system(documents)

        # Generate answers
        results = generate_answers(dataset, rag, output_path)

        # Save results
        save_results(results, output_path)

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"Total questions processed: {len(results)}")

        successful = sum(1 for r in results if not r.get(
            "generated_answer", "").startswith("ERROR"))
        logger.info(f"Successful generations: {successful}")
        logger.info(f"Failed generations: {len(results) - successful}")
        logger.info(f"Output saved to: {output_path}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
