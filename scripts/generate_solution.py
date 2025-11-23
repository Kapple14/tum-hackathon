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

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv

# Ensure src/ is on PYTHONPATH before imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_eval.resources.rag_prototype import (  # noqa: E402
    EmbeddingModel,
    LLMModel,
    ParseMode,
    RAGConfig,
    RAGPrototype,
    RetrievalMode,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


def validate_required_keys() -> None:
    """Ensure mandatory env vars are present."""
    required = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "JINA_API_KEY": os.getenv("JINA_API_KEY"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
        "QDRANT_URL": os.getenv("QDRANT_URL"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )
    logger.info("✅ All required API keys loaded successfully!")


def initialize_rag_system() -> RAGPrototype:
    """
    Initialize the multimodal RAG system using the Allplan manual,
    mirroring the setup from dual_evaluator_rag_pipeline.ipynb.

    This will:
    1. Load the Allplan PDF with multimodal LlamaParse (vision-based parsing)
    2. Create embeddings with Jina v4 (multimodal embeddings)
    3. Store in Qdrant vector database
    4. Set up retriever with vision-capable Claude for generation
    """

    logger.info(
        "Initializing multimodal RAG system with Allplan manual corpus...")
    validate_required_keys()

    # Check for optional LlamaParse API key
    llamaparse_key = os.getenv("LLAMAPARSE_API_KEY")
    if not llamaparse_key:
        logger.warning(
            "⚠️  LLAMAPARSE_API_KEY not found. "
            "Multimodal parsing requires this key. "
            "Will fall back to basic text extraction."
        )

    config = RAGConfig(
        retrieval_mode=RetrievalMode.MULTIMODAL,
        parse_mode=ParseMode.MULTIMODAL_LVM,
        enable_vision=True,
        vision_llm_model=LLMModel.CLAUDE_SONNET_3_5,
        llm_model=LLMModel.CLAUDE_HAIKU_4_5,
        embedding_model=EmbeddingModel.JINA_V4,
        collection_name="allplan_docs_collection",
        chunk_size=1024,
        chunk_overlap=200,
        top_k=5,
        force_recreate=False,  # Reuse existing embeddings if available
    )

    rag = RAGPrototype(config=config)

    # Deploy embeddings with force_reload to ensure multimodal parsing is used
    # This will reload the documents with LlamaParse if not already done
    rag.ensure_manual_embeddings(force_reload=True)

    logger.info("✅ RAG system initialized and ready for solution generation")
    logger.info(f"   📚 Documents: {len(rag._default_documents)}")
    logger.info(f"   🔍 Retrieval: top-{config.top_k}")
    logger.info(
        f"   🧠 LLM: {config.llm_model.value} (vision: {config.vision_llm_model.value})")
    logger.info(f"   📐 Embeddings: {config.embedding_model.value}")

    return rag


def main():
    """Main execution function."""
    # Define paths
    load_dotenv()

    project_root = PROJECT_ROOT
    dataset_path = project_root / "data" / "generated_qa_data_tum.json"
    output_path = project_root / "data" / "solution.json"

    try:
        logger.info("="*60)
        logger.info("STARTING SOLUTION GENERATION")
        logger.info("="*60)

        # Load dataset
        dataset = load_qa_dataset(dataset_path)

        # Initialize RAG system
        rag = initialize_rag_system()

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
