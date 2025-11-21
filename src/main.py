# %% [markdown]
# # Some quick start code for TUM Hackathon

# %%
from ai_eval.resources import deepeval_scorer as deep
from ai_eval.resources.rag_template import FAISSRAG
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from ai_eval.resources import eval_dataset_builder as eval
from ai_eval.services.file import JSONService
from ai_eval.resources.preprocessor import Preprocessor
from langchain.document_loaders import PyPDFLoader
from ai_eval.config import global_config as glob

filename = "Allplan_2020_Manual.pdf"

loader = PyPDFLoader(f"{glob.DATA_PKG_DIR}/{filename}")

raw_data = loader.load()

texts = [page.page_content for page in raw_data]

print(f"Number of docs: {len(texts)}")

# %% [markdown]
# ## (Optional) Preprocess and load data:

# %%

filename = "Allplan_2020_Manual.pdf"

pre = Preprocessor()

docs = pre.fetch_documents(
    blob_path=f"{glob.DATA_PKG_DIR}/{filename}", source="local"
)

documents = pre.chunk_documents(documents=docs)

print(f"Number of processed document chunks: {len(documents)}")

# %% [markdown]
# ## Get annotated data:

# %%

json = JSONService(path="generated_qa_data_tum.json",
                   root_path=glob.DATA_PKG_DIR, verbose=True)

qa_data = json.doRead()
print(f"Number of evaluation data samples: {len(qa_data)}")

# %% [markdown]
# ### Fit RAG model on the generated data and create evaluation dataset

# %%

ground_truth_contexts = [item["context"] for item in qa_data]
sample_queries = [item["question"] for item in qa_data]
expected_responses = [item["answer"] for item in qa_data]

# %% [markdown]
# Example: using Vertex AI models

# %%

chat_model = ChatVertexAI(
    project=glob.GCP_PROJECT,
    model_name="gemini-2.5-flash",
    temperature=0.1,
    max_retries=2,
)

embedding_model = VertexAIEmbeddings(
    project=glob.GCP_PROJECT,
    model_name="text-embedding-005",
)

# %%

vectorstore = FAISS.from_documents(documents, embedding_model)

# 1. Create your RAG instance
rag = FAISSRAG(chat_model, documents, k=3,
               vectorstore=vectorstore)    # some vanilla example
# rag = TFIDFRAG(models.qa_generator, documents, k=3)                 # our (naive) hackathon baseline

query = "What is Allplan?"

the_relevant_docs = rag.retrieve(question=query)

answer, relevant_docs = rag.answer(question=query)

# %%
# 2. Create the builder with the RAG instance
builder = eval.EvalDatasetBuilder(rag)

# 3. Build the evaluation dataset
evaluation_dataset = builder.build_evaluation_dataset(
    input_contexts=ground_truth_contexts,
    sample_queries=sample_queries,
    expected_responses=expected_responses,
)

# %%

scorer = deep.DeepEvalScorer(evaluation_dataset)

results = scorer.calculate_scores()
print(results)

# %%
scorer.get_overall_metrics()

# %%
scorer.get_summary(save_to_file=True)

# %%
