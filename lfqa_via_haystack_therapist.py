from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.nodes import Seq2SeqGenerator, TransformersSummarizer
from haystack.pipelines import GenerativeQAPipeline, SearchSummarizationPipeline
from haystack.utils import print_documents, print_answers
from haystack.pipelines import DocumentSearchPipeline

import os
import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

datafolder = "data_therapist_files"
db_name = "faiss_therapist"

if not os.path.exists(f"{db_name}.json"):
    document_store = FAISSDocumentStore(sql_url=f"sqlite:///faiss_doc_store_{db_name}.db", embedding_dim=128, faiss_index_factory_str="Flat")

    from haystack.utils import convert_files_to_docs, fetch_archive_from_http, clean_wiki_text

    doc_dir = datafolder
    docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    document_store.write_documents(docs)

    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
        passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
    )
    document_store.update_embeddings(retriever)

    document_store.save(f"{db_name}")
else:
    document_store = FAISSDocumentStore(faiss_index_path=f"{db_name}", faiss_config_path=f"{db_name}.json")

    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
        passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
    )


# p_retrieval = DocumentSearchPipeline(retriever)
# res = p_retrieval.run(query="Tell me something about Arya Stark?", params={"Retriever": {"top_k": 10}})
# print_documents(res, max_text_len=512)

generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
# summarizer = TransformersSummarizer(model_name_or_path="google/pegasus-xsum")
summarizer = TransformersSummarizer(model_name_or_path="philschmid/bart-large-cnn-samsum")

pipe = GenerativeQAPipeline(generator, retriever)
#pipe = SearchSummarizationPipeline(summarizer, retriever, generate_single_summary=True)

result = pipe.run(
    query="There are many times when I think about dying. My heart beats often. How can I handle this?", 
    params={"Retriever": {"top_k": 5}, "Generator": {"top_k": 1}}
)
print(result.keys())
print_documents(result, max_text_len=512, print_name=True, print_meta=True)
print_answers(result, max_text_len=512)

result = pipe.run(
    query="I feel bad. How can I handle this?", 
    params={"Retriever": {"top_k": 5}, "Generator": {"top_k": 1}}
)
print_documents(result, max_text_len=512, print_name=True, print_meta=True)
print_answers(result, max_text_len=512)
