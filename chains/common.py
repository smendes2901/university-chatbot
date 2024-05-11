from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from chains.utils import format_docs
from prompt import BASIC_CHAT_TEMPLATE
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma


class CommonChain:

    def __init__(self, collection_name, embeddings) -> None:
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory="data/chromadb",
            embedding_function=embeddings,
        )

    def process_prompt(self, llm, prompt: str) -> str:
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | BASIC_CHAT_TEMPLATE
            | llm
            | StrOutputParser()
        )
        retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )
        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        return rag_chain_with_source.invoke(prompt)
