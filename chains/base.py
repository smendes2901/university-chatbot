from abc import ABC, abstractmethod
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from chains.utils import format_docs
from prompt import chat_template
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma


class CommonChain:

    def __init__(self, filename, embeddings) -> None:
        graduate_education_df = pd.read_csv(filename)
        docs = [
            Document(page_content=text)
            for text in graduate_education_df["text"].tolist()
        ]
        self.vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    def process_prompt(self, llm, prompt: str) -> str:
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | chat_template
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
