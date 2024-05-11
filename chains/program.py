from prompt import BASIC_CHAT_TEMPLATE
from chains.common import CommonChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


META_FIELD_INFO = [
    AttributeInfo(
        name="content_type",
        description="""
        The type of the content for each department 
        Can contain the following options individually or together [Faculty, Certificate, Graduate, Undergraduate]
        Ensure to do a case insensitive match
        """.strip(),
        type="string",
    ),
    AttributeInfo(
        name="program_name",
        description="""
        The name of the Program
        Sample program names are Compute Science and Machine Learning
        Ensure to do a case insensitive match
        """.strip(),
        type="string",
    ),
    AttributeInfo(
        name="department",
        description="The department or school that has that program",
        type="string",
    ),
    AttributeInfo(
        name="program_type",
        description="""
        The type of the program offered by the university ['Masters', 'Bachelors', 'Certificate', 'PhD', 'Engineer']
        MS/Master stands for Masters
        BS/Bachelor stands for Bachelors
        Doctoral/Doctrate stands for PhD
        Engg stands for Engineer 
        Ensure to do a case insensitive match
        """.strip(),
        type="string",
    ),
]
DOCUMENT_CONTENT_DESCRIPTION = "Information regarding the program"


class ProgramChain(CommonChain):

    def process_prompt(self, llm, prompt: str) -> str:

        retriever = SelfQueryRetriever.from_llm(
            llm,
            self.vectorstore,
            DOCUMENT_CONTENT_DESCRIPTION,
            META_FIELD_INFO,
            structured_query_translator=ChromaTranslator(),
            enable_limit=True,
        )

        combine_docs_chain = create_stuff_documents_chain(llm, BASIC_CHAT_TEMPLATE)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        retriever_prompt = f"""
        Pick the top 5 result for the following question: ```{prompt}```
        """.strip()

        return retrieval_chain.invoke({"input": retriever_prompt, "question": prompt})
