from chains.common import CommonChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question-answering tasks related to Stevens Institute Of Technology.",
        ),
        (
            "human",
            """ 
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        If the topic is related to a course then ensure to mention to course numbers.
        Question: {question}
        Context: {context}
        Answer:""",
        ),
    ]
)

META_FIELD_INFO = [
    AttributeInfo(
        name="program_name",
        description="""
        The program_name of the Program
        Sample program name are Computer Engineering and Machine Learning
        Ensure to do a case insensitive match
        """.strip(),
        type="string",
    ),
    AttributeInfo(
        name="program_name",
        description="""
        The program_acronym of the Program
        Sample program acronyms are CPE for Computer Engineering and ML for Machine Learning
        Ensure to do a case insensitive match
        """.strip(),
        type="string",
    ),
    AttributeInfo(
        name="course_number",
        description="The number for that particular course. This is a float field",
        type="float",
    ),
    AttributeInfo(
        name="course_credits",
        description="""
        The number of credits obtained on completion of the course. This is a float field
        """.strip(),
        type="float",
    ),
]
DOCUMENT_CONTENT_DESCRIPTION = "Information regarding the course"


class CourseChain(CommonChain):

    def process_prompt(self, llm, prompt: str) -> str:

        retriever = SelfQueryRetriever.from_llm(
            llm,
            self.vectorstore,
            DOCUMENT_CONTENT_DESCRIPTION,
            META_FIELD_INFO,
            structured_query_translator=ChromaTranslator(),
            enable_limit=True,
        )

        combine_docs_chain = create_stuff_documents_chain(llm, chat_template)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        retriever_prompt = f"{prompt}".strip()

        return retrieval_chain.invoke({"input": retriever_prompt, "question": prompt})
