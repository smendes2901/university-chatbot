from chains.common import CommonChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from prompt import COURSE_CHAT_TEMPLATE

META_FIELD_INFO = [
    AttributeInfo(
        name="course_name",
        description="""
        The name of the Course
        Sample course name are Computer Engineering and Machine Learning
        Ensure to do a case insensitive match
        """.strip(),
        type="string",
    ),
    AttributeInfo(
        name="course_acronym",
        description="""
        The acronym of the Course
        The acronym of the Course[SYS,PME,EE,HSS,MA,CM,DS,FE,CHE,NE,CH,HAR,ACC,HPL,MT,CPE,CE,CAL,ENGR,SSW,HLI,PEP,FIN,COMM,OE,HST,EM,CS,BIA,ME,FA,NIS,BIOE,TG,LFR,HMU,EN,ELC,MGT,HHS,MIS,SM,BT,AAI,SES,BIO,BME,CLK,TM,TE,IPD,LTL,DE,QF,LSP,NANO,ISE,EMT,GEN,IDE,PIN,ES,ECON,HTH,LCH,SEF,HONR,SOC,HUM,PAE]
        """.strip(),
        type="string",
    ),
    AttributeInfo(
        name="course_number",
        description="The number for that particular course",
        type="integer",
    ),
    AttributeInfo(
        name="course_credits",
        description="""
        The number of credits obtained on completion of the course
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

        combine_docs_chain = create_stuff_documents_chain(llm, COURSE_CHAT_TEMPLATE)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        retriever_prompt = f"""
        ```{prompt}```
        Sample filters are:
        and(eq("course_acronym", "CPE"),  eq("course_number", 100))
        or(eq("course_acronym", "ML"), eq("course_number", 678))""".strip()

        return retrieval_chain.invoke({"input": retriever_prompt, "question": prompt})
