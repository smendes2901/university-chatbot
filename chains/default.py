from prompt import DEFAULT_CHAT_TEMPLATE
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class DefaultChain:

    def process_prompt(self, llm, prompt: str) -> str:
        retreival_chain = (
            {"question": RunnablePassthrough()}
            | DEFAULT_CHAT_TEMPLATE
            | llm
            | StrOutputParser()
        )
        return {"answer": retreival_chain.invoke(prompt)}
