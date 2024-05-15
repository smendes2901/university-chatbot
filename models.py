from langchain_openai import ChatOpenAI


def initialize_llm(name) -> ChatOpenAI:
    model = {"gpt3": "gpt-3.5-turbo-0613", "gpt4": "gpt-4-1106-preview"}
    return ChatOpenAI(
        model=model[name],
        temperature=0,
    )
