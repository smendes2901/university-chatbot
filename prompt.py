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
        If the topic is related to a course then ensure to mention to course numbers and display the result as a table.
        Answer in markdown format and render tables without code 
        Question: {question}
        Context: {context}
        Answer:""".strip(),
        ),
    ]
)
