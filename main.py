import json
from chains.common import CommonChain
from chains.program import ProgramChain
from chains.course import CourseChain
from chains.default import DefaultChain
from embedding import initialize_embeddings
from models import initialize_llm
from router import PromptRouter

llm = initialize_llm("gpt4")
embeddings = initialize_embeddings()

chain_info = json.load(open("chains/chain_info.json"))

chains = {
    "financing_a_stevens_education": CommonChain(
        "financing_a_stevens_education", embeddings
    ),
    "tuition_fees_and_other_expenses_for_undergraduate_students": CommonChain(
        "tuition_fees_and_other_expenses_for_undergraduate_students", embeddings
    ),
    "tuition_fees_and_other_expenses_for_graduate_students": CommonChain(
        "tuition_fees_and_other_expenses_for_graduate_students", embeddings
    ),
    "student_life_at_stevens": CommonChain("student_life_at_stevens", embeddings),
    "student_services_at_stevens": CommonChain(
        "student_services_at_stevens", embeddings
    ),
    "graduate_education_at_stevens": CommonChain(
        "graduate_education_at_stevens", embeddings
    ),
    "undergraduate_education_at_stevens": CommonChain(
        "undergraduate_education_at_stevens", embeddings
    ),
    "program_info_at_stevens": ProgramChain("department_info", embeddings),
    "off_campus_employment": CommonChain("off_campus_employment", embeddings),
    "course_info": CourseChain("course_info", embeddings),
}

print(chains["course_info"].vectorstore._collection.count())

router = PromptRouter(llm=llm, chain_info=chain_info)

import streamlit as st

st.title("UNIGUIDE🦆")

with st.form("my_form"):
    text = st.text_area("Enter text:", "Hello")
    chain_name = router.get_chain_name(prompt=text)
    print(chain_name)
    curr_chain = chains.get(chain_name, DefaultChain())
    submitted = st.form_submit_button("Submit")
    response = curr_chain.process_prompt(llm, prompt=text)
    st.markdown(response["answer"])
