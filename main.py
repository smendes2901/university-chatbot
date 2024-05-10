import json
from langchain_community.llms import OpenAI


from chains.common import CommonChain
from chains.program import ProgramChain
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
    "course_info": CommonChain("course_info", embeddings),
}

router = PromptRouter(llm=llm, chain_info=chain_info)

import streamlit as st

st.title("Stevens WebUI")


def generate_response(input_text):
    llm = OpenAI(temperature=0)
    st.info(llm(input_text))


with st.form("my_form"):
    text = st.text_area("Enter text:")
    chain_name = router.get_chain_name(prompt=text)
    curr_chain = chains.get(chain_name)
    submitted = st.form_submit_button("Submit")
    if curr_chain:
        response = curr_chain.process_prompt(llm, prompt=text)
        st.markdown(response["answer"])
    else:
        generate_response(text)
