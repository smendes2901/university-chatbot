import json
from langchain_community.llms import OpenAI
import streamlit as st

from chains.base import CommonChain
from embedding import initialize_embeddings
from models import initialize_llm
from router import PromptRouter

llm = initialize_llm("gpt3")
embeddings = initialize_embeddings()

chain_info = json.load(open("chains/chain_info.json"))
# chain_info = dict((c["name"], c["description"]) for c in chain_info)

finance_chain = CommonChain("data/financing-a-stevens-education.csv", embeddings)
graudate_chain = CommonChain("./data/graduate-education.csv", embeddings)
undergraduate_chain = CommonChain("data/undergraduate-education.csv", embeddings)
student_life = CommonChain("data/student-life.csv", embeddings)
student_services = CommonChain("data/student-services.csv", embeddings)
graudate_tution_chain = CommonChain(
    "data/tuition-fees-and-other-expenses-for-graduate-students.csv", embeddings
)
undergraudate_tution_chain = CommonChain(
    "data/tuition-fees-and-other-expenses-for-undergraduate-students.csv", embeddings
)

chains = {
    "financing_a_stevens_education": finance_chain,
    "tuition_fees_and_other_expenses_for_undergraduate_students": undergraudate_tution_chain,
    "tuition_fees_and_other_expenses_for_graduate_students": graudate_tution_chain,
    "student_life_at_stevens": student_life,
    "student_services_at_stevens": student_services,
    "graduate_education_at_stevens": graudate_chain,
    "undergraduate_chain": undergraduate_chain,
}

router = PromptRouter(llm=llm, chain_info=chain_info)

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
