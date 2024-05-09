from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from typing import Dict


class PromptRouter:

    def __init__(self, llm, chain_info: Dict) -> None:
        destinations = [f"{p['name']}: {p['description']}" for p in chain_info]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
            destinations=destinations_str
        )
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )
        self.router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    def get_chain_name(self, prompt: str) -> str:
        return self.router_chain.invoke({"input": prompt})["destination"]
