from langchain_community.embeddings import HuggingFaceEmbeddings


def initialize_embeddings(
    model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = "cpu"
):
    model_kwargs = {"trust_remote_code": True, "device": device}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
