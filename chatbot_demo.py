from langchain import HuggingFaceHub, PromptTemplate, LLMChain
repo_id = "tiiuae/falcon-7b-instruct"
huggingfacehub_api_token = "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"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.7, "max_new_tokens":500})

template = """Question: {question}
              Answer: Let's give a detailed answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(prompt=prompt, llm=llm)
out = chain.run("Give me a way to make a videogame") #test 
print(out)
