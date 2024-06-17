import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

repo_id = "tiiuae/falcon-7b-instruct"
huggingfacehub_api_token = "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" #mytoken
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.7, "max_new_tokens":500})

template = """Question: {question}
Answer: Let's give a detailed answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(prompt=prompt, llm=llm)


@cl.on_chat_start
async def main():
    while True:
        #quétion
        res = await cl.AskUserMessage(content="What is your question").send()
        if res:
            # Chạy mô hình để lấy câu trả lời
            answer = chain.run(res['output'])
            await cl.Message(
                content=f"Answer: {answer}",
            ).send()
        
        #continue or quit
        res = await cl.AskActionMessage(
            content="Pick an action!",
            actions=
            [
                cl.Action(name="continue", value="continue", label="✅ Continue"),
                cl.Action(name="cancel", value="cancel", label="❌ Its enough"),
            ],
        ).send()

        if not res or res.get("value") == "cancel":
            await cl.Message(
                content="Ah okay cya",
            ).send()
            break
