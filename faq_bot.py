from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled, RunConfig
from dotenv import load_dotenv
from openai import AsyncOpenAI

import os

set_tracing_disabled(True)
load_dotenv(override=True)

openai_api_key=os.getenv("OPENAI_API_KEY")
openai_base_url=os.getenv("OPENAI_BASE_PATH")
openai_model_name=os.getenv("OPENAI_MODEL_NAME")

openai_client=AsyncOpenAI(api_key=openai_api_key,base_url=openai_base_url)
openai_model=OpenAIChatCompletionsModel(openai_client=openai_client, model=str(openai_model_name))

faq_agent: Agent = Agent(
    name= "faq_agent",
    instructions="""
You are a helpful chatbot that answers basic predefined questions like:
    - "What is your name?"
    - "What can you do?"
    
    If the user asks anything outside of this, reply with:
    "I'm sorry, I can only answer a few specific questions."
    """,
    model=openai_model,
)

prompt = input("Enter your question : ")
result = Runner.run_sync(faq_agent, prompt)
print(result.final_output)