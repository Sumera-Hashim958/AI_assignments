from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled, function_tool, RunConfig
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

@function_tool
def add(a:int, b:int):
    "add two numbers and return value"
    # print("add tool")
    return a+b

def main():
    print("➕Welcome to the Math tool agent")

math_agent: Agent = Agent(
    name= "Tool_math_agent",
    instructions="""
You are a math agent who can add numbers.
    If the user asks something like 'What is 5 + 7?', use the tool.
    """,
    model=openai_model,
    tools=[add]
)
while True:
    user_input = input("Enter your question:")
    if user_input.lower() in ['exit', 'quit']:
        print("👋Goodby")
        break

    result = Runner.run_sync(math_agent, user_input)
    print("↙Bot:", result.final_output)

# prompt = input("Enter your question : ")
# result = Runner.run_sync(math_agent, prompt)
# print(result.final_output)