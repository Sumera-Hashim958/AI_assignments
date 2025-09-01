import os
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool
from openai import AsyncOpenAI
from agents.run import RunConfig

#load openai key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("openai_api_key not found in .env!")

#openai compatible setup
external_client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url="https://api.openai.com/v1"
)
model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",
    openai_client=external_client
)

config = RunConfig(model=model, tracing_disabled=True)

#weather tool (mocked)
@function_tool
def get_weather(city: str) -> str:
    """Return mock weather info for the given city."""
    return f"The current temperature in {city} is 34C with clear skies."

#main cli
def main():
    print("☁ Welcom to the Weather info Agent!")
    print("Ask me about weather in any city. Type 'exit to 'quit'. \n")

agent = Agent(
    name="weatherbot",
    instructions="You are a weather assistant. Use the weather tool to provide information when asked about the weather in any city.",
    model=model,
    tools=[get_weather]
)
while True:
    user_input = input("Enter your question:")
    if user_input.lower() in ['exit', 'quit']:
        print("👋Goodby")
        break

    result = Runner.run_sync(agent, user_input, run_config=config)
    print("↙Bot:", result.final_output)


if __name__=="__main__":
    main()