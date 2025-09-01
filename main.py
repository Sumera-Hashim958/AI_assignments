import os
import chainlit as cl
from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variable.")

client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url="https://api.openai.com/v1"
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="My Chatbot").send()  # ✅ this line is now correct

@cl.on_message
async def on_message(message: cl.Message):
    try:
        response = await client.chat.completion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message.content}],
        )
        await cl.Message(content=response.choices[0].message.content).send()
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

    