export default function handler(req, res) {
  res.send(`

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

messages = [
    ("system", "You are a helpful {topic} assistant."),
    ("user", "Tell {number} things about {topic}.")
]

prompt_template = ChatPromptTemplate(messages=messages)
print("Prompt template:", prompt_template)

prompt = prompt_template.invoke({"topic": "AI", "number": "3"})
print("\nRendered Prompt:")
for msg in prompt.messages:
    print(f"{msg.type.capitalize()}: {msg.content}")
messages = [
    ("system", "You are a helpful {topic} assistant."),
    ("placeholder", "{placeholder}")  
]

prompt_template = ChatPromptTemplate(messages=messages)
print("\nPrompt template with placeholder:", prompt_template)

prompt = prompt_template.invoke({
    "placeholder": [
        ("user", "What is 2+2?"),
        ("ai", "4"),
        ("user", "What is 2+4?")
    ],
    "topic": "AI"
})

print("\nRendered Placeholder Prompt:")
for msg in prompt.messages:
    print(f"{msg.type.capitalize()}: {msg.content}")

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

chat_history = []

chat_history.append(SystemMessage(content="You are a helpful assistant in Mathematics."))

while True:
    human_message = input("Your query: ")
    if human_message.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=human_message))

    ai_message = llm.invoke(chat_history)

    print(ai_message.content)

    chat_history.append(AIMessage(content=ai_message.content))
`);
}
