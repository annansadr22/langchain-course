import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Tavily client
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ✅ Gemini-compatible tool
@tool
def tavily_search(query: str) -> str:
    """Search the web for real-time information."""
    return str(tavily.search(query=query))

tools = [tavily_search]

# ✅ create_agent (allowed)
agent = create_agent(
    model=llm,
    tools=tools
)

def main():
    print("Hello from langchain-course!")

    result = agent.invoke({
        "messages": [
            HumanMessage(content="What's the weather in Tokyo?")
        ]
    })

    # agent returns a state dict
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()