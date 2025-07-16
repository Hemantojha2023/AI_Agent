import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from openai import OpenAI
import requests
import uuid
import os

# Load OpenAI API key
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Custom tool: generate image & return URL
def generate_image_dalle(prompt: str) -> str:
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    return response.data[0].url

# Wrap as LangChain tool
dalle_tool = Tool(
    name="DalleImageGenerator",
    func=generate_image_dalle,
    description="Generates an image from a prompt using OpenAI DALL·E"
)

# Create LangChain LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create agent
agent = initialize_agent(
    tools=[dalle_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit UI
st.set_page_config(page_title="LangChain Agent with DALL·E")
st.title("LangChain Agent + DALL·E")
st.write("Type a prompt and let the agent generate an image using OpenAI DALL·E!")

prompt = st.text_input("Enter your image prompt:")

if st.button("Generate Image") and prompt:
    with st.spinner("Generating..."):
            # Tell agent to output only URL
            agent_prompt = f"Generate an image of: {prompt}. As final answer, return only the image URL."
            result = agent.run(agent_prompt)
            st.write("Agent raw output:", result)

            # Extract URL from agent response
            if isinstance(result, str) and "http" in result:
                url = result.split("http", 1)[1]
                url = "http" + url.strip()

                # Download the image
                response = requests.get(url)
                if response.status_code == 200:
                    filename = f"temp_{uuid.uuid4().hex}.png"
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    st.image(filename, caption="Generated Image")
                    # Optionally, remove file after displaying
                    os.remove(filename)
                else:
                    st.error("Failed to download image from URL")
            else:
                st.error("Agent did not return a valid URL")