from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from openai import AsyncOpenAI as OpenAI
import openai
import os
import json
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
import traceback
from tools import ask_from_internal_agent

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs, e.g., ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],  # This allows all methods, adjust as needed
    allow_headers=["*"],  # This allows all headers, adjust as needed
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory storage for chat history
chat_histories: Dict[str, List[Dict[str, str]]] = {}
tools = [
    openai.pydantic_function_tool(ask_from_internal_agent.model)
]
tool_name_to_tool = {
    ask_from_internal_agent.__name__: ask_from_internal_agent,
}

system_prompt = open("prompts/system_prompt.md", "r").read()
print("system_prompt", system_prompt)
#import pdb; pdb.set_trace()
class Message(BaseModel):
    session_id: str
    message: str

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/api/chat")
async def chat(message: Message):
    print("message", message)
    try:
        # Retrieve or initialize chat history for the session
        if message.session_id not in chat_histories:
            chat_histories[message.session_id] = [{"role": "system", "content": system_prompt}]

        # Add user message to chat history
        chat_histories[message.session_id].append({"role": "user", "content": message.message})
        stop = False
        while not stop:
            # Send the entire chat history to the OpenAI API
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    *chat_histories[message.session_id]
                ],
                tools=tools,
            )

            print("response", response.choices[0].message)
            if response.choices[0].finish_reason == "tool_calls":
                # TODO: handle multiple tool calls
                chat_histories[message.session_id].append(response.choices[0].message)
                tool_call = response.choices[0].message.tool_calls[0].function.name
                tool_input = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
                print("tool_input", tool_input)
                
                tool_output = tool_name_to_tool[tool_call](**tool_input).raw
                print("tool_output", tool_output)
                chat_histories[message.session_id].append({
                    "role": "tool",
                    "content": json.dumps(tool_output),
                    "tool_call_id": response.choices[0].message.tool_calls[0].id
                })
                print("tool_input", tool_input)
                print("tool_output", tool_output)
                
                bot_reply = tool_output
                #import pdb; pdb.set_trace()

            else:
                bot_reply = response.choices[0].message.content.strip()
                stop = True

        # Add bot reply to chat history
        chat_histories[message.session_id].append({"role": "assistant", "content": bot_reply})

        return {"reply": bot_reply}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error connecting to OpenAI API: {str(e)}")
        

# Add your other endpoints here
