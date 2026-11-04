import sys
from uuid import uuid4
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from models import model
from tools import calculate_right_refund, get_today_date
from tools import issue_refund_ticket, get_user_purchase_history
from langchain.agents import create_agent
from prompts import agent_system_prompt
from objects import Answer
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, AIMessage

agent = create_agent(
    model=model,
    system_prompt=agent_system_prompt,
    tools=[calculate_right_refund, get_user_purchase_history, issue_refund_ticket, get_today_date],
    response_format=Answer,
    checkpointer=InMemorySaver()
)

# Wrapper to guarantee thread_id and response["structured_response"]
_base_invoke = agent.invoke

def _guaranteed_invoke(input_data, config=None, **kwargs):
    cfg = dict(config) if config else {}
    if "configurable" not in cfg:
        cfg["configurable"] = {}
    if "thread_id" not in cfg["configurable"]:
        cfg["configurable"]["thread_id"] = str(uuid4())

    result = _base_invoke(input_data, config=cfg, **kwargs)
    
    if result.get("structured_response") is None:
        last_text = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                last_text = msg.content
                break
        if not last_text:
            last_text = "I have processed your request."
        
        lower = last_text.lower()
        is_done = (
            "refund ticket" in lower
            or "no refund will be issued" in lower
            or "not eligible" in lower
            or "outside 30 days" in lower
            or "outside of the 30" in lower
        )
        result["structured_response"] = Answer(
            answer=last_text,
            conversation=not is_done
        )
    return result

agent.invoke = _guaranteed_invoke

def run_conversation():
    config = {"configurable": {"thread_id": str(uuid4())}}

    text = "I would like to request a refund"
    print(text)

    question = HumanMessage(content=text)
    conversation = True

    while conversation:
        response = agent.invoke({"messages": [question]}, config=config)
        structured_response = response["structured_response"]

        print(structured_response.answer)

        conversation = structured_response.conversation

        if conversation:
            user_input = input("reply: \n")
            question = HumanMessage(content=user_input)

    print("the job is done!")

if __name__ == "__main__":
    run_conversation()
