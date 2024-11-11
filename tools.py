
import json
from pydantic import BaseModel
from crew import InternalSupportCrew


def tool(name: str):
    def decorator(func):
        # Create a Pydantic model class for the function parameters
        params = {
            k: (v, ...) for k, v in func.__annotations__.items() 
            if k != 'return'
        }
        
        # Create the model class
        model = type(
            f"{name}",
            (BaseModel,),
            {
                '__annotations__': {k: v[0] for k, v in params.items()},
                'function': func,
                '__doc__': func.__doc__
            }
        )

        # Store the model in the function for OpenAI
        func.model = model
        func.__name__ = name
        return func
    return decorator



@tool("DataExtractionTool")
def data_extraction_tool(order_id: int):
    """
    This function extracts data for a given order ID.
    """
    return json.dumps({
        str(order_id): {
            "order_date": "2024-01-01",
            "order_amount": 100,
            "order_items": ["item1", "item2", "item3"],
            "status": "intransit"
        }
    })

@tool("AskFromInternalAgent")
def ask_from_internal_agent(context: str, task: str):
    """
    If user is asking anything which needs to be answered by internal team, after checking any company policy, customer data, past 
    customer interactions, this function will be used. One should always consult this function before answering any query. Never make up 
    any answer,  always check and reply based on company policy, customer data. 
    if the user is asking for updating any information then this function can be used to guide user to self update the information or update the information through internal tools.
    """
    inputs = {
        'user_query': task,
        'user_info': context
    }
    crew_result =  InternalSupportCrew().crew().kickoff(inputs=inputs)
    return crew_result