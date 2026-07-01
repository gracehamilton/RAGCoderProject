from RAGService import RAGSession
from tools import tools
import inspect, pydantic

class Agent:
    def __init__(self, model_client=None, context = ""):
        self.model_client = model_client
        self.conversation_history = []
        self.context = context
    def ask(self, question):
        messages = {"role": "user", "content": question}
        self.conversation_history.append(messages)
        if self.model_client:
            while True:
                response = self.model_client.chat(messages=[
                    {"role": "system", "content": f"{self.context}. You have had the following conversation so far: {self.conversation_history}"},
                    {"role": "user", "content": question}], 
                    tools=[name for name, obj in inspect.getmembers(tools, inspect.isfunction) if obj.__module__ == tools.__name__],
                    stream=True)
                if response.done_reason == "tool call":
                    messages.append({"role": "assistant", "content": response.message.content})
                    tool_result = []