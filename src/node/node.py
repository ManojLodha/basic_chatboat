from src.state import State

class BasicChatboatnode:

    def __init__(self,model):
        self.llm=model

    def process(self, state: State)->dict:
        """
        Processes the input state and generates a chatboat response.
        """
        return {"message":self.llm.invoke(state['message'])}