from langchain.graph import StateGraph
from langchain.graph import START, END
from langchain.graph import StateGraph
from src.state.state import State
from src.node.node import BasicChatboatnode
from scr.llm.groqllm import GroqLLM

class GraphBuilder:
    def __init__(self,model):
        self.model=GroqLLM(model)
        self.graph=StateGraph()


    def basic_chatbot_build_graph(self):
        """
        Builds a basic chatbot graph using LangGraph.
        This method initializes a chatbot node using the `BasicChatbotNode` class 
        and integrates it into the graph. The chatbot node is set as both the 
        entry and exit point of the graph.
        """

        self.basic_chatbot_node=BasicChatbotNode(self.llm)

        self.graph_builder.add_node("chatbot",self.basic_chatbot_node.process)
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_edge("chatbot",END)