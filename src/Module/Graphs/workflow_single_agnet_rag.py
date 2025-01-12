from langgraph.graph import StateGraph
from src.Module.nodes import *
from src.Utils import *
from Config.rag_config import RAGConfig
from Config.output_pydantic import *

class WorkFlowSingleAgentRAG():
    def __init__(
        self, 
        query: str, 
        collection: str, 
        rag_config: RAGConfig,
    ):        
        # Create the workflow(graph)
        workflow = StateGraph(SingleState)
        
        # Add nodes to the workflow
        nodes = NodesSingleAgentRAG(query, collection, rag_config)
        workflow.add_node("run_node", nodes.run_node)
        workflow.set_entry_point("run_node")
        workflow.set_finish_point("run_node")
        
        # Compile
        self.graph = workflow.compile()
        
if __name__ == "__main__":
    workflow = WorkFlowSingleAgentRAG()
    print(workflow)