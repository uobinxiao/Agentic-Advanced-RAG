from langgraph.graph import StateGraph, END
from .nodes import *
from Utils import *
# from Frontend import *
from Config.rag_config import RAGConfig
from Config.output_pydantic import *

                
class HybridAgenticRAG:
    def __init__(self):
        # Create the workflow(graph)
        workflow = StateGraph(UnitQueryState)
        
        # Add nodes class
        nodes = NodesModularRAG()
        
        # Add nodes into the workflow
        workflow.add_node("user_query_classification_node", nodes.user_query_classification_node)
        
        workflow.add_node("global_retrieval_node", nodes.global_retrieval_node)
        workflow.add_node("local_retrieval_node", nodes.local_retrieval_node)
        workflow.add_node("detail_retrieval_node", nodes.detail_retrieval_node)
        
        workflow.add_node("local_reranking_node", nodes.local_reranking_node)
        workflow.add_node("detail_reranking_node", nodes.detail_reranking_node)
        workflow.add_node("local_reducing_node", nodes.local_reducing_node)
        workflow.add_node("detail_reducing_node", nodes.detail_reducing_node)
            
        workflow.add_node("global_mapping_node", nodes.global_mapping_node)
        workflow.add_node("global_reducing_node", nodes.global_reducing_node)
        
        workflow.add_node("local_community_retrieval_node", nodes.local_community_retrieval_node)
        workflow.add_node("local_hyde_node", nodes.local_hyde_node)

        workflow.add_node("information_organization_node", nodes.information_organization_node)
        workflow.add_node("generation_node", nodes.generation_node)
        
        # Draw the workflow
        workflow.set_entry_point("user_query_classification_node")
        workflow.add_conditional_edges(
            "user_query_classification_node",
            nodes.is_retrieval_needed_edges,
            {
                "retrieval_not_needed": "generation_node",
                "retrieval_needed_for_global_topic_searching": "global_retrieval_node",
                "retrieval_needed_for_local_topic_searching": "local_retrieval_node",
            }
        )
        workflow.add_conditional_edges("global_retrieval_node", nodes.global_mapping_dispatch_edges, ["global_mapping_node"])
        workflow.add_edge("local_retrieval_node", "local_reranking_node")
        workflow.add_edge("global_mapping_node", "global_reducing_node")
        workflow.add_edge("global_reducing_node", "detail_retrieval_node")
        workflow.add_edge("local_reranking_node", "local_reducing_node")
        workflow.add_conditional_edges("local_reducing_node", 
            nodes.is_local_retrieval_empty_edges, 
            {
                "local_retrieval_not_empty": "detail_retrieval_node",
                "local_retrieval_empty": "local_community_retrieval_node",
            }
        )
        workflow.add_conditional_edges("local_community_retrieval_node", 
            nodes.is_local_retrieval_empty_edges, 
            {
                "local_retrieval_not_empty": "detail_retrieval_node",
                "local_retrieval_empty": "local_hyde_node",
            }
        )
        workflow.add_edge("local_hyde_node", "detail_retrieval_node")
        workflow.add_edge("detail_retrieval_node", "detail_reranking_node")
        workflow.add_edge("detail_reranking_node", "detail_reducing_node")
        workflow.add_edge("detail_reducing_node", "information_organization_node")
        workflow.add_edge("information_organization_node", "generation_node")

        # Compile
        self.graph = workflow.compile()
                
class DatasetRecord:
    def __init__(self):
        # Create the workflow(graph)
        workflow = StateGraph(QueriesState)
        
        # Add nodes class
        nodes = NodesDataset()
        
        # Add subgraph
        unit_query_subgraph = HybridAgenticRAG().graph
        
        # Add nodes into the workflow
        workflow.add_node("update_next_query_node", nodes.update_next_query_node)
        workflow.add_node("unit_query_subgraph", unit_query_subgraph)
        workflow.add_node("store_results_node", nodes.store_results_node)

        # Draw the workflow
        workflow.set_entry_point("update_next_query_node")
        workflow.add_edge("update_next_query_node", "unit_query_subgraph")
        workflow.add_edge("unit_query_subgraph", "store_results_node")
        workflow.add_conditional_edges(
            "store_results_node",
            nodes.is_dataset_unfinished_edges,
            {
                "dataset_unfinished": "update_next_query_node",
                "dataset_finished": END,
            }
        )
        # Compile
        self.graph = workflow.compile()
        
class ParallelDatasetRecord:
    def __init__(self):
        workflow = StateGraph(QueriesParallelState)
                              
        nodes = NodesParallelDataset()
                              
        unit_query_subgraph = HybridAgenticRAG().graph
                              
        workflow.add_node("prepare_batches_node", nodes.prepare_batches)
        workflow.add_node("unit_query_subgraph", unit_query_subgraph)
        workflow.add_node("sort_results_by_query_id", nodes.sort_results_by_query_id)

        workflow.set_entry_point("prepare_batches_node")
        workflow.set_finish_point("sort_results_by_query_id")
        workflow.add_conditional_edges("prepare_batches_node", nodes.dispatch_dataset_edges, ["unit_query_subgraph"])
        workflow.add_edge("unit_query_subgraph", "sort_results_by_query_id")
        self.graph = workflow.compile()
        
class WorkFlowMultiAgentRAG():
    def __init__(self, 
                query: str, 
                collection: str, 
                rag_config: RAGConfig,
                ):        
        # Create the workflow(graph)
        workflow = StateGraph(OverallState)
        
        # Add nodes to the workflow
        nodes = NodesMultiAgentRAG(query, collection, rag_config)
        workflow.add_node("overall_node", nodes.overall_node)
        workflow.set_entry_point("overall_node")
        workflow.set_finish_point("overall_node")
        
        # Compile
        self.graph = workflow.compile()
        
class WorkFlowSingleAgentRAG():
    def __init__(self, 
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
    workflow = WorkFlowModularHybridRAG()
    print(workflow)