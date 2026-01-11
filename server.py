import torch
import copy

class Aggregator:
    def __init__(self, model):
        self.global_model = model

    def aggregate(self, client_models):
        """
        Averages the weights of client models to update the global model.
        Args:
            client_models: List of client models (state_dicts or models)
        """
        global_dict = self.global_model.state_dict()
        
        # Initialize global_dict with zeros
        for k in global_dict.keys():
            global_dict[k] = torch.zeros_like(global_dict[k], dtype=torch.float32) # Ensure float32 for summation
            
        # Sum up weights
        for client_model in client_models:
            if isinstance(client_model, dict):
                client_state = client_model
            else:
                client_state = client_model.state_dict()
                
            for k in global_dict.keys():
                global_dict[k] += client_state[k]
                
        # Average
        num_clients = len(client_models)
        for k in global_dict.keys():
            global_dict[k] = torch.div(global_dict[k], num_clients)
            
        self.global_model.load_state_dict(global_dict)
        return copy.deepcopy(self.global_model)
    
    def get_global_model(self):
        return self.global_model
