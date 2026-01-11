import torch
import copy
import torch.nn as nn
import torch.optim as optim
from utils import get_device, train_one_epoch
# torch.func is available in standard torch builds since 2.0
from torch.func import functional_call

class FLClient:
    def __init__(self, client_id, model, train_loader, personalization_epochs=1):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.train_loader = train_loader
        self.device = get_device()
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.personalization_epochs = personalization_epochs

    def set_parameters(self, global_model):
        """
        Standard FedAvg: Overwrite local with global.
        """
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.device)

    def adapt_global_model(self, global_model, ala_epochs=3, lr=0.1):
        """
        FedALA: Adaptive Local Aggregation at Channel Granularity.
        Blends Global and Local models: W_new = alpha * W_global + (1 - alpha) * W_local
        """
        global_model.to(self.device)
        self.model.to(self.device)
        
        # 1. Initialize Alphas
        # We need an alpha for each weight tensor.
        # If weight is [Out, In, H, W], we want alpha to be [Out, 1, 1, 1] for channel-wise.
        # If weight is 1D (bias), alpha is [Out].
        
        local_state = dict(self.model.named_parameters())
        global_state = dict(global_model.named_parameters())
        
        alphas = {}
        for name, param in local_state.items():
            if 'weight' in name and param.dim() == 4:
                # Conv layer: Channel-wise alpha (output channels)
                # param.shape[0] is output channels
                alphas[name] = torch.ones(param.shape[0], 1, 1, 1, device=self.device, requires_grad=True)
            elif 'bias' in name:
                alphas[name] = torch.ones(param.shape[0], device=self.device, requires_grad=True)
            else:
                # Fallback for other logical parameters (scalar alpha)
                alphas[name] = torch.ones(1, device=self.device, requires_grad=True)
                
        # Optimizer for Alphas
        optimizer = optim.Adam(alphas.values(), lr=lr)
        
        # 2. ALA Optimization Loop
        # We use a few batches of local data to learn the best mixing weights
        self.model.eval() # We are not updating model batchnorm stats here, just alphas
        
        data_iter = iter(self.train_loader)
        
        for _ in range(ala_epochs):
            try:
                images = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                images = next(data_iter)
            
            images = images.to(self.device)
            
            optimizer.zero_grad()
            
            # Construct blended parameters with functional logic
            blended_params = {}
            for name in local_state.keys():
                w_local = local_state[name]
                w_global = global_state[name]
                alpha = alphas[name]
                
                # W_new = alpha * W_global + (1 - alpha) * W_local
                # To ensure stability, we can clam alpha or just let it float.
                # Usually standard linear interpolation is fine.
                blended_params[name] = alpha * w_global + (1 - alpha) * w_local
            
            # Functional call to compute output with blended weights
            # Note: functional_call expects a dict keying 'param_name' to tensor
            outputs = functional_call(self.model, blended_params, (images,))
            
            loss = self.criterion(outputs, images) # Reconstruction loss
            loss.backward()
            optimizer.step()
            
        # 3. Apply best alphas to update local model permanently
        with torch.no_grad():
            final_state = self.model.state_dict()
            for name in local_state.keys():
                w_local = local_state[name]
                w_global = global_state[name]
                alpha = alphas[name]
                
                final_state[name] = alpha * w_global + (1 - alpha) * w_local
            
            self.model.load_state_dict(final_state)
            
        print(f"Client {self.client_id}: ALA adaptation complete.")


    def train(self, epochs=1, lr=1e-3):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        epoch_losses = []
        for _ in range(epochs):
            loss = train_one_epoch(self.model, self.train_loader, self.criterion, optimizer, self.device)
            epoch_losses.append(loss)
            
        return self.model.state_dict(), sum(epoch_losses) / len(epoch_losses)

    def personalize(self, lr=1e-4):
        """
        Further fine-tune on local data for personalization.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for _ in range(self.personalization_epochs):
             train_one_epoch(self.model, self.train_loader, self.criterion, optimizer, self.device)
             
    def get_model(self):
        return self.model

    def detect_anomaly(self, image_tensor, threshold=0.1):
        """
        Returns anomaly score and boolean prediction.
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0) # Add batch dim
            
        with torch.no_grad():
            output = self.model(image_tensor)
            loss = self.criterion(output, image_tensor)
            score = loss.item()
            
        return score, score > threshold
