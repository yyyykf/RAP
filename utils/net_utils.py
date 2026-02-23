import torch
import torch.nn as nn


class PrunePredictor(nn.Module):
    def __init__(self, input_dim=15, hidden_dims=[32, 32, 16], use_bn=False, use_dropout=False, dropout_p=0.1, use_softmax=True):
        super(PrunePredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(p=dropout_p))
            prev_dim = hidden_dim

        # Final output layer (2 logits for binary classification)
        layers.append(nn.Linear(prev_dim, 2))
        if use_softmax:
            layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def save_model(self, save_path="prune_predictor.pth"):
        torch.save(self.state_dict(), save_path)
        print(f"Model saved at {save_path}")

    def load_model(self, load_path="prune_predictor.pth", device="cuda:0"):
        self.load_state_dict(torch.load(load_path, map_location=device))
        print(f"Model loaded from {load_path}")