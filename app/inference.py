import os
import torch
import numpy as np
import torch_geometric
import torch.nn.functional as F 
from fastapi.logger import logger
from model.base_model import Model
from typing import List, Optional, Dict, Any 

class InferenceUtils:
    def __init__(self, device) -> None:
        self.device = device

    @property
    def _load_edge_index(
        self, add_self_loops: bool = True, normalize_edge: bool = False
    ) -> torch.Tensor:
        """_summary_

        Args:
            add_self_loops (bool, optional): Whether to add self loops in the nodes or not . Defaults to True.
            normalize_edge (bool, optional): Whether to normalize the edges or not . Defaults to True.

        Returns:
            torch.Tensor: Loading the edge index (This will remain constant for all the graphs)
        """

        pose_indices = [
            (0, 1),
            (0, 6),
            (0, 2),
            (1, 3),
            (1, 7),
            (1, 0),
            (2, 4),
            (2, 0),
            (3, 5),
            (3, 1),
            (4, 2),
            (5, 3),
            (6, 0),
            (6, 7),
            (6, 8),
            (7, 1),
            (7, 6),
            (7, 9),
            (8, 6),
            (8, 10),
            (9, 7),
            (9, 11),
            (10, 8),
            (10, 12),
            (11, 9),
            (11, 13),
            (12, 10),
            (13, 11),
        ]

        source_indices: List[int] = [pose[0] for pose in pose_indices]
        target_indices: List[int] = [pose[1] for pose in pose_indices]

        edge_indices: np.ndarray = np.array([source_indices, target_indices], dtype=np.float32)
        edge_index: torch.Tensor = torch.tensor(edge_indices, dtype=torch.long)

        if add_self_loops:
            edge_index: torch.Tensor = torch_geometric.utils.add_self_loops(edge_index)[0]

        if normalize_edge:
            edge_index: torch.Tensor = edge_index / len(source_indices)

        return edge_index.to(self.device)

    @property
    def get_classes(self):
        pose_dir_names = [
            "downdog",
            "goddess",
            "plank",
            "tree",
            "warrior2"
        ]
        return pose_dir_names

    def load_model(
        self,
        in_features: Optional[int] = 3,
        hidden_features1: Optional[int] = 64,
        hidden_features2: Optional[int] = 32,
        num_classes: Optional[int] = 5,
        model_path=None,
    ):
        """Returns the model based on the provided configuration

        Args:
            in_features (int, optional): _description_. Defaults to 3.
            hidden_features1 (int, optional): _description_. Defaults to 64.
            hidden_features2 (int, optional): _description_. Defaults to 32.
            num_classes (int, optional): _description_. Defaults to 5.
            model_path (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        model = Model(
            in_features=in_features,
            hidden_features=hidden_features1,
            out_features=hidden_features2,
            num_classes=num_classes,
        ).to(self.device)

        model_path = "model/base_model.pth" if model_path is None else model_path
        model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

        logger.info("=> Model loaded successfully")
        return model


class ModelInference(InferenceUtils):
    def __init__(
        self,
        device: str,
        input_features: Optional[int]=3,
        hidden_features1: Optional[int]=64,
        hidden_features2: Optional[int]=32,
        num_classes: Optional[int]=5,
        model_path: Optional[str]=None
    ) -> None:
        super(ModelInference, self).__init__(device)

        self.model = self.load_model(
            in_features=input_features,
            hidden_features1=hidden_features1,
            hidden_features2=hidden_features2,
            num_classes=num_classes,
            model_path=model_path,
        )

        self.edge_index = self._load_edge_index
    
    def get_results(self, request : torch.Tensor):
        """Get the inference results
        ### NOTE: We are assuming for now that the batch size is set to 1

        Args:
            request (torch.Tensor): The incoming request containnig the tensor values of keypoints 

        Returns:
            _type_: _description_
        """
        request = request.to(self.device)
        with torch.no_grad():
            logits = self.model(request, self.edge_index, torch.tensor([0]))
        probabilities = F.softmax(logits).detach().cpu().numpy()[0].tolist()
        response_dict = dict(zip(self.get_classes, probabilities))
        return response_dict
        


if __name__ == "__main__":
    import json 

    infer = ModelInference(device="cpu")
    x = torch.rand(14, 3)
    response = infer.get_results(x)
    print(json.dumps(response, indent=4))