import cv2
import numpy as np

class SpatialHeuristicFixer:
    """
    Forces left/right separated features into their correct classes based on image halves.
    Since faces are mostly centered, this instantly fixes the 0.99 L/R merged similarity 
    caused by intentional horizontal flipping.
    """
    def __init__(self):
        # (Merged List) -> (Left ID, Right ID)
        self.rules = [
            ([6, 7], 7, 6),    # Eyebrows
            ([4, 5], 5, 4),    # Eyes
            ([8, 9], 9, 8),    # Ears
        ]

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        H, W = mask.shape
        mid_x = W // 2
        result = mask.copy()

        for merge_classes, left_id, right_id in self.rules:
            binary = np.zeros((H, W), dtype=np.uint8)
            for c in merge_classes:
                binary[mask == c] = 1
                
            if binary.sum() == 0: continue
                
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                cx, cy = centroids[i]
                if cx < mid_x:
                    result[labels == i] = left_id
                else:
                    result[labels == i] = right_id
                    
        return result
