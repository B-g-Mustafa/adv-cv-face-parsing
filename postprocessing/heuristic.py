"""
Spatial Heuristic Post-processor.

"Hacks" the left/right ambiguity caused by horizontal flip augmentation
by forcing classes to their correct left/right IDs based on simple x-coordinate
centroids. This exploits the fact that CelebAMask-HQ faces are perfectly centered.
"""
import cv2
import numpy as np


class SpatialHeuristicFixer:
    """
    Forces left/right separated features into their correct classes based on image halves.
    
    Pairs (Left side of image -> Right side of image):
    - Eyebrows: 7 -> 6
    - Eyes: 5 -> 4
    - Ears: 9 -> 8
    """

    def __init__(self):
        # Maps (merged_class_list) -> (left_class_id, right_class_id)
        self.rules = [
            # Eyebrows: if the model predicts either 6 or 7, we relabel them spatially
            ([6, 7], 7, 6),
            # Eyes: if the model predicts either 4 or 5, we relabel them spatially
            ([4, 5], 5, 4),
            # Ears: if the model predicts either 8 or 9, we relabel them spatially
            ([8, 9], 9, 8),
        ]

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        """
        Fixes the mask in-place or returns a new fixed mask.
        mask: (H, W) int32
        """
        H, W = mask.shape
        mid_x = W // 2
        
        result = mask.copy()

        for merge_classes, left_id, right_id in self.rules:
            # Create a binary mask of anywhere the model predicted ANY of the paired classes
            binary = np.zeros((H, W), dtype=np.uint8)
            for c in merge_classes:
                binary[mask == c] = 1
                
            if binary.sum() == 0:
                continue
                
            # Find independent connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            # Skip background (label 0)
            for i in range(1, num_labels):
                cx, cy = centroids[i]
                
                # Rule: if centroid is on the left half of the image, it's the left class
                if cx < mid_x:
                    result[labels == i] = left_id
                else:
                    result[labels == i] = right_id
                    
        return result
