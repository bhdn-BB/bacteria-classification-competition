import numpy as np
from sklearn.linear_model import LinearRegression
from global_config import COLORS_CHANNEL

def balance_color_channel(image_original, target_channel):
    channel_indices = {str(value): i for i, value in enumerate(COLORS_CHANNEL)}
    target_idx = channel_indices[target_channel]
    other_idxs = [i for i in range(len(channel_indices)) if i != target_idx]
    image_float = image_original.astype(np.float32)
    X = np.stack([image_float[:, :, i].ravel() for i in other_idxs], axis=1)
    target = image_float[:, :, target_idx].ravel()
    regressor = LinearRegression()
    regressor.fit(X, target)
    predicted_channel = regressor.predict(X).reshape(image_original.shape[:2])
    balanced_image = image_float.copy()
    balanced_image[:, :, target_idx] = predicted_channel
    balanced_image = np.clip(balanced_image, 0, 255).astype(np.uint8)
    return balanced_image