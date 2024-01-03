import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from mlops_cookiecutter.data.mnist_dataloader import mnist
from mlops_cookiecutter.models.model import MyAwesomeModel

# import mlops_cookiecutter

# print(dir(mlops_cookiecutter))
model = MyAwesomeModel()
print(model)
model.load_state_dict(torch.load("models/trained_models/trained_model_v1.pt"))
model.eval()

train_set, _ = mnist(32)
for batch in train_set:
    data = batch[0]

    break
# print(model.state_dict())
# print(model)

layer = model.conv_layer[5]

# Step 2: Register a hook
features = []


def hook_function(module, input, output):
    features.append(output)


hook = layer.register_forward_hook(hook_function)

# Step 3: Forward pass
model(data)

# Step 4: Extract features
extracted_features = features[0]

# Don't forget to remove the hook when you're done
hook.remove()


feature_image = torch.sum(extracted_features[0], dim=0)
feature_image = torch.squeeze(feature_image, 0)
feature_image = feature_image.detach().numpy()

feature_image -= feature_image.min()  # Translate so the minimum value is 0.
feature_image /= feature_image.max()
plt.imshow(feature_image, cmap="gray")
plt.savefig("reports/figures/feature_extract.png")


num_samples, c, h, w = extracted_features.size()
flattened_features = extracted_features.view(num_samples, -1).detach().numpy()

# Step 2: Apply t-SNE
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(flattened_features)

# Step 3: Visualize with Matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
plt.xlabel("t-SNE feature 1")
plt.ylabel("t-SNE feature 2")
plt.title("t-SNE Visualization of CNN Features")
plt.savefig("reports/figures/t-SNE.png")
