# Manifolk
Did you think that wouldn't it be nice to have a tool that would plot T-SNE graphs at each epoch, and only of specific labels you want to visualize? Look no more!

Manifolk is a Plotly+Dash+MySQLite web application that does exactly that. During your model training process, you generate T-SNE embeddings using your favourite package (e.g. sklearn's T-SNE) and then use Manifolk's API to store all of them into an SQLite database. After the training is over the Dash app can be launched to visualize the 3D T-SNE embeddings. beware the T-SNE embeddings you generate must be 3D, not 2D (because why not have an extra dimension?).

## Installation Instructions
```bash

```
## Python API
## Screenshots
### Changing Epochs
The below screenshot shows how to change epochs and visualize the T-SNE at every epoch

![](readme_images/changing_epochs.gif)

### Changing Labels
The below screenshot shows how to change labels

![](readme_images/changing_labels.gif)

### Marking a unique datapoint
The below image shows how to mark a unique datapoint which you want to track across epochs and different labels in the TSNE cloud

![](readme_images/marking_uid.gif)

## Use Cases