"""
# Dashboard & GUI Application

This package contains the interactive interfaces for the Animal Classification project.

## Modules
- **dashboard**: The main Gradio-based web interface. It allows you to:
    - Train models (VGG16/VGG11) interactively.
    - Visualize training progress (Loss/Accuracy curves).
    - Run inference and view metrics (Confusion Matrix, ROC Curves).

## How to Run
You can launch the dashboard from the terminal using:
```bash
python -m animal_classification.app.dashboard
```
Or if you installed the project:

```bash
animal-dashboard
```
"""