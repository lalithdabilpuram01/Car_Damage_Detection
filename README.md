Project: Automated Car Damage Detection for Vroom Car Rentals
Status: In Progress (Active Development)
The Problem
The current vehicle return process at Vroom Car Rentals relies heavily on manual inspections, which are often subjective and time-consuming. My goal with this project is to build an objective, automated system that identifies and localizes exterior damage—such as dents, scratches, and cracks—using computer vision. By streamlining this assessment, the system aims to reduce dispute times and improve operational efficiency at rental hubs.

My Technical Approach
I am developing this project with a production-first mindset, focusing on balancing high-accuracy detection with a model architecture that can be deployed across different environments.

1. Leveraging Transfer Learning (ImageNet)
Rather than training a model from scratch, I am utilizing ImageNet pre-trained backbones. Since ImageNet provides a robust foundation for recognizing general shapes and textures, it allows the model to converge faster on the specialized task of identifying automotive damage. I am currently fine-tuning these weights to distinguish between actual structural damage and common "noise" like shadows or reflections on metallic surfaces.

2. Development and Framework (PyTorch)
The core logic is being built in PyTorch. I chose this framework for its flexibility in experimentation, specifically for handling the custom loss functions needed to detect "fine-grained" damage (like small scratches) that standard object detection models might overlook.

3. Optimization for Deployment (ONNX)
To make the system useful for Vroom’s field operations, the final model must be portable. I am incorporating ONNX as a bridge between development and production. By exporting the PyTorch models to the ONNX format, the system can run with high performance on various hardware—ranging from cloud servers to the mobile devices used by rental agents on the lot.

Current Progress
The project is currently in the Model Refinement stage. I have established the training pipeline and am iterating on the following:

Fine-tuning the ImageNet-based backbone on a curated dataset of damaged vehicles.

Developing the logic to categorize damage by severity (e.g., minor scratch vs. major dent).

Setting up the initial ONNX export scripts to benchmark inference speeds.

Next Steps
Validation: Testing the model against edge cases like low-light conditions and wet car surfaces.

Quantization: Exploring model quantization within the ONNX runtime to further reduce latency for real-time applications.

API Integration: Wrapping the model in a service that can generate an automated damage report for Vroom’s check-in system.
