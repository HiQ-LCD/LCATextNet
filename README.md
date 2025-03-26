# LCATextNet

A deep learning-based environmental impact prediction model that integrates residual networks, self-attention mechanisms, and system boundary classifiers into a unified neural network architecture.

## Project Overview

LCATextNet is a deep learning-based environmental impact prediction model that can predict life cycle environmental impact indicators based on product text descriptions. The model innovatively combines residual networks, self-attention mechanisms, and system boundary classifiers, significantly improving prediction accuracy and model generalization capability.

This project uses the Ecoinvent 3.10 database for training, focusing on predicting 25 environmental impact categories such as climate change (global warming potential), acidification, eutrophication, and more.

## Model Architecture

LCATextNet adopts a novel neural network architecture, mainly consisting of the following components:

1. **Text Feature Extraction Branch**: Processes multi-dimensional text embedding vectors, enhancing feature extraction capability through residual blocks
2. **System Boundary Classifier**: Predicts the system boundary classification of products in the life cycle assessment methodology
3. **Self-Attention Mechanism**: Captures relationships and dependencies between different text features
4. **Feature Fusion Module**: Integrates text features and system boundary information to generate comprehensive representations

Model Input:
- Activity Name: 768-dimensional text embedding vector
- Reference Product Name: 768-dimensional text embedding vector
- CPC Classification: 768-dimensional text embedding vector
- Product Information: 768-dimensional text embedding vector
- System Boundary: 768-dimensional text embedding vector
- General Comment: 768-dimensional text embedding vector
- Technology Comment: 768-dimensional text embedding vector

Model Output:
- Predictions for 25 environmental impact category indicators based on the EF methodology

## Experimental Results

On the Ecoinvent 3.10 dataset, LCATextNet achieved significant prediction performance:

- The Mean Relative Error (MRE) for Climate Change (GWP) indicators outperformed traditional methods
- Prediction accuracy was significantly improved for products with differentiated system boundaries
- Demonstrated good generalization capability for unseen product categories

For detailed experimental results and comparative analysis, please refer to the paper.

## Model Innovations

1. **Unified Neural Network Architecture**: Integrates text feature extraction, system boundary classification, and environmental impact prediction into an end-to-end model
2. **Residual Connections and Self-Attention Mechanism**: Significantly enhances feature representation capability and prediction performance
3. **System Boundary Awareness**: Captures system boundary information through a dedicated classifier, improving life cycle assessment
4. **Text Feature Fusion**: Effectively utilizes various product text descriptions for comprehensive feature extraction


## Contact Information

If you have any questions or suggestions, please contact us through:

- Email: luobiao@hiqlcd.com
- GitHub Issues: https://github.com/HiQ-LCD/LCATextNet/issues
