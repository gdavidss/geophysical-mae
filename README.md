# Geophysical Data Embedding with Vision Transformer-Based Masked Autoencoders

Incorporating geophysical data enhances our ability to analyze and interpret subsurface structures. This project explores the use of Vision Transformer (ViT) based Masked Autoencoders (MAEs) to generate compact and robust embeddings of geophysical datasets. 

We work with 2D survey magnetic maps derived from simulated 3D voxel datasets containing density ellipsoids. The 3D data is transformed into 2D by aggregating along the z-axis. We adapt MAEs for both single-channel and multi-channel representations of this transformed geophysical data and determine embedding dimensions that provide optimal compression while maintaining reconstruction quality.

Our ViT-based implementation achieves excellent reconstruction even with high masking ratios up to 95%, demonstrating a reconstruction loss of $1.77 \times 10^{-2}$. The quality of these learned embeddings is crucial for efficient downstream processing and analysis of large-scale geophysical surveys.
