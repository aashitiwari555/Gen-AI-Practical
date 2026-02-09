# Gen-AI-Practical

This repository documents my learning journey in working with image and text data using Python, with a focus on preprocessing, analysis, visualization, and understanding how TensorFlow executes computations.  
The goal is to build strong foundations required before training generative AI models.

Rather than focusing on model training, this project emphasizes **how data is prepared, explored, and reasoned about**, which is a critical step in any real-world GenAI pipeline.

---

## Overview

The repository explores two widely used datasets:

- **MNIST** – handwritten digit images used for image-based analysis  
- **20 Newsgroups** – a large text corpus used for natural language analysis  

Using these datasets, I implemented workflows for:
- loading and inspecting data
- computing statistics
- removing duplicate samples
- preprocessing image and text data
- visualizing meaningful patterns
- understanding TensorFlow execution modes

---

## Repository Structure
```
GENAI/
├── Practice_1/
│   ├── img_dataset/
│   │   └── mnist_loader.py
│   ├── txt_dataset/
│   │   └── newsgroup_loader.py
│   ├── img_analysis.py
│   ├── txt_analysis.py
│   ├── duplicate_removal.py
│   ├── preprocessing_pipeline.py
│   ├── visualization_txt_length.py
│   ├── visualization_patterns.py
│   └── tensorflow_demo.py
├── .gitignore
└── .gitattributes
```
Each script is modular and focuses on a single concept to keep experimentation and understanding clear.

---

## What Each Part Explores

### Image Data (MNIST)
- Loading and visualizing sample images
- Understanding image shape and pixel structure
- Computing basic statistics such as mean, minimum, and maximum pixel values
- Normalizing pixel values before model usage
- Visualizing pixel intensity distributions and image patterns

### Text Data (20 Newsgroups)
- Loading raw text data into Pandas DataFrames
- Counting total and unique words
- Identifying and removing duplicate text samples
- Cleaning text by lowercasing and removing punctuation
- Analyzing text length distributions and trends

### Preprocessing Pipeline
- Image normalization with quantitative verification using intensity distributions
- Text cleaning with before/after comparisons
- Visualization used as validation rather than decoration

### Visualization
Multiple visualization techniques are used to analyze patterns:
- Histograms for distributions
- Line plots for trends
- Image grids for visual structure
These techniques help in understanding data characteristics before any learning step.

### TensorFlow Execution
A small TensorFlow example demonstrates:
- **Eager execution** (default, immediate evaluation)
- **Computational graph execution** using `@tf.function`

The emphasis is on understanding *how* TensorFlow runs operations rather than training models.

---

## Key Learning Takeaways

- Data preprocessing changes **numerical properties**, not necessarily visual appearance
- Visualization must be designed carefully to avoid misleading interpretations
- Text and image data require different representations and tools
- TensorFlow execution modes affect performance, not output correctness
- Clean, modular code makes experimentation and debugging easier

---

## Tools and Libraries Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- scikit-learn  
- TensorFlow  

---

## Notes

This repository is intentionally kept simple and focused on clarity.  
The emphasis is on *understanding the data and execution flow*, which forms the backbone of any future work involving generative AI models.

---

## Status

Actively learning and refining concepts.
More experiments and extensions will be added as understanding deepens.
