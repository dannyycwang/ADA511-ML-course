# ADA511-Datavitenskap og KI-prototyping
Course slides repository

Machine Learning Part
---

## 🧠 Machine Learning I — Foundations & Neural Networks

**Machine Learning I** introduces ML as a **function learning problem**.

### Topics

- ML as prediction and function approximation
- Mathematical functions → learned functions
- Data representations: scalar, vector, matrix, tensor
- Neural Networks (MLP)
  - Neurons, weights, bias
  - Activation functions (ReLU, Sigmoid, etc.)
  - Universal Approximation Theorem
- Forward propagation & backpropagation
- Loss functions
  - Mean Squared Error
  - Cross-Entropy
- Gradient descent & optimizers
  - SGD, Adam
- Softmax and probabilistic outputs
- Hyperparameters vs model parameters
- Introduction to overfitting and underfitting

**Goal:**  
Understand *what neural networks compute* and *how learning happens in parameter space*.

---

## 📊 Machine Learning II — Evaluation, Generalization & Calibration

**Machine Learning II** focuses on **how we evaluate models** and make predictions trustworthy.

### Topics

#### Model Evaluation
- Training / Validation / Test splits
- Confusion matrix
- Classification metrics:
  - Accuracy, Precision, Recall, F1-score
- Regression metrics:
  - MAE, MSE, RMSE, R²
- Macro vs weighted averaging (imbalanced data)

#### Overfitting & Underfitting
- Bias–variance tradeoff
- Diagnosing learning curves
- Mitigation techniques:
  - L1 / L2 regularization
  - Dropout
  - Batch normalization
  - Early stopping
  - Data augmentation

#### Model Calibration
- Why softmax ≠ probability
- Reliability diagrams
- Expected Calibration Error (ECE)
- Post-hoc calibration:
  - Temperature scaling
  - Platt scaling
  - Histogram calibration

**Goal:**  
Move from *accurate models* to *reliable, decision-aware models*.

---

## 🤖 Large Language Models & RAG

The course also introduces modern **Large Language Models (LLMs)**:

- Core concepts of LLMs
- Prediction-based language modeling
- Retrieval-Augmented Generation (RAG)
- LLM evaluation metrics:
  - Perplexity
  - BLEU
  - BERTScore

Connections between **probability, prediction, and utility** are emphasized.

---

## 🛠 Hands-on Components

- Neural network playgrounds
- Forward & backward propagation exercises
- Hyperparameter tuning experiments
- Regularization and calibration practice
- Visualization tools for optimization dynamics

---

## 🎓 Learning Outcomes

After completing this course, students will be able to:

- Formulate ML problems as function learning tasks
- Design and train neural networks from first principles
- Diagnose and fix overfitting and underfitting
- Choose appropriate evaluation metrics
- Interpret and calibrate model confidence
- Understand strengths and limitations of LLM-based systems

---

## 📖 References

- Vapnik, *The Nature of Statistical Learning Theory*
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017
- Geman et al., *Neural Networks and the Bias/Variance Dilemma*
- Russell & Norvig, *Artificial Intelligence: A Modern Approach*

---

## 📜 License

This repository is intended for **educational and non-commercial use**.

---

## 🤝 Contributing

This material is under active development.  
Issues, suggestions, and pull requests are welcome.
