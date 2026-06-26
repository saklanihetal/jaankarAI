# JaankaarAI: Multimodal Multilingual Regional Fake News Detection

## Overview
Fake news and visual manipulation have emerged as critical threats to the society and digital ecosystem, especially for regional languages where automated tools for verification of news are limited. The misinformation being spread through social media is not just confined to text, it often combines text with unrelated or manipulated images including deepfakes to make the news more convincing to the audience reading it. 

This project proposes a **Multi-modal Multilingual Regional Fake News Detection System** for detecting regional fake news. The proposed system supports regional languages and integrates:
- Text-based fake news detection
- Text-image consistency analysis
- Image-based deepfake detection

Transformer-based language models are used to analyze textual content in regional languages and English, while deep learning-based image models are used to detect manipulated images. By jointly analyzing text and images, the proposed system improves the reliability of fake news detection in regional contexts.

## Architecture Pipeline
The JaankaarAI pipeline follows a six-stage architecture:
1. **Input Ingestion and Preprocessing**: Language Detection and Translation (NLLB).
2. **Query Generation and Evidence Retrieval**: Keywords extraction and live retrieval via News APIs.
3. **Evidence Ranking and Natural Language Inference**: Semantic similarity using LaBSE and entailment/contradiction reasoning via BART-Large-MNLI.
4. **Multimodal Verification**: Image-text consistency (CLIP), Deepfake detection (CNN), and Facial recognition (AWS Rekognition).
5. **Decision Fusion and Gated Classification**: Fusion of support/contradiction gates.
6. **Final Verdict and User Interface**: Streamlit-based Web UI.
