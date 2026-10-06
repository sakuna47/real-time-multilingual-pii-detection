# real-time-multilingual-pii-detection
A real-time, explainable PII detection system for AI prompts, with Sri Lankan regional PII support, multilingual Sinhala/Tamil/English detection, implicit PII detection, and XAI.

# Real-Time Multilingual PII Detection for AI Prompts

This project focuses on detecting Personally Identifiable Information (PII) in AI prompts before the prompt is submitted to an AI system.

The main focus of the project is to develop a real-time PII detection system that supports Sri Lankan PII formats and multilingual inputs, including English, Sinhala, Tamil, and code-switched text.

The system will detect sensitive information such as names, NIC numbers, phone numbers, email addresses, and other personal information. It will also investigate contextual or implicit PII that may identify a person even when there is no direct identifier.

## Main Objectives

- Detect PII in AI prompts in real time.
- Support Sri Lankan NIC and phone number formats.
- Support Sinhala, Tamil, English, and mixed-language prompts.
- Compare rule-based and machine learning approaches.
- Detect implicit or contextual PII.
- Provide explanations for why information was detected as PII.
- Provide users with options to edit, mask, or continue with a detected PII input.

## Proposed Approach

The system will use a combination of rule-based and machine learning techniques.

The initial layer will use regular expressions to detect structured information such as Sri Lankan NIC numbers, phone numbers, email addresses, and other patterns.

A transformer-based NER model, mainly XLM-RoBERTa, will then be used to detect contextual PII such as names and addresses.

GLiNER will also be investigated as a zero-shot PII detection approach.

An additional classification model will be developed to identify implicit PII. This is intended to detect cases where several pieces of information together could identify a person.

Explainable AI techniques such as SHAP will be investigated to provide information about why a particular part of a prompt was detected.

## System

The planned system will contain:

1. Regex-based PII detection
2. Transformer-based NER
3. Multilingual and code-switched text handling
4. Implicit PII detection
5. Explainable AI
6. Real-time warning and masking interface

The user interface is planned as a Chrome extension, with a FastAPI backend used for the detection services.

## Technologies

- Python
- FastAPI
- Hugging Face Transformers
- XLM-RoBERTa
- GLiNER
- BERT
- Microsoft Presidio
- spaCy
- SHAP
- scikit-learn
- Pandas
- Faker
- Chrome Extension APIs

## Dataset

The project will use the AI4Privacy PII-Masking-300k dataset as one of the main training resources.

A separate synthetic Sri Lankan PII dataset will also be created for this project. It will include Sri Lankan NIC numbers, phone numbers, names, multilingual sentences, code-switched prompts, and contextual PII examples.

Only synthetic personal information will be used for the project.

## Evaluation

The system will be evaluated using:

- Precision
- Recall
- F1-score
- Entity-level F1
- False positive rate
- Detection latency

Different system configurations will also be compared to understand the contribution of each component.

The planned experiments include comparisons between Presidio, spaCy, custom regex, XLM-R, GLiNER, and the complete proposed system.

## Project Structure

The repository will contain the implementation, experiments, notebooks, backend, browser extension, and testing code.

```text
src/
backend/
chrome-extension/
notebooks/
tests/
data/
models/
experiments/
requirements.txt
README.md
