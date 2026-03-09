---
layout: page
title: Projects
permalink: /projects/
---

## Chatbot

*Goal-oriented chatbot with casual conversation function.*

[View on GitHub](https://github.com/aramakus/Chatbot.git)

![Schematic diagram of the chatbot](/assets/images/projects/chatbot/scheme.png)

Ever since the Turing test was formulated by Alan Turing in 1950, ever more sophisticated chatbots were created for both commercial and research applications. In the context of recent advances in Natural Language Processing (NLP), modern chatbots are increasingly more accurate at inferring and addressing user queries on a large variety of topics. While state-of-the-art systems, such as IBM Watson, can be incredibly sophisticated, most chatbots share basic building blocks that perform a set of simple functions. The chatbot I developed demonstrates how some of these basic blocks work together to find a code-related query through a casual conversation and provide the user with a relevant link to a Stack Overflow thread.

The objective of the chatbot is to pick up a software related question, suggest a relevant thread from Stack Overflow, while being able to support a casual conversation. This requirement determines the first step, where a decision must be made if the user query is a software-related question, or a part of a casual conversation.

**Architecture:**

- **Intent classification** — 99% accuracy using Scikit-learn
- **Tag classification** — 80% accuracy
- **Embeddings** — word lookup tables; fine-tuning with RoBERTa for article references (see [this TDS article](https://towardsdatascience.com/fine-tuning-bert-and-roberta-for-high-accuracy-text-classification-in-pytorch-c9e63cf64646))
- **Casual conversation** — [ChatterBot library](https://chatterbot.readthedocs.io/en/stable/), trained on the Cornell Movie Dialogue Corpus and native corpus with [StarSpace](https://github.com/facebookresearch/StarSpace) embeddings

Online version deployed on AWS Elastic Cluster. Local version available on [GitHub](https://github.com/aramakus/Chatbot.git) — run with Docker or Docker Compose.
