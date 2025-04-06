# [MIE1517] Project-YoungHee (Team 20)

![End-to-end Pipeline](figures/project_description.png)

This project implements an AI-powered version of the popular "Red Light, Green Light" game (also known as "YoungHee" game), inspired by the one seen in the series "Squid Game." The system uses computer vision and machine learning to detect player movements and specific actions.

Overview
Young-hee is an interactive game that combines real-time pose detection using MediaPipe with action recognition using deep learning. The game alternates between "Red Light" and "Green Light" phases:

During Green Light: Players must perform a specific action (boxing, handclapping, handwaving, or walking) to earn points
During Red Light: Players must remain still, or they'll be penalized

Features

Real-time pose detection using MediaPipe
Custom-trained deep learning model (Bidirectional LSTM with attention mechanism) for human action recognition
Three difficulty levels (easy, medium, hard)
Visual feedback on player performance
Score tracking system

References
- Zhou, Q., & Wu, H. (2018, October). NLP at IEST 2018: BiLSTM-attention and LSTM-attention via soft voting in emotion classification. In Proceedings of the 9th workshop on computational approaches to subjectivity, sentiment and social media analysis (pp. 189-194).
