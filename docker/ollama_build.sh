#!/bin/sh
#Start Ollama engine in the background
ollama serve &
#Wait for server inits
sleep 5
#Pull frontier models for inference
ollama pull llama3.2
ollama pull gemma4:e2b
#Pull embedding models
ollama pull nomic-embed-text
ollama pull embeddinggemma
# Kill engine to finish image layer processing
pkill ollama