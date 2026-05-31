FROM ollama/ollama:latest

# Copy initialization script
COPY ./ollama_build.sh /tmp/builder.sh

# Make executable and run inside docker build
RUN chmod +x /tmp/builder.sh && /tmp/builder.sh

EXPOSE 11434