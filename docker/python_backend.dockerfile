FROM python/python:3.13.13-alpine
WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /src/data

RUN su -;useradd -m rag_admin;echo "rag_admin:P455w0rd" | chpasswd;usermod -aG sudo rag_admin
RUN su rag_admin

COPY RAGService.py /src
COPY .env /src
COPY ./data ../mnt/data

CMD ["python",  "/src/RAGService.py"]