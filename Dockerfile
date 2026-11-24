FROM flwr/superexec:1.31.0

USER root
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir .

USER app
ENTRYPOINT ["flower-superexec"]
