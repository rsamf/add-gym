ARG BASE_IMAGE
FROM $BASE_IMAGE

# Install uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app


COPY pyproject.toml .
RUN uv sync --no-dev

COPY add_gym/ add_gym/
COPY sagemaker-entrypoint.sh /sagemaker-entrypoint.sh

RUN chmod +x /sagemaker-entrypoint.sh

# Override the base image entrypoint for SageMaker compatibility
ENTRYPOINT ["/sagemaker-entrypoint.sh"]
CMD ["python", "-m", "add_gym.main"]
