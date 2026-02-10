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
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies (cached unless pyproject.toml or uv.lock change)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

# Copy application code (changes here don't re-install deps)
COPY add_gym/ add_gym/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-editable

COPY sagemaker-entrypoint.sh /sagemaker-entrypoint.sh

RUN chmod +x /sagemaker-entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"

# Override the base image entrypoint for SageMaker compatibility
ENTRYPOINT ["/sagemaker-entrypoint.sh"]
CMD ["python", "-m", "add_gym.main"]
