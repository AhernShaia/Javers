# Use the official Python image from the Docker Hub
FROM python:3.12-slim-buster

# Install Poetry
RUN pip install poetry

# Set the WORKDIR to /app so all following commands run in /app
WORKDIR /app

# Copy the project's poetry.lock file to the WORKDIR
COPY poetry.lock pyproject.toml /app/

# Disable the creation of the virtual environment
# Then install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of the project to WORKDIR
COPY . /app

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn

# Expose port 8000
EXPOSE 8000

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]