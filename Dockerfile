# syntax=docker/dockerfile:1

FROM python:3.11

WORKDIR /code

# Copy the requirements.txt file
COPY requirements.txt .

COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application


# Expose port for Streamlit
EXPOSE 8501

# Set the entrypoint to run Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py"]
