# Use an official Python runtime as a parent image
FROM python:3.11.6

RUN mkdir /app

# Set the working directory to /app
WORKDIR /app

# RUN bash -c "ls"
RUN echo $(ls .)

# # Copy the current directory contents into the container at /app
COPY requirements.txt /app

# Install packages from the package list in requirements.txt
# RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN pip install -r requirements.txt

# Install the local pkg
# RUN pip install .


# Run app.py when the container launches
CMD ["ls"]
# CMD ["python", "/app/app.py"]