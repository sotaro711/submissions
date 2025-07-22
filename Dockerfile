# これらのうち 1 行だけコメントアウトしない
 FROM python:3.12.3-bullseye
# FROM python:3.11.9-bullseye
# FROM python:3.10.14-bullseye
# FROM python:3.9.19-bullseye
# FROM python:3.8.19-bullseye

RUN pip install poetry
RUN poetry config virtualenvs.create false

WORKDIR /code
COPY ./pyproject.toml /code/
RUN poetry install --no-root

COPY . /code/
ENV PYTHONPATH=/code

CMD ["python", "main.py"]
