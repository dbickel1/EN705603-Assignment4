FROM python:3.8
COPY . ./
RUN pip3 install -r requirements.txt
CMD ["assignment4_djb.py"]
ENTRYPOINT ["python"]