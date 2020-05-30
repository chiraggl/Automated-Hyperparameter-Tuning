FROM centos

RUN yum install python3 -y

RUN pip3 install --upgrade pip

RUN pip3 install --upgrade setuptools

RUN pip3 install keras

RUN pip3 install tensorflow

RUN pip3 install pandas

RUN pip3 install pillow

RUN pip3 install opencv-python

ENTRYPOINT [ "python3" ]

CMD [ "-h" ]

