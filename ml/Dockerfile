FROM centos

RUN yum install python3 -y

RUN pip3 install --upgrade pip

RUN pip3 install --upgrade setuptools

RUN pip3 install scikit-learn

RUN pip3 install pandas

RUN pip3 install numpy

RUN pip3 install opencv-python

RUN pip3 install matplotlib

ENTRYPOINT [ "python3" ]

CMD [ " -h " ]

