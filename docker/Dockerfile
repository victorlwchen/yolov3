FROM horovod/horovod:0.18.2-tf2.0.0-torch1.3.0-mxnet1.5.0-py3.6-gpu
 
ENV PASSWORD ocis#1234
 
RUN apt-get update 
RUN pip install numpy==1.15.1 Pillow==5.3.0 scipy==1.1.0 wget==3.2 seaborn==0.9.0
RUN pip install opencv-contrib-python easydict tqdm
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0
RUN pip install jupyter
RUN pip install notebook
WORKDIR /opt
RUN git clone https://github.com/victorlwchen/yolov3.git
WORKDIR yolov3

COPY jupyter.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/jupyter.sh
RUN sed -i -e 's/\r$//' /usr/local/bin/jupyter.sh

RUN mkdir -p /root/notebooks
WORKDIR /root/notebooks

#for ssh
RUN apt-get install -y openssh-server

#RUN mkdir /var/run/sshd
RUN echo root:$PASSWORD | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
  
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
  
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
  
COPY runssh.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/runssh.sh
EXPOSE 22 8888

#set ssh key between containers
RUN mkdir -p /root/.ssh/
COPY ssh/* /root/.ssh/
RUN chmod 600 -R /root/.ssh/id_rsa

CMD ["/usr/sbin/sshd", "-D"]

WORKDIR /opt/yolov3
RUN echo 'dummy for reset git pull cache'
RUN git pull


