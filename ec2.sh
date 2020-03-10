ssh -i ~/venv/automatedfishdetection/Study-is-happy.pem ubuntu@ec2-18-188-102-3.us-east-2.compute.amazonaws.com

cat << EOF | sudo tee --append /etc/modprobe.d/blacklist.conf
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist nvidiafb
blacklist rivatv
EOF
cat << EOF | sudo tee --append /etc/default/grub
GRUB_CMDLINE_LINUX="rdblacklist=nouveau"
EOF
sudo update-grub
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

rm cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
echo 'export PATH=/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1${PATH:+:${PATH}}' >> ~/.bashrc
sudo apt install -y python3-pip
pip3 install virtualenv

sudo reboot

# ssh-keygen -t rsa -b 4096 -C "zhang.zhiyo@husky.neu.edu"
# eval "$(ssh-agent -s)"
# ssh-add ~/.ssh/id_rsa
# sudo apt-get install xclip
# xclip -sel clip < ~/.ssh/id_rsa.pub

virtualenv venv
cd venv
source bin/activate
pip install numpy
pip install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip install torchvision
pip install opencv-python
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

git clone https://gitlab.com/neufieldrobotics/automatedfishdetection.git

cd automatedfishdetection/
git submodule init
git submodule update
cd detectron2/
python setup.py build develop
echo 'export PYTHONPATH=$PYTHONPATH:~/venv/automatedfishdetection/detectron2' >> ~/.bashrc
source ~/.bashrc
cd ..

python init.py

# upload update_instances to ec2
scp -i ~/venv/automatedfishdetection/Study-is-happy.pem ~/datasets/fish_detection/update/instances.json ubuntu@ec2-18-188-102-3.us-east-2.compute.amazonaws.com:~/fish_detection/update/instances.json

# upload update.zip to ec2
scp -i ~/venv/automatedfishdetection/Study-is-happy.pem ~/datasets/fish_detection/update.zip ubuntu@ec2-18-188-102-3.us-east-2.compute.amazonaws.com:~/fish_detection/

# unzip update
unzip update.zip

# download metrics.json to local
scp -i ~/venv/automatedfishdetection/Study-is-happy.pem ubuntu@ec2-18-188-102-3.us-east-2.compute.amazonaws.com:~/fish_detection/outputs/metrics.json ~/datasets/fish_detection/outputs/

 # download model.pth to local
scp -i ~/venv/automatedfishdetection/Study-is-happy.pem ubuntu@ec2-18-188-102-3.us-east-2.compute.amazonaws.com:~/fish_detection/outputs/model_0067999.pth ~/datasets/fish_detection/outputs/

# zip predict
zip -r predict.zip predict/

# download predict.zip to local
scp -i ~/venv/automatedfishdetection/Study-is-happy.pem ubuntu@ec2-18-188-102-3.us-east-2.compute.amazonaws.com:~/fish_detection/predict.zip ~/datasets/fish_detection/

# nohup
nohup python train.py > nohup.out 2>&1 &