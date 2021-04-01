pip install -U pip
pip install -U setuptools
pip install -r requirements.txt
ln -s /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200
pip install git+git://github.com/deepmind/dm_control.git
pip install git+git://github.com/denisyarats/dmc2gym.git
pip install hydra-core --upgrade