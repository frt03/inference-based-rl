# How to use
0. `docker build . -t`
1. `docker run --runtime=nvidia -it --privileged -v /path/to/yourworkspace/workspace:/root/workspace -p 9999:9999`
2. `cd setup`
3. `source setup.sh`