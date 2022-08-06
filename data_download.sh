"""
Modified version of download.sh in https://github.com/naver-ai/StyleMapGAN
"""
# !/bin/bash

FILE_NAME="data.zip"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tQmWrjRv7SZ03MFQpxQgCxNz1ZAgbkop' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tQmWrjRv7SZ03MFQpxQgCxNz1ZAgbkop" -O $FILE_NAME && rm -rf /tmp/cookies.txt
unzip $FILE_NAME
rm $FILE_NAME