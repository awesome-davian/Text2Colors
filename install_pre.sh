conda install matplotlib --yes
yes | pip install scikit-image
mkdir data
cd data
wget --load-cookies=/tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies=/tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CtYnsTASXFZZ4RCJXZaxEWvybz_4AMa4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CtYnsTASXFZZ4RCJXZaxEWvybz_4AMa4" -O data.zip && rm -rf /tmp/cookies.txt
unzip data.zip
rm -f data.zip
