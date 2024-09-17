echo "============================================"
ls
echo "============================================"
unzip SharkSeagrass.zip
rm SharkSeagrass.zip

tar -xzvf TC256.tar.gz
tar -xzvf cv${1}_256_clip.tar.gz
rm TC256.tar.gz
mv TC256 ./SharkSeagrass/
rm cv${1}_256_clip.tar.gz
mv cv${1}_256_clip ./SharkSeagrass/
mv d3f64_tsv1.pth ./SharkSeagrass/

cd SharkSeagrass
mkdir results
echo "============================================"
ls
# echo "============================================"
# Install PyTorch Lightning
# python -m pip install pytorch-lightning
# echo "============================================"
# Check if the installation succeeded
# python -m pip show pytorch-lightning
# echo "============================================"
# pip list
echo "============================================"
python UNetUNet_v1_py5_step2.py --cross_validation $1
echo "============================================"
# python UNetUNet_v1_py2_train.py
du -lh -d 1
rm -r TC256
rm -r cv${1}_256
mv cache ./results/
tar -czvf UNetUnet_256_cv${1}_step2.tar.gz results
echo "============================================"
ls
echo "============================================"
mv UNetUnet_256_cv${1}_step2.tar.gz ../
echo "============================================"
ls
echo "============================================"