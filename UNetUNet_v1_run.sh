unzip SharkSeagrass.zip
rm SharkSeagrass.zip

tar -xzvf TC256_v2.tar.gz
rm TC256_v2.tar.gz
mv TC256_v2 ./SharkSeagrass/
mv vq_f4_noattn_nn.pth ./SharkSeagrass/

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
# python UNetUNet_v1_py2_train.py --cross_validation $1
python UNetUNet_v1_py2_train_acs.py --cross_validation $1
echo "============================================"
# python UNetUNet_v1_py2_train.py
du -lh -d 1
rm -r TC256_v2
mv cache ./results/
tar -czvf UNetUnet_256_cv$1.tar.gz results
echo "============================================"
ls
echo "============================================"
mv UNetUnet_256_cv$1.tar.gz ../
echo "============================================"
ls
echo "============================================"