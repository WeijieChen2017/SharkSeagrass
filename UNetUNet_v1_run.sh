unzip SharkSeagrass.zip
rm SharkSeagrass.zip

tar -xzvf TC256.tar.gz
rm TC256.tar.gz
mv TC256 ./SharkSeagrass/
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
python UNetUNet_v1_py2_train.py --cross_validation $1
echo "============================================"
# python UNetUNet_v1_py2_train.py
du -lh -d 1
rm -r TC256
mv cache ./results/
tar -czvf SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar.gz results
echo "============================================"
ls
echo "============================================"
mv SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar.gz ../
echo "============================================"
ls
echo "============================================"