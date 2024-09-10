unzip SharkSeagrass.zip
rm SharkSeagrass.zip

tar -xzvf TOFNAC_CTAC_hash.tar.gz
rm TOFNAC_CTAC_hash.tar.gz
mv TOFNAC_CTAC_hash ./SharkSeagrass/
mv vq_f4-noattn.ckpt ./SharkSeagrass/

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
rm -r TOFNAC_CTAC_hash
mv cache ./results/
tar -czvf SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar.gz results
echo "============================================"
ls
echo "============================================"
mv SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar.gz ../
echo "============================================"
ls
echo "============================================"