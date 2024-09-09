unzip SharkSeagrass.zip
rm SharkSeagrass.zip

tar -xzvf fold_0.tar.gz
tar -xzvf fold_1.tar.gz
tar -xzvf fold_2.tar.gz
tar -xzvf fold_3.tar.gz
tar -xzvf fold_4.tar.gz
rm fold_0.tar.gz
rm fold_1.tar.gz
rm fold_2.tar.gz
rm fold_3.tar.gz
rm fold_4.tar.gz
mv fold_0.hdf5 ./SharkSeagrass/
mv fold_1.hdf5 ./SharkSeagrass/
mv fold_2.hdf5 ./SharkSeagrass/
mv fold_3.hdf5 ./SharkSeagrass/
mv fold_4.hdf5 ./SharkSeagrass/
mv vq_f4-noattn.ckpt ./SharkSeagrass/

cd SharkSeagrass
mkdir results
ls
pip install pytorch-lightning
echo "============================================"
python UNetUNet_v1_py1_prepare_dataset.py --train_fold $1 --val_fold $2  --test_fold $3
# python UNetUNet_v1_py2_train.py
du -lh -d 1
mv data_div.json ./results/
rm -rf dataset
rm fold_0.hdf5
rm fold_1.hdf5
rm fold_2.hdf5
rm fold_3.hdf5
rm fold_4.hdf5
mv cache ./results/
tar -czvf SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar.gz results
echo "============================================"
ls
mv SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar.gz ../
echo "============================================"
ls
