pip list
tar -xvzf tsv1_ct.tar.gz
unzip SharkSeagrass.zip
rm SharkSeagrass.zip
rm tsv1_ct.tar.gz
mv tsv1_ct ./SharkSeagrass/
mv model_best_181_state_dict.pth ./SharkSeagrass/
cd SharkSeagrass
python train_v1.py
rm -rf tsv1_ct
cd ../
tar -czvf SharkSeagrass_results.tar
rm -rf SharkSeagrass
