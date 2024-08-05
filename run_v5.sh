pip list
tar -xzf crop.tar
unzip SharkSeagrass.zip
rm SharkSeagrass.zip
rm crop.tar
mv crop ./SharkSeagrass/
mv model_best_3350_state_dict_2.pth ./SharkSeagrass/
cd SharkSeagrass
ls
echo "============================================"
python train_v5.py
rm -rf crop
mv cache ./results/
tar -czvf SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar.gz results
echo "============================================"
ls
mv SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar.gz ../
echo "============================================"
ls
