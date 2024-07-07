pip list
tar -xvf tsv1_ct_80.tar
unzip SharkSeagrass.zip
rm SharkSeagrass.zip
rm tsv1_ct_80.tar
mv tsv1_ct_80 ./SharkSeagrass/
mv model_best_181_state_dict.pth ./SharkSeagrass/
cd SharkSeagrass
ls
echo "============================================"
python train_v4_pyramid.py
rm -rf tsv1_ct_80
mv cache ./results/
tar -czvf SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar.gz results
echo "============================================"
ls
mv SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar.gz ../
echo "============================================"
ls
