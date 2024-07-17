pip list
tar -xvzf tsv1_ct.tar.gz
unzip SharkSeagrass.zip
rm SharkSeagrass.zip
rm tsv1_ct.tar.gz
mv tsv1_ct ./SharkSeagrass/
mv latest_model_2400_state_dict.pth ./SharkSeagrass/
mv latest_optimizer_2400_state_dict.pth ./SharkSeagrass/
cd SharkSeagrass
ls
echo "============================================"
python train_v4_universal.py
rm -rf tsv1_ct
mv cache ./results/
tar -czvf SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar.gz results
echo "============================================"
ls
mv SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar.gz ../
echo "============================================"
ls
