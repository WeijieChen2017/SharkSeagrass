pip list
tar -xvzf tsv1_ct.tar.gz
unzip SharkSeagrass.zip
rm SharkSeagrass.zip
rm tsv1_ct.tar.gz
mv tsv1_ct ./SharkSeagrass/
mv model_best_181_state_dict.pth ./SharkSeagrass/
cd SharkSeagrass
python train_v2_vq.py
rm -rf tsv1_ct
find . -name wandb-metadata.json
find / -name wandb-metadata.json
mv cache ./results/
tar -czvf SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar results
rm -rf SharkSeagrass
