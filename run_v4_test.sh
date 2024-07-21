pip list
tar -xvzf tsv1_ct.tar.gz
unzip SharkSeagrass.zip
rm SharkSeagrass.zip
rm tsv1_ct.tar.gz
mv tsv1_ct ./SharkSeagrass/
mv model_best_1900_state_dict_2.pth ./SharkSeagrass/
cd SharkSeagrass
ls
echo "============================================"
python test_v4.py
rm -rf tsv1_ct
mv cache ./results/
tar -czvf SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar.gz results
echo "============================================"
ls
mv SharkSeagrass_results_$(date +"%m_%d_%H_%M").tar.gz ../
echo "============================================"
ls
