pip list
tar -xvzf tsv1_ct.tar.gz
unzip SharkSeagrass.zip
rm SharkSeagrass.zip
rm tsv1_ct.tar.gz
mv tsv1_ct ./SharkSeagrass/
mv model_best_181_state_dict.pth ./SharkSeagrass/
cd SharkSeagrass
mkdir cache
cd cache
mkdir wandb
cd ..
export WANDB_API_KEY="41c33ee621453a8afcc7b208674132e0e8bfafdb"
export WANDB_DIR="cache/wandb"
python train_v2_vq.py
rm -rf tsv1_ct
cd ../
tar -czvf SharkSeagrass_results.tar
rm -rf SharkSeagrass
