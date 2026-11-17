# Next steps

## 1. Rerun with the new fixes (road alignment + rail detection)
```
python lane_extract.py data/drive0001.npy --x-range 0 60 --out results
```
The output now prints the road heading correction and how many rails were found; rails are drawn as grey dashed lines. Use `--no-align` to keep the original frame (needed for `eval_synthetic.py`).

## 2. ML classifier (local)
Needs a second drive to test on. Download drive 0011 into kitti/ (same aria2c command, drive_0011), unzip with tar, then:
```
python accumulate.py --base kitti --date 2011_09_26 --drive 0011 --n 40 --step 2 --out data/drive0011.npy
python train_classifier.py --train data/drive0001.npy --test data/drive0011.npy --out results/ml
```
Takes a few minutes. Outputs: model, metrics.json, fabric_metrics.csv, two prediction maps.

## 3. Azure Machine Learning
1. Sign up for **Azure for Students** with your UM email (no credit card needed; check the current offer on the signup page).
2. Azure portal > create an **Azure Machine Learning workspace**. Open it in Azure ML studio, download **config.json** (workspace menu, top right) into `azure/`.
3. Install the Azure CLI, then: `pip install azure-ai-ml azure-identity` and `az login`.
4. Submit: `python azure/submit_job.py --train data/drive0001.npy --test data/drive0011.npy`
5. Screenshot the job's Metrics tab and the registered outputs for the README and LinkedIn.

## 4. Microsoft Fabric + Power BI
1. Start the Fabric free trial (needs a work or school account, so use the UM email).
2. Create a workspace > **Lakehouse**. Upload `results/ml/fabric_metrics.csv` and the lane JSON results, load them as tables.
3. Build a Power BI report on the Lakehouse: holdout vs unseen-drive scores, feature importance, lanes found per drive.
4. Screenshot it for the README.
