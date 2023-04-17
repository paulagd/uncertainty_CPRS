# Leveraging uncertainty in recommender systems for solving the Contextual Pure Cold-Start Problem: The case of PSM

_We propose an innovative approach that considers uncertainty to provide fairer and more variate recommendations in a purely contextual cold-start scenario from a Public Service Media (PSM)._

## Starting 🚀


### PSM dataset


- The original dataset can be found in [zenodo](https://zenodo.org/record/7801273) in order to reproduce the results. 
- The file should be placed in a folder like "**data/name_dataset/**" with the name *contextual_dataset_PSM.csv*, which already comes with the datase once you download it.


### 📋  Project building

_This project is built with Pytorch and Python==3.6.9. You can insall the dependencies by running:_

```
pip install tensorboardX torch==1.7.0 torchvision==0.8.0 
pip install torch_sparse==0.6.10 torch-scatter==2.0.7
```

### ⚙️ Reproducing the code

* Data needs to be placed on `data/dataset` folder (eg. `data/non-logged`, as it is the default argument dataset).
* We have many **MODELS** available among which you can choose: `[ random, itempop, context_itempop, cemb, mlp ]`


1. You can run the code by doing (add `--save` flag to save model weights):
```
python main.py --model {@chosenmodel} --dataset {@chosendataset} --save
```

2. You can perform inference or tune hyperparams by adding more flags. Check `parse_args()` in *utils.py* file.
```
python main.py --model {@chosenmodel} --dataset {@chosendataset} --inference  
```

3. In order to reproduce the experiments of applying our ranker, proposed in the paper *"Leveraging uncertainty in recommender systems for solving the Contextual Pure Cold-Start Problem: The case of PSM"*, you can run the next command once any model is trained:

```
python main.py --model {@chosenmodel} --dataset {@chosendataset} --inference --aleatoric_ranker 
```

> For any other option, please check `parse_args()` function in the file *utils.py* . 

