from classes import *
from sklearn.model_selection import KFold,train_test_split
import torch as T
if __name__ == "__main__":
    path_old = "./DataSet_old/"
    path = "./Gender_Age_DataSet/"
    model_path ="./folds/"
    B = 16
    N =  8
    K = 5 
    FC = 2
    AP = 20
    epochs = 1
    k_folds = 10

    ds = ImageDataset(path_old)

    ds_labels = list(ds.classes_dict.keys())
    

    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(ds)):
        if fold == 8:
            train_subsampler = T.utils.data.SubsetRandomSampler(train_idx)
            valid_subsampler = T.utils.data.SubsetRandomSampler(valid_idx)


            train_dl = DataLoader(ds, 64, num_workers=2, pin_memory=True,sampler=train_subsampler)
            val_dl = DataLoader(ds, 64, num_workers=2, pin_memory=True,sampler=valid_subsampler)

            lrs = [0.001]#,0.0001]
            wds = [0.0001]#, 0.001]

            model_dir = os.path.join(model_path,"B{}_N{}_FC{}_K{}_AP{}_fold{}".format(B,N,FC,K,AP,fold))
            if os.path.isdir(model_dir) == False:
                os.mkdir(model_dir)
            model_name = os.path.join(model_dir,"B{}_N{}_FC{}_K{}_AP{}_fold{}".format(B,N,FC,K,AP,fold))
            model = ResNet(num_classes=4,num_channel=B,num_blocks=N,num_fc_layers=FC,avg_pool_size=AP)

            save_best_path = os.path.join(model_path,model_name)

            learn = Learner(train_dl, val_dl, model,labels_name=ds_labels,save_best=model_name,log_path=model_name)
            for lr in lrs:
                for wd in wds:  
                    learn.train_eval(epochs=epochs,lr=lr,wd = wd)


            dsp = ImageDataset(path)
            ds_files = dsp.files

            ds_labels = list(dsp.classes_dict.keys())
            
            test_dl = DataLoader(dsp, 64, num_workers=2, pin_memory=True)

            all_predictions, all_labels, all_probabilites = learn.predict(test_dl)

            df = pd.DataFrame(
                {"path": ds_files,
                "true_label":all_labels,
                "pred": all_predictions,
                "prob": all_probabilites}
            )
            df.to_csv("pred_resutls.csv")
