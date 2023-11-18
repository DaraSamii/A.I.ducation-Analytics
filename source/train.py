from classes import *


if __name__ == "__main__":
    path = "./DataSet/"
    model_path ="./models/"
    ds = ImageDataset(path)

    ds_labels = list(ds.classes_dict.keys())
    train_set, val_set, test_set = T.utils.data.random_split(ds, [0.7,0.15,0.15])

    train_dl = DataLoader(train_set, 64, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_set, 64, shuffle=True, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_set, 64, shuffle=True, num_workers=2, pin_memory=True)


    print("Full DataSet image count:", len(ds))
    print("Train dataset: ", len(train_set), "Valid dataset: ", len(val_set),"Valid dataset: ",len(test_set))



    block_sizes = [16,32,64]
    num_layers =  [4,8]
    kernels = [3,5,7]
    AP = 20
    epochs = 100

    lr = 0.001
    wd = 0.0001
    for B in block_sizes:
        for n in num_layers:
            for k in kernels:    
                F = int(n/3)   
                model_dir = os.path.join(model_path,"B{}FC{}K{}AP{}".format(n,F,k,AP))
                os.mkdir(model_dir)
                model_name = os.path.join(model_dir,"B{}FC{}K{}AP{}".format(n,F,k,AP))
                model = ResNet(num_classes=4,num_channel=B,num_blocks=n,num_fc_layers=F,avg_pool_size=AP)

                save_best_path = os.path.join(model_path,model_name)
                learn = Learner(train_dl, val_dl, model,labels_name=ds_labels,save_best=model_name,log_path=model_name)

                learn.train_eval(epochs=epochs,lr=lr,wd = wd)

                learn.plot_metrics_macro(model_name + "metrics_mocro.svg")
                learn.plot_metrics_micro(model_name + "metrics_micro.svg")
                learn.plot_loss(model_name + "Loss.svg")
                
                print("creating confusion matrix for validation")
                learn.predict(val_dl)
                learn.plotConfisuionMatrix(model_name + "confMatrixVal.svg")

                print("creating confusion matrix for Test")
                learn.predict(test_dl)
                learn.plotConfisuionMatrix(model_name + "confMatrixTest.svg")
