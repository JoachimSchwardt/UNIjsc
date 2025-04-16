PATH=r"/media/ag_budich1/CARL2/joachim_150224/"
for cdir in os.listdir(PATH):
    for subdir in os.listdir(PATH+cdir):
        for data in os.listdir(PATH+cdir+"/"+subdir):
            path = r"/home/ag_budich1/JoachimSchwardt/MasterThesisLL/MA_Code/data_joachim/" + subdir + "_beta1/"
            if data.endswith(".npy") or data == "jobscript" or (data.startswith("data_MPI_SBA") and data.endswith("h0.030000.txt")):
                if not os.path.exists(path):
                    os.mkdir(path)
                shutil.copy(PATH+cdir+"/"+subdir+"/" + data, path + data)
