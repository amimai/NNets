data_file='all_data_223k_3y_m5.csv'
file_truth='PyTorch/data/Finance_4b_4f/t_msd.csv'
file_mean='PyTorch/data/Finance_4b_4f/t_mean.csv'
file_std='PyTorch/data/Finance_4b_4f/t_std.csv'

def pred_denorm(pred,file_truth,file_mean,file_std,forward,back):
    df_p = pd.DataFrame(pred)
    dat_m = pd.read_csv(file_mean, header=None, names=['keys', 'values'], index_col='keys')
    dat_s = pd.read_csv(file_std, header=None, names=['keys', 'values'], index_col='keys')
    dat = pd.read_csv(file_truth, index_col='date')

    df_p.columns = dat.columns

    for i in range(len(df_p.columns)):
        df_p[df_p.columns[i]] = df_p[df_p.columns[i]] * dat_s['values'][i]  +dat_m['values'][i]
    mod = forward+back-1
    df_p.index = dat.index[mod:len(df_p.index) + mod]
    return df_p


if __name__ == '__main__':
    from PyTorch.data_analysis.old.test_model5 import *
    savedir = 'PyTorch/data_analysis/saved_mod/test_5'
    train = trainer()
    train.boot(train_loader, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers, backwards=backwards)
    train.load(savedir, 465)
    pred, targ, loss = train.evaluate(test_loader, pred=True)