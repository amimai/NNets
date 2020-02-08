import torch

#d2mean more punishing to models that predict the mean
#but are ignorant of directioanlity
class d2mse_loss(torch.nn.Module):
    def __init__(self):
        super(d2mse_loss,self).__init__()

    def forward(self,x,y):
        totloss = torch.nn.functional.mse_loss(x**2,y**2)
        return totloss

# d3 mean aggressively correct models for predicting the average
class L3_loss(torch.nn.Module):
    def __init__(self):
        super(L3_loss,self).__init__()

    def forward(self,x,y):
        return torch.nn.functional.mse_loss(x**3,y**3)

class shrink_loss(torch.nn.Module):
    def __init__(self):
        super(shrink_loss,self).__init__()

    def forward(self,x,y):
        return torch.nn.functional.mse_loss(x,torch.add(x,y)/2)

class scaled_error(torch.nn.Module):
    def __init__(self):
        super(scaled_error,self).__init__()

    def forward(self,x,y):
        meloss = torch.add(torch.nn.functional.mse_loss(x,y),torch.nn.functional.l1_loss(x,y))
        totloss = torch.nn.functional.mse_loss(x ** 2, y ** 2)
        return meloss**3 *totloss

class scaled_error_d(torch.nn.Module):
    def __init__(self):
        super(scaled_error_d,self).__init__()

    def forward(self,x,y):
        meloss = torch.add(torch.nn.functional.mse_loss(x,y),torch.nn.functional.l1_loss(x,y))
        totloss = torch.nn.functional.mse_loss(x ** 3, y ** 3)**.25
        return meloss**3 *totloss

class STD_MSE(torch.nn.Module):
    def __init__(self):
        super(STD_MSE,self).__init__()

    def forward(self,x,y):
        std = ((torch.std(x)-torch.std(y))**2)
        mse = torch.nn.functional.mse_loss(x, y)
        return std*mse

class L3_STD_MSE(torch.nn.Module):
    def __init__(self):
        super(L3_STD_MSE,self).__init__()

    def forward(self,x,y):
        std = ((torch.std(x)-torch.std(y))**2)
        mse = torch.nn.functional.mse_loss(x,y)
        mae = torch.nn.functional.l1_loss(x,y)
        l3  = torch.nn.functional.mse_loss(x ** 3, y ** 3)**.25
        mme = torch.add(mse,mae)
        return l3*std*mme**4

class L3_MSTD_MSE_old(torch.nn.Module):
    def __init__(self):
        super(L3_MSTD_MSE_old,self).__init__()

    def forward(self,x,y):
        std = (torch.std(x)-torch.std(y))**2
        mean= (torch.mean(x)-torch.mean(y))**2
        mstd = torch.add(std,mean)

        mse = torch.nn.functional.mse_loss(x,y)
        mae = torch.nn.functional.l1_loss(x,y)
        msea = torch.add(mse,mae)

        l3  = torch.nn.functional.mse_loss(x ** 3, y ** 3)**.25

        return torch.add(l3*msea**4,mstd)

class L3_MSTD_MSE(torch.nn.Module):
    def __init__(self):
        super(L3_MSTD_MSE,self).__init__()

    def forward(self,x,y):
        std = (torch.std(x)-torch.std(y))**2
        mean= (torch.mean(x)-torch.mean(y))**2
        mstd = torch.add(std,mean)

        mse = torch.nn.functional.mse_loss(x,y)
        mae = torch.nn.functional.l1_loss(x,y)
        msea = torch.add(mse,mae)

        l3  = torch.nn.functional.mse_loss(x ** 3, y ** 3)**.25

        return torch.add(l3*msea,mstd)

class L3_MSE_2(torch.nn.Module):
    def __init__(self):
        super(L3_MSE_2,self).__init__()

    def forward(self,x,y):

        mse = torch.nn.functional.mse_loss(x,y)
        mae = torch.nn.functional.l1_loss(x,y)
        msea = torch.add(mse,mae)

        l3  = torch.nn.functional.mse_loss(x ** 3, y ** 3)**.25

        return l3*msea

class L3_MSE(torch.nn.Module):
    def __init__(self):
        super(L3_MSE,self).__init__()

    def forward(self,x,y):

        mse = torch.nn.functional.mse_loss(x,y)
        mae = torch.nn.functional.l1_loss(x,y)
        msea = torch.add(mse,mae)

        l3  = torch.nn.functional.mse_loss(x ** 3, y ** 3)**.25

        return l3*msea

class BL3_MSTD_MSE(torch.nn.Module):
    def __init__(self):
        super(BL3_MSTD_MSE,self).__init__()

    def forward(self,x,y):
        std = (torch.std(x)-torch.std(y))**2
        mean= (torch.mean(x)-torch.mean(y))**2
        mstd = torch.add(std,mean)

        mse = torch.nn.functional.mse_loss(x,y)
        mae = torch.nn.functional.l1_loss(x,y)
        msea = torch.add(mse,mae)

        l3  = torch.nn.functional.mse_loss(x ** 3, y ** 3)

        bin = torch.nn.functional.l1_loss(torch.nn.functional.threshold(-torch.nn.functional.threshold(x,0.,-1.),0.,0.),torch.nn.functional.threshold(-torch.nn.functional.threshold(y,0.,-1.),0.,0.))

        loss = torch.add(l3*msea,mstd)
        return torch.add(loss,loss*bin)

class MSTD_MSEA(torch.nn.Module):
    def __init__(self):
        super(MSTD_MSEA,self).__init__()

    def forward(self,x,y):
        std = (torch.std(x) - torch.std(y)) ** 2
        mean = (torch.mean(x) - torch.mean(y)) ** 2
        mstd = torch.add(std,mean)
        mse = torch.nn.functional.mse_loss(x,y)
        mae = torch.nn.functional.l1_loss(x,y)
        mme = torch.add(mse,mae)
        return torch.add(mstd,mme)

class B_MSTD_MsaE(torch.nn.Module):
    def __init__(self):
        super(B_MSTD_MsaE,self).__init__()

    def forward(self,x,y):
        std = (torch.std(x) - torch.std(y)) ** 2
        mean = (torch.mean(x) - torch.mean(y)) ** 2
        mstd = torch.add(std,mean)
        mse = torch.nn.functional.mse_loss(x,y)
        mae = torch.nn.functional.l1_loss(x,y)
        msae = torch.add(mse,mae)
        bin = torch.nn.functional.l1_loss(torch.nn.functional.threshold(-torch.nn.functional.threshold(x,0.,-1.),0.,0.),torch.nn.functional.threshold(-torch.nn.functional.threshold(y,0.,-1.),0.,0.))
        mme = torch.add(mstd,msae)
        return torch.add(mme,mme*bin) #msea*bin to prioritise the correct directionality over raw accuracy

def learn(out,target, iter=0):
    #simulates learning
    ret = torch.add(out,target) / 2
    for _ in range(iter):
        ret = torch.add(ret,target) / 2
    return ret


if __name__ == '__main__':
    out = torch.rand(10, 10, 10)
    target = torch.rand(10, 10, 10)
    ones = torch.ones(10, 10, 10) * .5
    zero = torch.zeros(10, 10, 10)

    crits = [
             torch.nn.MSELoss(),
             torch.nn.L1Loss()
             ]

    preds = [out, zero, ones, learn(out, target, iter=0), learn(out, target, iter=1), learn(out, target, iter=2),
             target]
    preds0 = [out, zero, ones, learn(zero, target, iter=0), learn(zero, target, iter=1), learn(zero, target, iter=2),
              target]

    loss = []
    for crit in crits:
        loss.append([crit(each, target) for each in preds])
    for crit in crits:
        loss.append([crit(each, target) for each in preds0])
    for each in loss:
        print(each)

    from matplotlib import pyplot as plt
    def learn(loss):
        x=[]
        y=[]
        pred = torch.rand(10000)
        targ = torch.rand(10000)
        los = loss()
        lr = 0.05
        for i in range(100):
            y.append(los(pred,targ).item())
            x.append(i)
            pred = torch.add(pred,targ*lr)/(1+lr)
        plt.plot(x,y)

    def learn_mean(loss):
        x=[]
        y=[]
        pred = torch.rand(10000)
        targ = torch.rand(10000)
        mm = torch.zeros(10000)
        los = loss()
        lr = 0.05
        for i in range(100):
            y.append(los(pred,targ).item())
            x.append(i)
            pred = torch.add(pred,targ*lr)/(1+lr)
            mm = torch.add(mm,targ*lr)/(1+lr)
            pred = torch.add(mm,pred*lr)/(1+lr)
        plt.plot(x,y)

    fig1= plt.figure()
    learn(torch.nn.MSELoss)
    learn(ls.ex_L3)
    fig2 = plt.figure()
    learn_mean(torch.nn.MSELoss)
    learn_mean(ls.ex_L3)